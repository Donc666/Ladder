from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Tuple, Literal, Optional, Any
import dataclasses
import inspect
import re
import xml.etree.ElementTree as ET
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from pxr import Sdf, Usd, UsdGeom, UsdShade


def _accepted_cfg_fields(cls) -> set[str]:
    names: set[str] = set()
    try:
        names.update(f.name for f in fields(cls))
    except TypeError:
        pass
    try:
        names.update(inspect.signature(cls).parameters.keys())
    except (TypeError, ValueError):
        pass
    names.discard("self")
    names.discard("args")
    names.discard("kwargs")
    return names


def _urdf_has_movable_joints(urdf_path: str | None) -> bool:
    if not urdf_path:
        return False
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return False
    for joint in root.findall("joint"):
        if (joint.attrib.get("type") or "").lower() in {"revolute", "prismatic", "continuous"}:
            return True
    return False


def _urdf_joint_limits(urdf_path: str | None) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    if not urdf_path:
        return limits
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return limits
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "prismatic"}:
            continue
        name = joint.attrib.get("name") or ""
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        try:
            lo = float(limit.attrib.get("lower", "0"))
            hi = float(limit.attrib.get("upper", "0"))
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        limits[name] = (lo, hi)
    return limits


def _clamp_joint_pos_to_urdf(urdf_path: str | None, joint_pos: dict[str, float]) -> dict[str, float]:
    """Keep init joint_pos inside URDF limits so Articulation._validate_cfg does not crash."""
    if not joint_pos:
        return {}
    limits = _urdf_joint_limits(urdf_path)
    if not limits:
        return dict(joint_pos)
    out: dict[str, float] = {}
    for key, val in joint_pos.items():
        try:
            q = float(val)
        except (TypeError, ValueError):
            continue
        matched = [name for name in limits if re.fullmatch(key, name)]
        if not matched:
            out[key] = q
            continue
        for name in matched:
            lo, hi = limits[name]
            clamped = min(hi, max(lo, q))
            if abs(clamped - q) > 1e-9:
                print(
                    f"[SceneBuilder] clamp joint {name}: {q:.4f} -> {clamped:.4f} "
                    f"(limits [{lo:.4f}, {hi:.4f}])"
                )
            out[name] = clamped
    return out


def _make_urdf_file_cfg(**kwargs: Any) -> sim_utils.UrdfFileCfg:
    """Build UrdfFileCfg, dropping fields the installed Isaac Lab version does not accept.

    Local Isaac Lab was patched with `physical_material`; stock server builds reject it.
    """
    accepted = _accepted_cfg_fields(sim_utils.UrdfFileCfg)
    if "physical_material" in kwargs and "physical_material" not in accepted:
        if "physics_material" in accepted and "physics_material" not in kwargs:
            kwargs["physics_material"] = kwargs["physical_material"]
        kwargs.pop("physical_material", None)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[SceneBuilder] UrdfFileCfg ignored unsupported fields: {dropped}")
    return sim_utils.UrdfFileCfg(**filtered)


def bind_diffuse_texture(prim_path: str, texture_path: str, roughness: float = 0.55) -> bool:
    """Override a prim's look with a repeating albedo texture."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    # Unique look prim so a later random texture does not reuse the previous file input.
    stem = Path(texture_path).stem.replace(" ", "_")
    material_path = f"{prim_path}/Looks/Albedo_{stem}"
    mat = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    st = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarST")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    targets = [prim]
    for child in Usd.PrimRange(prim):
        if child.GetPath() == prim.GetPath():
            continue
        if "Looks" in str(child.GetPath()):
            continue
        if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Gprim):
            targets.append(child)
    ok = False
    for target in targets:
        try:
            ok = bool(
                bind_visual_material(
                    str(target.GetPath()),
                    material_path,
                    stage=stage,
                    stronger_than_descendants=True,
                )
            ) or ok
        except ValueError:
            continue
    return ok


@dataclass
class AssetCfg:
    """物体资产参数类：包含导入和材质应用所需的所有数据。"""
    # 基础信息
    name: str
    pos: Tuple[float, float, float]
    rot: Tuple[float, float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass : float = 0.1
    # kg/m^3 used when the URDF link has no inertial mass. 0 keeps the importer default.
    link_density: float = 0.0
    disable_gravity : bool =False 

    # 资产来源: urdf / cuboid（方块）/ usd（已带碰撞的 USD）
    source_kind: Literal["urdf", "cuboid", "usd"] = "urdf"
    urdf_path: Optional[str] = None
    usd_path: Optional[str] = None
    cuboid_size: Optional[Tuple[float, float, float]] = None
    cuboid_color: Tuple[float, float, float] = (1.0, 0.5, 0.2)

    fix_base: bool = True
    asset_kind: Literal["auto", "articulation", "rigid"] = "auto"
    # convex_hull fills cavities (microwave oven). Use convex_decomposition for hollow shells.
    collider_type: Literal["convex_hull", "convex_decomposition"] = "convex_hull"

    # visual_material_cfg
    diffuse_color: Optional[Tuple[float, float, float]] = None
    roughness: Optional[float] = None
    metallic: Optional[float] = None
    opacity: Optional[float] = None

    # articulation drive
    stiffness: float = 0.0
    damping: float = 0.0

    # physical_material_cfg
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    compliant_contact_stiffness: float = 0.0
    compliant_contact_damping: float = 0.0

    # mdl_cfg
    use_mdl: bool = False
    mdl_path: str = "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl"
    texture_scale: Tuple[float, float] = (1.0, 1.0)
    albedo_brightness: float = 1.0
    # Local albedo jpg/png. Bound after spawn and overrides URDF materials.
    texture_file: Optional[str] = None

    target_links: list[str] = field(default_factory=list)
    prim_path: str = None
    joint_pos: dict[str, float] | None = field(default_factory=dict)
    force_usd_conversion: Optional[bool] = None


class SceneBuilder:
    def __init__(
        self,
        assets: list[list[AssetCfg]],
        base_scene_cfg: InteractiveSceneCfg = None,
        workspace_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.cfg = base_scene_cfg if base_scene_cfg is not None else InteractiveSceneCfg()
        self.assets_2d = assets
        self.asset_key_map = {}
        ox, oy, oz = workspace_offset

        
        for env_id, env_assets in enumerate(self.assets_2d):
            for asset in env_assets:
                # 1) 唯一 key（防 setattr 覆盖）
                scene_key = f"{asset.name}__env{env_id}"
                # 2) 强制绑到对应环境
                prim_path = f"/World/envs/env_{env_id}/{asset.name}"
                # 3) 复制一份 asset（避免改原对象），并套上工作空间偏移
                px, py, pz = asset.pos
                a = dataclasses.replace(
                    asset,
                    name=scene_key,
                    prim_path=prim_path,
                    pos=(px + ox, py + oy, pz + oz),
                )
                kind = a.asset_kind
                if kind == "auto":
                    kind = "articulation" if (a.source_kind == "urdf" and "mobility.urdf" in (a.urdf_path or "").lower()) else "rigid"
                cfg = self.create_articulation_cfg(a) if kind == "articulation" else self.create_rigid_cfg(a)
                setattr(self.cfg, scene_key, cfg)
                self.asset_key_map[(asset.name, env_id)] = scene_key
            # kind = asset.asset_kind
            # if kind == "auto":
            #     print("i am in the auto")
            #     # cuboid 必然 rigid；URDF 根据路径启发式判定
            #     if asset.source_kind == "cuboid":
            #         kind = "rigid"
            #     else:
            #         urdf = (asset.urdf_path or "").lower()
            #         kind = "articulation" if "mobility.urdf" in urdf else "rigid"

            # if kind == "articulation":
            #     print("i am in the articulation")
            #     cfg = self.create_articulation_cfg(asset)
            # else:
            #     print("i am in the rigid")
            #     cfg = self.create_rigid_cfg(asset)

            # setattr(self.cfg, asset.name, cfg)

        self.scene = InteractiveScene(self.cfg)
        self.apply_visual_textures()

    def apply_visual_textures(self) -> None:
        """Bind albedo overrides. Call again after sim.reset(), which restores URDF looks."""
        stage = get_current_stage()
        for env_id, env_assets in enumerate(self.assets_2d):
            for asset in env_assets:
                tex = getattr(asset, "texture_file", None)
                if not tex:
                    continue
                prim_path = f"/World/envs/env_{env_id}/{asset.name}"
                if not stage.GetPrimAtPath(prim_path).IsValid():
                    print(f"[SceneBuilder] skip texture, missing prim {prim_path}")
                    continue
                if bind_diffuse_texture(
                    prim_path,
                    tex,
                    roughness=asset.roughness if asset.roughness is not None else 0.55,
                ):
                    print(f"[SceneBuilder] bound texture {Path(tex).name} -> {prim_path}")

    def _initialize_physics_materials(self):
        for i in range(self.scene.num_envs):
            for asset in self.assets:
                self.apply_physical_material(asset, env_index=i)
        print(f"Successfully initialized {len(self.assets)} assets with physical materials.")

    @staticmethod
    def create_articulation_cfg(asset: AssetCfg) -> ArticulationCfg:
        print("i am already in the articulation ")
        if asset.source_kind != "urdf":
            raise ValueError(f"{asset.name}: articulation only supports source_kind='urdf'")
        if not asset.urdf_path:
            raise ValueError(f"{asset.name}: urdf_path is required for articulation asset")

        if asset.use_mdl:
            _visual_material_cfg = sim_utils.MdlFileCfg(
                mdl_path=asset.mdl_path,
                texture_scale=asset.texture_scale,
                albedo_brightness=asset.albedo_brightness,
                project_uvw=True,
            )
        else:
            _visual_material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=asset.diffuse_color,
                roughness=asset.roughness,
                metallic=asset.metallic,
                opacity=asset.opacity,
            )

        spawn_kwargs = dict(
            asset_path=asset.urdf_path,
            scale=asset.scale,
            fix_base=asset.fix_base,
            collider_type=asset.collider_type,
            # Instanceable USDs can yield a 0-DOF PhysX view ("Available strings: []").
            make_instanceable=False,
            force_usd_conversion=True if asset.force_usd_conversion is None else bool(asset.force_usd_conversion),
            physical_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=asset.static_friction,
                dynamic_friction=asset.dynamic_friction,
                restitution=asset.restitution,
                compliant_contact_stiffness=asset.compliant_contact_stiffness,
                compliant_contact_damping=asset.compliant_contact_damping,
            ),
            self_collision=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=asset.disable_gravity,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            link_density=asset.link_density,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=asset.stiffness,
                    damping=asset.damping,
                )
            ),
        )
        if asset.diffuse_color is not None:
            spawn_kwargs["visual_material"] = _visual_material_cfg

        joint_pos = _clamp_joint_pos_to_urdf(asset.urdf_path, dict(asset.joint_pos or {}))
        has_joints = bool(joint_pos) or _urdf_has_movable_joints(asset.urdf_path)
        if not has_joints:
            spawn_kwargs["joint_drive"] = None
            print(f"[SceneBuilder] {asset.name} URDF has no movable joints; skip .* actuators")
        actuators = {}
        if has_joints:
            actuators = {
                f"{asset.name}_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=asset.stiffness,
                    damping=asset.damping,
                ),
            }
        print(f"[SceneBuilder] articulation {asset.name} urdf={asset.urdf_path} joint_pos={list(joint_pos)}")
        return ArticulationCfg(
            prim_path = asset.prim_path if getattr(asset, "prim_path", None) else f"{{ENV_REGEX_NS}}/{asset.name}",
            spawn=_make_urdf_file_cfg(**spawn_kwargs),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=asset.pos,
                rot=asset.rot,
                joint_pos=joint_pos,
                joint_vel={},
            ),
            actuators=actuators,
        )

    @staticmethod
    def create_rigid_cfg(asset: AssetCfg) -> RigidObjectCfg:
        # 允许 URDF 或 cuboid 两种 rigid 来源
        if asset.source_kind == "cuboid":
            if asset.cuboid_size is None:
                raise ValueError(f"{asset.name}: cuboid_size is required when source_kind='cuboid'")
            spawn_cfg = sim_utils.CuboidCfg(
                size=asset.cuboid_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=asset.cuboid_color,
                    roughness=0.2,
                    metallic=asset.metallic,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
        elif asset.source_kind == "usd":
            if not asset.usd_path:
                raise ValueError(f"{asset.name}: usd_path is required when source_kind='usd'")
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=asset.usd_path,
                scale=asset.scale,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=asset.disable_gravity,
                    kinematic_enabled=asset.fix_base,
                    retain_accelerations=False,
                    linear_damping=2.0,
                    angular_damping=4.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=asset.mass),
            )
        else:
            if not asset.urdf_path:
                raise ValueError(f"{asset.name}: urdf_path is required when source_kind='urdf'")
            spawn_cfg = _make_urdf_file_cfg(
                asset_path=asset.urdf_path,
                scale=asset.scale,
                fix_base=asset.fix_base,
                collider_type=asset.collider_type,
                force_usd_conversion=asset.collider_type != "convex_hull",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=asset.disable_gravity,
                    retain_accelerations=False,
                    linear_damping=2.0,
                    angular_damping=4.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=asset.mass),
                physical_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=asset.static_friction,
                    dynamic_friction=asset.dynamic_friction,
                    restitution=asset.restitution,
                    compliant_contact_stiffness=asset.compliant_contact_stiffness,
                    compliant_contact_damping=asset.compliant_contact_damping,
                ),
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=asset.stiffness,
                        damping=asset.damping,
                    )
                ),
            )

        return RigidObjectCfg(
            prim_path = asset.prim_path if getattr(asset, "prim_path", None) else f"{{ENV_REGEX_NS}}/{asset.name}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=asset.pos,
                rot=asset.rot,
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )

    @staticmethod
    def apply_physical_material(asset: AssetCfg, env_index: int = 0):
        stage = get_current_stage()
        env_ns = f"/World/envs/env_{env_index}"
        root_prim_path = f"{env_ns}/{asset.name}"
        material_path = f"{root_prim_path}/physics_material"

        rigid_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=asset.static_friction,
            dynamic_friction=asset.dynamic_friction,
            restitution=asset.restitution,
            compliant_contact_stiffness=asset.compliant_contact_stiffness,
            compliant_contact_damping=asset.compliant_contact_damping,
        )
        rigid_material_cfg.func(material_path, rigid_material_cfg)

        for link_name in (asset.target_links or []):
            link_prim_path = f"{root_prim_path}/{link_name}"
            if stage.GetPrimAtPath(link_prim_path).IsValid():
                success = bind_physics_material(link_prim_path, material_path, stage=stage)
                if success:
                    print(f"[INFO] env_{env_index} {asset.name}.{link_name} material bound.")
            else:
                print(f"[WARNING] link not found: {link_prim_path}")