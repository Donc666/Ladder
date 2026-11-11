
from dataclasses import dataclass
from typing import Tuple
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


@dataclass
class AssetCfg:
    """物体资产参数类：包含导入和材质应用所需的所有数据。"""
    # 基础信息
    name: str
    urdf_path: str
    pos: Tuple[float, float, float]
    rot: Tuple[float, float, float, float]
    scale: Tuple[float, float, float]
    fix_base:bool = True 
    # visual_material_cfg
    diffuse_color: Tuple[float, float, float] = None
    roughness: float = None
    metallic: float = None
    opacity: float = None
    
    # physical_material_cfg
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    compliant_contact_stiffness: float = 0.0
    compliant_contact_damping: float = 0.0

    # mdl_cfg
    use_mdl: bool = False  # 标志位：是否使用 MDL 材质
    mdl_path: str = "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl"
    texture_scale: Tuple[float, float] = (1.0, 1.0)
    albedo_brightness: float = 1.0

    target_links: list[str] = None

class SceneBuilder:
    def __init__(self, assets: list[AssetCfg], base_scene_cfg: InteractiveSceneCfg = None):
        """
        初始化时即构建整个场景。
        
        Args:
            assets: 包含所有要导入物体的参数类列表
            base_scene_cfg: 基础场景配置（如包含机器人、地面等），若为 None 则新建
        """
        # 1. 准备场景配置
        self.cfg = base_scene_cfg if base_scene_cfg is not None else InteractiveSceneCfg()
        self.assets = assets

        # 2. 动态注册所有物体配置
        for asset in self.assets:
            # 调用之前的静态方法生成 ArticulationCfg
            articulation_cfg = self.create_articulation_cfg(asset)
            # 动态绑定到配置类变量上
            setattr(self.cfg, asset.name, articulation_cfg)

        # 3. 【关键步】实例化场景（此时物体被真正导入到物理世界）
        self.scene = InteractiveScene(self.cfg)

        # 4. 场景创建后，立即绑定物理材质
        self._initialize_physics_materials()

    def _initialize_physics_materials(self):
        """遍历所有资产并应用物理材质绑定。"""
        # 注意：InteractiveScene 创建后，可以通过 self.scene.num_envs 获取并行环境数量
        for i in range(self.scene.num_envs):
            for asset in self.assets:
                self.apply_physical_material(asset, env_index=i)
        print(f"Successfully initialized {len(self.assets)} assets with physical materials.")

    @staticmethod
    def create_articulation_cfg(asset: AssetCfg) -> ArticulationCfg:
        """生成配置（逻辑同前，填补所有 MISSING 字段）"""

        if asset.use_mdl:
            # 使用 MDL 材质配置
            visual_material_cfg = sim_utils.MdlFileCfg(
                mdl_path=asset.mdl_path,
                texture_scale=asset.texture_scale,
                albedo_brightness=asset.albedo_brightness,
                project_uvw=True
            )
        else:
            # 回退到普通的预览表面配置
            visual_material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=asset.diffuse_color,
                roughness=asset.roughness,
                metallic=asset.metallic,
                opacity=asset.opacity,
            )
        return ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{asset.name}",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=asset.urdf_path,
                scale=asset.scale,
                fix_base=asset.fix_base,
                visual_material=visual_material_cfg,
                self_collision = False,
                collision_props = sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    # 缩小缓冲层，防止微小间隙触发巨大排斥力
                    contact_offset=0.05,  # 5毫米的检测缓冲
                    # 允许轻微的物理重叠，这能让静止状态更稳定
                    rest_offset=-0.1,       # 或者设为 -0.001 
                ),
                
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # physics_material=my_friction_mtl,
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=asset.pos, rot=asset.rot),
            actuators={
                f"{asset.name}_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=0.0, damping=0.0
                ),
            },
        )

    @staticmethod
    def apply_physical_material(asset: AssetCfg, env_index: int = 0):
        """执行物理材质绑定（逻辑同前）"""
        stage = get_current_stage()
        env_ns = f"/World/envs/env_{env_index}"
        prim_path = f"{env_ns}/{asset.name}"
        material_path = f"{prim_path}/physics_material"

        rigid_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=asset.static_friction,
            dynamic_friction=asset.dynamic_friction,
            restitution=asset.restitution,
            compliant_contact_stiffness=asset.compliant_contact_stiffness,
            compliant_contact_damping=asset.compliant_contact_damping,
        )
        # 创建并绑定
        rigid_material_cfg.func(material_path, rigid_material_cfg)
        bind_physics_material(prim_path, material_path, stage=stage)