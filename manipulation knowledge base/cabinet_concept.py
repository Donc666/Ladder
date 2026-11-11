from my_utils import *
from set_interface_try import *
import random
import xml.etree.ElementTree as ET

random_degree = 10.0
# Shared with the PartNet cabinet so the concept can sit beside it.
CABINET_SPAWN_XYZ = (
    random.uniform(0.80, 0.95),
    random.uniform(-0.20, -0.10),
    0.58,
)


def _concept_frame_lift(urdf_path: str, scale_z: float) -> float:
    """URDF frame origin z, scaled. PartNet cabinets have no such lift."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return 0.0
    for joint in root.findall("joint"):
        if not (joint.attrib.get("name") or "").startswith("world_to_"):
            continue
        origin = joint.find("origin")
        if origin is None or not origin.attrib.get("xyz"):
            return 0.0
        try:
            z = float(origin.attrib["xyz"].split()[2])
        except (TypeError, ValueError):
            return 0.0
        return z * float(scale_z)
    return 0.0


def closed_drawer_joint_pos(urdf_path: str) -> dict[str, float]:
    """Put every prismatic drawer at the limit nearer to 0, which is closed."""
    pos: dict[str, float] = {}
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return pos
    for joint in root.findall("joint"):
        if (joint.attrib.get("type") or "").lower() != "prismatic":
            continue
        name = joint.attrib.get("name")
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
        pos[name] = lo if abs(lo) <= abs(hi) else hi
    return pos
class cabinet_concept ():
    def __init__(self,):
        self.scale = (0.3,0.3,0.3)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs 
    ):
        self.num_envs = num_envs
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "Cabinet"
        concept_urdf_fallback= str(PARTNET_CONCEPT_ROOT / "Cabinet" / "5ff7e1be775072e5824662341ce2b233" / "Cabinet.urdf")


        self.random_degree = random_degree
        self.scene_key = instance_name
        micwave_cfg = AssetCfg(
            fix_base=True,
            name=self.scene_key,
            urdf_path=urdf_path,
            pos=CABINET_SPAWN_XYZ,
            rot=(1,0,0,0),
            scale=self.scale,
            asset_kind = "articulation",
            stiffness=0.0,
            damping=0.0,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
        )
        # 3) concept urdf 
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=micwave_cfg.urdf_path,
            concept_root=concept_root,
            category=category,
        )
        if resolved_concept_urdf is None:
            print(f"[WARN] model_id={model_id} not found, fallback: {concept_urdf_fallback}")
            resolved_concept_urdf = concept_urdf_fallback
        else:
            print(f"[INFO] model_id={model_id}, concept URDF: {resolved_concept_urdf}")
        #concept urdf will be used in the atomic task
        self.concept_urdf_path = resolved_concept_urdf
        # PartNet root is Rx(90) then Rz(-90). The concept root is only Rx(90)
        # and is lifted by the frame origin, so yaw it back 90 deg and drop it.
        z = CABINET_SPAWN_XYZ[2]
        lift = _concept_frame_lift(resolved_concept_urdf, self.scale[2])
        # Keep the height that the handle mapping expects. Park XY far from the arms.
        concept_pos = (1000.0, 1000.0, z - lift)
        concept_yaw = float(self.random_degree) - 90.0
        rot_concept = xyzw_to_wxyz(R.from_euler("z", concept_yaw, degrees=True).as_quat())
        print(
            f"[cabinet_concept] parked away from the cabinet at {tuple(round(v, 3) for v in concept_pos)} "
            f"yaw={concept_yaw:.1f} frame_lift={lift:.3f}"
        )
        concept_laptop_cfg = AssetCfg(
            name="my_concept_cabinet",
            urdf_path=resolved_concept_urdf,
            pos=concept_pos,
            rot=rot_concept,
            scale=self.scale,
            asset_kind = "articulation",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
            joint_pos=closed_drawer_joint_pos(resolved_concept_urdf),
        )
        return  [concept_laptop_cfg]



 