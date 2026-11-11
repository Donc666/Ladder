from my_utils import *
from set_interface_try import *
import random
import xml.etree.ElementTree as ET

# Real window yaw. The concept URDF's fixed joint is only Rx(90); the PartNet
# window is Rx(90) then Rz(-90), so the concept root is yawed 90 deg less to
# make the glass face the same way.
random_degree2 = np.random.uniform(-20, 20)
random_degree = 90.0 + random_degree2
# Concept frame origin is z=0.715 in the URDF. At scale 0.35 that lifts the
# mesh 0.25 m above the root, so the concept root is spawned that much lower.
_CONCEPT_FRAME_LIFT = 0.715 * 0.35
WINDOW_SPAWN_XYZ = (
    float(np.random.uniform(0.40, 0.55)),
    float(np.random.uniform(0.70, 0.78)),
    0.70,
)
# Park XY far from the arms. Height stays tied to the real window so the
# handle mapping still copies the right z.
CONCEPT_SPAWN_XYZ = (
    1000.0,
    1000.0,
    WINDOW_SPAWN_XYZ[2] - _CONCEPT_FRAME_LIFT,
)
CONCEPT_URDF_PATH = None
CONCEPT_URDF_FALLBACK = str(
    PARTNET_CONCEPT_ROOT / "Window" / "d62e7ca4-856b-4fcd-a578-ff2c58d9c6f9" / "Window.urdf"
)


def _safe_default_joint_pos(urdf_path: str) -> dict[str, float]:
    pos: dict[str, float] = {}
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return pos
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "prismatic"}:
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
        if lo - 1e-6 <= 0.0 <= hi + 1e-6:
            continue
        pos[name] = float(min(hi, max(lo, 0.0)))
    return pos


class window_concept:
    def __init__(self):
        self.scale = (0.35, 0.35, 0.35)

    def import2sim(self, urdf_path: str, instance_name: str, num_envs):
        global CONCEPT_URDF_PATH
        self.num_envs = num_envs
        self.random_degree = float(random_degree2)
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=urdf_path,
            concept_root=str(PARTNET_CONCEPT_ROOT),
            category="Window",
            allow_fallback=False,
        )
        if resolved_concept_urdf is None:
            print(f"[WARN] model_id={model_id} has no paired Window concept, fallback: {CONCEPT_URDF_FALLBACK}")
            resolved_concept_urdf = CONCEPT_URDF_FALLBACK
        CONCEPT_URDF_PATH = resolved_concept_urdf
        self.concept_urdf_path = resolved_concept_urdf
        r_z = R.from_euler("z", self.random_degree, degrees=True)
        rot_concept = xyzw_to_wxyz(r_z.as_quat())
        cx, cy, cz = CONCEPT_SPAWN_XYZ
        print(
            f"[window_concept] paired concept for {urdf_path} -> {resolved_concept_urdf} "
            f"pos=({cx:.2f},{cy:.2f},{cz:.2f}) yaw={self.random_degree:.1f} "
            f"(real yaw={float(random_degree):.1f})"
        )
        concept_cfg = AssetCfg(
            name="my_concept_window",
            urdf_path=resolved_concept_urdf,
            pos=(cx, cy, cz),
            rot=rot_concept,
            scale=self.scale,
            asset_kind="articulation",
            fix_base=True,
            stiffness=0.0,
            damping=1.0,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
            joint_pos=_safe_default_joint_pos(resolved_concept_urdf),
        )
        return [concept_cfg]
