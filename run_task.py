# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""
# import warp as wp
# wp.init()
import argparse
import re
import time
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent

from isaaclab.app import AppLauncher
# from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
# from curobo.types import ContentPath, GoalToolPose, JointState, Pose
# from curobo._src.geom.collision import SceneCollisionCfg
# from curobo._src.geom.types import SceneCfg,Cuboid,Obstacle,Mesh,PointCloud,DeviceCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--anti_interference",
    dest="anti_interference",
    action="store_true",
    help="Data-collection anti-interference: drop test and live end-effector pose, together.",
)
parser.add_argument(
    "--no_anti_interference",
    dest="anti_interference",
    action="store_false",
    help="Turn off drop test and live end-effector pose together.",
)
parser.set_defaults(anti_interference=False)
parser.add_argument(
    "--record",
    action="store_true",
    help="Record the parquet dataset and export episode videos. Off by default.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=0,
    help="Stop after this many simulation frames and save the video. 0 keeps running.",
)
parser.add_argument(
    "--max_seconds",
    type=int,
    default=0,
    help="Stop this run after this many seconds from process start, then save. 0 keeps running.",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="",
    help="LeRobot folder for env_0, the environment with green markers. Default: data/.",
)
parser.add_argument(
    "--dataset_root_env1",
    type=str,
    default="",
    help="LeRobot folder for env_1, the environment without markers. Default: the same folder as env_0.",
)
parser.add_argument(
    "--episode_index",
    type=int,
    default=0,
    help="Episode number written for this run. env_1 uses the same number when it has its own folder.",
)
parser.add_argument(
    "--preview_dir",
    type=str,
    default="",
    help="If set, save the last third-person frame of each env here.",
)
parser.add_argument("--urdf_path", type=str, default="", help="Optional URDF path override.")
parser.add_argument(
    "--task",
    type=str,
    default="",
    help="Task name. Loads tasks/wooden_box, tasks/rigid, or tasks/articulated/<name>.json. Accepts a name with or without .json, or a json path. Default: tasks/rigid/task_plan.json.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# import warp as wp
# wp.init()

"""Rest everything follows."""
# import sys
# import os

# # 强制将安装了 cuRobo 的路径添加到 Python 的搜索路径中
# curobo_root = "/home/jiacheng/curobo"
# if curobo_root not in sys.path:
#     sys.path.append(curobo_root)
# import warp as wp
import torch
import random
import imageio.v2 as imageio
import numpy as np 
import open3d as o3d
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.geom.types import SceneCfg, Mesh,Cuboid
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.core.utils.bounds import compute_combined_aabb
from scipy.spatial.transform import Rotation as R
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Gf
try:
    import omni.physx.scripts.utils as physx_utils
except Exception:  # pragma: no cover - Isaac runtime only
    physx_utils = None
import kb_path  # noqa: F401
from set_interface_try import AssetCfg, SceneBuilder
from microwave import *
from dispenser import *
from auto_collect_data import *
from collect_data_utils import *
from collect_data_utils import _copy_camera_rgb
from task_executor import *
from runtime.world_state import WorldState
from runtime.atom_compiler import AtomCompiler
from runtime.grasp_monitor import GraspMonitorCfg
from runtime.task_runtime import TaskRuntime, TaskRuntimeCfg
from runtime.scene_markers import ScenePlanMarkers

# Star-imports (my_utils via auto_collect_data) must not replace this with a .py path.
PROJECT_ROOT = Path(__file__).resolve().parent


# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
# from curobo.geom.types import WorldConfig

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

VERBOSE = False
HANDLE_LINK_CANDIDATES = [
    "Trifold_Handle",
    "Cuboidal_Handle",
    "Trifold_Curve_Handle",
    "Curve_Handle",
]
DOOR_LINK_CANDIDATES = ["Sunken_Door", "Cuboidal_Door"]


def log_debug(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _build_link_candidates(target_link: str):
    if target_link in HANDLE_LINK_CANDIDATES:
        return [target_link] + [x for x in HANDLE_LINK_CANDIDATES if x != target_link]
    if target_link in DOOR_LINK_CANDIDATES:
        return [target_link] + [x for x in DOOR_LINK_CANDIDATES if x != target_link]
    return [target_link]


def map_concept_point_to_real_xy(
    point_xyz: np.ndarray, concept_root_pos_w: np.ndarray, real_root_pos_w: np.ndarray
) -> np.ndarray:
    """
    Map a point from concept-world frame to real-world frame using XY translation only.
    Assumption (as discussed): same axes, same yaw, no fixed frame bias, only XY offset.
    """
    mapped = np.array(point_xyz, dtype=np.float64).copy()
    mapped[0] += float(real_root_pos_w[0] - concept_root_pos_w[0])
    mapped[1] += float(real_root_pos_w[1] - concept_root_pos_w[1])
    return mapped


def _parse_xyz_attr(xyz_text: str):
    values = [float(v) for v in xyz_text.strip().split()]
    if len(values) != 3:
        raise ValueError(f"invalid xyz string: {xyz_text}")
    return np.array(values, dtype=np.float64)

def _urdf_xyz_to_sim_yzx(vec_xyz: np.ndarray) -> np.ndarray:
    """Convert URDF xyz ordering to simulator yzx ordering."""
    v = np.asarray(vec_xyz, dtype=np.float64)
    return np.array([v[2], v[0], v[1]], dtype=np.float64)

def get_door_to_handle_offset_from_urdf(urdf_path: str):
    """
    Parse fixed/revolute/prismatic joint origins and return the first matched
    door->handle offset in URDF local frame.
    """
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        log_debug(f"failed to parse urdf for offset: {urdf_path}, err={exc}")
        return None, None, None

    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        origin_elem = joint.find("origin")
        if parent_elem is None or child_elem is None or origin_elem is None:
            continue
        parent_name = parent_elem.attrib.get("link", "")
        child_name = child_elem.attrib.get("link", "")
        if parent_name not in DOOR_LINK_CANDIDATES or child_name not in HANDLE_LINK_CANDIDATES:
            continue
        xyz_text = origin_elem.attrib.get("xyz")
        if not xyz_text:
            continue
        try:
            offset = _parse_xyz_attr(xyz_text)
        except ValueError:
            continue
        return offset, parent_name, child_name

    return None, None, None


def get_door_hinge_from_urdf(urdf_path: str):
    """
    Parse URDF and return hinge definition for the first matched door joint:
    - child link must be a door candidate
    - parent link can be any link (e.g. Cuboidal_Body/world)
    Returns: (origin_xyz, axis_xyz, parent_link, child_link) in parent frame.
    """
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None, None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        log_debug(f"failed to parse urdf for door hinge: {urdf_path}, err={exc}")
        return None, None, None, None

    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")
        if parent_elem is None or child_elem is None or origin_elem is None:
            continue
        child_name = child_elem.attrib.get("link", "")
        if child_name not in DOOR_LINK_CANDIDATES:
            continue
        parent_name = parent_elem.attrib.get("link", "")
        xyz_text = origin_elem.attrib.get("xyz")
        if not xyz_text:
            continue
        try:
            origin_xyz = _parse_xyz_attr(xyz_text)
        except ValueError:
            continue
        # default axis for revolute door if missing
        axis_xyz = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if axis_elem is not None and axis_elem.attrib.get("xyz"):
            try:
                axis_xyz = _parse_xyz_attr(axis_elem.attrib.get("xyz"))
                axis_xyz = _urdf_xyz_to_sim_yzx(axis_xyz)
            except ValueError:
                pass
        norm = np.linalg.norm(axis_xyz)
        if norm > 1e-8:
            axis_xyz = axis_xyz / norm
        return origin_xyz, axis_xyz, parent_name, child_name
    return None, None, None, None


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def transform_joint_axis_to_world(parent_pos_w, parent_quat_wxyz, joint_origin_parent, axis_parent):
    """
    Convert joint origin/axis from parent frame to world frame.
    Returns (axis_point_world, axis_dir_world).
    """
    rot_parent = _quat_wxyz_to_rotmat(np.asarray(parent_quat_wxyz, dtype=np.float64))
    p_parent = np.asarray(parent_pos_w, dtype=np.float64)
    axis_point_w = p_parent + rot_parent @ np.asarray(joint_origin_parent, dtype=np.float64)
    axis_dir_w = rot_parent @ np.asarray(axis_parent, dtype=np.float64)
    norm = np.linalg.norm(axis_dir_w)
    if norm > 1e-8:
        axis_dir_w = axis_dir_w / norm
    return axis_point_w, axis_dir_w


def extract_model_id_from_urdf(urdf_path: str) -> str | None:
    """Read URDF robot name and derive model_id."""
    path = Path(urdf_path)
    if not path.exists():
        print(f"[WARN] URDF 不存在: {urdf_path}")
        return None
    try:
        # Robot name is usually in first few lines.
        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:10])
    except OSError as exc:
        print(f"[WARN] 读取 URDF 失败: {urdf_path}, error={exc}")
        return None

    match = re.search(r'<robot\s+name="([^"]+)"', head)
    if not match:
        print(f"[WARN] 未在 URDF 中解析到 robot name: {urdf_path}")
        return None

    robot_name = match.group(1).strip()
    if robot_name.startswith("partnet_"):
        model_id = robot_name[len("partnet_") :]
    else:
        # Fallback: pick id-like token from robot_name, otherwise use full name.
        id_match = re.search(r"([0-9a-fA-F]{8,})", robot_name)
        model_id = id_match.group(1) if id_match else robot_name
    return model_id


def resolve_concept_urdf_path(
    source_urdf_path: str,
    concept_root: str = str(PARTNET_CONCEPT_ROOT),
    category: str = "Refrigerator",
) -> tuple[str | None, str | None]:
    """Resolve concept URDF by model_id parsed from source URDF."""
    model_id = extract_model_id_from_urdf(source_urdf_path)
    if model_id is None:
        return None, None

    base_dir = Path(concept_root) / category / model_id
    candidates = [
        base_dir / "Microwave.urdf",
        base_dir / "mobility.urdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), model_id

    # Last fallback: any urdf in model folder.
    if base_dir.exists():
        urdf_files = sorted(base_dir.glob("*.urdf"))
        if urdf_files:
            return str(urdf_files[0]), model_id

    return None, model_id


# my_friction_mtl = sim_utils.RigidBodyMaterialCfg(
#     static_friction=100,   # 静摩擦系数
#     dynamic_friction=100,  # 动摩擦系数
#     restitution=0.0        # 恢复系数（0为无弹力）
# )
# my_material = RigidBodyMaterialCfg(
#     friction_combine_mode="average",
#     restitution_combine_mode="average",
#     static_friction=0.8,
#     dynamic_friction=0.5,
#     restitution=0.0,
# )
# world_cfg = WorldConfig(
#     obstacles={
#         "microwave": Box(
#             name="microwave",
#             pose=Pose.from_numpy([0.65, 0.4, 0.25], [1.0, 0.0, 0.0, 0.0]),
#             dims=torch.tensor([0.3, 0.3, 0.3]),
#         ),
#         "cube": Box(
#             name="cube",
#             pose=Pose.from_numpy([0.5, -0.3, 0.025], [1.0, 0.0, 0.0, 0.0]),
#             dims=torch.tensor([0.05, 0.05, 0.05]),
#         )
#     }
# my_cuboid = Cuboid(
#     name="my_cuboid",
#     pose=[0.3,0.2,0.3],
#     scale = [1.5,1.5,1.5]
# )
# scene_planner_cfg = SceneCfg(
#     cuboid = [my_cuboid]
# )

# #右侧
# x_micwave = random.uniform(0.5, 0.6)
# range_coordinate = [(0.4, 0.7), (-0.5, -0.4)]
# index = random.randint(0, 1)
# # selected_range = random.choice([(0.2, 0.7), (-0.6, -0.4)])
# y_micwave = random.uniform(*range_coordinate[1])

x_micwave = random.uniform(0.6, 0.7)
y_micwave = random.uniform(0.1, 0.2)
# range_coordinate = [(0.0, 0.5), (-0.4, -0.0)]
# index = random.randint(0, 1)
# selected_range = random.choice([(0.2, 0.7), (-0.6, -0.4)])
# y_micwave = random.uniform(*selected_range)
# if index == 0:
#     x_cube = random.uniform(0.3,0.5)
#     y_cube = y_micwave - random.uniform(0.5,0.7)
# else:
#     x_cube = random.uniform(0.3,0.5)
#     y_cube = y_micwave + random.uniform(0.5,0.7)
scale_micwave = 0.3
x_cube = 0.6
y_cube = 0.0

KITCHEN_USD = PROJECT_ROOT / "Lightwheel_Kitchen" / "Collected_KitchenRoom" / "KitchenRoom.usd"
ENABLE_KITCHEN_USD = True
ENABLE_RANDOM_BACKDROP = False
# Grid rooms and the small warehouses keep the table/arms workspace open.
# Hospital / office / full warehouse stay in BACKDROP_CATALOG_HEAVY.
BACKDROP_CATALOG = (
    {
        "name": "warehouse",
        "usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        "pos": (0.0, -4.0, 0.02),
        "kitchen_layout": False,
    },
)
BACKDROP_CATALOG_HEAVY = (
    {
        "name": "warehouse_shelves",
        "usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        "pos": (0.0, -4.0, 0.02),
        "kitchen_layout": False,
    },
    {
        "name": "office",
        "usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
        "pos": (0.0, -3.5, 0.02),
        "kitchen_layout": False,
    },
    {
        "name": "hospital",
        "usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
        "pos": (0.0, -5.0, 0.02),
        "kitchen_layout": False,
    },
)


def _backdrop_available(scene: dict) -> bool:
    usd = Path(str(scene.get("usd", "")))
    # Nucleus / HTTP assets are resolved by Isaac; only skip missing local files.
    if usd.is_absolute() and usd.is_relative_to(PROJECT_ROOT):
        return usd.exists()
    return True


def pick_backdrop_scene() -> dict:
    """Pick one visual backdrop for this episode."""
    catalog = [s for s in BACKDROP_CATALOG if _backdrop_available(s)]
    if not catalog:
        catalog = list(BACKDROP_CATALOG)
    if not ENABLE_RANDOM_BACKDROP:
        return catalog[0]
    scene = random.choice(catalog)
    print(f"[Backdrop] selected {scene['name']}: {scene['usd']}", flush=True)
    return scene


BACKDROP_Z_OFFSET = 0.1


def make_backdrop_cfg(scene: dict) -> AssetBaseCfg:
    x, y, z = tuple(scene["pos"])
    return AssetBaseCfg(
        prim_path="/World/Kitchen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(scene["usd"]),
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z + BACKDROP_Z_OFFSET)),
    )


# Slide the island (and fruit bowl on it) aside so the table/robots have floor space.
KITCHEN_ISLAND_OFFSET = (0.0, 2.2, 0.1)
KITCHEN_ISLAND_PRIM_KEYS = (
    "insular",
    "island",
    "orange",
    "disk001",
    "disk002",
    "flavour",
    "fruit",
)
KITCHEN_CEILING_NAME_KEYS = (
    "ceiling",
    "ceil",
    "roof",
    "rooftop",
    "topslab",
    "droppedceiling",
    "gypsum",
    "天花",
    "屋顶",
)


def _kitchen_child_groups(kitchen):
    """Direct editable groups under the spawned kitchen USD."""
    children = list(kitchen.GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    if len(children) == 1:
        nested = list(children[0].GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
        if nested:
            children = nested
    return children


def _prim_world_aabb(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    size = mx - mn
    center = (mn + mx) * 0.5
    return center, size


def _is_island_or_fruit(prim) -> bool:
    name = prim.GetName().lower()
    if any(key in name for key in KITCHEN_ISLAND_PRIM_KEYS):
        return True
    aabb = _prim_world_aabb(prim)
    if aabb is None:
        return False
    center, size = aabb
    # Skip walls / floor / whole-room meshes.
    if float(size[0]) > 3.5 or float(size[1]) > 3.5 or float(size[2]) < 0.04:
        return False
    in_workspace = (
        -0.6 <= float(center[0]) <= 1.6
        and -1.2 <= float(center[1]) <= 1.2
        and 0.05 <= float(center[2]) <= 1.6
    )
    looks_like_island = float(size[0]) > 0.5 and float(size[1]) > 0.5 and 0.35 < float(size[2]) < 1.6
    looks_like_fruit = float(size[0]) < 0.8 and float(size[1]) < 0.8 and float(center[2]) > 0.55
    return in_workspace and (looks_like_island or looks_like_fruit)


def _authorable_xform(prim, stop_prim):
    cur = prim
    last = None
    while cur.IsValid() and cur.GetPath() != stop_prim.GetPath():
        if (not cur.IsInstanceProxy()) and (not cur.IsInPrototype()) and UsdGeom.Xformable(cur):
            last = cur
            break
        cur = cur.GetParent()
    return last


def _set_clearance_translate(prim, offset) -> bool:
    """Set an extra translate op so calling this twice does not double the shift."""
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return False
    dx, dy, dz = float(offset[0]), float(offset[1]), float(offset[2])
    target = Gf.Vec3d(dx, dy, dz)
    for op in xform.GetOrderedXformOps():
        op_name = op.GetOpName() if hasattr(op, "GetOpName") else op.GetName()
        if "workspaceClear" in str(op_name):
            op.Set(target)
            return True
    op = xform.AddTranslateOp(opSuffix="workspaceClear")
    op.Set(target)
    return True


def _uninstance_kitchen(kitchen) -> int:
    """Flatten instanceable kitchen props so physics APIs can be authored."""
    count = 0
    for _ in range(8):
        instances = [p for p in Usd.PrimRange(kitchen) if p.IsInstanceable() or p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
            count += 1
    return count


_COLLISION_GROUP_NAMES = {"collision", "collisions", "collider", "colliders"}


def _set_usd_bool(attr, value: bool) -> bool:
    if not attr:
        return False
    try:
        attr.Set(bool(value))
        return True
    except Exception:
        return False


def _prim_usable(prim) -> bool:
    try:
        return bool(prim) and prim.IsValid()
    except RuntimeError:
        return False


def _is_collision_group_prim(prim, kitchen) -> bool:
    """True for the collision group root. Children go inactive with the parent."""
    if not _prim_usable(prim) or prim.GetPath() == kitchen.GetPath():
        return False
    return prim.GetName().lower() in _COLLISION_GROUP_NAMES


def _strip_prim_collision(prim) -> bool:
    """Stop PhysX from cooking this prim. Set(False) is required: CreateXAttr(False) is a no-op if the USD already authored True."""
    if not _prim_usable(prim):
        return False
    changed = False
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(prim)
            attr = api.GetCollisionEnabledAttr()
            if attr:
                _set_usd_bool(attr, False)
            else:
                _set_usd_bool(api.CreateCollisionEnabledAttr(), False)
            changed = True
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI(prim)
            attr = api.GetCollisionEnabledAttr() if hasattr(api, "GetCollisionEnabledAttr") else None
            if attr:
                _set_usd_bool(attr, False)
            changed = True
        if physx_utils is not None and changed:
            try:
                physx_utils.removeCollider(prim)
            except Exception:
                pass
        else:
            for api in (
                UsdPhysics.CollisionAPI,
                UsdPhysics.MeshCollisionAPI,
                PhysxSchema.PhysxCollisionAPI,
                PhysxSchema.PhysxConvexHullCollisionAPI,
                PhysxSchema.PhysxConvexDecompositionCollisionAPI,
                PhysxSchema.PhysxTriangleMeshCollisionAPI,
                PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI,
            ):
                if prim.HasAPI(api):
                    try:
                        prim.RemoveAPI(api)
                        changed = True
                    except Exception:
                        pass
    except RuntimeError:
        return changed
    return changed


def _kitchen_prim_paths(kitchen) -> list[str]:
    """Snapshot paths before mutating the stage. PrimRange handles expire if a parent is deactivated."""
    paths = []
    try:
        rng = Usd.PrimRange(kitchen)
    except RuntimeError:
        return paths
    for prim in rng:
        if not _prim_usable(prim):
            continue
        try:
            if prim.IsInstanceProxy() or prim.IsInPrototype():
                continue
            paths.append(str(prim.GetPath()))
        except RuntimeError:
            continue
    return paths


def _disable_kitchen_physics(stage, kitchen) -> dict[str, int]:
    """Visual-only backdrop: drop nested colliders/rigid bodies, same as the warehouse rooms."""
    stats = {
        "articulation": 0,
        "rigid": 0,
        "collision": 0,
        "joint": 0,
        "phys_scene": 0,
        "collision_prim": 0,
    }
    paths = _kitchen_prim_paths(kitchen)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not _prim_usable(prim) or not _is_collision_group_prim(prim, kitchen):
            continue
        try:
            prim.SetActive(False)
            stats["collision_prim"] += 1
        except Exception:
            continue
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not _prim_usable(prim):
            continue
        try:
            if prim.IsInstanceProxy() or prim.IsInPrototype() or not prim.IsActive():
                continue
            type_name = prim.GetTypeName()
        except RuntimeError:
            continue
        if type_name == "PhysicsScene" or prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
            stats["phys_scene"] += 1
            continue
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                physx_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                _set_usd_bool(
                    physx_api.GetArticulationEnabledAttr() or physx_api.CreateArticulationEnabledAttr(),
                    False,
                )
                try:
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                except Exception:
                    pass
                stats["articulation"] += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = (
                    UsdPhysics.RigidBodyAPI(prim)
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    else UsdPhysics.RigidBodyAPI.Apply(prim)
                )
                _set_usd_bool(rb.GetRigidBodyEnabledAttr() or rb.CreateRigidBodyEnabledAttr(), False)
                if physx_utils is not None:
                    try:
                        physx_utils.removePhysics(prim)
                    except Exception:
                        pass
                stats["rigid"] += 1
            if _strip_prim_collision(prim):
                stats["collision"] += 1
        except RuntimeError:
            continue
        if "Joint" in type_name:
            joint = UsdPhysics.Joint(prim)
            if joint:
                _set_usd_bool(joint.GetJointEnabledAttr() or joint.CreateJointEnabledAttr(), False)
            try:
                prim.SetActive(False)
            except Exception:
                pass
            stats["joint"] += 1
    return stats


def offset_kitchen_island(stage, offset=KITCHEN_ISLAND_OFFSET) -> list[str]:
    """Translate the island and fruit plate together so the table/robots have floor space."""
    kitchen = stage.GetPrimAtPath("/World/Kitchen")
    if not kitchen.IsValid():
        print("[Kitchen] /World/Kitchen not found, skip island offset")
        return []
    groups = _kitchen_child_groups(kitchen)
    print(f"[Kitchen] top groups: {[g.GetName() for g in groups[:40]]}")
    moved = []
    seen = set()
    for prim in groups:
        if not _is_island_or_fruit(prim):
            continue
        target = _authorable_xform(prim, kitchen) or prim
        path = str(target.GetPath())
        if path in seen or path == str(kitchen.GetPath()):
            continue
        if not _set_clearance_translate(target, offset):
            print(f"[Kitchen] failed to translate {path}")
            continue
        seen.add(path)
        moved.append(path)
    print(f"[Kitchen] island offset {offset} applied to: {moved}")
    return moved


def _is_ceiling(prim) -> bool:
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    if any(key in name or key in path for key in KITCHEN_CEILING_NAME_KEYS):
        return True
    aabb = _prim_world_aabb(prim)
    if aabb is None:
        return False
    center, size = aabb
    # Thin, wide, high slab. Skip walls / whole-room volumes / floor.
    if float(size[2]) > 0.5 or float(size[2]) < 0.01:
        return False
    if float(size[0]) < 2.5 or float(size[1]) < 2.5:
        return False
    min_z = float(center[2]) - 0.5 * float(size[2])
    return min_z > 1.6


def hide_kitchen_ceiling(stage) -> list[str]:
    """Hide the kitchen roof/ceiling so the world dome light can enter."""
    kitchen = stage.GetPrimAtPath("/World/Kitchen")
    if not kitchen.IsValid():
        print("[Kitchen] /World/Kitchen not found, skip ceiling hide")
        return []
    hidden = []
    seen = set()

    def _hide(prim) -> None:
        path = str(prim.GetPath())
        if path in seen or path == str(kitchen.GetPath()):
            return
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
        else:
            prim.SetActive(False)
        seen.add(path)
        hidden.append(path)

    for prim in _kitchen_child_groups(kitchen):
        if _is_ceiling(prim):
            _hide(prim)
    for prim in Usd.PrimRange(kitchen):
        if prim.IsInstanceProxy() or prim.IsInPrototype():
            continue
        if not prim.IsActive():
            continue
        if _is_ceiling(prim):
            _hide(prim)
    print(f"[Kitchen] hid ceiling: {hidden}")
    return hidden


def hide_oven_collision_visuals(stage) -> int:
    """Keep oven collision hulls for physics, but do not draw them.

    The fine hulls sit on the same surface as the shell. Drawing both makes the
    pixels flicker. Visibility does not turn off PhysX.
    """
    envs = stage.GetPrimAtPath("/World/envs")
    if not envs.IsValid():
        return 0
    hidden = 0
    for env in envs.GetChildren():
        for child in env.GetChildren():
            if "oven" not in child.GetName().lower():
                continue
            for prim in Usd.PrimRange(child, Usd.TraverseInstanceProxies()):
                if prim.GetName().lower() not in _COLLISION_GROUP_NAMES:
                    continue
                imageable = UsdGeom.Imageable(prim)
                if not imageable:
                    continue
                imageable.MakeInvisible()
                hidden += 1
    if hidden:
        print(f"[oven] hid {hidden} collision groups from the render")
    return hidden


def prepare_kitchen_backdrop(stage, offset=KITCHEN_ISLAND_OFFSET, kitchen_layout: bool = True) -> None:
    """Keep the room as visuals only. Call again after sim.reset(), which recooks PhysX."""
    kitchen = stage.GetPrimAtPath("/World/Kitchen")
    if not kitchen.IsValid():
        print("[Kitchen] /World/Kitchen not found, skip backdrop prep")
        return
    n_inst = _uninstance_kitchen(kitchen)
    stats = _disable_kitchen_physics(stage, kitchen)
    print(f"[Kitchen] uninstanced {n_inst} prims; stripped physics {stats}")
    if kitchen_layout:
        hide_kitchen_ceiling(stage)
        offset_kitchen_island(stage, offset)


VIEW_CAMERA_EYE = (1.8, 0.0, 0.75)
VIEW_CAMERA_TARGET = (0.6, 0.0, 0.5)


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Dual-arm tabletop inside the Lightwheel kitchen USD."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.52, 0.43, 0.35)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # Outdoor HDRI. visible_in_primary_ray so the sky shows through the open ceiling.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=300.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            texture_format="latlong",
            visible_in_primary_ray=True,
        ),
    )

    kitchen = None

   
    
    robot_right = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_right",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, -0.4, 0.2),
                rot=(1.0, 0.0, 0.0, 0.0),
        ))
    robot_left = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_left",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.4, 0.2),
                rot=(1.0, 0.0, 0.0, 0.0),
        ))




    hand_left_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot_left/panda_hand/front_cam",
    update_period=0.05,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"),
)
    hand_right_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot_right/panda_hand/front_cam",
    update_period=0.05,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"),
)
    head_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot_right/HeadCamera",
    update_period=0.05,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(-0.3, 0.4, 1.6),            # 你可调整
        rot=(0.9063, 0.0, 0.4226, 0.0),  # 略向下看，wxyz
        convention="world",
    ),
)
    
    side_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/SideCamera",
    update_period=0.05,
    height=1080,
    width=1920,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(
        # Side view: closer in x, offset in +y, looking at the table center.
        pos=(1.35, 1.20, 0.75),
        rot=(-0.4562, -0.1035, -0.0535, 0.8822),  # wxyz, world +X toward (0.50, 0.00, 0.40)
        convention="world",
    ),
)
    view_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ViewCamera",
        update_period=0.05,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=VIEW_CAMERA_EYE,
            rot=(0.7830, 0.6142, -0.0608, -0.0775),  # wxyz，约绕 X 76°、绕 Z -11°
            convention="world",
        ),
    )
def get_goal_pose(planner,step,x1,y1,z1,x3,y3,z3,rot_hand,random_degree):
    ee_goals = [
        # [0.5,-0.6,0.4,0.707, 0, 0.707, 0],
        [x1-0.15, y1 , z1 , rot_hand[0], rot_hand[1], rot_hand[2], rot_hand[3]],#1 水平去拉
        [x1, y1, z1 , rot_hand[0], rot_hand[1], rot_hand[2], rot_hand[3]],
    ]
    pose = ee_goals[step]
    pos = pose[:3]
    q = pose[3:7]
    # planner = MotionPlanner(config)
    # q_start = JointState.from_position(
    #     planner.default_joint_state.position.unsqueeze(0),
    #     joint_names=planner.joint_names,
    # )
    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
    )

    return goal_pose
def move_to_pose(planner,q_start,goal_pose):
    
    # config = MotionPlannerCfg.create(
    #     robot="franka.yml",  # 确保此文件存在于 curobo 的资产路径中
    #     scene_model="collision_test.yml", # 或者根据需s要创建 WorldConfig
    # )
    # planner = MotionPlanner(config)
    # planner.warmup(enable_graph=True, num_warmup_iterations=5)

    result = planner.plan_pose(goal_pose, q_start)
    
    if result == None:
        print("规划失败！")
        simulation_app.close()
        return result

    # 获取插值后的轨迹
    traj = result.get_interpolated_plan()
    trajectory_positions = traj.position.squeeze(0)
    joint_pos_des = trajectory_positions[0]
    return joint_pos_des
    # if waypoint_idx < joint_pos_des.shape[0]:
    #     joint_each_time_pos_des = joint_pos_des[waypoint_idx]
    #     robot.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
    #     waypoint_idx += 1
    #     return True
    # else:
    #     return False
def _hand_quat_angle_rad(q_wxyz_a, q_wxyz_b) -> float:
    dot = abs(
        float(q_wxyz_a[0]) * float(q_wxyz_b[0])
        + float(q_wxyz_a[1]) * float(q_wxyz_b[1])
        + float(q_wxyz_a[2]) * float(q_wxyz_b[2])
        + float(q_wxyz_a[3]) * float(q_wxyz_b[3])
    )
    return 2.0 * float(np.arccos(min(1.0, dot)))


def _hand_pose_reached(robot, goal_pose, ori_threshold=float(np.deg2rad(10.0)), pos_threshold=0.04) -> bool:
    """Position within pos_threshold meters and hand orientation within ori_threshold radians."""
    ee_link_idx, _ = robot.find_bodies("panda_hand")
    # body_state_w is (env, body, 13): xyz, quat wxyz, linear vel, angular vel.
    hand = robot.data.body_state_w[0, int(ee_link_idx[0])]
    current_ee_pos = hand[0:3]
    target_pos_tensor = torch.tensor(goal_pose[:3], device=hand.device, dtype=hand.dtype)
    if torch.norm(current_ee_pos - target_pos_tensor).item() >= float(pos_threshold):
        return False
    if len(goal_pose) < 7:
        return True
    return _hand_quat_angle_rad(hand[3:7], goal_pose[3:7]) < float(ori_threshold)


def whether_reach_left(scene, goal_pose, ori_threshold=float(np.deg2rad(10.0)), pos_threshold=0.04):
    return _hand_pose_reached(scene["robot_left"], goal_pose, ori_threshold, pos_threshold)


def whether_reach_right(scene, goal_pose, ori_threshold=float(np.deg2rad(10.0)), pos_threshold=0.04):
    return _hand_pose_reached(scene["robot_right"], goal_pose, ori_threshold, pos_threshold)


def _arms_still_moving(runtime, speed: float = 0.08) -> bool:
    """True while an arm joint is still moving. Gripper joints are ignored."""
    robots = getattr(runtime, "robots", None) or {}
    fingers = getattr(runtime, "finger_indices", None) or {}
    for name, robot in robots.items():
        vel = getattr(getattr(robot, "data", None), "joint_vel", None)
        if vel is None or vel.numel() == 0:
            return True
        row = vel[0]
        skip = {int(i) for i in fingers.get(name, [])}
        for i in range(int(row.shape[0])):
            if i in skip:
                continue
            if abs(float(row[i].item())) > speed:
                return True
    return False


def _append_cube_as_triangles(
    cube_prim: Usd.Prim,
    all_world_vertices: list,
    all_face_indices: list,
    vertex_count_offset: int,
) -> int:
    """Convert UsdGeom.Cube to world-space triangle mesh and append."""
    cube = UsdGeom.Cube(cube_prim)
    size_attr = cube.GetSizeAttr().Get()
    size = 1.0 if size_attr is None else float(size_attr)

    # Local cube geometry: 8 vertices + 12 triangles.
    local_vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ) * size
    triangle_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # -Z
            [4, 6, 5],
            [4, 7, 6],  # +Z
            [0, 4, 5],
            [0, 5, 1],  # -Y
            [1, 5, 6],
            [1, 6, 2],  # +X
            [2, 6, 7],
            [2, 7, 3],  # +Y
            [3, 7, 4],
            [3, 4, 0],  # -X
        ],
        dtype=np.int32,
    )

    world_transform: Gf.Matrix4d = cube.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    for pt in local_vertices:
        world_pt = world_transform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
        all_world_vertices.append([world_pt[0], world_pt[1], world_pt[2]])

    # Keep face indices flattened (same format as current Mesh path).
    all_face_indices.extend((triangle_faces + vertex_count_offset).reshape(-1).tolist())
    return vertex_count_offset + len(local_vertices)


def _T_from_body_state_row(body_state_row: torch.Tensor) -> np.ndarray:
    pos = body_state_row[0:3].detach().cpu().numpy().astype(np.float32)
    quat = body_state_row[3:7].detach().cpu().numpy().astype(np.float32)  # wxyz
    w, x, y, z = quat
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _transform_points_with_T(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([points.astype(np.float32), ones], axis=1)
    out = (T @ ph.T).T
    return out[:, :3]


def _merge_mesh_data(mesh_list: list[dict]) -> dict:
    merged_vertices = []
    merged_faces = []
    v_offset = 0
    for mesh_data in mesh_list:
        verts = np.asarray(mesh_data.get("vertices", []), dtype=np.float32)
        faces = np.asarray(mesh_data.get("faces", []), dtype=np.int32).reshape(-1)
        if verts.size == 0 or faces.size == 0:
            continue
        merged_vertices.extend(verts.tolist())
        merged_faces.extend((faces + v_offset).tolist())
        v_offset += verts.shape[0]
    return {"vertices": merged_vertices, "faces": merged_faces}


def extract_link_local_mesh_for_curobo(stage: Usd.Stage, prim_path: str, link_name: str) -> dict:
    """Extract one link's collision mesh in that link's local frame."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"【错误】路径 {prim_path} 在 Stage 中不存在！")

    link_prim = stage.GetPrimAtPath(f"{prim_path}/{link_name}")
    if not link_prim.IsValid():
        raise ValueError(f"【错误】路径 {prim_path}/{link_name} 在 Stage 中不存在！")

    T_world_link = np.array(
        UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float32,
    )
    T_link_world = np.linalg.inv(T_world_link)

    all_world_vertices = []
    all_face_indices = []
    vertex_count_offset = 0

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prim_range = Usd.PrimRange(root_prim, predicate)

    for prim in prim_range:
        path_string = prim.GetPath().pathString
        path_lower = path_string.lower()
        if "collision" not in path_lower:
            continue
        if f"/{link_name}/" not in path_string:
            continue

        if prim.IsA(UsdGeom.Mesh):
            print(f"成功进入！发现物理 Mesh 节点: {path_string}")

            usd_mesh = UsdGeom.Mesh(prim)
            raw_points = usd_mesh.GetPointsAttr().Get()
            raw_faces = usd_mesh.GetFaceVertexIndicesAttr().Get()

            if raw_points is None or raw_faces is None:
                continue

            np_points = np.array(raw_points, dtype=np.float32)
            np_faces = np.array(raw_faces, dtype=np.int32)

            if np_points.size == 0 or np_faces.size == 0:
                continue

            max_internal_idx = np.max(np_faces)
            if max_internal_idx >= len(np_points):
                actual_faces = np_faces.tolist()
            else:
                actual_faces = (np_faces + vertex_count_offset).tolist()

            world_transform: Gf.Matrix4d = usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for pt in np_points:
                world_pt = world_transform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                all_world_vertices.append([world_pt[0], world_pt[1], world_pt[2]])

            all_face_indices.extend(actual_faces)
            vertex_count_offset += len(np_points)
       

    return {"vertices": all_world_vertices, "faces": all_face_indices}


def extract_mesh_for_curobo(stage: Usd.Stage, prim_path: str, include_links: set[str] | None = None) -> dict:
    """
    Instanceable 穿透防爆版：完美解决 PartNet-Mobility 因实例化/模板化导致 
    IsA(UsdGeom.Mesh) 无法触发、进而引起内存分配崩溃的问题。
    """
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"【错误】路径 {prim_path} 在 Stage 中不存在！")
        
    all_world_vertices = []
    all_face_indices = []
    vertex_count_offset = 0

    # 【核心修改】使用能够穿透 Instance（实例化节点）和 Prototype 模板的专用断言谓词
    # 这样不管 PartNet 怎么隐藏子网格，都无法逃过扫描
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prim_range = Usd.PrimRange(root_prim, predicate)

    log_debug(f"\n==== [DEBUG] 开始穿透扫描资产: {prim_path} ====")

    for prim in prim_range:
        path_string = prim.GetPath().pathString
        path_lower = path_string.lower()
        if "visuals" not in path_lower:
            continue
        if include_links is not None:
            rel = path_string[len(prim_path) + 1 :] if path_string.startswith(prim_path + "/") else ""
            link_name = rel.split("/", 1)[0] if "/" in rel else None
            if link_name not in include_links:
                continue

        # 1) Mesh collision geometry
        if prim.IsA(UsdGeom.Mesh):
            log_debug(f"成功进入！发现物理 Mesh 节点: {path_string}")

            usd_mesh = UsdGeom.Mesh(prim)
            raw_points = usd_mesh.GetPointsAttr().Get()
            raw_faces = usd_mesh.GetFaceVertexIndicesAttr().Get()

            if raw_points is None or raw_faces is None:
                continue

            np_points = np.array(raw_points, dtype=np.float32)
            np_faces = np.array(raw_faces, dtype=np.int32)

            if np_points.size == 0 or np_faces.size == 0:
                continue

            max_internal_idx = np.max(np_faces)
            if max_internal_idx >= len(np_points):
                actual_faces = np_faces.tolist()
            else:
                actual_faces = (np_faces + vertex_count_offset).tolist()

            world_transform: Gf.Matrix4d = usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for pt in np_points:
                world_pt = world_transform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                all_world_vertices.append([world_pt[0], world_pt[1], world_pt[2]])

            all_face_indices.extend(actual_faces)
            vertex_count_offset += len(np_points)

        # 2) Box collision geometry (usually imported as UsdGeom.Cube)
        elif prim.IsA(UsdGeom.Cube):
            log_debug(f"成功进入！发现物理 Cube 节点: {path_string}")
            vertex_count_offset = _append_cube_as_triangles(
                prim, all_world_vertices, all_face_indices, vertex_count_offset
            )

    log_debug(f"==== [DEBUG] 穿透扫描结束，共抓取到顶点数: {len(all_world_vertices)} ====\n")

    # 5. 此时如果依然为空，说明路径真有问题或模型未加载
    if len(all_world_vertices) == 0:
        raise RuntimeError(f"未能从 {prim_path} 中提取到任何有效的 Mesh 顶点。请检查路径或模型是否加载成功。")

    return {
        "vertices": all_world_vertices,
        "faces": all_face_indices
    }
def visualize_mesh_data_open3d(mesh_data, show_frame=True):
    verts = np.asarray(mesh_data["vertices"], dtype=np.float64)
    faces = np.asarray(mesh_data["faces"], dtype=np.int32).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.8])
    geoms = [mesh]
    # 世界坐标轴
    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        geoms.append(frame)
    # mesh 包围盒 + 中心坐标轴
    aabb = mesh.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    geoms.append(aabb)
    center = mesh.get_center()
    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=center)
    geoms.append(center_frame)
    log_debug(f"[mesh] center = {center}")
    log_debug(f"[mesh] min/max = {aabb.get_min_bound()} / {aabb.get_max_bound()}")
    o3d.visualization.draw_geometries(geoms, window_name="mesh_data debug")


def debug_mesh_alignment(mesh_data, microwave_root_pos, handle_world_pos, obstacle_pose=None):
    """Print alignment diagnostics between extracted mesh and simulator poses."""
    verts = np.asarray(mesh_data["vertices"], dtype=np.float64)
    if verts.size == 0:
        log_debug("[ALIGN] mesh_data is empty.")
        return

    mesh_center = verts.mean(axis=0)
    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    microwave_root_pos = np.asarray(microwave_root_pos, dtype=np.float64)
    handle_world_pos = np.asarray(handle_world_pos, dtype=np.float64)

    log_debug("\n[ALIGN] ===== Mesh vs Simulator =====")
    log_debug(f"[ALIGN] mesh center      : {mesh_center}")
    log_debug(f"[ALIGN] mesh bbox min/max: {mesh_min} / {mesh_max}")
    log_debug(f"[ALIGN] microwave root   : {microwave_root_pos}")
    log_debug(f"[ALIGN] handle world pos : {handle_world_pos}")
    log_debug(f"[ALIGN] center-root diff : {mesh_center - microwave_root_pos}")
    log_debug(f"[ALIGN] center-handle diff: {mesh_center - handle_world_pos}")

    if obstacle_pose is not None:
        pose_t = np.asarray(obstacle_pose[:3], dtype=np.float64)
        log_debug(f"[ALIGN] cuRobo mesh pose translation: {pose_t}")
        if np.linalg.norm(pose_t) > 1e-6:
            log_debug("[ALIGN][WARN] vertices are already world-space; non-zero mesh pose will shift obstacle again.")
    log_debug("[ALIGN] =================================\n")
def get_link_prims_world_pose(
    stage,
    root_path: str,
    target_link: str,
    
):
    """
    在 root_path 下查找属于 target_link 的 prim，并返回它们的世界位姿。
    返回: List[dict], 每项含 path, T_world(4x4), pos(3), quat_wxyz(4)
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"路径不存在: {root_path}")
    results = []
    target_candidates = _build_link_candidates(target_link)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prim_range = Usd.PrimRange(root_prim, predicate)
    for prim in prim_range:
        path = prim.GetPath().pathString
        # path_lower = path.lower()
        # # 可选：只看 visuals / collisions 分支
        # if branch_keyword is not None and branch_keyword.lower() not in path_lower:
        #     continue
        # 关键：保证匹配的是“某个层级名 == target_link”
        # 例如 .../Sunken_Door/Trifold_Handle/visuals/...
        tokens = path.strip("/").split("/")
        if not any(candidate in tokens for candidate in target_candidates):
            continue
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        T = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        pos = T[:3, 3]
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        results.append(
            {
                "path": path,
                "T_world": T,
                "pos": pos,
                "quat_wxyz": quat_wxyz,
            }
        )
    return results
def get_handle_pose_physics(microwave_asset, scale_micwave, concept_urdf_path=None):
    # 1) 如果把手是独立 body，直接返回
    body_names = list(microwave_asset.body_names)
    for handle_name in HANDLE_LINK_CANDIDATES:
        if handle_name in body_names:
            hid, _ = microwave_asset.find_bodies(handle_name)
            hstate = microwave_asset.data.body_state_w[0, int(hid[0])]
            pos = hstate[0:3].detach().cpu().numpy()
            quat_wxyz = hstate[3:7].detach().cpu().numpy()
            log_debug(f"handle body matched: {handle_name}")
            return pos, quat_wxyz
    # 2) 否则用门的 physics 真值 + door->handle 的 joint origin 偏移
    found_door_name = None
    for door_name in DOOR_LINK_CANDIDATES:
        if door_name in body_names:
            found_door_name = door_name
            break
    if found_door_name is None:
        raise RuntimeError(f"未找到门link，候选: {DOOR_LINK_CANDIDATES}")
    did, _ = microwave_asset.find_bodies(found_door_name)
    dstate = microwave_asset.data.body_state_w[0, int(did[0])]
    door_pos = dstate[0:3].detach().cpu().numpy()
    door_quat_wxyz = dstate[3:7].detach().cpu().numpy()  # wxyz

    offset_from_urdf = None
    if concept_urdf_path:
        parsed_offset, parsed_door, parsed_handle = get_door_to_handle_offset_from_urdf(concept_urdf_path)
        if parsed_offset is not None:
            offset_from_urdf = parsed_offset * scale_micwave
            print(f"have a offest pose{parsed_offset}")
            log_debug(
                f"door->handle offset from urdf: door={parsed_door}, handle={parsed_handle}, "
                f"offset={offset_from_urdf}"
            )

    # fallback to historical default when URDF pair is missing
    if offset_from_urdf is None:
        offset_from_urdf = np.array(
            [1.290500 * scale_micwave, -0.094500 * scale_micwave, 0.052500 * scale_micwave],
            dtype=np.float64,
        )
        log_debug(f"use fallback door->handle offset: {offset_from_urdf}")

    qw, qx, qy, qz = door_quat_wxyz  
    R_world_door = R.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: xyzw
    handle_pos = door_pos + R_world_door @ offset_from_urdf
    handle_quat_wxyz = door_quat_wxyz.copy()  # 固定关节且rpy=0时与门同向
    return handle_pos, handle_quat_wxyz
def get_link_mesh_world_ymax(stage, root_path: str, target_link: str, use_visuals=True):
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"路径不存在: {root_path}")
    ymin = np.inf
    hit_paths = []
    target_candidates = _build_link_candidates(target_link)
    matched_link_name = None
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prim_range = Usd.PrimRange(root_prim, predicate)
    for prim in prim_range:
        path = prim.GetPath().pathString
        path_lower = path.lower()
        
        if (not use_visuals) and "collisions" not in path_lower:
            continue
        # 只保留目标 link 路径下的 prim
        tokens = path.strip("/").split("/")
        matched = next((c for c in target_candidates if c in tokens), None)
        if matched is None:
            continue
        if matched_link_name is None:
            matched_link_name = matched
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_local = mesh.GetPointsAttr().Get()
        if pts_local is None:
            continue
        T_world = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # 只算 y 最大，不需要保存全部点
        for p in pts_local:
            p_w = T_world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            if p_w[1] < ymin:
                ymin = p_w[1]
        hit_paths.append(path)
        log_debug(f"the path of mesh:{path}")
    if ymin == np.inf:
        raise RuntimeError(f"未找到目标link网格，候选: {target_candidates}")
    if matched_link_name is not None and matched_link_name != target_link:
        log_debug(f"link fallback: requested={target_link}, matched={matched_link_name}")
    return float(ymin), hit_paths
def generate_handle_arc_points_about_axis(
    axis_point,
    axis_dir,
    handle_pos0,
    max_deg=80,
    step_deg=40,
):
    """
    Generate handle trajectory by rotating initial handle position around an arbitrary axis
    using Rodrigues' rotation formula.
    """
    c = np.asarray(axis_point, dtype=np.float64)
    u = np.asarray(axis_dir, dtype=np.float64)
    p0 = np.asarray(handle_pos0, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-8:
        raise ValueError("axis_dir norm is too small")
    u = u / norm_u
    r0 = p0 - c

    pts = []
    for deg in range(0, max_deg + 1, step_deg):
        th = np.deg2rad(deg)
        r = (
            r0 * np.cos(th)
            + np.cross(u, r0) * np.sin(th)
            + u * np.dot(u, r0) * (1.0 - np.cos(th))
        )
        p = c + r
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    return pts
def plan_concat_segments(planner, q_start, arc_points, quat=[0.707, 0.0, 0.707, 0.0]):
    segs = []
    q_curr = q_start
    for i in range(len(arc_points) - 1):
        target = arc_points[i + 1]   # 从当前到下一个点
        goal_pose = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor([[[[target]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[quat]]]], device="cuda", dtype=torch.float32),
        )
        result = planner.plan_pose(goal_pose, q_curr)
        if result is None:
            print(f"[WARN] segment {i}->{i+1} planning failed")
            break
        traj = result.get_interpolated_plan().position.squeeze(0)  # [T, dof]
        joint_pos_des = traj[0]

        log_debug(f"segment trajectory length: {joint_pos_des.shape[0]}")

        # 拼接时去掉后续段第一个点，避免重复
        if len(segs) > 0 and traj.shape[0] > 1:
            joint_pos_des = joint_pos_des[1:]
        segs.append(joint_pos_des)
        # 下一段起点 = 本段终点
        joint_each_time_pos_des = joint_pos_des[joint_pos_des.shape[0]-1]
        joint_each_time_pos_des[:7].unsqueeze(0)
        q_curr = JointState.from_position(
            joint_each_time_pos_des[:7].unsqueeze(0),
            joint_names=planner.joint_names,
        )
    if len(segs) == 0:
        return None
    return torch.cat(segs, dim=0)
def _collision_world_points(stage, prim_path: str) -> list[list[float]]:
    """World-space samples of a prim's collision meshes and collision cubes."""
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return []
    points: list[list[float]] = []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange(root, predicate):
        path_l = prim.GetPath().pathString.lower()
        if "collision" not in path_l and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            raw = mesh.GetPointsAttr().Get()
            if not raw:
                continue
            world_tf = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for pt in raw:
                w = world_tf.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                points.append([float(w[0]), float(w[1]), float(w[2])])
        elif prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            size = cube.GetSizeAttr().Get()
            half = 1.0 if size is None else float(size) * 0.5
            world_tf = cube.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for x in (-half, half):
                for y in (-half, half):
                    for z in (-half, half):
                        w = world_tf.Transform(Gf.Vec3d(x, y, z))
                        points.append([float(w[0]), float(w[1]), float(w[2])])
    return points


def _asset_prim_path(stage, scene_key: str, env_id: int = 0):
    """Scene key is ``table_1__env0``; the USD prim is ``/World/envs/env_0/table_1``."""
    name = str(scene_key)
    marker = f"__env{env_id}"
    asset_name = name[: -len(marker)] if name.endswith(marker) else name
    for candidate in (
        f"/World/envs/env_{env_id}/{asset_name}",
        f"/World/envs/env_{env_id}/{name}",
    ):
        if stage.GetPrimAtPath(candidate).IsValid():
            return candidate, asset_name
    return None, asset_name


def named_collision_cuboid(scene, kind: str):
    """Axis-aligned cuboid of a spawned asset's collision geometry right now."""
    stage = scene.stage
    prim_path = None
    prim_name = None
    for key in scene.keys():
        name = str(key)
        low = name.lower()
        if "concept" in low:
            continue
        if kind == "table" and "table" not in low:
            continue
        if kind == "microwave" and "mic" not in low:
            continue
        candidate, asset_name = _asset_prim_path(stage, name)
        if candidate is not None:
            prim_path = candidate
            prim_name = asset_name
            break
    if prim_path is None:
        print(f"[curobo] no {kind} prim; obstacle stays off")
        return None
    pts = _collision_world_points(stage, prim_path)
    if not pts:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedBox()
        mn, mx = box.GetMin(), box.GetMax()
        if mn[0] > mx[0]:
            print(f"[curobo] {kind} {prim_path} has no collision bounds")
            return None
        mins = [float(mn[0]), float(mn[1]), float(mn[2])]
        maxs = [float(mx[0]), float(mx[1]), float(mx[2])]
        source = "bbox"
    else:
        arr = np.asarray(pts, dtype=np.float64)
        mins = arr.min(axis=0).tolist()
        maxs = arr.max(axis=0).tolist()
        source = f"collision pts={len(pts)}"
    dims = [max(maxs[i] - mins[i], 0.01) for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    print(
        f"[curobo] {kind} obstacle {prim_name} {source} center={[round(v, 3) for v in center]} "
        f"dims={[round(v, 3) for v in dims]}"
    )
    return Cuboid(
        name=kind,
        pose=[center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0],
        dims=dims,
    )


def _scene_prim(scene, kind: str):
    stage = scene.stage
    for key in scene.keys():
        name = str(key)
        low = name.lower()
        if "concept" in low:
            continue
        if kind == "table" and "table" not in low:
            continue
        if kind == "microwave" and "mic" not in low:
            continue
        candidate, asset_name = _asset_prim_path(stage, name)
        if candidate is not None:
            return stage, candidate, asset_name
    return stage, None, None


def _mesh_world_triangles(mesh: UsdGeom.Mesh):
    raw = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not raw or not counts or not indices:
        return None, None
    world_tf = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    vertices = []
    for pt in raw:
        w = world_tf.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
        vertices.append([float(w[0]), float(w[1]), float(w[2])])
    faces = []
    cursor = 0
    for count in counts:
        n = int(count)
        poly = [int(indices[cursor + i]) for i in range(n)]
        cursor += n
        for k in range(1, n - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    if not faces:
        return None, None
    return vertices, faces


def named_collision_meshes(scene, kind: str):
    """Each live collision mesh, in world coordinates, so the open door stays hollow."""
    stage, prim_path, prim_name = _scene_prim(scene, kind)
    if prim_path is None:
        print(f"[curobo] no {kind} prim; mesh obstacle stays off")
        return []
    meshes = []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    root = stage.GetPrimAtPath(prim_path)
    scan_id = int(getattr(named_collision_meshes, "_scan_id", 0)) + 1
    named_collision_meshes._scan_id = scan_id
    for prim in Usd.PrimRange(root, predicate):
        path_l = prim.GetPath().pathString.lower()
        if "visual" in path_l and "collision" not in path_l:
            continue
        if "collision" not in path_l and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        vertices, faces = _mesh_world_triangles(UsdGeom.Mesh(prim))
        if vertices is None:
            continue
        meshes.append(
            Mesh(
                name=f"{kind}_{scan_id}_{len(meshes)}",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=vertices,
                faces=faces,
            )
        )
    n_verts = sum(len(mesh.vertices) for mesh in meshes)
    n_faces = sum(len(mesh.faces) for mesh in meshes)
    print(f"[curobo] {kind} mesh obstacle {prim_name} parts={len(meshes)} verts={n_verts} faces={n_faces}")
    return meshes


def table_collision_cuboid(scene):
    return named_collision_cuboid(scene, "table")


def _asset_collision_bounds(scene, scene_key: str, env_id: int = 0):
    """World AABB of an asset's collision geometry. Falls back to the prim bounds."""
    stage = scene.stage
    prim_path, _asset_name = _asset_prim_path(stage, scene_key, env_id)
    if prim_path is None:
        return None
    pts = _collision_world_points(stage, prim_path)
    if pts:
        arr = np.asarray(pts, dtype=np.float64)
        return arr.min(axis=0), arr.max(axis=0)
    aabb = _prim_world_aabb(stage.GetPrimAtPath(prim_path))
    if aabb is None:
        return None
    center, size = aabb
    center = np.array([float(center[i]) for i in range(3)], dtype=np.float64)
    size = np.array([float(size[i]) for i in range(3)], dtype=np.float64)
    return center - 0.5 * size, center + 0.5 * size


def _ray_hit_aabb(origin, direction, mins, maxs):
    """First hit of a ray on an axis-aligned box. None when the ray misses."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    tmin, tmax = 0.0, 1.0e6
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-8:
            if origin[axis] < mins[axis] or origin[axis] > maxs[axis]:
                return None
            continue
        inv = 1.0 / float(direction[axis])
        t1 = (mins[axis] - origin[axis]) * inv
        t2 = (maxs[axis] - origin[axis]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0.0:
        return None
    travel = tmin if tmin >= 0.0 else 0.0
    return origin + direction * travel


def _grasp_marks(compiler, scene, atomic_plan, bounds_by_asset: dict):
    """One mark per atom: the pose where the gripper first closes, and the surface hit."""
    marks = []
    for index, step in enumerate(atomic_plan):
        try:
            waypoints = compiler.compile_step(step, scene, index)
        except Exception as exc:
            print(f"[grasp-vis] skip {step.get('asset')}.{step.get('atom')}: {exc}")
            continue
        previous = None
        chosen = None
        for wp in waypoints:
            grip = wp.get("gripper", "open")
            if grip == "close" and previous == "open":
                chosen = wp
                break
            previous = grip
        if chosen is None:
            continue
        pose = list(chosen["pose"])
        pos = np.asarray(pose[:3], dtype=np.float64)
        quat = np.asarray(pose[3:7], dtype=np.float64)
        hand_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        into_object = hand_rot[:, 2]
        asset_id = str(step.get("asset", ""))
        affordance = pos
        bounds = bounds_by_asset.get(asset_id)
        if bounds is not None:
            hit = _ray_hit_aabb(pos, into_object, bounds[0], bounds[1])
            if hit is None:
                hit = _ray_hit_aabb(pos, -into_object, bounds[0], bounds[1])
            if hit is not None:
                affordance = hit
        arm = str(chosen.get("executor", ""))
        marks.append(
            {
                "name": asset_id,
                "atom": str(step.get("atom", "")),
                "arm": arm,
                "pos": pos,
                "quat": quat,
                "affordance": affordance,
            }
        )
    return marks


def _prepare_head_grasp_scene(scene, world, compiler, atomic_plan) -> dict:
    """Collision boxes and the first close-gripper pose of each atom. Computed once."""
    objects = []
    bounds_by_asset = {}
    for asset_id in world.knowledge_map:
        if "concept" in str(asset_id).lower():
            continue
        try:
            _asset, scene_key = world.resolve_asset(asset_id, 0)
        except Exception:
            continue
        bounds = _asset_collision_bounds(scene, scene_key, 0)
        if bounds is None:
            continue
        bounds_by_asset[str(asset_id)] = bounds
        objects.append({"name": str(asset_id), "mins": bounds[0], "maxs": bounds[1]})
    grasps = _grasp_marks(compiler, scene, atomic_plan, bounds_by_asset)
    return {"objects": objects, "grasps": grasps}


def _head_camera_ros_pose(scene, cam):
    """Camera position and ROS axes taken from the live USD camera prim.

    USD cameras look along local -Z with +Y up. ROS looks along +Z with +Y down.
    """
    stage = scene.stage
    raw = str(getattr(getattr(cam, "cfg", None), "prim_path", ""))
    resolved = raw.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    candidates = [resolved]
    if "HeadCamera" in resolved:
        candidates.append("/World/envs/env_0/Robot_right/HeadCamera")
    prim = None
    for path in candidates:
        if not path:
            continue
        found = stage.GetPrimAtPath(path)
        if found.IsValid():
            prim = found
            break
    if prim is None:
        pos = cam.data.pos_w[0].detach().cpu().numpy()
        quat = cam.data.quat_w_ros[0].detach().cpu().numpy()
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        return pos, rot
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = xf.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    right = xf.Transform(Gf.Vec3d(1.0, 0.0, 0.0)) - origin
    up = xf.Transform(Gf.Vec3d(0.0, 1.0, 0.0)) - origin
    forward = -(xf.Transform(Gf.Vec3d(0.0, 0.0, 1.0)) - origin)
    down = -up
    pos = np.array([float(origin[0]), float(origin[1]), float(origin[2])], dtype=np.float64)
    axes = [right, down, forward]
    cols = []
    for axis in axes:
        vec = np.array([float(axis[0]), float(axis[1]), float(axis[2])], dtype=np.float64)
        n = np.linalg.norm(vec)
        cols.append(vec / n if n > 1e-8 else vec)
    return pos, np.stack(cols, axis=1)


def _save_head_grasp_annotation(scene, prepared: dict, task_name: str) -> bool:
    """Save one head-camera frame with collision boxes, grasp axes, and affordance dots."""
    from PIL import Image as PILImage

    try:
        cam = scene["head_camera"]
    except Exception:
        print("[grasp-vis] no head_camera")
        return False
    rgb = cam.data.output.get("rgb")
    if rgb is None:
        return False
    frame = rgb[0, ..., :3]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.size == 0:
        return False
    if frame.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(frame)) <= 1.5 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    if float(frame.mean()) < 2.0:
        return False
    cam_pos, cam_rot = _head_camera_ros_pose(scene, cam)
    intrinsic = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
    annotated = render_head_grasp_annotation(
        frame,
        cam_pos,
        None,
        intrinsic,
        prepared["objects"],
        prepared["grasps"],
        cam_rot=cam_rot,
    )
    safe = re.sub(r"[^\w.\-]+", "_", str(task_name or "task")).strip("._") or "task"
    out_path = PROJECT_ROOT / f"head_grasp_{safe}.png"
    PILImage.fromarray(annotated).save(out_path, format="PNG")
    print(
        f"[grasp-vis] {out_path} objects={len(prepared['objects'])} "
        f"grasps={len(prepared['grasps'])} image={annotated.shape[1]}x{annotated.shape[0]} "
        f"cam={[round(float(v), 3) for v in cam_pos]}"
    )
    return True


def _mark_from_waypoint(wp: dict, bounds_by_asset: dict, asset_id: str | None = None) -> dict:
    """Grasp arrow from the waypoint pose, affordance where that axis meets the object."""
    pose = list(wp["pose"])
    pos = np.asarray(pose[:3], dtype=np.float64)
    quat = np.asarray(pose[3:7], dtype=np.float64)
    hand_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    into_object = hand_rot[:, 2]
    name = str(asset_id or wp.get("grasp_asset") or "")
    affordance = pos
    bounds = bounds_by_asset.get(name)
    if bounds is not None:
        hit = _ray_hit_aabb(pos, into_object, bounds[0], bounds[1])
        if hit is None:
            hit = _ray_hit_aabb(pos, -into_object, bounds[0], bounds[1])
        if hit is not None:
            affordance = hit
    return {
        "name": name,
        "atom": str(wp.get("atom", "")),
        "arm": str(wp.get("executor", "")),
        "pos": pos,
        "quat": quat,
        "affordance": affordance,
    }


def _first_close_waypoint(tasks: list) -> dict | None:
    previous = None
    for wp in tasks:
        grip = wp.get("gripper", "open")
        if grip == "close" and previous == "open":
            return wp
        previous = grip
    return None


def _runtime_grasp_marks(runtime, objects) -> list:
    """Hand pose at each grasp close, and the point where that axis meets the object."""
    compiled = getattr(runtime, "compiled", None)
    if compiled is None or not getattr(compiled, "waypoints", None):
        return []
    bounds = {
        str(obj["name"]): (np.asarray(obj["mins"], dtype=np.float64), np.asarray(obj["maxs"], dtype=np.float64))
        for obj in objects
    }
    asset_id = str(getattr(compiled, "asset_id", "") or "")
    if getattr(compiled, "simultaneous", False):
        packs = [list(runtime.left_arm.tasks or []), list(runtime.right_arm.tasks or [])]
    else:
        packs = [list(compiled.waypoints)]
    marks = []
    for tasks in packs:
        wp = _first_close_waypoint(tasks)
        if wp is None and tasks and tasks[0].get("gripper") == "close":
            wp = tasks[0]
        if wp is None or "pose" not in wp:
            continue
        marks.append(_mark_from_waypoint(wp, bounds, wp.get("grasp_asset") or asset_id))
    return marks


def _live_hand_marks(scene, objects) -> list:
    """Current panda_hand pose. Red arrow follows the fingers as the arm moves."""
    marks = []
    for key in ("robot_left", "robot_right"):
        try:
            robot = scene[key]
            ids, _ = robot.find_bodies("panda_hand")
            state = robot.data.body_state_w[0, int(ids[0])]
        except Exception:
            continue
        pos = np.asarray(state[0:3].detach().cpu().numpy(), dtype=np.float64)
        quat = np.asarray(state[3:7].detach().cpu().numpy(), dtype=np.float64)
        hand_rot = R.from_quat([float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]).as_matrix()
        direction = hand_rot[:, 2]
        affordance = None
        best = None
        for obj in objects:
            name = str(obj.get("name", "")).lower()
            if "table" in name:
                continue
            hit = _ray_hit_aabb(pos, direction, obj["mins"], obj["maxs"])
            if hit is None:
                continue
            dist = float(np.linalg.norm(hit - pos))
            if dist > 0.2:
                continue
            if best is None or dist < best:
                best = dist
                affordance = hit
        marks.append(
            {
                "name": key,
                "atom": "ee",
                "arm": key,
                "pos": pos,
                "quat": quat,
                "affordance": affordance,
                "label": False,
            }
        )
    return marks


def _place_can_stage(runtime, saved: set[str]) -> str | None:
    """Five moments: approach, grasp, lift, then each can lowered into the box."""
    compiled = getattr(runtime, "compiled", None)
    if compiled is None:
        return None
    atom = str(compiled.atom or "")
    if compiled.simultaneous and "pick" in atom:
        left = int(runtime.left_arm.step_idx)
        right = int(runtime.right_arm.step_idx)
        if left >= 2 and right >= 2 and "01_approach" not in saved:
            return "01_approach"
        if left >= 3 and right >= 3 and "02_grasp" not in saved:
            return "02_grasp"
        if left >= 4 and right >= 4 and "03_lift" not in saved:
            return "03_lift"
        return None
    step = int(runtime._seq.step)
    if atom == "into_1" and step >= 2 and "04_place_can1" not in saved:
        return "04_place_can1"
    if atom == "into_2" and step >= 2 and "05_place_can2" not in saved:
        return "05_place_can2"
    return None


def _roller_stage(runtime, saved: set[str]) -> str | None:
    """Hover, lower onto the roller, close, then lift. Both hands share one roller."""
    compiled = getattr(runtime, "compiled", None)
    if compiled is None or not compiled.simultaneous or "grap" not in str(compiled.atom or ""):
        return None
    left = int(runtime.left_arm.step_idx)
    right = int(runtime.right_arm.step_idx)
    if left >= 1 and right >= 1 and "01_approach" not in saved:
        return "01_approach"
    if left >= 2 and right >= 2 and "02_down" not in saved:
        return "02_down"
    if left >= 3 and right >= 3 and "03_grasp" not in saved:
        return "03_grasp"
    if left >= 4 and right >= 4 and "04_lift" not in saved:
        return "04_lift"
    return None


def _roller_marks(runtime, bounds_by_asset: dict) -> list:
    marks = []
    for tasks in (runtime.left_arm.tasks, runtime.right_arm.tasks):
        wp = tasks[1] if len(tasks) > 1 else _first_close_waypoint(tasks)
        if wp is not None:
            marks.append(_mark_from_waypoint(wp, bounds_by_asset, "roller_1"))
    return marks


def _usd_root_rigid(stage, prim_path: str):
    """Root translation and rotation. Scale stays in the point coordinates."""
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    origin = xf.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    cols = []
    for vec in (Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        axis = xf.Transform(vec) - origin
        arr = np.array([float(axis[0]), float(axis[1]), float(axis[2])], dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        cols.append(arr / norm if norm > 1e-8 else arr)
    pos = np.array([float(origin[0]), float(origin[1]), float(origin[2])], dtype=np.float64)
    return pos, np.stack(cols, axis=1)


def _collision_body_points(stage, prim_path: str):
    """Collision samples in the rigid-body frame, with the asset scale kept.

    USD world points and the USD root pose are a matching pair, so removing that
    pose leaves a local cloud that is still valid after the body moves.
    """
    cache = getattr(_collision_body_points, "_cache", None)
    if cache is None:
        cache = {}
        _collision_body_points._cache = cache
    if prim_path in cache:
        return cache[prim_path]
    pts = _collision_world_points(stage, prim_path)
    if not pts:
        cache[prim_path] = None
        return None
    origin, rot = _usd_root_rigid(stage, prim_path)
    cache[prim_path] = (np.asarray(pts, dtype=np.float64) - origin) @ rot
    return cache[prim_path]


def _physics_collision_objects(scene, world) -> list:
    """Collision AABBs placed at each asset's physics root on this frame."""
    objects = []
    stage = scene.stage
    for asset_id in world.knowledge_map:
        if "concept" in str(asset_id).lower():
            continue
        try:
            asset, scene_key = world.resolve_asset(asset_id, 0)
        except Exception:
            continue
        prim_path, _asset_name = _asset_prim_path(stage, scene_key, 0)
        body = None if prim_path is None else _collision_body_points(stage, prim_path)
        if body is None or not hasattr(asset.data, "root_pos_w"):
            bounds = _asset_collision_bounds(scene, scene_key, 0)
            if bounds is None:
                continue
            objects.append({"name": str(asset_id), "mins": bounds[0], "maxs": bounds[1]})
            continue
        pos = np.asarray(asset.data.root_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
        quat = np.asarray(asset.data.root_quat_w[0].detach().cpu().numpy(), dtype=np.float64)
        rot = R.from_quat([float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]).as_matrix()
        world_pts = body @ rot.T + pos
        if len(world_pts) > 400:
            world_pts = world_pts[:: max(1, len(world_pts) // 400)]
        objects.append(
            {
                "name": str(asset_id),
                "mins": world_pts.min(axis=0),
                "maxs": world_pts.max(axis=0),
                "points": world_pts,
            }
        )
    return objects


def _roller_live_marks(scene, world, bounds_by_asset: dict) -> list:
    """Same hand offsets as grap, rebuilt from the roller's pose on this frame."""
    try:
        asset, _scene_key = world.resolve_asset("roller_1", 0)
    except Exception:
        return []
    pos = np.asarray(asset.data.root_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
    quat = np.asarray(asset.data.root_quat_w[0].detach().cpu().numpy(), dtype=np.float64)
    hand_rot = R.from_quat(wxyz_to_xyzw(quat)) * R.from_euler("y", 90, degrees=True)
    hand_quat = np.asarray(xyzw_to_wxyz(hand_rot.as_quat()), dtype=np.float64)
    marks = []
    for local_x, arm in ((-0.1, "simultaneous_left"), (0.1, "simultaneous_right")):
        contact = pos + hand_rot.apply([local_x, 0.0, 0.0])
        contact = np.array([contact[0], contact[1], pos[2] + 0.08], dtype=np.float64)
        wp = {
            "pose": [float(contact[0]), float(contact[1]), float(contact[2]), *hand_quat.tolist()],
            "atom": "grap_simulataneously",
            "executor": arm,
        }
        marks.append(_mark_from_waypoint(wp, bounds_by_asset, "roller_1"))
    return marks


def _place_can_marks(runtime, stage: str, bounds_by_asset: dict) -> list:
    if stage.startswith("0") and stage[1] in "123":
        marks = []
        for tasks, asset_id in (
            (runtime.left_arm.tasks, "can_1"),
            (runtime.right_arm.tasks, "can_2"),
        ):
            wp = _first_close_waypoint(tasks)
            if wp is not None:
                marks.append(_mark_from_waypoint(wp, bounds_by_asset, asset_id))
        return marks
    compiled = runtime.compiled
    if compiled is None or len(compiled.waypoints) < 2:
        return []
    asset_id = "plasticbox_1" if "plasticbox_1" in bounds_by_asset else str(compiled.asset_id)
    return [_mark_from_waypoint(compiled.waypoints[1], bounds_by_asset, asset_id)]


def _live_collision_objects(scene, world) -> list:
    """Collision boxes from the USD world pose. Kept as the backup annotation."""
    objects = []
    for asset_id in world.knowledge_map:
        if "concept" in str(asset_id).lower():
            continue
        try:
            _asset, scene_key = world.resolve_asset(asset_id, 0)
        except Exception:
            continue
        bounds = _asset_collision_bounds(scene, scene_key, 0)
        if bounds is None:
            continue
        objects.append({"name": str(asset_id), "mins": bounds[0], "maxs": bounds[1]})
    return objects


def _save_place_can_keyframe(scene, world, runtime, prepared: dict, stage: str, task_name: str) -> bool:
    usd_bounds = {obj["name"]: (obj["mins"], obj["maxs"]) for obj in prepared["objects"]}
    live_objects = _physics_collision_objects(scene, world)
    live_bounds = {obj["name"]: (obj["mins"], obj["maxs"]) for obj in live_objects}
    if task_name == "grap_roller":
        usd_marks = _roller_marks(runtime, usd_bounds)
        live_marks = _roller_live_marks(scene, world, live_bounds)
    else:
        usd_marks = _place_can_marks(runtime, stage, usd_bounds)
        live_marks = _place_can_marks(runtime, stage, live_bounds)
    _save_head_grasp_annotation(
        scene,
        {"objects": prepared["objects"], "grasps": usd_marks},
        f"{task_name}_{stage}_planned",
    )
    return _save_head_grasp_annotation(
        scene,
        {"objects": live_objects, "grasps": live_marks},
        f"{task_name}_{stage}",
    )


def _marker_origin(scene) -> tuple[float, float, float]:
    """Green marks stay on env_0. Later envs are the same scene with no overlay."""
    origins = getattr(scene, "env_origins", None)
    if origins is None or len(origins) < 2:
        return (0.0, 0.0, 0.0)
    origin = tuple(float(v) for v in origins[0].detach().cpu().tolist())
    print(
        f"[INFO] {len(origins)} envs: green markers only on env_0 at "
        f"{[round(v, 3) for v in origin]}; other envs have none"
    )
    return origin


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    task_ctx,
    
):
    """Runs the simulation loop with live world sensing and atom-level replan."""






    # items = get_link_prims_world_pose(
    # stage=scene.stage,
    # root_path="/World/envs/env_0/my_concept_micwave",
    # target_link="Trifold_Handle",
    # )
    # log_debug(f"handle prim matched count: {len(items)}")
    # if VERBOSE:
    #     for it in items:
    #         print(it["path"], it["pos"], it["quat_wxyz"])

    
    # micwave_pos = microwave_asset.data.root_pos_w[0]

    # x3 = micwave_pos[0].item()
    # y3 = micwave_pos[1].item()
    # z3 = micwave_pos[2].item()
    # log_debug(f"microwave root position: {micwave_pos}")
    
    # # 提取单个数值
    # x_old = current_handle_pos[0].item()
    # y_old = current_handle_pos[1].item()
    # z_old = current_handle_pos[2].item()
    # print(f"a crucial pose:{x_old},{y_old},{z_old}")

    # current_cube_pos = cube_asset.data.root_pos_w[0]
    # x2 = current_cube_pos[0].item()
    # y2 = current_cube_pos[1].item()
    # z2 = current_cube_pos[2].item()
    # log_debug(f"cube position: {current_cube_pos}")



    # microwave_prim_path = "/World/envs/env_0/my_laptop"
  

    # static_mesh_data = extract_mesh_for_curobo(
    #     scene.stage, microwave_prim_path
    # )
    # # 2) Door local mesh (link_0), will be transformed by live body pose
    # door_local_mesh = extract_link_local_mesh_for_curobo(
    #     scene.stage, microwave_prim_path, "link_0"
    # )
    # link0_body_idx, _ = microwave_asset.find_bodies("link_0")
    # link0_body_idx = int(link0_body_idx[0])
    # # Calibrate body->link transform once (body_state_w may use COM frame).
    # link0_prim = scene.stage.GetPrimAtPath(f"{microwave_prim_path}/link_0")
    # T_world_link0_init = np.array(
    #     UsdGeom.Xformable(link0_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
    #     dtype=np.float32,
    # )
    # T_world_body0_init = _T_from_body_state_row(microwave_asset.data.body_state_w[0, link0_body_idx])
    # T_body0_to_link0 = np.linalg.inv(T_world_body0_init) @ T_world_link0_init

    # T_world_body0 = _T_from_body_state_row(microwave_asset.data.body_state_w[0, link0_body_idx])
    # T_world_link0 = T_world_body0 @ T_body0_to_link0
    # door_world_vertices = _transform_points_with_T(
    #     np.asarray(door_local_mesh["vertices"], dtype=np.float32), T_world_link0
    # )
    # door_world_mesh = {
    #     "vertices": door_world_vertices.tolist(),
    #     "faces": door_local_mesh["faces"],
    # }
    # mesh_data = _merge_mesh_data([static_mesh_data, door_local_mesh])
    # if VERBOSE:
    #     visualize_mesh_data_open3d(mesh_data)


    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.prim_path = "/Visuals/FrameMarker"
    # frame_marker = VisualizationMarkers(frame_cfg)
    # obstacle_pose = [x_micwave-10000, y_micwave-10000, 0.0, 1, 0, 0, 0]

#     curobo_obstacle = Mesh(
#     name="my_cart",
#     vertices=static_mesh_data["vertices"],
#     faces=static_mesh_data["faces"],
#     pose = obstacle_pose,
#     scale=None # 因为顶点已经是真实世界尺寸了
# )
    far_cube = Cuboid(
        name="mic_bottom",
        pose=[100, 100+ 0.4, 0.3 - 0.2, 1.0, 0.0, 0.0, 0.0],
        dims=[5, 5, 0.001],  # 长, 宽, 高 (单位: 米)
        color=[0.4, 0.4, 0.4, 1.0],  # 灰色
    )
    empty_scene = SceneCfg(cuboid=[far_cube])
    table_cube = table_collision_cuboid(scene)
    table_scene = SceneCfg(cuboid=[table_cube]) if table_cube is not None else None
    curobo_scene_model = empty_scene

    # Mesh slots stay allocated from the start. The live microwave scan fills them later.
    collision_cache = {"cuboid": 8, "mesh": 256}
    config_left = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        collision_cache=collision_cache,
        self_collision_check=False,
        optimizer_collision_activation_distance=0.02,
        num_ik_seeds=64,
        position_tolerance = 0.06,
        orientation_tolerance=0.08,
    )
    config_right = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        collision_cache=collision_cache,
        self_collision_check=False,
        optimizer_collision_activation_distance=0.02,
        position_tolerance = 0.06,
        orientation_tolerance=0.08,
        num_ik_seeds=64,
    )
    planner_left = MotionPlanner(config_left)
    planner_left.warmup(enable_graph=True, num_warmup_iterations=5)
    link_list=["panda_leftfinger","panda_rightfinger","panda_hand"]
    planner_left.disable_link_collision(disable_collision_links=link_list)
    planner_left.table_scene = table_scene
    planner_left.empty_scene = empty_scene
    planner_left.sim_scene = scene
    planner_left.collision_cuboid = named_collision_cuboid
    planner_left.collision_meshes = named_collision_meshes
    planner_left.base_offset = (0.0, 0.4, 0.2)
    planner_left._obstacle_mode = "empty"

    planner_right = MotionPlanner(config_right)
    planner_right.warmup(enable_graph=True, num_warmup_iterations=5)
    link_list=["panda_leftfinger","panda_rightfinger","panda_hand"]
    planner_right.disable_link_collision(disable_collision_links=link_list)
    planner_right.table_scene = table_scene
    planner_right.empty_scene = empty_scene
    planner_right.sim_scene = scene
    planner_right.collision_cuboid = named_collision_cuboid
    planner_right.collision_meshes = named_collision_meshes
    planner_right.base_offset = (0.0, -0.4, 0.2)
    planner_right._obstacle_mode = "empty"

    robot_left = scene["robot_left"]
    robot_right = scene["robot_right"]
    cam = scene["head_camera"]
    cam_pos = cam.data.pos_w[0]
    cam_quat = cam.data.quat_w_world[0]   # 你这版 IsaacLab 是 quat_w_world

    # pos = root_pos.unsqueeze(0).to(dtype=torch.float32)    # [1,3]
    # quat = root_quat.unsqueeze(0).to(dtype=torch.float32)  # [1,4]
    # frame_marker.visualize(pos, quat)

    q_start_left = JointState.from_position(
        torch.tensor([[ 0.0000, -1.3000,  0.0000, -2.5000,  0.0000,  1.5000,  0.8000]],
       device='cuda:0'),
        joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'],
    )
    q_start_right = JointState.from_position(
        torch.tensor([[ 0.0000, -1.3000,  0.0000, -2.5000,  0.0000,  1.5000,  0.8000]],
       device='cuda:0'),
        joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'],
    )


    all_joint_names = robot_left.data.joint_names
    log_debug(f"joint names: {all_joint_names}")
    finger_indices = [
        i for i, name in enumerate(all_joint_names) if "finger" in name
    ]

    record_stride = 1
    physics_dt = sim.get_physics_dt()
    fps = max(1, int(round(1.0 / (physics_dt * record_stride))))
    video_fps = max(1, int(round(1.0 / physics_dt)))
    record_episode = bool(args_cli.record)
    video_writer = None
    collector = None
    object_tracker = None
    object_track_errors = 0
    if task_ctx.task_name in ("grap_roller", "place_bread_basket", "scan_object", "place_dual_shoe", "place_dual_shoes"):
        object_tracker = ThirdViewObjectTracker(
            dataset_root=str(PROJECT_ROOT / "data"),
            task_name=task_ctx.task_name,
            id_to_type=getattr(task_ctx, "id_to_type", {}),
        )
    def _dataset_dir(raw: str):
        if not raw:
            return PROJECT_ROOT / "data"
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    dataset_root = _dataset_dir(args_cli.dataset_root)
    dataset_root_env1 = _dataset_dir(args_cli.dataset_root_env1) if args_cli.dataset_root_env1 else dataset_root
    separate_dataset_roots = dataset_root_env1 != dataset_root
    recorders = []
    task_prompt = green_object_task_text(task_ctx.task_name, task_ctx.task_json)
    if record_episode:
        print(f"[INFO] task prompt: {task_prompt}")
        video_writer = EpisodeVideoWriter(dataset_root=str(dataset_root), fps=fps)
        collector = DualArmParquetCollector(
            CollectorConfig(
                output_root=str(dataset_root),
                task_text=task_prompt,
                fps=fps,
                arm_dof=7,
            )
        )
        recorders.append((0, collector, video_writer))
        n_env = int(getattr(scene, "num_envs", 1))
        if n_env > 1:
            import collect_data_utils as _collect_mod

            collector_env1 = DualArmParquetCollector(
                CollectorConfig(
                    output_root=str(dataset_root_env1),
                    task_text=task_prompt,
                    fps=fps,
                    arm_dof=7,
                )
            )
            _collect_mod._ACTIVE_COLLECTOR = collector
            video_env1 = EpisodeVideoWriter(
                dataset_root=str(dataset_root_env1),
                fps=fps,
                stream_prefix="env1_",
            )
            recorders.append((1, collector_env1, video_env1))
            print(
                "[INFO] env_0 annotated dataset "
                f"{dataset_root}; env_1 plain dataset {dataset_root_env1}"
            )
        print(f"[INFO] recording dataset and video under {dataset_root}")
    else:
        collector = None
        video_writer = None
        print("[INFO] dataset recording and video export are off")
    episode_index = int(args_cli.episode_index)
    frame_index = 0
    sim_time_s = 0.0
    sim_dt = sim.get_physics_dt()

    finger_open_left = torch.tensor([0.04, 0.04], device=robot_left.device, dtype=torch.float32)
    finger_open_right = torch.tensor([0.04, 0.04], device=robot_right.device, dtype=torch.float32)
    finger_close_left = torch.tensor([0.0, 0.0], device=robot_left.device, dtype=torch.float32)
    finger_close_right = torch.tensor([0.0, 0.0], device=robot_right.device, dtype=torch.float32)

    world = WorldState(scene=scene, knowledge_map=task_ctx.knowledge_map, env_id=0)
    # Keep the spawned poses. Restoring import poses drops the env origin and
    # pulls env_0 assets behind the arms.
    world.refresh()
    compiler = AtomCompiler(
        knowledge_map=task_ctx.knowledge_map,
        id_to_type=task_ctx.id_to_type,
        dispatch=DISPATCH,
        allowed_atoms=ALLOWED_ATOMS,
        env_id=0,
        import_pose_lookup=task_ctx.import_pose_lookup,
        live_pose_enabled=args_cli.anti_interference,
    )
    runtime = TaskRuntime(
        scene=scene,
        world=world,
        compiler=compiler,
        atomic_plan=task_ctx.atomic_plan,
        planners={"robot_left": planner_left, "robot_right": planner_right},
        robots={"robot_left": robot_left, "robot_right": robot_right},
        q_starts={"robot_left": q_start_left, "robot_right": q_start_right},
        finger_indices={"robot_left": finger_indices, "robot_right": finger_indices},
        finger_open={"robot_left": finger_open_left, "robot_right": finger_open_right},
        finger_close={"robot_left": finger_close_left, "robot_right": finger_close_right},
        move_to_pose_fn=plan_pose,
        link_list=link_list,
        cfg=TaskRuntimeCfg(
            env_id=0,
            max_atom_retries=4,
            hold_frames=40,
            plan_every_n=1,
            drop_test_enabled=args_cli.anti_interference,
            live_pose_enabled=args_cli.anti_interference,
            grasp_check_enabled=args_cli.anti_interference,
        ),
        grasp_cfg=GraspMonitorCfg(),
        reach_fns={"robot_left": whether_reach_left, "robot_right": whether_reach_right},
        scene_markers=ScenePlanMarkers(origin=_marker_origin(scene)),
    )
    planner_left.scene_markers = runtime.scene_markers
    planner_right.scene_markers = runtime.scene_markers
    scene.task_runtime = runtime
    scene.world_state = world
    # Live edit example (anytime during the loop):
    #   runtime.set_pose("bottle_1", pos=(0.55, 0.10, 0.35))
    #   runtime.queue_set_pose("block_1", pos=(0.60, -0.12, 0.32))
    #   runtime.set_state("microwave_1", joint_pos=[0.4])

    settle_count = 0
    still_count = 0
    # After the plan says finished, keep recording while the arms are still
    # moving so the last action is in the video. 100 frames is the shortest
    # tail, 2000 frames is the longest.
    settle_min_frames = 100
    settle_still_frames = 50
    settle_max_frames = 2000
    count=0
    status = "running"
    should_dump = False
    exec_started = time.perf_counter()
    last_preview_rgb: dict[int, np.ndarray] = {}
    grasp_vis_done = False
    grasp_vis_scene = None
    place_can_shots: set[str] = set()
    keyframe_task = task_ctx.task_name in ("place_cans_plasticbox", "grap_roller")
    while simulation_app.is_running():
        status = runtime.tick()
        count+=1
        if runtime.tick_count % 250 == 0:
            snaps = runtime.snapshot()
            compact = {k: [round(x, 3) for x in v["pos"]] for k, v in snaps.items()}
            print(f"[World] tick={runtime.tick_count} atom={runtime.atom_index} {compact}")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if args_cli.preview_dir:
            from collect_data_utils import _copy_camera_rgb

            for preview_env in range(int(getattr(scene, "num_envs", 1))):
                try:
                    last_preview_rgb[preview_env] = _copy_camera_rgb(scene["side_camera"], preview_env)
                except Exception:
                    pass

        if object_tracker is not None:
            try:
                side = scene["side_camera"]
                rgb = side.data.output.get("rgb") if side.data.output is not None else None
                intrinsics = getattr(side.data, "intrinsic_matrices", None)
                if rgb is None or intrinsics is None:
                    raise RuntimeError("side camera has no rgb yet")
                side_pos, side_rot = _head_camera_ros_pose(scene, side)
                side_k = intrinsics[0].detach().cpu().numpy()
                objs = _physics_collision_objects(scene, world)
                try:
                    grasps = _runtime_grasp_marks(runtime, objs)
                except Exception:
                    grasps = []
                try:
                    live_hands = _live_hand_marks(scene, objs)
                except Exception:
                    live_hands = []
                object_tracker.observe(
                    _copy_camera_rgb(side, 0),
                    side_pos,
                    side_rot,
                    side_k,
                    objs,
                    grasps,
                    live_hands,
                )
            except Exception as exc:
                object_track_errors += 1
                if object_track_errors == 1 or object_track_errors % 200 == 0:
                    print(f"[WARN] third-view object track skipped: {exc}")

        if object_tracker is None and keyframe_task:
            if task_ctx.task_name == "grap_roller":
                stage = _roller_stage(runtime, place_can_shots)
            else:
                stage = _place_can_stage(runtime, place_can_shots)
            if stage is not None and stage not in place_can_shots and len(place_can_shots) < 5:
                try:
                    prepared = {"objects": _live_collision_objects(scene, world)}
                    if _save_place_can_keyframe(
                        scene, world, runtime, prepared, stage, task_ctx.task_name
                    ):
                        place_can_shots.add(stage)
                except Exception as exc:
                    print(f"[WARN] place-can keyframe {stage} failed: {exc}")
                    place_can_shots.add(stage)
        elif object_tracker is None and not grasp_vis_done and count >= 8:
            try:
                if grasp_vis_scene is None:
                    grasp_vis_scene = _prepare_head_grasp_scene(
                        scene, world, compiler, task_ctx.atomic_plan
                    )
                grasp_vis_done = _save_head_grasp_annotation(
                    scene, grasp_vis_scene, task_ctx.task_name
                )
            except Exception as exc:
                print(f"[WARN] head grasp annotation failed: {exc}")
                grasp_vis_done = True
            if count >= 40:
                grasp_vis_done = True

        if record_episode:
            for env_id, rec_collector, rec_video in recorders:
                if frame_index % record_stride == 0:
                    rec_video.append_from_scene(scene=scene, env_id=env_id)
                    rec_collector.append_frame_from_sim(
                        scene=scene,
                        env_id=env_id,
                        episode_index=episode_index + env_id,
                        frame_index=frame_index,
                        sim_time_s=sim_time_s,
                        robot_left_key="robot_left",
                        robot_right_key="robot_right",
                        cam_head_key="head_camera",
                        cam_hand_left_key="hand_left_camera",
                        cam_hand_right_key="hand_right_camera",
                        left_ee_body="panda_hand",
                        right_ee_body="panda_hand",
                        left_tcp_body=None,
                        right_tcp_body=None,
                        left_armbase_body=None,
                        right_armbase_body=None,
                        robot_frame_pose7=None,
                    )
        frame_index += 1
        sim_time_s += sim_dt

        if status in ("finished",):
            settle_count += 1
            if _arms_still_moving(runtime):
                still_count = 0
            else:
                still_count += 1
            arms_stopped = settle_count >= settle_min_frames and still_count >= settle_still_frames
            if arms_stopped or settle_count >= settle_max_frames:
                print(
                    f"[INFO] task finished; recording kept until the arms stopped "
                    f"(extra_frames={settle_count})"
                )
                should_dump = True
                break
        if args_cli.max_frames > 0 and frame_index >= int(args_cli.max_frames):
            print(f"[INFO] reached max_frames={int(args_cli.max_frames)}; saving video")
            should_dump = True
            break
        if args_cli.max_seconds > 0 and (time.perf_counter() - exec_started) >= float(args_cli.max_seconds):
            print(f"[INFO] reached max_seconds={int(args_cli.max_seconds)}; saving and closing")
            should_dump = True
            break
        # elif count == 5000:
        #         should_dump = True
        #         break
        

    if object_tracker is not None:
        try:
            object_tracker.finish()
        except Exception as exc:
            print(f"[WARN] third-view track export failed: {exc}")

    if record_episode and should_dump:
        print(f"[INFO] dumping episode status={status} frames={frame_index}")
        for env_id, rec_collector, rec_video in recorders:
            ep = episode_index if separate_dataset_roots else episode_index + env_id
            try:
                rec_collector.dump_episode(ep)
            except Exception as exc:
                print(f"[WARN] dataset dump failed env_{env_id}: {exc}")
            try:
                rec_video.dump_episode(ep)
            except Exception as exc:
                print(f"[WARN] video dump failed env_{env_id}: {exc}")
        try:
            if len(recorders) > 1:
                collector.episode_lengths.extend(recorders[1][1].episode_lengths)
            collector.dump_info_json()
            if separate_dataset_roots and len(recorders) > 1:
                recorders[1][1].dump_info_json()
        except Exception as exc:
            print(f"[WARN] info.json dump failed: {exc}")
    elif record_episode:
        collector.dump_info_json()
        if separate_dataset_roots and len(recorders) > 1:
            recorders[1][1].dump_info_json()
    if args_cli.preview_dir:
        preview_root = Path(args_cli.preview_dir)
        if not preview_root.is_absolute():
            preview_root = PROJECT_ROOT / preview_root
        labels = ("annotated", "plain")
        n_env = int(getattr(scene, "num_envs", 1))
        from collect_data_utils import _copy_camera_rgb

        for env_id in range(n_env):
            try:
                rgb = last_preview_rgb.get(env_id)
                if rgb is None:
                    rgb = _copy_camera_rgb(scene["side_camera"], env_id)
                label = labels[env_id] if env_id < len(labels) else f"env{env_id}"
                out_path = (
                    preview_root
                    / label
                    / str(task_ctx.task_name)
                    / f"episode_{episode_index:06d}.png"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(out_path, rgb)
                print(f"[INFO] third-person last frame: {out_path}")
            except Exception as exc:
                print(f"[WARN] third-person preview failed env_{env_id}: {exc}")

def resolve_task_json(task: str) -> Path:
    """Map a CLI task name to tasks/wooden_box, tasks/rigid, or tasks/articulated/<name>.json."""
    import json

    raw = (task or "").strip()
    root = PROJECT_ROOT / "tasks"
    if not raw:
        return root / "rigid" / "task_plan.json"
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    name = p.stem if p.suffix.lower() == ".json" else raw
    direct = [
        root / "wooden_box" / f"{name}.json",
        root / "rigid" / f"{name}.json",
        root / "articulated" / f"{name}.json",
        root / "rigid" / "task_aloha_agilex" / f"{name}.json",
        root / f"{name}.json",
    ]
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    stem_matches = []
    name_matches = []
    if root.is_dir():
        for candidate in sorted(root.rglob("*.json")):
            if candidate.stem == name:
                stem_matches.append(candidate)
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(data.get("task_name", "")) == name:
                name_matches.append(candidate)
    matches = stem_matches or name_matches
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise SystemExit(f"[ERROR] no task json for {raw!r} under {root}")
    listed = "\n".join(f"  {m}" for m in matches)
    raise SystemExit(f"[ERROR] multiple task jsons match {raw!r}:\n{listed}")


_RUN_STARTED = 0.0


def main():
    """Main function."""
    global _RUN_STARTED
    _RUN_STARTED = time.perf_counter()
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([0.2, -1.5, 0.8], [0.5, 0.0, 0.4])
    # sim.set_camera_view([1.8, 0, 0.65], [0.2, 0.0, 0.5])
    # Two envs sit on +x and -x. Frame both; env_0 (+x) is the one with green marks.
    if args_cli.num_envs > 1:
        sim.set_camera_view([2.6, -3.6, 2.3], [0.0, 0.0, 0.45])
    else:
        sim.set_camera_view(list(VIEW_CAMERA_EYE), list(VIEW_CAMERA_TARGET))


    for n in range(5):
        if n >=1:
            stage = get_current_stage()
            world = stage.GetPrimAtPath("/World")
            envs = stage.GetPrimAtPath("/World/envs")
            if not world.IsValid() or not envs.IsValid():
                raise RuntimeError("World or /World/envs not found")
            # 1) 删全局节点（按需）
            for child in list(world.GetChildren()):
                name = child.GetName()
                if name.startswith("defaultGroundPlane") or name.startswith("DomeLight") or name.startswith("Kitchen"):
                    stage.RemovePrim(child.GetPath())
            # 2) 遍历每个 env，删除你想删的对象
            for env_prim in list(envs.GetChildren()):
                env_name = env_prim.GetName()   # env_0 / env_1 ...
                env_path = f"/World/envs/{env_name}"
                # 删任务资产（my_开头）
                for child in list(env_prim.GetChildren()):
                    name = child.GetName()
                    if name.startswith("my_") or name.startswith("Cu") or name.startswith("Side") or name.startswith("View"):
                        stage.RemovePrim(child.GetPath())
                # 删手爪下你加的相机/frame（示例）
                robot_left_hand = stage.GetPrimAtPath(f"{env_path}/Robot_left/panda_hand")
                if robot_left_hand.IsValid():
                    for child in list(robot_left_hand.GetChildren()):
                        name = child.GetName()
                        if name.startswith("front_cam") :
                            stage.RemovePrim(child.GetPath())
                robot_right_hand = stage.GetPrimAtPath(f"{env_path}/Robot_right/panda_hand")
                robot_right_hand2 = stage.GetPrimAtPath(f"{env_path}/Robot_right")
                if robot_right_hand.IsValid():
                    for child in list(robot_right_hand.GetChildren()):
                        name = child.GetName()
                        if name.startswith("front_cam")  :
                            stage.RemovePrim(child.GetPath())
                    for child in list(robot_right_hand2.GetChildren()):
                        name = child.GetName()
                        if name.startswith("front_cam") or name.startswith("HeadCamera") :
                            stage.RemovePrim(child.GetPath())

        spacing = 3.6 if args_cli.num_envs > 1 else 2.0
        base_scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=spacing, replicate_physics=False)
        backdrop = pick_backdrop_scene() if ENABLE_KITCHEN_USD else None
        if backdrop is not None:
            base_scene_cfg.kitchen = make_backdrop_cfg(backdrop)

        task_json_path = resolve_task_json(args_cli.task)
        print(f"[INFO] task json: {task_json_path}")
        task_ctx = build_task_context(
            task_json_path=str(task_json_path),
            sim=sim,
            base_scene_cfg=base_scene_cfg,
            post_spawn_fn=(
                (
                    lambda scene=backdrop: prepare_kitchen_backdrop(
                        get_current_stage(),
                        kitchen_layout=bool(scene and scene.get("kitchen_layout")),
                    )
                )
                if backdrop is not None
                else None
            ),
        )

        sim.reset()
        hide_oven_collision_visuals(get_current_stage())
        if task_ctx.apply_visual_textures is not None:
            task_ctx.apply_visual_textures()
        if backdrop is not None and backdrop.get("kitchen_layout"):
            offset_kitchen_island(get_current_stage())
        print("[INFO]: Setup complete...")
        run_simulator(
            sim,
            task_ctx.scene,
            task_ctx=task_ctx,
        )
        


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()