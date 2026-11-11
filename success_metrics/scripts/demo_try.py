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
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher
# from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
# from curobo.types import ContentPath, GoalToolPose, JointState, Pose
# from curobo._src.geom.collision import SceneCollisionCfg
# from curobo._src.geom.types import SceneCfg,Cuboid,Obstacle,Mesh,PointCloud,DeviceCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--urdf_path", type=str, default="/home/jiacheng/7167/mobility.urdf", help="微波炉 USD 文件的路径")
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
from isaaclab.sensors import CameraCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.core.utils.bounds import compute_combined_aabb
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, Gf
import kb_path  # noqa: F401
from set_interface_try import AssetCfg, SceneBuilder
from microwave import *
from dispenser import *
from auto_collect_data import *
from collect_data_utils import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    concept_root: str = "/home/jiacheng/Partnet_Mobility_urdf",
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

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # my_material = sim_utils.PreviewSurfaceCfg(
    #     diffuse_color=(0.2, 0.5, 0.8), # 蓝色
    #     roughness=0.2,                 # 光滑度
    #     metallic=0.5                   # 金属感
    # )
    # ground plane
    # cfg_ground = sim_utils.GroundPlaneCfg()
    # cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.2,0.6,0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # lights: use a dedicated dome light for global ambient illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    # # mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    #     ),
    # )
    # path = "/home/jiacheng/41083/mobility/mobility.urdf"
    # urdf_impoter_articulation(path,x+1,y+1,1)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 10, 6), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.961, 0.871, 0.702),roughness=0.2, metallic=0.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-3.5, 0, 3.0),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )

   
    
    robot_right = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_right",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, -0.2, 0.2),   # x, y, z
                rot=(1.0, 0.0, 0.0, 0.0),  # qw, qx, qy, qz
        ))
    robot_left = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_left",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.4, 0.2),   # x, y, z
                rot=(1.0, 0.0, 0.0, 0.0),  # qw, qx, qy, qz
        ))




    hand_left_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot_left/panda_hand/front_cam",
    update_period=0.0,  # 每物理步更新
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
    update_period=0.0,  # 每物理步更新
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
    update_period=0.0,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(-0.1, 0.2, 1.0),            # 你可调整
        rot=(0.9063, 0.0, 0.4226, 0.0),  # 略向下看，wxyz
        convention="world",
    ),
)
    
    side_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/SideCamera",
    update_period=0.0,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=1.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 50.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 2.0, 0.9),            # 固定在桌子侧上方
        rot=(0.000, -0.087, 0.000, 0.996),  # 约俯视45°（wxyz）
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
    record_curobo_plan(planner, result, arm_dof=6)
    return joint_pos_des
    # if waypoint_idx < joint_pos_des.shape[0]:
    #     joint_each_time_pos_des = joint_pos_des[waypoint_idx]
    #     robot.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
    #     waypoint_idx += 1
    #     return True
    # else:
    #     return False
def whether_reach_left(scene,goal_pose):   
    ee_goals = goal_pose
    robot = scene["robot_left"]
    # 1. 初始化：获取末端索引
    ee_link_name = "panda_hand"
    ee_link_idx, _ = robot.find_bodies(ee_link_name)
    pos_threshold = 0.1
    # if step == 1:
    #     pos_threshold = 0.03

    ee_state = robot.data.body_state_w[:, ee_link_idx, :]
    print(f"ee_state:{ee_state}")
    current_ee_pos = ee_state[:, 0:3]
    current_ee_pos = current_ee_pos[0][0]
    # current_ee_pos[0] -=1
    # current_ee_pos[1] +=1
    print(f"current ee pos:{current_ee_pos[0:3]}")
        
        # 3. 假设 target_pos 是从你的 ee_goals 中提取的当前目标点 [3]
        # 注意：需要确保 target_pos 也是 cuda 上的 tensor
    target_pos_tensor = torch.tensor(ee_goals[:3], device=robot.device)
    print(f"target ee pos:{target_pos_tensor}")
        # 4. 计算距离
        # 使用 torch.norm 计算 L2 范数（欧式距离）
    distance = torch.norm(current_ee_pos[0:3] - target_pos_tensor)
        
        # 5. 闭环判断逻辑
    if distance < pos_threshold:
        return True  # 切换到下一个任务
        waypoint_idx = 0 # 重置轨迹索引
    else:
        return False
def whether_reach_right(scene,goal_pose):   
    ee_goals = goal_pose
    robot = scene["robot_right"]
    # 1. 初始化：获取末端索引
    ee_link_name = "panda_hand"
    ee_link_idx, _ = robot.find_bodies(ee_link_name)
    pos_threshold = 0.1
    # if step == 1:
    #     pos_threshold = 0.03

    ee_state = robot.data.body_state_w[:, ee_link_idx, :]
    print(f"ee_state:{ee_state}")
    current_ee_pos = ee_state[:, 0:3]
    current_ee_pos = current_ee_pos[0][0]
    # current_ee_pos[0] -=1
    # current_ee_pos[1] +=1
    print(f"current ee pos:{current_ee_pos[0:3]}")
        
        # 3. 假设 target_pos 是从你的 ee_goals 中提取的当前目标点 [3]
        # 注意：需要确保 target_pos 也是 cuda 上的 tensor
    target_pos_tensor = torch.tensor(ee_goals[:3], device=robot.device)
    print(f"target ee pos:{target_pos_tensor}")
        # 4. 计算距离
        # 使用 torch.norm 计算 L2 范数（欧式距离）
    distance = torch.norm(current_ee_pos[0:3] - target_pos_tensor)
        
        # 5. 闭环判断逻辑
    if distance < pos_threshold:
        return True  # 切换到下一个任务
        waypoint_idx = 0 # 重置轨迹索引
    else:
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
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    pose_pack,
    
):
    """Runs the simulation loop."""






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
    frame_marker = VisualizationMarkers(frame_cfg)
    obstacle_pose = [x_micwave-10000, y_micwave-10000, 0.0, 1, 0, 0, 0]

#     curobo_obstacle = Mesh(
#     name="my_cart",
#     vertices=static_mesh_data["vertices"],
#     faces=static_mesh_data["faces"],
#     pose = obstacle_pose,
#     scale=None # 因为顶点已经是真实世界尺寸了
# )
    cube = Cuboid(
        name="mic_bottom",
        pose=[100, 100, 0.25, 1.0, 0.0, 0.0, 0.0], 
        dims=[0.1, 0.1, 0.1], # 长, 宽, 高 (单位: 米)
        color=[0.4, 0.4, 0.4, 1.0] # 灰色
    )
    curobo_scene_model = SceneCfg(
        # mesh=[curobo_obstacle]
        cuboid=[cube]
        # mesh=[microwave_mesh]
        
    )

    config_left = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        self_collision_check=False,
        optimizer_collision_activation_distance=1000,
        num_ik_seeds=512,
        position_tolerance = 0.1,
        orientation_tolerance= 0.1,
    )
    config_right = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        self_collision_check=False,
        optimizer_collision_activation_distance=1000,
        position_tolerance = 0.2,
        orientation_tolerance= 0.1,
        num_ik_seeds=512,
    )
    planner_left = MotionPlanner(config_left)
    planner_left.warmup(enable_graph=True, num_warmup_iterations=5)
    link_list=["panda_leftfinger","panda_rightfinger","panda_hand"]
    planner_left.disable_link_collision(disable_collision_links=link_list)

    planner_right = MotionPlanner(config_right)
    planner_right.warmup(enable_graph=True, num_warmup_iterations=5)
    link_list=["panda_leftfinger","panda_rightfinger","panda_hand"]
    planner_right.disable_link_collision(disable_collision_links=link_list)

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
    # 筛选包含 'finger' 的关节索引
    finger_indices = [
        i for i, name in enumerate(all_joint_names) if "finger" in name
    ]
    fps = int(round(1.0 / sim.get_physics_dt()))
    
    video_writer = EpisodeVideoWriter(dataset_root=str(PROJECT_ROOT / "data"), fps=fps)
    collector = DualArmParquetCollector(
    CollectorConfig(
        output_root=str(PROJECT_ROOT / "data"),
        fps=fps,
        arm_dof=7,  # 对齐目标数据集
    )
)
    episode_index = 0
    frame_index = 0
    sim_time_s = 0.0

    sim_dt = sim.get_physics_dt()

    

 # 1. 预先定义好手指指令

    finger_close_left = torch.tensor([0.0, 0.0], device=robot_left.device, dtype=torch.float32)
    finger_close_right = torch.tensor([0.0, 0.0], device=robot_right.device, dtype=torch.float32)
    waypoint_idx = 0
    step = 0 
    joint_pos_des = None

    gripper_cmd = "open"            
    need_hold_this_step = False     
    hold_counter = 0
    hold_frames = 40
    prev_gripper_cmd = "open"
    count = 0
    count2 = 0 
    frames = []
    env_id = 0
    # joint_pos_des = trajectory_positions[0]
# 2. 在循环内统一逻辑
    while simulation_app.is_running():
        pos = cam_pos.unsqueeze(0).to(dtype=torch.float32)    # [1,3]
        quat = cam_quat.unsqueeze(0).to(dtype=torch.float32)  # [1,4]
        frame_marker.visualize(pos, quat)
        # cam = scene["camera"]
        # rgb = cam.data.output["rgb"]              # [num_envs, H, W, 4] (RGBA)
        # frame = rgb[env_id, ..., :3].detach().cpu().numpy().astype(np.uint8)  # RGB
        # frames.append(frame)
        count2 += 1
        if count2 % 2 ==0:

            if step < len(pose_pack):
                is_first_close_of_run = (gripper_cmd == "close" and prev_gripper_cmd != "close")
                task = pose_pack[step]
                executor = task.get("executor", "robot_right")
                if executor == "robot_left":
                    if joint_pos_des is None and step < len(pose_pack):   
                        task = pose_pack[step]         
                        pose = task["pose"]
                        pos = torch.tensor([pose[0:3]], dtype=torch.float32, device="cuda")
                        quat = torch.tensor([pose[3:7]], dtype=torch.float32, device="cuda")
                        # frame_marker.visualize(pos, quat)
                        gripper_cmd = task["gripper"]   # "open" or "close"
                        use_oa = task.get("use_obstacle_avoidance", True)
                        if use_oa:
                            planner_left.enable_link_collision(enable_collision_links=link_list)
                        else:
                            planner_left.disable_link_collision(disable_collision_links=link_list)
                        pos = pose[:3]
                        pos[1] = pose[1]-0.4
                        pos[2] = pose[2]-0.2
                        q = pose[3:7]
                        print(f"the whole pack pose :{pose_pack}")
                        print(f"the goal pose {pose}")
                        goal_pose = GoalToolPose(
                            tool_frames=planner_left.tool_frames,
                            position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
                        )
                        print(f"the goal pose:{pose}")
                        joint_pos_des = move_to_pose(planner_left, q_start_left, goal_pose)
                        if joint_pos_des == None:
                            return

                        print("have finished planning")

                        is_first_close_of_run = (gripper_cmd == "close" and prev_gripper_cmd != "close")
                        need_hold_this_step = is_first_close_of_run
                        hold_counter = 0
                        waypoint_idx = 0

                        if joint_pos_des is None:
                            print(f"[WARN] 目标点 {step} 规划失败，跳过")
                            step += 1
                            continue

                    
                    elif joint_pos_des is not None and waypoint_idx < joint_pos_des.shape[0]:
                            joint_each_time_pos_des = joint_pos_des[waypoint_idx]
                            robot_left.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
                            print(f"current waypoint_idx:{waypoint_idx}")
                            print(f"max waypoints : {joint_pos_des.shape[0]}")
                        
                            if need_hold_this_step and waypoint_idx == 0 and hold_counter < hold_frames:
                                    hold_counter += 1
                            else:
                                    waypoint_idx += 1

                    elif joint_pos_des is not None and waypoint_idx == joint_pos_des.shape[0] and whether_reach_left(scene=scene,goal_pose=pose):
                            q_start_left = JointState.from_position(
                                joint_each_time_pos_des[:7].unsqueeze(0),
                                joint_names=planner_left.joint_names,
                            )
                            step += 1
                            joint_pos_des = None 
                            prev_gripper_cmd = gripper_cmd
                    if step >= len(pose_pack) and joint_pos_des is None:
                            step = 0
                            # sum_step += step+1 


                    if gripper_cmd == "close":
                                current_finger_target = finger_close_left
                                robot_left.set_joint_position_target(current_finger_target, joint_ids=finger_indices)
                else:
                    if joint_pos_des is None and step < len(pose_pack):            
                        pose = task["pose"]
                        pose = task["pose"]
                        pos = torch.tensor([pose[0:3]], dtype=torch.float32, device="cuda")
                        quat = torch.tensor([pose[3:7]], dtype=torch.float32, device="cuda")
                        # frame_marker.visualize(pos, quat)
                        gripper_cmd = task["gripper"]   # "open" or "close"
                        use_oa = task.get("use_obstacle_avoidance", True)
                        if use_oa:
                            planner_right.enable_link_collision(enable_collision_links=link_list)
                        else:
                            planner_right.disable_link_collision(disable_collision_links=link_list)
                        pos = pose[:3]
                        pos[1] = pose[1]+0.2
                        pos[2] = pose[2]-0.2
                        q = pose[3:7]
                        print(f"the whole pack pose :{pose_pack}")
                        print(f"the goal pose {pose}")
                        goal_pose = GoalToolPose(
                            tool_frames=planner_right.tool_frames,
                            position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
                        )
                        joint_pos_des = move_to_pose(planner_right, q_start_right, goal_pose)
                        if joint_pos_des == None:
                            step == len(pose_pack) - 1
                            
                        print("have finished planning")

                        is_first_close_of_run = (gripper_cmd == "close" and prev_gripper_cmd != "close")
                        need_hold_this_step = is_first_close_of_run
                        hold_counter = 0
                        waypoint_idx = 0

                        if joint_pos_des is None:
                            print(f"[WARN] 目标点 {step} 规划失败，跳过")
                            step += 1
                            continue

                    
                    elif joint_pos_des is not None and waypoint_idx < joint_pos_des.shape[0]:
                            joint_each_time_pos_des = joint_pos_des[waypoint_idx]
                            robot_right.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
                            print(f"current waypoint_idx:{waypoint_idx}")
                            print(f"max waypoints : {joint_pos_des.shape[0]}")
                        
                            if need_hold_this_step and waypoint_idx == 0 and hold_counter < hold_frames:
                                    hold_counter += 1
                            else:
                                    waypoint_idx += 1

                    elif joint_pos_des is not None and waypoint_idx == joint_pos_des.shape[0] and whether_reach_right(scene=scene,goal_pose=pose):
                            q_start_right = JointState.from_position(
                                joint_each_time_pos_des[:7].unsqueeze(0),
                                joint_names=planner_right.joint_names,
                            )
                            step += 1
                            joint_pos_des = None 
                            prev_gripper_cmd = gripper_cmd
                    if step >= len(pose_pack) and joint_pos_des is None:
                            step = 0
                            # sum_step += step+1 


                    if gripper_cmd == "close":
                                current_finger_target = finger_close_right
                                robot_right.set_joint_position_target(current_finger_target, joint_ids=finger_indices)

            # if step == 1:
            #     fps = int(round(1.0 / sim.get_physics_dt()))
            #     imageio.mimsave("output_env0.mp4", frames, fps=fps, codec="libx264")

        video_writer.append_from_scene(
        scene=scene,
        env_id=0,
    )
        collector.append_frame_from_sim(
        scene=scene,
        env_id=0,
        episode_index=episode_index,
        frame_index=frame_index,
        sim_time_s=sim_time_s,
        robot_left_key="robot_left",
        robot_right_key="robot_right",
        cam_head_key="head_camera",
        cam_hand_left_key="hand_left_camera",
        cam_hand_right_key="hand_right_camera",
        left_ee_body="panda_hand",
        right_ee_body="panda_hand",
        left_tcp_body=None,   # 没有独立 tcp body 时用 None
        right_tcp_body=None,
        left_armbase_body=None,
        right_armbase_body=None,
        robot_frame_pose7=None,  # None = 用 left robot root 当 robot frame
    )
        frame_index += 1
        sim_time_s += sim_dt
        print(f"the step:{step}")
        if step == len(pose_pack) - 1 :
            count +=1
            if count == 40:
                count = 0
                collector.dump_episode(episode_index)
                video_writer.dump_episode(episode_index)
                episode_index += 1
                frame_index = 0
                sim_time_s = 0.0
                return

        scene.write_data_to_sim()
   
        sim.step()

        scene.update(sim_dt)
    collector.dump_info_json()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.2, -1.5, 0.8], [0.5, 0.0, 0.4])
    sim.set_camera_view([1.8, 0, 1.0], [0.1, 0.0, 0.7])



    for n in range(2):
        if n >=1:
            stage = get_current_stage()
            world = stage.GetPrimAtPath("/World")
            envs = stage.GetPrimAtPath("/World/envs")
            if not world.IsValid() or not envs.IsValid():
                raise RuntimeError("World or /World/envs not found")
            # 1) 删全局节点（按需）
            for child in list(world.GetChildren()):
                name = child.GetName()
                if name.startswith("defaultGroundPlane") or name.startswith("DomeLight"):
                    stage.RemovePrim(child.GetPath())
            # 2) 遍历每个 env，删除你想删的对象
            for env_prim in list(envs.GetChildren()):
                env_name = env_prim.GetName()   # env_0 / env_1 ...
                env_path = f"/World/envs/{env_name}"
                # 删任务资产（my_开头）
                for child in list(env_prim.GetChildren()):
                    name = child.GetName()
                    if name.startswith("my_") or name.startswith("Cu") or name.startswith("Side"):
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

        base_scene_cfg = TableTopSceneCfg(num_envs=4, env_spacing=2.0)

        scene,pose_pack = build_scene_and_pose_pack(task_json_path=str(PROJECT_ROOT / "tasks" / "task_plan.json"),sim=sim,base_scene_cfg=base_scene_cfg)

        sim.reset()
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(
            sim,
            scene,
            pose_pack =pose_pack,
        )
        


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()