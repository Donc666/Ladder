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

from isaaclab.app import AppLauncher
# from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
# from curobo.types import ContentPath, GoalToolPose, JointState, Pose
# from curobo._src.geom.collision import SceneCollisionCfg
# from curobo._src.geom.types import SceneCfg,Cuboid,Obstacle,Mesh,PointCloud,DeviceCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--urdf_path", type=str, default="/home/jiacheng/Partnet_mobility/box/100162/mobility.urdf", help= "urdf文件的路径")
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
import omni
import random
import isaaclab.sim as sim_utils
# from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
# from curobo.types import ContentPath, GoalToolPose, JointState, Pose
# from curobo._src.geom.collision import SceneCollisionCfg
# from curobo._src.geom.types import SceneCfg,Cuboid,Obstacle,Mesh
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import usd.schema.isaac.robot_schema as robot_schema
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo._src.geom.collision import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg,Cuboid,Obstacle,Mesh,PointCloud,DeviceCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners import materials
from isaaclab.assets import AssetBaseCfg,ArticulationCfg,RigidObjectCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply


# from omni.isaac.core.utils.bounds import compute_combined_aabb
# from from_ply_to_voxel import *
from from_ply_to_obs import *
from set_material import *
from urdf_impoter import *
from set_interface import *
import sys
import os

# 检查所有包含 'warp' 的路径
warp_paths = [p for p in sys.path if "warp" in p.lower()]
print("--- Warp Path Detection ---")
for i, path in enumerate(warp_paths):
    print(f"[{i}] {path}")

try:
    import warp as wp
    print(f"Current active Warp: {wp.__file__}")
    print(f"Warp Version: {wp.__version__ if hasattr(wp, '__version__') else 'Unknown'}")
    wp.init()
    print(f"Warp Runtime Status: {wp.context.runtime}")
except Exception as e:
    print(f"Warp Import/Init Error: {e}")
print("---------------------------")


# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
# from curobo.geom.types import WorldConfig

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip



aseet_pos_x = 0.5
aseet_pos_y =0.3
# aseet_pos_x = random.uniform(0.4, 0.9)
# range_coordinate = [(0.2, 0.7), (-0.5, -0.3)]
# index = random.randint(0, 1)
# # selected_range = random.choice([(0.2, 0.7), (-0.6, -0.4)])
# aseet_pos_y = random.uniform(*range_coordinate[index])
# if index == 0:
#     x_cube = random.uniform(0.1,1)
#     y_cube = y - random.uniform(0.4,0.7)
# else:
#     x_cube = random.uniform(0.1,1)
#     y_cube = y + random.uniform(0.4,0.7)
cube_pos_x = 0.5
cube_pos_y = -0.2
scale_box = 0.45
x_cart = 0.7
y_cart = -0.2
z_cart = 0.3
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.2,0.6,0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # robot.spawn = sim_utils.UsdFileCfg(
        #     usd_path=FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path, # 确保路径正确
        #     scale=(1.5, 0.51, 0.5)                             # 直接在这里设置 scale
        # )
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.01, 50.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"),
    )
# ee_goals = [
#         [0.35, 0.25, 0.2, 0.707, 0, 0.707, 0],#1
#         [0.4, 0.25, 0.2, 0.707, 0, 0.707, 0],#2
#         [0.3, 0.33, 0.2, 0.707, 0, 0.707, 0],#3
#         [0.25, 0.4, 0.2, 0.707, 0, 0.707, 0],#4
#         [0.2, 0.45, 0.2, 0.707, 0, 0.707, 0],#5
#         [0.15, 0.5, 0.2, 0.707, 0, 0.707, 0],#6
#         [0.1, 0.5, 0.2, 0.707, 0, 0.707, 0],#7
#         [0.1, 0.25, 0.2, 0.707, 0, 0.707, 0],#8
#         [0.25, 0.2, 0.2, 0.707, 0, 0.707, 0],#9
#         [0.5,-0.3, 0.3, 0, 1, 0, 0],#10
#         [0.5, -0.3, 0.2, 0, 1, 0, 0],#11
#         [0.5, -0.3, 0.2, 0.707, 0, 0.707, 0],#12
#         [0.3, 0.5, 0.2, 0.707, 0, 0.707, 0],#13
#         [0.45, 0.5, 0.3, 0.707, 0, 0.707, 0],#14
#         [0.25, 0.5, 0.3, 0.707, 0, 0.707, 0]
#     ]
# def create_pointcloud_obstacle():
#     # 1. 加载点云
#     ply_path = "/home/jiacheng/7167/point_sample/ply-10000.ply"
#     points = load_ply_to_tensor(ply_path) # 返回 [N, 3] 的 Tensor
    
#     # 2. 实例化 PointCloud 类
#     # 注意：这里的参数名必须是 'points'，对应你提供的源码
#     pc_obstacle = PointCloud(
#         name="microwave_pc",
#         points=points,
#         pose=[0.65, 0.4, 0.25, 1, 0, 0, 0], # 根据你的微波炉实际位置设置
#         color=[0.0, 1.0, 0.0, 1.0]
#     )
    
#     return pc_obstacle
import numpy as np
import torch
from pxr import Usd, UsdGeom, Gf

def extract_mesh_for_curobo(stage: Usd.Stage, prim_path: str) -> dict:
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

    print(f"\n==== [DEBUG] 开始穿透扫描资产: {prim_path} ====")

    for prim in prim_range:
        # 1. 检查节点类型，确保是几何 Mesh 
        if prim.IsA(UsdGeom.Mesh):
            path_string = prim.GetPath().pathString
            path_lower = path_string.lower()
            
            # 2. 精准过滤：只要是物理碰撞分支，且剔除 Looks 材质与局部 GeomSubset
            if "collisions" in path_lower :
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

                # 3. 自适应全局与局部索引，彻底掐死 bad_array_new_length 数组越界
                max_internal_idx = np.max(np_faces)
                if max_internal_idx >= len(np_points):
                    actual_faces = np_faces.tolist()
                else:
                    actual_faces = (np_faces + vertex_count_offset).tolist()

                # 4. 级联世界坐标变换：即便节点在 Prototype 代理中，也能正确算出现实世界坐标
                time_code = Usd.TimeCode.Default()
                world_transform: Gf.Matrix4d = usd_mesh.ComputeLocalToWorldTransform(time_code)
                
                for pt in np_points:
                    world_pt = world_transform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                    all_world_vertices.append([world_pt[0], world_pt[1], world_pt[2]])
                    
                all_face_indices.extend(actual_faces)
                vertex_count_offset += len(np_points)

    print(f"==== [DEBUG] 穿透扫描结束，共抓取到顶点数: {len(all_world_vertices)} ====\n")

    # 5. 此时如果依然为空，说明路径真有问题或模型未加载
    if len(all_world_vertices) == 0:
        raise RuntimeError(f"未能从 {prim_path} 中提取到任何有效的 Mesh 顶点。请检查路径或模型是否加载成功。")

    return {
        "vertices": all_world_vertices,
        "faces": all_face_indices
    }
def add_object_to_scene(cfg: TableTopSceneCfg,
                        name: str, 
                        urdf_path: str, 
                        pos: tuple, 
                        rot: tuple,
                        scale: tuple,
                        diffuse_color: tuple,
                        roughness:float,
                        metallic:float,
                        opacity:float):
    """
    动态往配置类中添加一个 Articulation 变量。
    
    Args:
        cfg: 场景配置实例
        name: 变量名 (如 "microwave", "oven", "fridge")
        urdf_path: URDF 文件路径
        pos: 初始位置
        rot: 初始姿态 (w, x, y, z)
        scale: 缩放比例
    """
    
    # 创建 Articulation 配置
    object_cfg = ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}", # 动态设置场景路径名
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            scale=scale,
            fix_base=True,
            visual_material = materials.PreviewSurfaceCfg(
                diffuse_color= diffuse_color, # 设置 RGB 颜色 (0.0-1.0)
                roughness=roughness,                 # 设置粗糙度 (0: 光滑, 1: 粗糙)
                metallic=metallic,                  # 设置金属度 (0: 绝缘体, 1: 金属)
                opacity= opacity
            ),
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, # 填补报错中提到的 MISSING 字段
                    damping=1.0    # 建议同时指定 damping
                )
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, 
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
        actuators={
            f"{name}_actuator": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                stiffness=0.0,
                damping=1.0,
            ),
        },
    )
    
    # 【核心操作】使用 setattr 动态绑定变量名
    # 效果等同于 cfg.name = object_cfg
    setattr(cfg, name, object_cfg)

# --- 使用方法 ---

# 1. 创建空的场景配置
# # my_cfg = TableTopSceneCfg()

# # 2. 通过接口添加不同命名的物体
# add_object_to_scene(
#     cfg=my_cfg, 
#     name="my_custom_microwave",  # 变量名变为 cfg.my_custom_microwave
#     urdf_path="/path/to/microwave.urdf",
#     pos=(1.0, 0.5, 0.2),
#     rot=(1, 0, 0, 0),
#     scale=(0.3, 0.3, 0.3)
# )

# add_object_to_scene(
#     cfg=my_cfg, 
#     name="another_box",          # 变量名变为 cfg.another_box
#     urdf_path="/path/to/box.urdf",
#     pos=(0.5, -0.5, 0.1),
#     rot=(1, 0, 0, 0),
#     scale=(1.0, 1.0, 1.0)
# )

# # 3. 传入场景加载
# scene = InteractiveScene(my_cfg)

def get_goal_pose(planner,step,x_cart,y_cart,z_cart,scale_num):
    ee_goals = [
        [x_cart-1.1*scale_num, y_cart, z_cart+0.3*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.75*scale_num, y_cart, z_cart+0.3*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.5*scale_num, y_cart, z_cart+1.0*scale_num, 0.707, 0, 0.707, 0],
        [x_cart, y_cart, z_cart+1.0*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.4, y_cart, z_cart+0.17, 0.707, 0, 0.707, 0]
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
        return

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
def grasp_pose(planner,q_start,grasp_poses,disable_collision_links):
    result = planner.plan_grasp(
        current_state=q_start,
        grasp_poses=grasp_poses,
        grasp_approach_offset=0.1,
        grasp_lift_offset=0.1,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=False,
        grasp_lift_in_tool_frame=True,
        disable_collision_links = disable_collision_links,
    )
    traj_list = []
    if result.approach_trajectory is not None:
        # cuRobo 轨迹的 position 形状通常是 [1, Timestep, Num_Joints]
        # squeeze(0) 变成 [Timestep, Num_Joints]
        app_pos = result.approach_trajectory.position.squeeze(0)
        traj_list.append(app_pos)
    if result.grasp_trajectory is not None:
        print("have a grasp traj")
        grasp_pos = result.grasp_trajectory.position.squeeze(0)
        traj_list.append(grasp_pos)
    if len(traj_list) > 0:
        joint_pos_des = torch.cat(traj_list, dim=0)
    # traj = result.get_interpolated_plan()
    # trajectory_positions = traj.position.squeeze(0)
    joint_pos_des_new = joint_pos_des[0]
    return joint_pos_des_new


def whether_reach(scene,x_cart,y_cart,z_cart,scale_num,step):
    ee_goals = [
        [x_cart-1.1*scale_num, y_cart, z_cart+0.3*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.75*scale_num, y_cart, z_cart+0.3*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.5*scale_num, y_cart, z_cart+1.0*scale_num, 0.707, 0, 0.707, 0],
        [x_cart, y_cart, z_cart+1.0*scale_num, 0.707, 0, 0.707, 0],
        [x_cart-0.4, y_cart, z_cart+0.17, 0.707, 0, 0.707, 0]
    ]
    robot = scene["robot"]
    # 1. 初始化：获取末端索引
    ee_link_name = "panda_hand"
    ee_link_idx, _ = robot.find_bodies(ee_link_name)
    pos_threshold = 0.04
    # if step == 1:
    #     pos_threshold = 0.03

    ee_state = robot.data.body_state_w[:, ee_link_idx, :]
    print(f"ee_state:{ee_state}")
    current_ee_pos = ee_state[:, 0:3]
    current_ee_pos = current_ee_pos[0][0]
    print(f"current ee pos:{current_ee_pos[0:3]}")
        
        # 3. 假设 target_pos 是从你的 ee_goals 中提取的当前目标点 [3]
        # 注意：需要确保 target_pos 也是 cuda 上的 tensor
    target_pos_tensor = torch.tensor(ee_goals[step][:3], device=robot.device)
    print(f"target ee pos:{target_pos_tensor}")
        # 4. 计算距离
        # 使用 torch.norm 计算 L2 范数（欧式距离）
    distance = torch.norm(current_ee_pos[0:3] - target_pos_tensor)
        
        # 5. 闭环判断逻辑
    if distance < pos_threshold:
        print(f"检测到已到达目标点 {step}，误差: {distance:.4f}m")
        return True  # 切换到下一个任务
        waypoint_idx = 0 # 重置轨迹索引
    else:
        return False

def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    """变换点云。
    Args:
        points: 点云，形状为 (..., N, 3)。
        pos: 平移向量，形状为 (..., 3)。默认为 None。
        quat: 旋转四元数 (w, x, y, z)，形状为 (..., 4)。默认为 None。
    Returns:
        变换后的点云，形状为 (..., N, 3)。
    """
    if quat is not None:
        # 这一步利用了你代码里已有的 quat_apply
        points = quat_apply(quat.unsqueeze(-2).repeat(*(1,) * (len(quat.shape) - 1), points.shape[-2], 1), points)
    if pos is not None:
        points = points + pos.unsqueeze(-2)
    return points

def get_camera_pointcloud(camera, filter_robot=False, robot_id=None):
    """
    从相机传感器获取实时点云并转换至世界坐标系。
    
    Args:
        camera: scene["camera"] 对象
        filter_robot: 是否根据语义过滤特定物体
        robot_id: 需要保留的语义 ID
    Returns:
        points_world: [N, 3] torch.Tensor，世界坐标系下的点云
    """
    # 1. 获取基础数据
    depth = camera.data.output["distance_to_image_plane"] # [num_envs, H, W]
    K = camera.data.intrinsic_matrices # [num_envs, 3, 3]
    cam_pos_w = camera.data.pos_w
    cam_quat_w = camera.data.quat_w
    
    # 2. 生成像素坐标网格
    rows, cols = torch.meshgrid(
        torch.arange(camera.cfg.height, device=camera.device),
        torch.arange(camera.cfg.width, device=camera.device),
        indexing='ij'
    )
    u = cols.float()
    v = rows.float()

    # 3. 反投影：从 2D 像素 + 深度 到 3D 局部空间
    # 公式: x = (u - cx) * depth / fx, y = (v - cy) * depth / fy
    z = depth
    x = (u - K[:, 0, 2].unsqueeze(-1).unsqueeze(-1)) * z / K[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    y = (v - K[:, 1, 2].unsqueeze(-1).unsqueeze(-1)) * z / K[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    
    points_cam = torch.stack([x, y, z], dim=-1).reshape(depth.shape[0], -1, 3)

    # 4. 转换到世界坐标系
    points_world = transform_points(points_cam, cam_pos_w, cam_quat_w)

    # 5. 可选：语义过滤 (只获取 Articulation)
    if filter_robot and "semantic_seg" in camera.data.output:
        semantic_mask = camera.data.output["semantic_seg"].reshape(depth.shape[0], -1)
        # 仅保留匹配 robot_id 的点
        if robot_id is not None:
            mask = (semantic_mask == robot_id)
            # 注意：这会改变形状，通常返回特定 env 的点
            return points_world[mask] 

    return points_world
#     return gripper_prim_path
def create_custom_scene(x3,y3,z3,mesh_data):

#     microwave_mesh = Mesh(
#     name="microwave_detailed",
#     file_path="/home/jiacheng/7167/microwave_model.obj",
#     pose=[x3+0.15, y3, z3, 1, 0, 0, 0],
#     scale=[0.3, 0.3, 0.3], # 对应你 Isaac Lab 中的 scale
#     device_cfg=DeviceCfg(device="cuda:0")
# )   
    cube = Cuboid(
        name="mic_bottom",
        pose=[x_cart, y_cart, 0.25, 1.0, 0.0, 0.0, 0.0], 
        dims=[0.5, 0.5, 0.5], # 长, 宽, 高 (单位: 米)
        color=[0.4, 0.4, 0.4, 1.0] # 灰色
    )
    curobo_obstacle = Mesh(
    name="my_cart",
    vertices=mesh_data["vertices"],
    faces=mesh_data["faces"],
    pose = [x_cart,y_cart,z_cart,1,0,0,0],
    scale=None, # 因为顶点已经是真实世界尺寸了
    device_cfg=DeviceCfg(device="cuda:0")
)
    # --- 步骤 2: 组装成 SceneCfg ---
    
    # 注意: cuboid 参数接收的是一个列表 [List]
    custom_scene = SceneCfg(
        mesh=[curobo_obstacle],
        # cuboid=[cube]
        
    )
    
    return custom_scene
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,asset_cfg):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    cart_asset = scene["my_trashcan"]
 

    cart_pos = cart_asset.data.root_pos_w[0]

    x_cart = cart_pos[0].item()
    y_cart = cart_pos[1].item()
    z_cart = cart_pos[2].item()
    print(f"the pos of cube{cart_pos}")
    name = asset_cfg.name
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    path = os.path.join("/World/envs/env_0/",name)
    mesh_data = extract_mesh_for_curobo(scene.stage, path)
#     curobo_obstacle = Mesh(
#     name="my_cart",
#     vertices=mesh_data["vertices"],
#     faces=mesh_data["faces"],
#     pose = [x_cart,y_cart,z_cart,1,0,0,0],
#     scale=None, # 因为顶点已经是真实世界尺寸了
#     device_cfg=DeviceCfg(device="cuda:0")
# )
#     cube = Cuboid(
#         name="mic_bottom",
#         pose=[x_cart, y_cart, z_cart, 1.0, 0.0, 0.0, 0.0], 
#         dims=[0.4, 0.4, 0.4], # 长, 宽, 高 (单位: 米)
#         color=[0.4, 0.4, 0.4, 1.0] # 灰色
#     )
    curobo_scene_model = create_custom_scene(x_cart,y_cart,z_cart,mesh_data)
    # my_scene_model = create_custom_scene(x_cart,y_cart,z_cart)
    config = MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model=curobo_scene_model, # 这里直接传入对象
    self_collision_check=True,
    optimizer_collision_activation_distance=0.01,
    num_ik_seeds = 32
)
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5) # 预热 CUDA 图
    
    q_start = JointState.from_position(
        torch.tensor([[ 0.0000, -1.3000,  0.0000, -2.5000,  0.0000,  1.5000,  0.8000]],
       device='cuda:0'),
        joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'],
    )
    # print(f"the defaut para:{planner.default_joint_state.position.unsqueeze(0)}")
    # print(f"the defaut para2423525:{planner.joint_names}")
    goal_pose = get_goal_pose(planner,0,x_cart,y_cart,z_cart,scale_box)
    

    disable_collision_links=["panda_leftfinger", "panda_rightfinger","panda_hand"]
    link_list=["panda_link0","panda_link1","panda_link2","panda_link3","panda_link4","panda_link5","panda_hand","panda_leftfinger", "panda_rightfinger"]
    planner.disable_link_collision(disable_collision_links=link_list)
    joint_pos_des = move_to_pose(planner,q_start,goal_pose)


    all_joint_names = robot.data.joint_names
    print(f"the names:{all_joint_names}")
    # 筛选包含 'finger' 的关节索引
    finger_indices = [
        i for i, name in enumerate(all_joint_names) if "finger" in name
    ]

    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
 # 1. 预先定义好手指指令
    finger_open = torch.tensor([0.04, 0.04], device=robot.device, dtype=torch.float32)
    finger_close = torch.tensor([0.0, 0.0], device=robot.device, dtype=torch.float32)
    # alpha=0.05 
    waypoint_idx = 0
    step = 0 
    step_copy = 0
    count = 0
    # joint_pos_des = trajectory_positions[0]
# 2. 在循环内统一逻辑
    while simulation_app.is_running():
        count += 1
        if count > 500:
            if waypoint_idx < joint_pos_des.shape[0]:

                    # 获取当前路点并发送给机器人
                    joint_each_time_pos_des = joint_pos_des[waypoint_idx]
                    # print(f"the size of all traj:{trajectory_positions.shape}")
                    print(f"the size of single traj:{joint_pos_des.shape[0]}")
                    print(f"the size of single traj 22:{joint_pos_des[waypoint_idx].unsqueeze(0).shape}")
                    print(joint_pos_des[waypoint_idx])
                    print(f"the step of sim:{waypoint_idx}")
                    print(f"the step of motion:{step}")
                    robot.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
                    waypoint_idx += 1
            if waypoint_idx == joint_pos_des.shape[0] and whether_reach(scene,x_cart,y_cart,z_cart,scale_box,step):
                step_copy = step
                step +=1
                waypoint_idx = 0
            if step == 0 :
                current_finger_target = finger_open
                robot.set_joint_position_target(current_finger_target, joint_ids=finger_indices)
            # if step == 2 or step == 3 or step == 4 or step == 5 or step == 11 or step == 12 or step == 13 :
            #     current_finger_target = finger_close
            #     robot.set_joint_position_target(current_finger_target, joint_ids=finger_indices)
            if step == 2 :
                current_finger_target = finger_close
                robot.set_joint_position_target(current_finger_target, joint_ids=finger_indices)
            if step != step_copy:
                step_copy += 1
                goal_pose = get_goal_pose(planner,step,x_cart,y_cart,z_cart,scale_box)
                print(f"the goal pose now is :{goal_pose}")
                q_start = JointState.from_position(
                    joint_each_time_pos_des[:7].unsqueeze(0),
                    joint_names=planner.joint_names,
                )
                joint_pos_des = move_to_pose(planner,q_start,goal_pose)
            # if step == 14:
            #     simulation_app.close()
                # apply actions
            
            scene.write_data_to_sim()
                # perform step
            sim.step()
                # update sim-time
            # count += 1
                # update buffers
            scene.update(sim_dt)

            # obtain quantities from simulation
        # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        #     # update marker positions
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])



        # # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        # # print("-------------------------------")
        # print(scene["height_scanner"])
        # print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")
        # print(scene["contact_forces"])
        # print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # camera_prim_path = "/World/envs/env_0/Robot/panda_hand/front_cam"
    # viewport_api = viewport_utils.get_viewport_interface()
    # viewport_window = viewport_api.get_viewport_window()

    # if viewport_window:
    #     viewport_window.set_active_camera(camera_prim_path)
    cart_cfg = AssetCfg(
        name = "my_trashcan",
        urdf_path="/home/jiacheng/Partnet_mobility/TrashCan/12447/mobility.urdf",
        pos = (x_cart, y_cart, z_cart),
        rot = (1, 0, 0, 0),
        scale = (scale_box, scale_box,scale_box),
        diffuse_color=(1.0,0.5,0.5),
        # roughness=0.5,
        # metallic=0,
        # opacity=1,
        static_friction=0.8,
        dynamic_friction= 0.8,
        restitution= 0,
        compliant_contact_damping=0,
        compliant_contact_stiffness=0,
        fix_base = False
    )
    base_scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    builder = SceneBuilder(assets=[cart_cfg],base_scene_cfg=base_scene_cfg)
    actual_scene = builder.scene

    sim.reset()
    # wp.init()

    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, actual_scene,cart_cfg)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()