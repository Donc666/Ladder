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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from dataclasses import dataclass
from typing import Any, Callable

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
parser.add_argument(
    "--task_jsons",
    type=str,
    nargs="+",
    required=True,
    help="每个环境使用的任务json列表",
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
from auto_collect_data import ASSET_FACTORY, ALLOWED_ATOMS, DISPATCH


# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
# from curobo.geom.types import WorldConfig

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip




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
    # cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.05, 0.05, 0.05), # 方块大小：5cm x 5cm x 5cm
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
    #         collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #             pos=(x_cube, y_cube, 0.025),
    #             rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
    #             lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
    #             ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
    #         ),
    # )

   
    
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


    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_left/panda_hand/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.01, 50.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"),
    )

CONVERTED_ROOT = Path(__file__).resolve().parent.parent / "converted_assets"
def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_")
def _find_asset_dir(asset: dict, root: Path) -> Path:
    """
    资产目录匹配优先级:
    1) asset["name"]
    2) asset["id"]
    3) asset["type"]
    支持:
      - 精确匹配: 001_bottle
      - 后缀匹配: *_bottle
      - 包含匹配: contains 'bottle'
    """
    keys = [asset.get("name"), asset.get("id"), asset.get("type")]
    keys = [k for k in keys if isinstance(k, str) and k.strip()]
    dirs = [p for p in root.iterdir() if p.is_dir()]
    norm_names = {d: _norm(d.name) for d in dirs}
    for key in keys:
        k = _norm(key)
        exact = [d for d in dirs if norm_names[d] == k]
        if exact:
            return exact[0]
        suffix = [d for d in dirs if norm_names[d].endswith(f"_{k}")]
        if suffix:
            return random.choice(suffix)
        contain = [d for d in dirs if k in norm_names[d]]
        if contain:
            return random.choice(contain)
    raise FileNotFoundError(f"Cannot match asset dir for asset={asset} in {root}")
def pick_random_urdf(asset_item: dict, root: Path = CONVERTED_ROOT) -> str:
    """
    按资产信息在 converted_assets 中随机挑一个 urdf。
    优先级：
    1) asset_item["asset_name"] 精确匹配目录
    2) asset_item["id"] 精确匹配目录
    3) 按 asset_item["type"] 匹配 *_type 目录（如 *_bottle）
    """
    if not root.exists():
        raise FileNotFoundError(f"converted_assets root not found: {root}")
    # 可用于匹配目录名的候选关键字
    keys = []
    for k in ("asset_name", "id", "type"):
        v = asset_item.get(k)
        if isinstance(v, str) and v.strip():
            keys.append(v.strip())
    candidate_dirs = []
    # 先精确目录名匹配
    for key in keys:
        p = root / key
        if p.is_dir():
            candidate_dirs.append(p)
    # 再按 type 模糊匹配，比如 001_bottle, 093_brush-pen
    a_type = asset_item.get("type", "")
    if isinstance(a_type, str) and a_type:
        candidate_dirs.extend([p for p in root.glob(f"*_{a_type}") if p.is_dir()])
    # 去重
    candidate_dirs = list(dict.fromkeys(candidate_dirs))
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No asset directory found for item={asset_item}. "
            f"Tried keys={keys} under {root}"
        )
    # 随机选一个对象目录，再随机选一个 urdf
    chosen_dir = random.choice(candidate_dirs)
    urdf_dir = chosen_dir / "urdf"
    urdf_files = sorted(urdf_dir.glob("*.urdf")) if urdf_dir.is_dir() else sorted(chosen_dir.rglob("*.urdf"))
    if not urdf_files:
        raise FileNotFoundError(f"No .urdf found in {chosen_dir}")
    return str(random.choice(urdf_files))
@dataclass
class ArmState:
    tasks: list
    step_idx: int = 0
    joint_pos_des: Any = None
    waypoint_idx: int = 0
    gripper_cmd: str = "open"
    prev_gripper_cmd: str = "open"
    need_hold: bool = False
    hold_counter: int = 0
    q_start: JointState | None = None
def split_pose_pack_by_executor(pose_pack: list[dict]):
    left, right = [], []
    for t in pose_pack:
        ex = t.get("executor", "robot_right")
        if ex == "robot_left":
            left.append(t)
        else:
            right.append(t)
    return left, right
def set_joint_target_one_env(robot, q_row: torch.Tensor, env_id: int, joint_ids=None):
    env_ids = torch.tensor([env_id], device=robot.device, dtype=torch.long)
    try:
        robot.set_joint_position_target(q_row.unsqueeze(0), joint_ids=joint_ids, env_ids=env_ids)
    except TypeError:
        # 兼容不支持 env_ids 的版本：构造全env命令
        cur = robot.data.joint_pos.detach().clone()
        if joint_ids is None:
            cur[env_id, : q_row.numel()] = q_row
            robot.set_joint_position_target(cur, joint_ids=None)
        else:
            cur[env_id, joint_ids] = q_row
            robot.set_joint_position_target(cur[:, joint_ids], joint_ids=joint_ids)
def move_to_pose(planner, q_start, goal_pose):
    result = planner.plan_pose(goal_pose, q_start)
    if result is None:
        return None
    traj = result.get_interpolated_plan()
    return traj.position.squeeze(0)[0]  # [T,7]
def reach_env(scene, robot_key: str, goal_pose, env_id: int, pos_threshold=0.10):
    robot = scene[robot_key]
    ee_link_idx, _ = robot.find_bodies("panda_hand")
    ee_state = robot.data.body_state_w[:, ee_link_idx, :]
    ee_state[0]-=1
    ee_state[1]+=1
    print("i am still judging")
    cur = ee_state[env_id, 0, 0:3]
    tgt = torch.tensor(goal_pose[:3], device=robot.device, dtype=torch.float32)
    return bool(torch.norm(cur - tgt) < pos_threshold)

def whether_reach(scene,goal_pose):   
    ee_goals = goal_pose
    robot = scene["robot_left"]
    # 1. 初始化：获取末端索引
    ee_link_name = "panda_hand"
    ee_link_idx, _ = robot.find_bodies(ee_link_name)
    pos_threshold = 0.1
    # if step == 1:
    #     pos_threshold = 0.03

    ee_state = robot.data.body_state_w[:, ee_link_idx, :]

    # print(f"ee_state:{ee_state}")
    current_ee_pos = ee_state[:, 0:3]
    current_ee_pos = current_ee_pos[0][0]
    current_ee_pos[0] -=1
    current_ee_pos[1] +=1

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
def tick_arm_env(
    env_id: int,
    arm: ArmState,
    scene,
    planner,
    robot,
    robot_key: str,
    link_list,
    finger_indices,
    finger_open,
    finger_close,
    hold_frames,
    y_offset,
    z_offset,
):
    if arm.step_idx >= len(arm.tasks):
        set_joint_target_one_env(robot, finger_open, env_id, joint_ids=finger_indices)
        return
    task = arm.tasks[arm.step_idx]
    pose = task["pose"]
    if len(pose) != 7:
        raise ValueError(f"[env{env_id}] invalid pose len={len(pose)}: {pose}")
    arm.gripper_cmd = task["gripper"]
    # 1) planning
    if arm.joint_pos_des is None:
        use_oa = task.get("use_obstacle_avoidance", True)
        if use_oa:
            planner.enable_link_collision(enable_collision_links=link_list)
        else:
            planner.disable_link_collision(disable_collision_links=link_list)
        pos = pose[:3].copy()
        pos[1] += y_offset
        pos[2] += z_offset

        print(f"the goal pose:{pos}")
        q = pose[3:7]
        goal_pose = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
        )
        arm.joint_pos_des = move_to_pose(planner, arm.q_start, goal_pose)
        is_first_close = (arm.gripper_cmd == "close" and arm.prev_gripper_cmd != "close")
        arm.need_hold = is_first_close
        arm.hold_counter = 0
        arm.waypoint_idx = 0
        if arm.joint_pos_des is None:
            print(f"[WARN][env{env_id}][{robot_key}] plan failed, skip step {arm.step_idx}")
            arm.step_idx += 1
            return
    # 2) execute
    if arm.waypoint_idx < arm.joint_pos_des.shape[0]:
        q_way = arm.joint_pos_des[arm.waypoint_idx]
        set_joint_target_one_env(robot, q_way, env_id, joint_ids=None)
        if arm.need_hold and arm.waypoint_idx == 0 and arm.hold_counter < hold_frames:
            arm.hold_counter += 1
        else:
            arm.waypoint_idx += 1
    # 3) transition
    if arm.waypoint_idx >= arm.joint_pos_des.shape[0]:
        if whether_reach(scene,  pose):
            last_q = arm.joint_pos_des[-1]
            arm.q_start = JointState.from_position(
                last_q[:7].unsqueeze(0),
                joint_names=planner.joint_names,
            )
            arm.step_idx += 1
            arm.joint_pos_des = None
            arm.prev_gripper_cmd = arm.gripper_cmd
    # 4) gripper
    if arm.gripper_cmd == "close":
        set_joint_target_one_env(robot, finger_close, env_id, joint_ids=finger_indices)
    else:
        set_joint_target_one_env(robot, finger_open, env_id, joint_ids=finger_indices)


def build_scene_and_env_pose_packs(task_json_paths: list[str], sim, base_scene_cfg):
    """
    输入多个 task json，返回：
      scene: 统一场景（资产并集）
      env_pose_packs: 每个json对应一个pose_pack
    """
    tasks = []
    for p in task_json_paths:
        with open(p, "r", encoding="utf-8") as f:
            t = json.load(f)
        if "assets" not in t or "atomic_plan" not in t:
            raise ValueError(f"task json missing fields: {p}")
        tasks.append(t)
    # ---------- A) 合并资产（并集） ----------
    # key: asset_id -> {"type":..., "urdf_path":...}
    merged_assets = {}
    for t in tasks:
        for a in t["assets"]:
            aid = a["id"]
            urdf_path = pick_random_urdf(a)
            spec = {"type": a["type"], "urdf_path": urdf_path}
            
            if aid in merged_assets:
                # 冲突保护：同id不能映射到不同type/path
                old = merged_assets[aid]
                if old["type"] != spec["type"] or old["urdf_path"] != spec["urdf_path"]:
                    raise ValueError(
                        f"asset id conflict on '{aid}': {old} vs {spec}. "
                        f"请在json里改成不同id（如 bottle_1/bottle_2）"
                    )
            else:
                merged_assets[aid] = spec
    # ---------- B) 创建知识对象与导入资产 ----------
    knowledge_map: dict[str, Any] = {}
    id_to_type: dict[str, str] = {}
    assets_cfg = []
    for aid, spec in merged_assets.items():
        a_type = spec["type"]
        urdf_path = spec["urdf_path"]
        if a_type not in ASSET_FACTORY:
            raise ValueError(f"Unknown asset type: {a_type}")
        obj = ASSET_FACTORY[a_type]()
        knowledge_map[aid] = obj
        id_to_type[aid] = a_type
        # 给每个实例唯一scene名字，避免同类型多实例冲突
        scene_name = f"my_{a_type}_{aid}"
        # 兼容两种签名：import2sim(urdf_path, instance_name) / import2sim(urdf_path)
       
        imported = obj.import2sim(urdf_path, instance_name=scene_name)

        if isinstance(imported, (list, tuple)):
            assets_cfg.extend(imported)
        else:
            assets_cfg.append(imported)
    if len(assets_cfg) == 0:
        raise RuntimeError("No AssetCfg imported.")
    builder = SceneBuilder(assets=assets_cfg, base_scene_cfg=base_scene_cfg)
    scene = builder.scene
    sim.reset()
    scene.update(sim.get_physics_dt())
    # ---------- C) 为每个task编译对应pose_pack ----------
    env_pose_packs: list[list[dict]] = []
    for env_id, t in enumerate(tasks):
        pose_pack = []
        for step in t["atomic_plan"]:
            aid = step["asset"]
            atom = step["atom"]
            
            params = dict(step.get("params", {}))  # 拷贝，避免污染原json
            params["_env_id"] = env_id             # 这里塞当前环境id

            if aid not in knowledge_map:
                raise ValueError(f"Unknown asset id in atomic_plan: {aid}")
            a_type = id_to_type[aid]
            if atom not in ALLOWED_ATOMS.get(a_type, set()):
                raise ValueError(f"Atom '{atom}' not allowed for '{a_type}'")
            key = (a_type, atom)
            if key not in DISPATCH:
                raise ValueError(f"No dispatcher for {key}")
            obj = knowledge_map[aid]
            
            seg = DISPATCH[key](obj, scene, params)
            if seg is None:
                continue
            if isinstance(seg, list):
                # 若atomic_plan里给了executor，覆盖到每个子pose
                ex = step.get("executor", None)
                if ex is not None:
                    for s in seg:
                        s["executor"] = ex
                pose_pack.extend(seg)
            else:
                ex = step.get("executor", None)
                if ex is not None and isinstance(seg, dict):
                    seg["executor"] = ex
                pose_pack.append(seg)
        env_pose_packs.append(pose_pack)
    return scene, env_pose_packs
def run_simulator(sim, scene, env_pose_packs: list[list[dict]]):
    # planners
    cube = Cuboid(
        name="mic_bottom",
        pose=[100, 100, 0.25, 1.0, 0.0, 0.0, 0.0],
        dims=[0.1, 0.1, 0.1],
        color=[0.4, 0.4, 0.4, 1.0],
    )
    curobo_scene_model = SceneCfg(cuboid=[cube])
    planner_left = MotionPlanner(MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        self_collision_check=False,
        optimizer_collision_activation_distance=1000,
        num_ik_seeds=512,
        position_tolerance=0.1,
        orientation_tolerance=0.1,
    ))
    planner_left.warmup(enable_graph=True, num_warmup_iterations=5)
    planner_right = MotionPlanner(MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=curobo_scene_model,
        self_collision_check=False,
        optimizer_collision_activation_distance=1000,
        num_ik_seeds=512,
        position_tolerance=0.2,
        orientation_tolerance=0.1,
    ))
    planner_right.warmup(enable_graph=True, num_warmup_iterations=5)
    link_list = ["panda_leftfinger", "panda_rightfinger", "panda_hand"]
    planner_left.disable_link_collision(disable_collision_links=link_list)
    planner_right.disable_link_collision(disable_collision_links=link_list)
    robot_left = scene["robot_left"]
    robot_right = scene["robot_right"]
    num_envs = scene.num_envs
    # env -> task mapping（任务数量不足时循环复用）
    if len(env_pose_packs) == 0:
        raise ValueError("env_pose_packs is empty")
    # init per-env states
    q0 = torch.tensor([[0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]], device="cuda:0")
    jn = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"]
    env_states = []
    for env_id in range(num_envs):
        pp = env_pose_packs[env_id % len(env_pose_packs)]
        l_tasks, r_tasks = split_pose_pack_by_executor(pp)
        env_states.append({
            "left": ArmState(tasks=l_tasks, q_start=JointState.from_position(q0.clone(), joint_names=jn)),
            "right": ArmState(tasks=r_tasks, q_start=JointState.from_position(q0.clone(), joint_names=jn)),
        })
    all_joint_names = robot_left.data.joint_names
    finger_indices = [i for i, n in enumerate(all_joint_names) if "finger" in n]
    finger_open_l = torch.tensor([0.04, 0.04], device=robot_left.device, dtype=torch.float32)
    finger_close_l = torch.tensor([0.0, 0.0], device=robot_left.device, dtype=torch.float32)
    finger_open_r = torch.tensor([0.04, 0.04], device=robot_right.device, dtype=torch.float32)
    finger_close_r = torch.tensor([0.0, 0.0], device=robot_right.device, dtype=torch.float32)
    sim_dt = sim.get_physics_dt()
    hold_frames = 60
    left_y_offset, left_z_offset = -0.4, -0.2
    right_y_offset, right_z_offset = +0.2, -0.2
    while simulation_app.is_running():
        all_done = True
        for env_id in range(num_envs):
            print(f"the env id:{env_id}")
            st = env_states[env_id]
            tick_arm_env(
                env_id, st["left"], scene, planner_left, robot_left, "robot_left",
                link_list, finger_indices, finger_open_l, finger_close_l,
                hold_frames, left_y_offset, left_z_offset
            )
            tick_arm_env(
                env_id, st["right"], scene, planner_right, robot_right, "robot_right",
                link_list, finger_indices, finger_open_r, finger_close_r,
                hold_frames, right_y_offset, right_z_offset
            )
            left_done = st["left"].step_idx >= len(st["left"].tasks) and st["left"].joint_pos_des is None
            right_done = st["right"].step_idx >= len(st["right"].tasks) and st["right"].joint_pos_des is None
            all_done = all_done and left_done and right_done
        if all_done:
            print("[INFO] all env tasks finished")
            break
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

def main():

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.2, -1.5, 0.8], [0.5, 0.0, 0.4])

    base_scene_cfg = TableTopSceneCfg(num_envs=4, env_spacing=2.0)

    scene, env_pose_packs = build_scene_and_env_pose_packs(
        task_json_paths=args_cli.task_jsons,
        sim=sim,
        base_scene_cfg=base_scene_cfg,
    )

    sim.reset()
    print("[INFO] Setup complete...")

    run_simulator(sim, scene, env_pose_packs)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()