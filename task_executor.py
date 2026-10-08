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
from pathlib import Path
import xml.etree.ElementTree as ET

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
from dataclasses import dataclass
from typing import Any

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


# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
# from curobo.geom.types import WorldConfig

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip








@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


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
# def move_to_pose(planner,q_start,goal_pose):
    
#     # config = MotionPlannerCfg.create(
#     #     robot="franka.yml",  # 确保此文件存在于 curobo 的资产路径中
#     #     scene_model="collision_test.yml", # 或者根据需s要创建 WorldConfig
#     # )
#     # planner = MotionPlanner(config)
#     # planner.warmup(enable_graph=True, num_warmup_iterations=5)

#     result = planner.plan_pose(goal_pose, q_start)
    
#     if result == None:
#         print("规划失败！")
#         simulation_app.close()
#         return

#     # 获取插值后的轨迹
#     traj = result.get_interpolated_plan()
#     trajectory_positions = traj.position.squeeze(0)
#     joint_pos_des = trajectory_positions[0]
#     return joint_pos_des
#     # if waypoint_idx < joint_pos_des.shape[0]:
#     #     joint_each_time_pos_des = joint_pos_des[waypoint_idx]
#     #     robot.set_joint_position_target(joint_each_time_pos_des.unsqueeze(0), joint_ids=None)
#     #     waypoint_idx += 1
#     #     return True
#     # else:
#     #     return False
def _quat_angle_rad(q_wxyz_a, q_wxyz_b) -> float:
    dot = abs(
        float(q_wxyz_a[0]) * float(q_wxyz_b[0])
        + float(q_wxyz_a[1]) * float(q_wxyz_b[1])
        + float(q_wxyz_a[2]) * float(q_wxyz_b[2])
        + float(q_wxyz_a[3]) * float(q_wxyz_b[3])
    )
    dot = min(1.0, dot)
    return 2.0 * float(np.arccos(dot))


def _ee_reached(robot, goal_pose, pos_threshold=0.08, ori_threshold=0.25) -> bool:
    """True only when both position and orientation have arrived."""
    ee_link_idx, _ = robot.find_bodies("panda_hand")
    ee_state = robot.data.body_state_w[0, int(ee_link_idx[0])]
    cur_pos = ee_state[0:3]
    tgt_pos = torch.tensor(goal_pose[:3], device=cur_pos.device, dtype=cur_pos.dtype)
    if torch.norm(cur_pos - tgt_pos).item() >= pos_threshold:
        return False
    if len(goal_pose) < 7:
        return True
    cur_q = ee_state[3:7]
    return _quat_angle_rad(cur_q, goal_pose[3:7]) < ori_threshold


def whether_reach_left(scene, goal_pose):
    return _ee_reached(scene["robot_left"], goal_pose)


def whether_reach_right(scene, goal_pose):
    return _ee_reached(scene["robot_right"], goal_pose)





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




# def extract_mesh_for_curobo(stage: Usd.Stage, prim_path: str, include_links: set[str] | None = None) -> dict:
#     """
#     Instanceable 穿透防爆版：完美解决 PartNet-Mobility 因实例化/模板化导致 
#     IsA(UsdGeom.Mesh) 无法触发、进而引起内存分配崩溃的问题。
#     """
#     root_prim = stage.GetPrimAtPath(prim_path)
#     if not root_prim.IsValid():
#         raise ValueError(f"【错误】路径 {prim_path} 在 Stage 中不存在！")
        
#     all_world_vertices = []
#     all_face_indices = []
#     vertex_count_offset = 0

#     # 【核心修改】使用能够穿透 Instance（实例化节点）和 Prototype 模板的专用断言谓词
#     # 这样不管 PartNet 怎么隐藏子网格，都无法逃过扫描
#     predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
#     prim_range = Usd.PrimRange(root_prim, predicate)

#     log_debug(f"\n==== [DEBUG] 开始穿透扫描资产: {prim_path} ====")

#     for prim in prim_range:
#         path_string = prim.GetPath().pathString
#         path_lower = path_string.lower()
#         if "visuals" not in path_lower:
#             continue
#         if include_links is not None:
#             rel = path_string[len(prim_path) + 1 :] if path_string.startswith(prim_path + "/") else ""
#             link_name = rel.split("/", 1)[0] if "/" in rel else None
#             if link_name not in include_links:
#                 continue

#         # 1) Mesh collision geometry
#         if prim.IsA(UsdGeom.Mesh):
#             log_debug(f"成功进入！发现物理 Mesh 节点: {path_string}")

#             usd_mesh = UsdGeom.Mesh(prim)
#             raw_points = usd_mesh.GetPointsAttr().Get()
#             raw_faces = usd_mesh.GetFaceVertexIndicesAttr().Get()

#             if raw_points is None or raw_faces is None:
#                 continue

#             np_points = np.array(raw_points, dtype=np.float32)
#             np_faces = np.array(raw_faces, dtype=np.int32)

#             if np_points.size == 0 or np_faces.size == 0:
#                 continue

#             max_internal_idx = np.max(np_faces)
#             if max_internal_idx >= len(np_points):
#                 actual_faces = np_faces.tolist()
#             else:
#                 actual_faces = (np_faces + vertex_count_offset).tolist()

#             world_transform: Gf.Matrix4d = usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
#             for pt in np_points:
#                 world_pt = world_transform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
#                 all_world_vertices.append([world_pt[0], world_pt[1], world_pt[2]])

#             all_face_indices.extend(actual_faces)
#             vertex_count_offset += len(np_points)

#         # 2) Box collision geometry (usually imported as UsdGeom.Cube)
#         elif prim.IsA(UsdGeom.Cube):
#             log_debug(f"成功进入！发现物理 Cube 节点: {path_string}")
#             vertex_count_offset = _append_cube_as_triangles(
#                 prim, all_world_vertices, all_face_indices, vertex_count_offset
#             )

#     log_debug(f"==== [DEBUG] 穿透扫描结束，共抓取到顶点数: {len(all_world_vertices)} ====\n")

#     # 5. 此时如果依然为空，说明路径真有问题或模型未加载
#     if len(all_world_vertices) == 0:
#         raise RuntimeError(f"未能从 {prim_path} 中提取到任何有效的 Mesh 顶点。请检查路径或模型是否加载成功。")

#     return {
#         "vertices": all_world_vertices,
#         "faces": all_face_indices
#     }

@dataclass
class ArmExecState:
    name: str                     # "robot_left" / "robot_right"
    tasks: list                   # 该机械臂自己的任务队列
    step_idx: int = 0
    joint_pos_des: Any = None
    waypoint_idx: int = 0
    gripper_cmd: str = "open"
    prev_gripper_cmd: str = "open"
    need_hold_this_step: bool = False
    hold_counter: int = 0
    current_pose: list | None = None
    last_sync_done: str | None = None
    plan_fail_count: int = 0
    executed_any: bool = False
    reach_wait: int = 0

    def abort_trajectory(self) -> None:
        self.joint_pos_des = None
        self.waypoint_idx = 0
        self.need_hold_this_step = False
        self.hold_counter = 0
        self.plan_fail_count = 0
        self.reach_wait = 0


def _joint_path_length(traj) -> float:
    if traj is None or traj.numel() == 0:
        return 1e18
    pos = traj.reshape(-1, traj.shape[-1])
    if pos.shape[0] < 2:
        return 0.0
    return float(torch.linalg.norm(pos[1:] - pos[:-1], dim=-1).sum().item())


def _last_tstep_for_seed(last_tstep, batch_i, seed_i, n_seeds, horizon):
    if last_tstep is None:
        return int(horizon)
    t = last_tstep
    if isinstance(t, torch.Tensor):
        t = t.reshape(-1)
        idx = int(batch_i) * int(n_seeds) + int(seed_i)
        if idx >= t.numel():
            idx = min(idx, t.numel() - 1)
        end = int(t[idx].item())
    else:
        end = int(t)
    if end <= 1:
        end = int(horizon)
    return max(2, min(end, int(horizon)))


def _drop_trailing_holds(pos, eps=1e-5):
    if pos is None or pos.ndim != 2 or pos.shape[0] < 3:
        return pos
    d = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1)
    moving = torch.nonzero(d > eps, as_tuple=False)
    if moving.numel() == 0:
        return pos[:2]
    end = int(moving[-1].item()) + 2
    return pos[:end]


def _seed_trajs_from_js(js, success, include_failed=False, last_tstep=None):
    if js is None or getattr(js, "position", None) is None:
        return []
    pos = js.position
    out = []
    if pos.ndim == 2:
        t = _drop_trailing_holds(pos)
        out.append(t)
    elif pos.ndim == 3:
        n = int(pos.shape[0])
        succ = None if (include_failed or success is None) else success.reshape(-1)
        for i in range(n):
            if succ is not None and i < succ.numel() and not bool(succ[i].item()):
                continue
            end = _last_tstep_for_seed(last_tstep, 0, i, n, pos.shape[1])
            out.append(_drop_trailing_holds(pos[i, :end]))
    elif pos.ndim == 4:
        b, s = int(pos.shape[0]), int(pos.shape[1])
        succ = None if (include_failed or success is None) else success.reshape(b, s)
        for i in range(b):
            for j in range(s):
                if succ is not None and not bool(succ[i, j].item()):
                    continue
                end = _last_tstep_for_seed(last_tstep, i, j, s, pos.shape[2])
                out.append(_drop_trailing_holds(pos[i, j, :end]))
    return out


def _traj_from_position(pos):
    if pos is None or pos.numel() == 0:
        return None
    t = pos
    while t.ndim > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 3:
        t = t[0]
    if t.ndim != 2:
        return None
    return t


def _interpolated_plan_traj(result):
    """Same extraction the original plan_pose used, even if success is False."""
    try:
        interp = result.get_interpolated_plan()
    except Exception:
        interp = getattr(result, "interpolated_trajectory", None)
    if interp is None or getattr(interp, "position", None) is None:
        return None
    return _traj_from_position(interp.position)


def _shortest_same_pose_traj(result):
    """Same GoalToolPose. Rank trimmed interpolated seeds by joint path."""
    if result is None:
        return None
    succ = getattr(result, "success", None)
    has_success = succ is not None and int(torch.count_nonzero(succ).item()) > 0
    include_failed = not has_success
    last_tstep = getattr(result, "interpolated_last_tstep", None)
    candidates = _seed_trajs_from_js(
        getattr(result, "interpolated_trajectory", None),
        succ,
        include_failed=include_failed,
        last_tstep=last_tstep,
    )
    scored = []
    for traj in candidates:
        t = _traj_from_position(traj)
        if t is None:
            continue
        scored.append((_joint_path_length(t), t))
    if scored:
        scored.sort(key=lambda x: x[0])
        tag = "success" if has_success else "failed"
        best = scored[0][1]
        print(
            f"[plan_pose] same pose, picked shortest of {len(scored)} "
            f"{tag} seeds path={scored[0][0]:.3f} traj={int(best.shape[0])}"
        )
        return best
    fallback = _interpolated_plan_traj(result)
    if fallback is not None:
        fallback = _drop_trailing_holds(fallback)
        if not has_success:
            print(f"[plan_pose] no official success, using interpolated plan traj={int(fallback.shape[0])}")
        return fallback
    return None


def _plan_pose_all_trajopt_seeds(planner, goal_pose, q_start):
    """Keep IK/graph from plan_pose, but return every TrajOpt seed for ranking."""
    trajopt = planner.trajopt_solver
    n_seeds = int(getattr(trajopt.config, "num_seeds", 1) or 1)
    orig = trajopt.solve_pose

    def _solve_pose_all_seeds(*args, **kwargs):
        kwargs["return_seeds"] = n_seeds
        return orig(*args, **kwargs)

    trajopt.solve_pose = _solve_pose_all_seeds
    try:
        return planner.plan_pose(goal_pose, q_start)
    finally:
        trajopt.solve_pose = orig


def plan_pose(planner, q_start, goal_pose):
    """Same waypoint pose. Rank joint seeds by shortest path, including failed ones."""
    result = _plan_pose_all_trajopt_seeds(planner, goal_pose, q_start)
    traj = _shortest_same_pose_traj(result)
    if traj is None:
        print("[WARN] plan_pose failed")
        return None
    return traj


def build_goal_pose_sequential(planner, arm_name2, pose_7d: list[float]):
    """World-frame EE goal used by the original single-arm execution path."""
    pos = list(pose_7d[:3])
    if arm_name2 == "robot_left":
        pos[1] = pos[1] - 0.4
        pos[2] = pos[2] - 0.2
    else:
        pos[1] = pos[1] + 0.4
        pos[2] = pos[2] - 0.2
    q = pose_7d[3:7]
    return GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
    )
def split_pose_pack(pose_pack: list[dict]) -> tuple[list[dict], list[dict]]:
    left_tasks, right_tasks = [], []
    for t in pose_pack:
        ex = t.get("executor")
        if ex == "simultaneous_left":
            left_tasks.append(t)
        elif ex == "simultaneous_right":
            print(f"the value of ex {ex}")
            print(f"the pose {t}")
            right_tasks.append(t)
    return left_tasks, right_tasks
def build_goal_pose_with_offset(planner,arm_name2, pose_7d: list[float]):
    # pose_7d: [x,y,z,qw,qx,qy,qz]
    pos = list(pose_7d[:3])
    if arm_name2 == "robot_left":
        pos[1] = pos[1] - 0.4
        pos[2] = pos[2] - 0.2
        
    if arm_name2 == "robot_right":
        pos[1] = pos[1] + 0.4
        pos[2] = pos[2] - 0.2
    q = pose_7d[3:7]
    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[q]]]], device="cuda", dtype=torch.float32),
    )
    return goal_pose
def tick_one_arm(
    arm,
    scene,
    planner,
    robot,
    q_start,
    count2,
    move_to_pose_fn,
    reach_fn,
    link_list,
    finger_indices,
    finger_open_target,
    finger_close_target,
    hold_frames,
    arm_name,
    sync_ctx,   # 新增：共享同步状态字典
    arm_joint_ids=None,
    reach_wait_frames=120,
):
    # 保证每次调用都有返回值
    # if count2 % 2 != 0:
    #     return q_start, arm.step_idx  # waiting=False
    # 任务完成
    if arm.step_idx >= len(arm.tasks):
        # robot.set_joint_position_target(finger_open_target, joint_ids=finger_indices)
        return q_start, arm.step_idx
    task = arm.tasks[arm.step_idx]
    pose = task["pose"]
    arm.gripper_cmd = task["gripper"]
    # --- 同步屏障 ---
    # 任务里可写 "sync_id": "pick_1"；左右手同一个 sync_id 表示此步要同步
    sync_id = task.get("sync_id")
    if sync_id:
        me = "left" if arm_name == "robot_left" else "right"
        other = "right" if me == "left" else "left"
        # 已经放行过这个sync点：直接通过
        if sync_ctx.get("released") == sync_id:
            arm.last_sync_done = sync_id
        else:
            # 正常登记到达
            sync_ctx[me] = sync_id
            if sync_ctx.get(other) != sync_id:
                return q_start, arm.step_idx  # 等待另一侧
            # 双方到齐 -> 放行（先不清空 left/right）
            sync_ctx["released"] = sync_id
            arm.last_sync_done = sync_id
    # 1) 规划
    if arm.joint_pos_des is None:
        use_oa = task.get("use_obstacle_avoidance", True)
        if use_oa:
            planner.enable_link_collision(enable_collision_links=link_list)
        else:
            planner.disable_link_collision(disable_collision_links=link_list)
        goal_pose = build_goal_pose_with_offset(planner=planner, pose_7d=pose, arm_name2=arm_name)
        arm.joint_pos_des = move_to_pose_fn(planner, q_start, goal_pose)
        arm.need_hold_this_step = (arm.gripper_cmd == "close" and arm.prev_gripper_cmd != "close")
        arm.hold_counter = 0
        arm.waypoint_idx = 0
        if arm.joint_pos_des is None:
            arm.plan_fail_count += 1
            if arm.plan_fail_count >= 8:
                print(
                    f"[TaskRuntime] skip {arm_name} waypoint {arm.step_idx} "
                    f"after {arm.plan_fail_count} plan failures "
                    f"xyz={[round(float(v), 3) for v in pose[:3]]}"
                )
                arm.step_idx += 1
                arm.plan_fail_count = 0
            elif arm.plan_fail_count == 1 or arm.plan_fail_count % 4 == 0:
                print(
                    f"[TaskRuntime] plan failed {arm_name} wp {arm.step_idx}/{len(arm.tasks)} "
                    f"try={arm.plan_fail_count} xyz={[round(float(v), 3) for v in pose[:3]]}"
                )
            return q_start, arm.step_idx
        arm.plan_fail_count = 0
        arm.executed_any = True
        print(
            f"[TaskRuntime] planned {arm_name} wp {arm.step_idx}/{len(arm.tasks)} "
            f"traj={int(arm.joint_pos_des.shape[0])} "
            f"xyz={[round(float(v), 3) for v in pose[:3]]} gripper={arm.gripper_cmd}"
        )
    # 2) 执行轨迹（只写手臂关节，避免规划里的开夹爪覆盖当前闭合）
    if arm.waypoint_idx < arm.joint_pos_des.shape[0]:
        q_way = arm.joint_pos_des[arm.waypoint_idx]
        q_way = q_way.reshape(-1)
        ids = arm_joint_ids
        if ids is None:
            name_to_i = {n: i for i, n in enumerate(robot.data.joint_names)}
            ids = [name_to_i[n] for n in planner.joint_names if n in name_to_i]
        expected = len(ids)
        if expected and q_way.numel() < expected:
            raise ValueError(
                f"[{arm_name}] trajectory dof({q_way.numel()}) < joint_ids({expected})."
            )
        if expected and q_way.numel() != expected:
            q_way = q_way[:expected]
        robot.set_joint_position_target(q_way.unsqueeze(0), joint_ids=ids)
        if arm.need_hold_this_step and arm.waypoint_idx == 0 and arm.hold_counter < hold_frames:
            arm.hold_counter += 1
        else:
            arm.waypoint_idx += 1
    # 3) 到达切步。姿态到不了时不要卡死在闭合点。
    if arm.waypoint_idx >= arm.joint_pos_des.shape[0]:
        last_q = arm.joint_pos_des[-1]
        dof = len(planner.joint_names)
        if reach_fn(scene=scene, goal_pose=pose):
            q_start = JointState.from_position(last_q[:dof].unsqueeze(0), joint_names=planner.joint_names)
            print(
                f"[TaskRuntime] reached {arm_name} wp {arm.step_idx}/{len(arm.tasks)} "
                f"gripper={arm.gripper_cmd} xyz={[round(float(v), 3) for v in pose[:3]]}"
            )
            arm.step_idx += 1
            arm.joint_pos_des = None
            arm.prev_gripper_cmd = arm.gripper_cmd
            arm.reach_wait = 0
        else:
            arm.reach_wait += 1
            if arm.reach_wait == 1 or arm.reach_wait % 50 == 0:
                print(
                    f"[TaskRuntime] waiting to reach {arm_name} waypoint {arm.step_idx} "
                    f"frames={arm.reach_wait}/{int(reach_wait_frames)}"
                )
            if arm.reach_wait >= int(reach_wait_frames):
                print(
                    f"[TaskRuntime] skip {arm_name} waypoint {arm.step_idx} after reach timeout"
                )
                q_start = JointState.from_position(last_q[:dof].unsqueeze(0), joint_names=planner.joint_names)
                arm.step_idx += 1
                arm.joint_pos_des = None
                arm.prev_gripper_cmd = arm.gripper_cmd
                arm.reach_wait = 0
    # 4) 夹爪
    if arm.gripper_cmd == "close":
        robot.set_joint_position_target(finger_close_target, joint_ids=finger_indices)
    else:
        robot.set_joint_position_target(finger_open_target, joint_ids=finger_indices)
    return q_start, arm.step_idx
