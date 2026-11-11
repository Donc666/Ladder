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


def _ee_reached(robot, goal_pose, pos_threshold=0.04, ori_threshold=0.25) -> bool:
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


def plan_pose_pos_tol(planner, q_start, goal_pose, pos_tol_m):
    """Plan one pose. A position error within pos_tol_m still counts as success."""
    saved = []
    for solver in (getattr(planner, "ik_solver", None), getattr(planner, "trajopt_solver", None)):
        cfg = getattr(solver, "config", None)
        if cfg is None or not hasattr(cfg, "position_tolerance"):
            continue
        saved.append((cfg, float(cfg.position_tolerance)))
        cfg.position_tolerance = float(pos_tol_m)
    try:
        return plan_pose(planner, q_start, goal_pose)
    finally:
        for cfg, old in saved:
            cfg.position_tolerance = old


def plan_pose_ori_tol(planner, q_start, goal_pose, ori_tol_deg):
    """Plan one pose. During door opening, orientation within ori_tol_deg still succeeds."""
    if ori_tol_deg is None:
        return plan_pose(planner, q_start, goal_pose)
    rad = float(np.deg2rad(float(ori_tol_deg)))
    saved = []
    for solver in (getattr(planner, "ik_solver", None), getattr(planner, "trajopt_solver", None)):
        cfg = getattr(solver, "config", None)
        if cfg is None or not hasattr(cfg, "orientation_tolerance"):
            continue
        saved.append((cfg, float(cfg.orientation_tolerance)))
        cfg.orientation_tolerance = rad
    try:
        return plan_pose(planner, q_start, goal_pose)
    finally:
        for cfg, old in saved:
            cfg.orientation_tolerance = old


def plan_pose(planner, q_start, goal_pose):
    """Same waypoint pose. Rank joint seeds by shortest path, including failed ones."""
    frames = list(getattr(planner, "tool_frames", None) or [])
    if frames and hasattr(planner, "update_tool_pose_criteria"):
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        planner.update_tool_pose_criteria({name: ToolPoseCriteria() for name in frames})
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
def _atom_key(task: dict):
    if task.get("atom_index") is not None:
        return task.get("atom_index")
    return (task.get("grasp_asset"), task.get("atom"), id(task))


def hide_finished_atom(planner, arm_name: str, tasks: list[dict], step_idx: int) -> None:
    """Remove the green line when this arm leaves an atom."""
    markers = getattr(planner, "scene_markers", None)
    if markers is None or not tasks or step_idx <= 0:
        return
    prev = tasks[step_idx - 1]
    if step_idx < len(tasks) and _atom_key(tasks[step_idx]) == _atom_key(prev):
        return
    shown = markers._shown_atom.get(arm_name)
    if shown is not None and shown != _atom_key(prev):
        return
    markers.clear_arm(arm_name)


def show_atom_trajectories(
    planner, arm_name, tasks, q_start, scene, link_list, move_to_pose_fn, goal_fn=None, *, exclusive: bool = False, shown_key=None
) -> None:
    """Draw every waypoint of this atom, replacing the previous atom on this arm.

    Planning here only builds the green line. It does not command the robot.
    """
    markers = getattr(planner, "scene_markers", None)
    if markers is None or not tasks:
        return
    key = _atom_key(tasks[0]) if shown_key is None else shown_key
    if markers._shown_atom.get(arm_name) == key:
        return
    from runtime.scene_markers import _ee_world_points, _gripper_asset_poses, extend_cubes_along_tool_z

    if goal_fn is None:
        goal_fn = build_goal_pose_with_offset
    q = q_start
    names = list(planner.joint_names)
    polylines = []
    label = f"{tasks[0].get('grasp_asset')}.{tasks[0].get('atom')}"
    try:
        for task in tasks:
            pose = task.get("pose")
            if not pose:
                continue
            apply_table_obstacle(planner, task, scene)
            if task.get("use_obstacle_avoidance", True):
                planner.enable_link_collision(enable_collision_links=link_list)
            else:
                planner.disable_link_collision(disable_collision_links=link_list)
            goal = goal_fn(planner=planner, arm_name2=arm_name, pose_7d=list(pose))
            ori_tol = task.get("ori_tol_deg")
            pos_tol = task.get("position_tolerance")
            try:
                if pos_tol is not None:
                    traj = plan_pose_pos_tol(planner, q, goal, pos_tol)
                elif ori_tol is None:
                    traj = move_to_pose_fn(planner, q, goal)
                else:
                    traj = plan_pose_ori_tol(planner, q, goal, ori_tol)
            except Exception as exc:
                print(f"[WARN] atom preview failed {arm_name} {label}: {exc}")
                continue
            if traj is None:
                print(f"[WARN] atom preview failed {arm_name} {label}")
                continue
            pts = _ee_world_points(planner, traj)
            if pts is not None and len(pts) >= 2:
                polylines.append(pts)
            last = traj[-1].detach().reshape(1, -1)
            if int(last.shape[-1]) >= len(names):
                q = JointState.from_position(last[:, : len(names)], joint_names=names)
    finally:
        planner._obstacle_mode = None
    cubes = _gripper_asset_poses(tasks)
    if len(tasks) == 1 and tasks[0].get("marker_per_waypoint"):
        pose = tasks[0].get("pose") or []
        if len(pose) >= 3:
            cubes = [[float(v) for v in list(pose)[:7]]]
    cubes = extend_cubes_along_tool_z(polylines, cubes)
    markers.show_atom(arm_name, key, polylines, cubes, exclusive=exclusive)
    print(f"[INFO] show atom {arm_name} {label} waypoints={len(tasks)} lines={len(polylines)}")


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
def _cuboid_in_base_frame(cube, offset):
    """Move a world-frame cuboid into the arm base frame used by cuRobo goals."""
    from curobo._src.geom.types import Cuboid

    ox, oy, oz = offset
    pose = [float(v) for v in cube.pose]
    pose[0] -= float(ox)
    pose[1] -= float(oy)
    pose[2] -= float(oz)
    return Cuboid(name=str(cube.name), pose=pose, dims=[float(v) for v in cube.dims])


def _mesh_in_base_frame(mesh, offset):
    from curobo._src.geom.types import Mesh

    ox, oy, oz = (float(v) for v in offset)
    vertices = [[float(v[0]) - ox, float(v[1]) - oy, float(v[2]) - oz] for v in mesh.vertices]
    return Mesh(
        name=str(mesh.name),
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vertices=vertices,
        faces=mesh.faces,
    )


def _save_curobo_obstacle_view(offset, cubes, meshes):
    """Write the exact base-frame obstacles just handed to cuRobo."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk_font).exists():
        font_manager.fontManager.addfont(cjk_font)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    oy = float(offset[1])
    arm = "left" if oy > 0.05 else "right" if oy < -0.05 else "base"
    if meshes and cubes:
        tag = "microwave_table"
    elif meshes:
        tag = "microwave"
    else:
        tag = "table"
    arm_label = {"left": "左臂", "right": "右臂", "base": "基座"}[arm]

    mesh_pack = []
    for i, mesh in enumerate(meshes):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            continue
        mesh_pack.append((verts, faces, i))
    boxes = []
    for cube in cubes:
        pose = [float(v) for v in cube.pose]
        boxes.append((np.array(pose[:3]), np.array([float(v) for v in cube.dims]), str(cube.name)))

    cmap = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(13.5, 6.4), dpi=140)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    polys, colors = [], []
    for verts, faces, i in mesh_pack:
        col = cmap(i % 20)
        for tri in faces:
            polys.append(verts[tri])
            colors.append(col)
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors=(0, 0, 0, 0.05), linewidths=0.03))
    span = []
    for center, dims, name in boxes:
        h = dims * 0.5
        corners = np.array([[x, y, z] for x in (center[0] - h[0], center[0] + h[0]) for y in (center[1] - h[1], center[1] + h[1]) for z in (center[2] - h[2], center[2] + h[2])])
        span.append(corners)
        for a, b in ((0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)):
            ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], color="#2e7d32", lw=1.2)
    ax.scatter([0], [0], [0], c="black", s=28, depthshade=False)
    chunks = [verts for verts, _, _ in mesh_pack]
    chunks.extend(span)
    if chunks:
        pts = np.vstack(chunks)
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.05
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect(np.maximum(hi - lo, 0.05))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{arm_label}基座坐标")
    ax.view_init(elev=24, azim=-125)

    ax2 = fig.add_subplot(1, 2, 2)
    if mesh_pack:
        z_cut = float(np.median(np.concatenate([verts[:, 2] for verts, _, _ in mesh_pack])))
        for verts, faces, i in mesh_pack:
            col = cmap(i % 20)
            for tri in faces:
                p = verts[tri]
                hits = []
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    za, zb = p[a, 2], p[b, 2]
                    if (za - z_cut) * (zb - z_cut) < 0:
                        t = (z_cut - za) / (zb - za)
                        hits.append(p[a, :2] + t * (p[b, :2] - p[a, :2]))
                    elif za == z_cut:
                        hits.append(p[a, :2])
                if len(hits) >= 2:
                    ax2.plot([hits[0][0], hits[1][0]], [hits[0][1], hits[1][1]], color=col, lw=1.15)
        ax2.set_title(f"水平切片  z={z_cut:.3f} m")
    else:
        ax2.set_title("水平范围")
    for center, dims, name in boxes:
        h = dims * 0.5
        rect = np.array([
            [center[0] - h[0], center[1] - h[1]],
            [center[0] + h[0], center[1] - h[1]],
            [center[0] + h[0], center[1] + h[1]],
            [center[0] - h[0], center[1] + h[1]],
            [center[0] - h[0], center[1] - h[1]],
        ])
        ax2.plot(rect[:, 0], rect[:, 1], color="#2e7d32", lw=1.4, ls="--")
    ax2.scatter([0], [0], c="black", s=28, zorder=5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(True, alpha=0.3)
    handles = []
    if mesh_pack:
        handles.append(Line2D([0], [0], color=cmap(0), lw=6, label=f"碰撞网格 {len(mesh_pack)} 块"))
    if boxes:
        handles.append(Line2D([0], [0], color="#2e7d32", lw=1.5, ls="--", label="包围盒"))
    handles.append(Line2D([0], [0], marker="o", color="black", lw=0, label="基座原点"))
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=3)
    fig.suptitle(
        f"导入 cuRobo 的障碍物  {arm_label}  offset={tuple(round(float(v), 3) for v in offset)}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    out = Path(__file__).resolve().parent / f"obstacle_curobo_{arm}_{tag}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[curobo] obstacle view {out}")


def _obstacle_scene_in_base(planner, cubes, meshes=None):
    from curobo._src.geom.types import SceneCfg

    offset = tuple(getattr(planner, "base_offset", (0.0, 0.0, 0.0)))
    shifted = [_cuboid_in_base_frame(cube, offset) for cube in cubes]
    shifted_meshes = [_mesh_in_base_frame(mesh, offset) for mesh in (meshes or [])]
    for cube in shifted:
        pose = cube.pose
        print(
            f"[curobo] {cube.name} in base frame offset={tuple(round(float(v), 3) for v in offset)} "
            f"center={[round(float(pose[0]), 3), round(float(pose[1]), 3), round(float(pose[2]), 3)]} "
            f"dims={[round(float(v), 3) for v in cube.dims]}"
        )
    if shifted_meshes:
        print(
            f"[curobo] meshes in base frame offset={tuple(round(float(v), 3) for v in offset)} "
            f"parts={len(shifted_meshes)}"
        )
    try:
        _save_curobo_obstacle_view(offset, shifted, shifted_meshes)
    except Exception as exc:
        print(f"[curobo] obstacle preview failed: {exc}")
    return SceneCfg(
        cuboid=shifted or None,
        mesh=shifted_meshes or None,
    )


def apply_table_obstacle(planner, task, scene=None) -> None:
    """World obstacles only while this waypoint asks for them.

    The table stays one collision cuboid. A microwave request sends each live
    collision mesh, door pose included, then both are shifted by this arm's base.
    """
    empty_scene = getattr(planner, "empty_scene", None)
    table_scene = getattr(planner, "table_scene", None)
    if empty_scene is None:
        return
    avoid_table = bool(task.get("avoid_table"))
    avoid_microwave = bool(task.get("avoid_microwave"))
    if avoid_microwave and avoid_table:
        want = "microwave+table"
    elif avoid_microwave:
        want = "microwave"
    elif avoid_table:
        want = "table"
    else:
        want = "empty"
    if getattr(planner, "_obstacle_mode", None) == want:
        return
    if want == "empty":
        planner.update_world(empty_scene)
    elif want == "table" and table_scene is not None and getattr(table_scene, "cuboid", None):
        planner.update_world(_obstacle_scene_in_base(planner, list(table_scene.cuboid)))
    else:
        sim_scene = scene if scene is not None else getattr(planner, "sim_scene", None)
        cubes = []
        meshes = []
        scan = getattr(planner, "collision_cuboid", None)
        scan_meshes = getattr(planner, "collision_meshes", None)
        if sim_scene is not None and scan is not None and avoid_table:
            cube = scan(sim_scene, "table")
            if cube is not None:
                cubes.append(cube)
        if sim_scene is not None and scan_meshes is not None and avoid_microwave:
            meshes = list(scan_meshes(sim_scene, "microwave") or [])
        if not cubes and not meshes:
            planner.update_world(empty_scene)
            want = "empty"
        else:
            planner.update_world(_obstacle_scene_in_base(planner, cubes, meshes))
    planner._obstacle_mode = want


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
        atom_tasks = []
        atom_key = _atom_key(task)
        shown_key = None
        if task.get("marker_per_waypoint"):
            atom_tasks = [task]
            shown_key = ("point", atom_key, arm.step_idx)
        else:
            for later in arm.tasks[arm.step_idx :]:
                if _atom_key(later) != atom_key:
                    break
                atom_tasks.append(later)
        show_atom_trajectories(
            planner, arm_name, atom_tasks, q_start, scene, link_list, move_to_pose_fn, shown_key=shown_key
        )
        apply_table_obstacle(planner, task, scene)
        use_oa = task.get("use_obstacle_avoidance", True)
        if use_oa:
            planner.enable_link_collision(enable_collision_links=link_list)
        else:
            planner.disable_link_collision(disable_collision_links=link_list)
        goal_pose = build_goal_pose_with_offset(planner=planner, pose_7d=pose, arm_name2=arm_name)
        ori_tol = task.get("ori_tol_deg")
        pos_tol = task.get("position_tolerance")
        if pos_tol is not None:
            arm.joint_pos_des = plan_pose_pos_tol(planner, q_start, goal_pose, pos_tol)
        elif ori_tol is None:
            arm.joint_pos_des = move_to_pose_fn(planner, q_start, goal_pose)
        else:
            arm.joint_pos_des = plan_pose_ori_tol(planner, q_start, goal_pose, ori_tol)
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
                hide_finished_atom(planner, arm_name, arm.tasks, arm.step_idx)
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
        ori_tol = task.get("ori_tol_deg")
        reach_kwargs = {}
        if ori_tol is not None:
            reach_kwargs["ori_threshold"] = float(np.deg2rad(float(ori_tol)))
        if task.get("reach_threshold") is not None:
            reach_kwargs["pos_threshold"] = float(task["reach_threshold"])
        reached = reach_fn(scene=scene, goal_pose=pose, **reach_kwargs)
        if reached:
            q_start = JointState.from_position(last_q[:dof].unsqueeze(0), joint_names=planner.joint_names)
            print(
                f"[TaskRuntime] reached {arm_name} wp {arm.step_idx}/{len(arm.tasks)} "
                f"gripper={arm.gripper_cmd} xyz={[round(float(v), 3) for v in pose[:3]]}"
            )
            arm.step_idx += 1
            hide_finished_atom(planner, arm_name, arm.tasks, arm.step_idx)
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
                hide_finished_atom(planner, arm_name, arm.tasks, arm.step_idx)
                arm.joint_pos_des = None
                arm.prev_gripper_cmd = arm.gripper_cmd
                arm.reach_wait = 0
    # 4) 夹爪
    if arm.gripper_cmd == "close":
        robot.set_joint_position_target(finger_close_target, joint_ids=finger_indices)
    else:
        robot.set_joint_position_target(finger_open_target, joint_ids=finger_indices)
    return q_start, arm.step_idx
