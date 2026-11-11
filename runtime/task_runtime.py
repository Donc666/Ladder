"""Execute an atomic plan with live world sensing, drop abort, and replan."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from curobo.types import GoalToolPose, JointState

from runtime.atom_compiler import AtomCompiler, CompiledAtom
from runtime.grasp_monitor import (
    GraspEvent,
    GraspMonitor,
    GraspMonitorCfg,
    _is_bowl_asset,
    _is_can_asset,
    _is_hamburg_asset,
)
from runtime.world_state import WorldState
from task_executor import (
    ArmExecState,
    build_goal_pose_sequential,
    build_goal_pose_with_offset,
    apply_table_obstacle,
    plan_pose_ori_tol,
    plan_pose_pos_tol,
    hide_finished_atom,
    show_atom_trajectories,
    split_pose_pack,
    tick_one_arm,
    _atom_key,
)


TickStatus = Literal["running", "atom_done", "dropped", "finished", "failed"]


@dataclass
class TaskRuntimeCfg:
    env_id: int = 0
    max_atom_retries: int = 4
    hold_frames: int = 40
    plan_every_n: int = 2
    ee_link: str = "panda_hand"
    reach_threshold: float = 0.04
    ori_threshold: float = math.radians(10.0)
    # Traj finished but EE never entered the reach ball: skip so the atom cannot hang.
    reach_wait_frames: int = 120
    # Drop-test hold if the object never lifts.
    drop_hold_max_frames: int = 300
    # Grasp at 0.1kg; after the object actually lifts off the table, wait N ticks then dump mass.
    drop_test_enabled: bool = False
    # When True, grasp waypoints are rebuilt from the object's current pose.
    # When False, they stay on the pose recorded at import.
    live_pose_enabled: bool = False
    # When False, do not judge whether a grasp lifted, slipped, or fell back.
    grasp_check_enabled: bool = False
    grasp_mass: float = 0.1
    drop_mass: float = 20.0
    drop_after_lift_frames: int = 20
    # Pick atoms are short; wait this many ticks after lift (not after gripper close).
    drop_after_pick_frames: int = 20
    # After dumping mass, keep holding so slip/drop can be detected before the next atom.
    drop_hold_after_inject_frames: int = 100
    drop_test_atom_keys: tuple[str, ...] = ("adjust", "pick", "stack")
    knock_confirm_frames: int = 20
    max_adjust_retries: int = 12
    knock_settle_ticks: int = 80
    # Fallen-bottle watch and the inserted adjust. Off until turned back on.
    bottle_watch_enabled: bool = False
    # After adjust/place, wait this many ticks so a tipping bottle is seen before the next atom.
    place_watch_frames: int = 80
    # Fallen when the bottle long axis is at least this far from world-up.
    # Works for both standing and lying import poses.
    fallen_min_deg: float = 45.0
    # Heartbeat in the terminal: current atom, waypoint, EE error.
    exec_log_every: int = 25
    ignore_fallen_bottle_after_index: int = -1


@dataclass
class _SeqState:
    step: int = 0
    joint_pos_des: Any = None
    waypoint_idx: int = 0
    gripper_cmd: str = "open"
    prev_gripper_cmd: str = "open"
    need_hold_this_step: bool = False
    hold_counter: int = 0
    pose: list | None = None
    plan_fail_count: int = 0
    executed_any: bool = False
    reach_wait: int = 0


class TaskRuntime:
    """
    Closed-loop controller sitting above cuRobo + Isaac Lab.

    Each physics tick:
      1. apply queued world mutations
      2. refresh asset snapshots
      3. if holding, check failed grasp / slip / back-on-table
      4. on miss or drop: abort the current trajectory, recompile this atom, retry
      5. otherwise advance the current compiled waypoints
      6. pick/adjust/grap atoms cannot finish until the object actually lifts
         (a planned final open / place is a release, not a miss or drop)
    """

    def __init__(
        self,
        scene: Any,
        world: WorldState,
        compiler: AtomCompiler,
        atomic_plan: list[dict],
        planners: dict[str, Any],
        robots: dict[str, Any],
        q_starts: dict[str, Any],
        finger_indices: dict[str, list[int]],
        finger_open: dict[str, torch.Tensor],
        finger_close: dict[str, torch.Tensor],
        move_to_pose_fn: Callable,
        link_list: list[str],
        cfg: TaskRuntimeCfg | None = None,
        grasp_cfg: GraspMonitorCfg | None = None,
        reach_fns: dict[str, Callable] | None = None,
        scene_markers=None,
    ):
        self.scene = scene
        self.scene_markers = scene_markers
        self.world = world
        self.compiler = compiler
        self.atomic_plan = list(atomic_plan)
        self.planners = planners
        self.robots = robots
        self.q_starts = dict(q_starts)
        self.finger_indices = finger_indices
        self.finger_open = finger_open
        self.finger_close = finger_close
        self.move_to_pose_fn = move_to_pose_fn
        self.link_list = link_list
        self.cfg = cfg or TaskRuntimeCfg()
        self.monitor = GraspMonitor(grasp_cfg)
        self.reach_fns = reach_fns or {}

        self.atom_index = 0
        self.compiled: CompiledAtom | None = None
        self.retry_count = 0
        self.tick_count = 0
        self.last_event: GraspEvent = GraspEvent.ok()
        self.finished = False
        self.failed = False
        # Per-asset one-shot for the whole episode: set 0.1 once, dump once, restore once.
        self._drop_test_started: set[str] = set()
        self._drop_mass_injected: set[str] = set()
        self._drop_test_done: set[str] = set()
        self._original_mass: dict[str, float] = {}
        self._heavy_assets: set[str] = set()
        self._lift_start_tick: int | None = None
        self._drop_inject_tick: int | None = None
        self._recovery_adjust: str | None = None
        self._paused_atom_index: int | None = None
        self._consumed_plan_indices: set[int] = set()
        self._adjust_idle_arm: str | None = None
        self._adjust_idle_next: int = 0
        self._pending_adjust: str | None = None
        self._place_watch_until: int | None = None
        self._adjust_retry = 0
        self._tilt_streak: dict[str, int] = {}
        self._atom_hold_frames = 0
        self._carry_miss_frames = 0
        self._bottle_long_axis: dict[str, tuple[float, float, float]] = {}

        self._seq = _SeqState()
        self.left_arm = ArmExecState(name="robot_left", tasks=[])
        self.right_arm = ArmExecState(name="robot_right", tasks=[])
        self.sync_ctx: dict[str, Any] = {"left": None, "right": None, "released": None}

        n = len(self.atomic_plan)
        print(f"[TaskRuntime] loaded {n} atoms:")
        for i, step in enumerate(self.atomic_plan):
            print(f"  {i:2d}  {step.get('asset')}.{step.get('atom')}")
        if self.atomic_plan:
            self._recompile_current(reason="init")

    def set_pose(self, asset_id: str, pos, quat_wxyz=None, env_id: int | None = None) -> None:
        self.world.set_pose(asset_id, pos=pos, quat_wxyz=quat_wxyz, env_id=env_id)

    def set_state(self, asset_id: str, **kwargs) -> None:
        self.world.set_state(asset_id, **kwargs)

    def set_mass(self, asset_id: str, mass: float, env_id: int | None = None, scale_inertia: bool = True) -> None:
        self.world.set_mass(asset_id, mass=mass, env_id=env_id, scale_inertia=scale_inertia)

    def get_mass(self, asset_id: str, env_id: int | None = None) -> float:
        return self.world.get_mass(asset_id, env_id=env_id)

    def queue_set_pose(self, asset_id: str, pos, quat_wxyz=None, env_id: int | None = None) -> None:
        self.world.queue_set_pose(asset_id, pos=pos, quat_wxyz=quat_wxyz, env_id=env_id)

    def queue_set_mass(self, asset_id: str, mass: float, env_id: int | None = None, scale_inertia: bool = True) -> None:
        self.world.queue_set_mass(asset_id, mass=mass, env_id=env_id, scale_inertia=scale_inertia)

    def snapshot(self) -> dict[str, Any]:
        return {k: v.as_dict() for k, v in self.world.get_all().items()}

    def _xyz3(self, pose) -> str:
        if pose is None or len(pose) < 3:
            return "?"
        return f"[{float(pose[0]):.3f},{float(pose[1]):.3f},{float(pose[2]):.3f}]"

    def _ee_pos(self, arm_name: str):
        robot = self.robots.get(arm_name)
        if robot is None:
            return None
        try:
            ids, _ = robot.find_bodies(self.cfg.ee_link)
            bid = int(ids[0])
        except Exception:
            return None
        idx = min(self.cfg.env_id, int(robot.data.body_state_w.shape[0]) - 1)
        return robot.data.body_state_w[idx, bid, 0:3]

    def _ee_err(self, arm_name: str, pose) -> tuple[float | None, float | None]:
        if pose is None or len(pose) < 3:
            return None, None
        cur = self._ee_pos(arm_name)
        if cur is None:
            return None, None
        tgt = torch.tensor(pose[:3], device=cur.device, dtype=cur.dtype)
        dist = float(torch.norm(cur - tgt).item())
        ori = None
        if len(pose) >= 7:
            robot = self.robots[arm_name]
            try:
                ids, _ = robot.find_bodies(self.cfg.ee_link)
                bid = int(ids[0])
            except Exception:
                return dist, None
            idx = min(self.cfg.env_id, int(robot.data.body_state_w.shape[0]) - 1)
            q = robot.data.body_state_w[idx, bid, 3:7]
            dot = abs(sum(float(q[i].item()) * float(pose[3 + i]) for i in range(4)))
            ori = math.degrees(2.0 * math.acos(min(1.0, dot)))
        return dist, ori

    def _err_txt(self, arm_name: str, pose) -> str:
        dist, ori = self._ee_err(arm_name, pose)
        if dist is None:
            return "d=?"
        if ori is None:
            return f"d={dist:.3f}m"
        return f"d={dist:.3f}m ori={ori:.0f}deg"

    def _log_compiled_waypoints(self, compiled: CompiledAtom) -> None:
        def _one(tag: str, wp: dict) -> None:
            pose = wp.get("pose")
            print(
                f"  {tag} {wp.get('executor', '?')} {wp.get('gripper', '?')} "
                f"goal={self._xyz3(pose)}"
            )

        if compiled.simultaneous:
            print(
                f"  dual-arm L={len(self.left_arm.tasks)} R={len(self.right_arm.tasks)}"
            )
            for i, wp in enumerate(self.left_arm.tasks[:8]):
                _one(f"L{i}", wp)
            for i, wp in enumerate(self.right_arm.tasks[:8]):
                _one(f"R{i}", wp)
            return
        for i, wp in enumerate(compiled.waypoints[:10]):
            _one(f"wp{i}", wp)

    def _focus_asset_ids(self) -> list[str]:
        ids: list[str] = []
        if self.compiled is not None:
            ids.extend(self.compiled.grouped_assets or [])
            ids.extend(self.compiled.grasp_assets)
            if self.compiled.asset_id:
                ids.append(self.compiled.asset_id)
        if self._recovery_adjust:
            ids.append(self._recovery_adjust)
        out: list[str] = []
        for aid in ids:
            if aid and aid not in out:
                out.append(aid)
        return out

    def _asset_z_txt(self) -> str:
        bits = []
        for aid in self._focus_asset_ids():
            try:
                snap = self.world.get(aid)
            except KeyError:
                continue
            rest = self.monitor.rest_z(aid, self.world.table_z(fallback=0.30))
            lifted = "up" if snap.pos[2] > rest + self.monitor.cfg.lift_confirm_dz else "table"
            bits.append(f"{aid} z={snap.pos[2]:.3f}({lifted})")
        return " ".join(bits)

    def _seq_phase(self) -> str:
        seq = self._seq
        n = len(self.compiled.waypoints) if self.compiled is not None else 0
        pose = seq.pose
        arm = self._seq_arm_name() or "?"
        grip = seq.gripper_cmd or "?"
        if seq.step >= n:
            if self._should_hold_atom_for_drop_test():
                return f"{arm} drop-hold {self._atom_hold_frames}/{self.cfg.drop_hold_max_frames} {grip}"
            return f"{arm} finishing {grip}"
        if seq.joint_pos_des is None:
            return f"{arm} planning wp {seq.step}/{n} {grip} goal={self._xyz3(pose)}"
        n_traj = int(seq.joint_pos_des.shape[0])
        if seq.need_hold_this_step and seq.waypoint_idx == 0:
            return (
                f"{arm} grasp-hold {seq.hold_counter}/{self.cfg.hold_frames} "
                f"wp {seq.step}/{n} {grip} {self._err_txt(arm, pose)}"
            )
        if seq.waypoint_idx < n_traj:
            return (
                f"{arm} traj {seq.waypoint_idx}/{n_traj} wp {seq.step}/{n} {grip} "
                f"goal={self._xyz3(pose)} {self._err_txt(arm, pose)}"
            )
        return (
            f"{arm} wait-reach {seq.reach_wait}/{self.cfg.reach_wait_frames} "
            f"wp {seq.step}/{n} {grip} {self._err_txt(arm, pose)}"
        )

    def _arm_phase(self, arm: ArmExecState) -> str:
        tag = "L" if arm.name == "robot_left" else "R"
        n = len(arm.tasks)
        if arm.step_idx >= n:
            return f"{tag} done"
        task = arm.tasks[arm.step_idx]
        pose = task.get("pose")
        grip = task.get("gripper", "?")
        sid = task.get("sync_id")
        if sid and self.sync_ctx.get("released") != sid:
            me = "left" if arm.name == "robot_left" else "right"
            other = "right" if me == "left" else "left"
            if self.sync_ctx.get(other) != sid:
                return f"{tag} sync-wait {sid} wp {arm.step_idx}/{n} {grip}"
        if arm.joint_pos_des is None:
            return f"{tag} planning wp {arm.step_idx}/{n} {grip} goal={self._xyz3(pose)}"
        n_traj = int(arm.joint_pos_des.shape[0])
        if arm.need_hold_this_step and arm.waypoint_idx == 0:
            return (
                f"{tag} grasp-hold {arm.hold_counter}/{self.cfg.hold_frames} "
                f"wp {arm.step_idx}/{n} {grip}"
            )
        if arm.waypoint_idx < n_traj:
            return (
                f"{tag} traj {arm.waypoint_idx}/{n_traj} wp {arm.step_idx}/{n} {grip} "
                f"{self._err_txt(arm.name, pose)}"
            )
        return (
            f"{tag} wait-reach {arm.reach_wait}/{self.cfg.reach_wait_frames} "
            f"wp {arm.step_idx}/{n} {grip} {self._err_txt(arm.name, pose)}"
        )

    def format_exec_status(self) -> str:
        n_plan = len(self.atomic_plan)
        if self.failed:
            return f"[Exec] t={self.tick_count} FAILED atom={self.atom_index}/{n_plan}"
        if self.finished:
            return f"[Exec] t={self.tick_count} FINISHED"
        if self._place_watch_until is not None:
            left = max(0, self._place_watch_until - self.tick_count)
            return (
                f"[Exec] t={self.tick_count} watching bottle ({left} ticks left) "
                f"next_atom={self.atom_index}/{n_plan}"
            )
        if self.compiled is None:
            return f"[Exec] t={self.tick_count} compiling atom={self.atom_index}/{n_plan}"
        label = f"{self.compiled.asset_id}.{self.compiled.atom}"
        if self.compiled.grouped_assets and len(self.compiled.grouped_assets) > 1:
            parts = []
            for i, aid in enumerate(self.compiled.grouped_assets):
                idx = int(self.compiled.start_index) + i
                atom = self.compiled.atom
                if 0 <= idx < len(self.atomic_plan):
                    atom = str(self.atomic_plan[idx].get("atom") or atom)
                parts.append(f"{aid}.{atom}")
            label = "+".join(parts)
        extra = ""
        if self._recovery_adjust:
            extra = f" adjust={self._recovery_adjust}#{self._adjust_retry}"
        if self.compiled.simultaneous:
            phase = f"{self._arm_phase(self.left_arm)} | {self._arm_phase(self.right_arm)}"
        else:
            phase = self._seq_phase()
        assets = self._asset_z_txt()
        return (
            f"[Exec] t={self.tick_count} atom {self.compiled.start_index}->{self.compiled.end_index} "
            f"({self.atom_index}/{n_plan}) {label} retry={self.retry_count}{extra}\n"
            f"       {phase}"
            + (f"\n       {assets}" if assets else "")
        )

    def _maybe_log_exec(self) -> None:
        every = max(1, int(self.cfg.exec_log_every))
        if self.tick_count <= 1 or self.tick_count % every == 0:
            print(self.format_exec_status())

    def _current_q(self, arm_name: str) -> Any:
        robot = self.robots[arm_name]
        planner = self.planners[arm_name]
        names = list(planner.joint_names)
        pos = robot.data.joint_pos
        idx = min(self.cfg.env_id, int(pos.shape[0]) - 1)
        name_to_i = {n: i for i, n in enumerate(robot.data.joint_names)}
        vals = []
        for n in names:
            if n not in name_to_i:
                raise KeyError(f"{arm_name} missing joint {n}")
            vals.append(float(pos[idx, name_to_i[n]].item()))
        return JointState.from_position(
            torch.tensor([vals], device=pos.device, dtype=pos.dtype),
            joint_names=names,
        )

    def _set_arm_q(self, arm_name: str, q_vec) -> None:
        """Write planner arm joints (7) without touching the 2 finger dofs."""
        robot = self.robots[arm_name]
        qv = q_vec.reshape(-1)
        n_robot = int(robot.data.joint_pos_target.shape[-1])
        if int(qv.numel()) == n_robot:
            robot.set_joint_position_target(qv.unsqueeze(0), joint_ids=None)
            return
        planner = self.planners[arm_name]
        joint_names = list(planner.joint_names)
        name_to_i = {n: i for i, n in enumerate(robot.data.joint_names)}
        ids = [name_to_i[n] for n in joint_names if n in name_to_i]
        n = min(len(ids), int(qv.numel()))
        if n:
            target = qv[:n].to(
                device=robot.data.joint_pos_target.device,
                dtype=robot.data.joint_pos_target.dtype,
            ).unsqueeze(0)
            robot.set_joint_position_target(target, joint_ids=ids[:n])

    def _mark_interaction_points(self, waypoints) -> None:
        markers = getattr(self, "scene_markers", None)
        if markers is None:
            return
        try:
            markers.add_waypoints(waypoints)
        except Exception as exc:
            print(f"[WARN] interaction markers failed: {exc}")

    def _plan_safe(self, planner, q_start, goal_pose):
        try:
            return self.move_to_pose_fn(planner, q_start, goal_pose)
        except Exception as exc:
            print(f"[TaskRuntime] plan failed: {exc}")
            return None

    def _remember_rest_heights(self, compiled: CompiledAtom) -> None:
        for aid in compiled.grasp_assets:
            try:
                z = self.world.get(aid).pos[2]
            except KeyError:
                continue
            self.monitor.remember_rest_z(aid, z)

    def _reset_executors(self) -> None:
        self._seq = _SeqState()
        self.left_arm = ArmExecState(name="robot_left", tasks=[])
        self.right_arm = ArmExecState(name="robot_right", tasks=[])
        self.left_arm.executed_any = False
        self.right_arm.executed_any = False
        self.sync_ctx = {"left": None, "right": None, "released": None}
        self.monitor.reset()
        self._lift_start_tick = None
        self._drop_inject_tick = None
        self._atom_hold_frames = 0
        self._carry_miss_frames = 0
        for arm_name in self.q_starts:
            self.q_starts[arm_name] = self._current_q(arm_name)

    def _recompile_current(self, reason: str) -> bool:
        while (
            self.atom_index < len(self.atomic_plan)
            and self.atom_index in self._consumed_plan_indices
        ):
            step = self.atomic_plan[self.atom_index]
            print(
                f"[TaskRuntime] skip atom {self.atom_index} "
                f"{step.get('asset')}.{step.get('atom')} (already done during adjust)"
            )
            self.atom_index += 1
        if self.atom_index >= len(self.atomic_plan):
            self.compiled = None
            self.finished = True
            return False
        self.world.refresh()
        atom_name = ""
        asset_id = ""
        if self.atom_index < len(self.atomic_plan):
            atom_name = str(self.atomic_plan[self.atom_index].get("atom") or "").lower()
            asset_id = str(self.atomic_plan[self.atom_index].get("asset") or "")
        a_type = str(self.compiler.id_to_type.get(asset_id, "") or "").lower()
        use_live_poses = self.cfg.live_pose_enabled and (
            str(reason).startswith("drop:")
            or str(reason).startswith("adjust:")
            or str(reason).startswith("resume_after_adjust")
            or "shake" in atom_name
            or a_type in ("bowl", "bread", "skillet", "frenchfries", "hamburg", "bell")
        )
        try:
            try:
                compiled = self.compiler.compile_group(
                    self.atomic_plan,
                    self.atom_index,
                    self.scene,
                    use_live_poses=use_live_poses,
                    skip_indices=self._consumed_plan_indices,
                )
            except TypeError:
                compiled = self.compiler.compile_group(
                    self.atomic_plan,
                    self.atom_index,
                    self.scene,
                    use_live_poses=use_live_poses,
                )
        except Exception as exc:
            print(f"[TaskRuntime] compile failed at atom={self.atom_index}: {exc}")
            self.failed = True
            return False
        if not compiled.waypoints:
            print(f"[TaskRuntime] empty waypoints for atom={self.atom_index}, skip")
            self.atom_index = compiled.end_index
            return self._recompile_current(reason="empty_skip")
        if self._is_bottle_adjust_atom(compiled):
            pack, active, idle = self._promote_adjust_dual_arm(compiled.waypoints)
            compiled.waypoints = pack
            compiled.simultaneous = True
            print(
                f"[TaskRuntime] bottle adjust dual-arm "
                f"{compiled.asset_id}.{compiled.atom} active={active} idle={idle} n={len(pack)}"
            )
        self.compiled = compiled
        self._mark_interaction_points(compiled.waypoints)
        self._reset_executors()
        self._remember_rest_heights(compiled)
        self._restore_grasp_mass(reason=f"recompile:{reason}", asset_ids=compiled.grasp_assets)
        if self._wants_drop_test(compiled):
            self._set_grasp_mass(self.cfg.grasp_mass, reason=reason, compiled=compiled)
        if compiled.simultaneous:
            left, right = split_pose_pack(compiled.waypoints)
            self.left_arm.tasks = left
            self.right_arm.tasks = right
        grouped = ",".join(compiled.grouped_assets) if compiled.grouped_assets else compiled.asset_id
        print(
            f"[TaskRuntime] compile atom[{compiled.start_index}:{compiled.end_index}] "
            f"{compiled.asset_id}.{compiled.atom} n={len(compiled.waypoints)} "
            f"sim={compiled.simultaneous} assets={grouped} reason={reason}"
            f"{' drop-test=on' if self._wants_drop_test(compiled) else ''}"
            f"{' open-loop' if self._is_mug_atom(compiled) else ''}"
            f"{' live-pose' if use_live_poses else ''}"
        )
        self._log_compiled_waypoints(compiled)
        return True

    def _is_drop_test_target(self, asset_id: str | None) -> bool:
        """Only bowl_2 dumps mass after lift. bowl_1 stacks once with no drop-test."""
        if not asset_id or _is_can_asset(asset_id):
            return False
        name = str(asset_id).lower()
        return "bowl_2" in name

    def _drop_test_pending_assets(self, compiled: CompiledAtom | None = None) -> list[str]:
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None:
            return []
        return [
            aid
            for aid in compiled.grasp_assets
            if aid
            and self._is_drop_test_target(aid)
            and aid not in self._drop_mass_injected
            and aid not in self._drop_test_done
        ]

    def _is_drop_test_atom(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if not self.cfg.drop_test_enabled or compiled is None:
            return False
        if self._recovery_adjust is not None:
            return False
        atom = str(compiled.atom or "").lower()
        if "shake" in atom:
            return False
        return any(self._is_drop_test_target(aid) for aid in compiled.grasp_assets)

    def _wants_drop_test(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if not self._is_drop_test_atom(compiled):
            return False
        return bool(self._drop_test_pending_assets(compiled))

    def _drop_wait_frames(self) -> int:
        if self.compiled is None:
            return self.cfg.drop_after_lift_frames
        atom = str(self.compiled.atom or "").lower()
        if "pick" in atom or "stack" in atom:
            return int(self.cfg.drop_after_pick_frames)
        return int(self.cfg.drop_after_lift_frames)

    def _current_mass(self, asset_id: str) -> float | None:
        try:
            return float(self.world.get_mass(asset_id))
        except Exception:
            return None

    def _mass_already(self, asset_id: str, mass: float) -> bool:
        cur = self._current_mass(asset_id)
        return cur is not None and abs(cur - float(mass)) < 1e-3

    def _write_mass(self, asset_id: str, mass: float, reason: str) -> bool:
        """Write mass only if this asset's drop-test cycle still allows it and value changed."""
        if asset_id in self._drop_test_done:
            return False
        if self._mass_already(asset_id, mass):
            return False
        try:
            self.world.set_mass(asset_id, mass)
            print(
                f"[TaskRuntime] drop-test {asset_id} mass -> {mass:.3f} kg ({reason})"
            )
            return True
        except Exception as exc:
            print(f"[TaskRuntime] set_mass failed for {asset_id}: {exc}")
            return False

    def _remember_original_mass(self, asset_id: str) -> None:
        if asset_id in self._original_mass:
            return
        self._original_mass[asset_id] = self._current_mass(asset_id) or float(self.cfg.grasp_mass)

    def _restore_mass_value(self, asset_id: str) -> float:
        return float(self._original_mass.get(asset_id, self.cfg.grasp_mass))

    def _mark_drop_test_done(self, asset_id: str, reason: str) -> None:
        if asset_id in self._drop_test_done:
            return
        self._drop_test_done.add(asset_id)
        self.monitor.mark_lifted(asset_id)
        print(
            f"[TaskRuntime] drop-test {asset_id} cycle done ({reason}); "
            "mass will not change again"
        )

    def _restore_grasp_mass(self, reason: str, asset_ids: list[str] | None = None) -> None:
        """Restore dumped objects once, then lock that asset for the rest of the episode."""
        ids = list(asset_ids) if asset_ids is not None else list(self._heavy_assets)
        for aid in ids:
            if aid in self._drop_test_done:
                continue
            if aid not in self._heavy_assets:
                if aid in self._drop_mass_injected:
                    self._mark_drop_test_done(aid, reason)
                continue
            mass = self._restore_mass_value(aid)
            self._write_mass(aid, mass, f"restore:{reason}")
            self._heavy_assets.discard(aid)
            self._mark_drop_test_done(aid, reason)

    def _set_grasp_mass(self, mass: float, reason: str, compiled: CompiledAtom | None = None) -> None:
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None or not self._wants_drop_test(compiled):
            return
        for aid in self._drop_test_pending_assets(compiled):
            if aid in self._drop_test_started:
                continue
            self._remember_original_mass(aid)
            self._write_mass(aid, mass, f"grasp:{reason}")
            self._drop_test_started.add(aid)

    def _should_hold_atom_for_drop_test(self) -> bool:
        """Do not finish until this object's one-shot dump has happened and slip can be seen.

        Pick-then-place (bowl stack) already opened the gripper; holding would
        close it again and look like a regrasp after the atom finished.
        """
        if not self._is_drop_test_atom() or self.compiled is None:
            return False
        if self._atom_ends_with_open():
            return False
        pending = self._drop_test_pending_assets()
        if pending:
            # Never lifted: that is a miss, not a reason to keep the atom alive.
            if self._is_pick_atom() and not any(self.monitor.ever_lifted(aid) for aid in pending):
                return False
            return True
        if self._drop_inject_tick is None:
            return False
        return (self.tick_count - self._drop_inject_tick) < int(self.cfg.drop_hold_after_inject_frames)

    def _hold_last_grasp_pose(self) -> None:
        """Keep the last close pose so a short pick does not end before the drop-test."""
        names = ("robot_left", "robot_right") if (self.compiled and self.compiled.simultaneous) else (self._seq_arm_name() or "robot_right",)
        for arm_name in names:
            if arm_name not in self.robots:
                continue
            robot = self.robots[arm_name]
            q = self.q_starts.get(arm_name)
            if q is not None:
                qv = q.position.reshape(-1)
                n_robot = int(robot.data.joint_pos_target.shape[-1])
                if int(qv.numel()) == n_robot:
                    robot.set_joint_position_target(qv.unsqueeze(0), joint_ids=None)
                else:
                    planner = self.planners[arm_name]
                    joint_names = list(getattr(q, "joint_names", None) or planner.joint_names)
                    name_to_i = {n: i for i, n in enumerate(robot.data.joint_names)}
                    ids = [name_to_i[n] for n in joint_names if n in name_to_i]
                    n = min(len(ids), int(qv.numel()))
                    if n:
                        target = qv[:n].to(
                            device=robot.data.joint_pos_target.device,
                            dtype=robot.data.joint_pos_target.dtype,
                        ).unsqueeze(0)
                        robot.set_joint_position_target(target, joint_ids=ids[:n])
            robot.set_joint_position_target(self.finger_close[arm_name], joint_ids=self.finger_indices[arm_name])

    def _hold_one_arm_close(self, arm_name: str) -> None:
        self._hold_one_arm(arm_name, gripper="close")

    def _hold_one_arm(self, arm_name: str, gripper: str = "close") -> None:
        robot = self.robots.get(arm_name)
        if robot is None:
            return
        q = self.q_starts.get(arm_name)
        if q is not None:
            qv = q.position.reshape(-1)
            n_robot = int(robot.data.joint_pos_target.shape[-1])
            if int(qv.numel()) == n_robot:
                robot.set_joint_position_target(qv.unsqueeze(0), joint_ids=None)
            else:
                planner = self.planners[arm_name]
                joint_names = list(getattr(q, "joint_names", None) or planner.joint_names)
                name_to_i = {n: i for i, n in enumerate(robot.data.joint_names)}
                ids = [name_to_i[n] for n in joint_names if n in name_to_i]
                n = min(len(ids), int(qv.numel()))
                if n:
                    target = qv[:n].to(
                        device=robot.data.joint_pos_target.device,
                        dtype=robot.data.joint_pos_target.dtype,
                    ).unsqueeze(0)
                    robot.set_joint_position_target(target, joint_ids=ids[:n])
        fingers = (
            self.finger_close[arm_name]
            if str(gripper) == "close"
            else self.finger_open[arm_name]
        )
        robot.set_joint_position_target(fingers, joint_ids=self.finger_indices[arm_name])

    def _hold_finished_arm(self, arm, arm_name: str) -> bool:
        """True if this arm is done and should just hold while the other catches up."""
        if not arm.tasks or arm.step_idx < len(arm.tasks):
            return False
        last = arm.tasks[-1]
        gripper = str(last.get("gripper") or "open")
        self._hold_one_arm(arm_name, gripper=gripper)
        return True

    def _mark_arm_finished_atoms(self, arm) -> None:
        """Mark plan atoms whose last waypoint this arm has already passed."""
        done_upto = int(arm.step_idx)
        last_idx: dict[int, int] = {}
        for i, wp in enumerate(arm.tasks):
            aidx = wp.get("atom_index")
            if isinstance(aidx, int) and aidx >= 0:
                last_idx[aidx] = i
        for aidx, last_i in last_idx.items():
            if last_i < done_upto and aidx not in self._consumed_plan_indices:
                step = self.atomic_plan[aidx] if aidx < len(self.atomic_plan) else {}
                self._consumed_plan_indices.add(aidx)
                print(
                    f"[TaskRuntime] arm {arm.name} finished atom {aidx} "
                    f"{step.get('asset', '?')}.{step.get('atom', '?')}"
                )

    def _consume_open_phase_atoms(self, arm) -> None:
        """If this arm is already opening to place, do not re-pick that object after adjust."""
        if not arm.tasks:
            return
        idx = max(int(arm.step_idx), 0)
        if idx >= len(arm.tasks):
            self._mark_arm_finished_atoms(arm)
            return
        aidx = arm.tasks[idx].get("atom_index")
        if not isinstance(aidx, int) or aidx < 0 or aidx in self._consumed_plan_indices:
            self._mark_arm_finished_atoms(arm)
            return
        atom = str(self.atomic_plan[aidx].get("atom") or "").lower() if aidx < len(self.atomic_plan) else ""
        last_i = max(
            (i for i, wp in enumerate(arm.tasks) if wp.get("atom_index") == aidx),
            default=idx,
        )
        # stack_bottom/middle is not done until every waypoint has been passed.
        if ("stack_bottom" in atom or "stack_middle" in atom) and idx <= last_i:
            self._mark_arm_finished_atoms(arm)
            return
        remaining = [wp for wp in arm.tasks[idx:] if wp.get("atom_index") == aidx]
        if remaining and all(str(wp.get("gripper") or "open") == "open" for wp in remaining):
            step = self.atomic_plan[aidx] if aidx < len(self.atomic_plan) else {}
            self._consumed_plan_indices.add(aidx)
            print(
                f"[TaskRuntime] arm {arm.name} already placing atom {aidx} "
                f"{step.get('asset', '?')}.{step.get('atom', '?')}; skip after adjust"
            )
        self._mark_arm_finished_atoms(arm)

    def _arm_drop_test_assets(self, arm) -> list[str]:
        ids = []
        for wp in getattr(arm, "tasks", []) or []:
            aid = wp.get("grasp_asset")
            if aid and aid not in ids and self._is_drop_test_target(aid):
                ids.append(aid)
        return ids

    def _should_freeze_arm_for_drop_test(self, arm) -> bool:
        """Do not freeze: dump mass happens mid-carry so the object can fall in motion."""
        return False

    def _should_freeze_seq_for_drop_test(self) -> bool:
        return False

    def _place_still_ahead(self, asset_id: str) -> bool:
        """True when this arm still has a planned open (place) after the current step."""
        if self.compiled is None:
            return False
        if self.compiled.simultaneous:
            for arm in (self.left_arm, self.right_arm):
                remaining = [
                    wp
                    for wp in arm.tasks[max(int(arm.step_idx), 0) :]
                    if str(wp.get("grasp_asset") or "") == str(asset_id)
                ]
                if not remaining:
                    continue
                return any(str(wp.get("gripper")) == "open" for wp in remaining)
            return False
        remaining = self.compiled.waypoints[max(int(self._seq.step), 0) :]
        return any(
            str(wp.get("grasp_asset") or "") == asset_id and str(wp.get("gripper")) == "open"
            for wp in remaining
        )

    def _maybe_finish_survived_drop_test(self) -> GraspEvent:
        """After dump-mass, restore original mass and regrasp (gravity experiment).

        The dump is supposed to make the object fall mid-motion. If the monitor
        has not already aborted, still restore and regrasp once the wait is over.
        """
        if self.compiled is None or self._drop_inject_tick is None:
            return GraspEvent.ok()
        wait = min(int(self.cfg.drop_hold_after_inject_frames), 50)
        if (self.tick_count - self._drop_inject_tick) < wait:
            return GraspEvent.ok()
        ids = [
            aid
            for aid in self.compiled.grasp_assets
            if aid in self._drop_mass_injected and aid not in self._drop_test_done
        ]
        if not ids:
            return GraspEvent.ok()
        aid = ids[0]
        try:
            z = float(self.world.get(aid).pos[2])
        except KeyError:
            z = 0.0
        print(
            f"[TaskRuntime] drop-test {aid}: restore mass and regrasp "
            f"after dump ({wait} frames)"
        )
        self._restore_grasp_mass(reason="drop_test_regrasp", asset_ids=[aid])
        return GraspEvent(
            dropped=True,
            reason="drop_test",
            asset_id=aid,
            object_z=z,
        )

    def _object_z_lifted(self, asset_id: str, dz: float | None = None) -> bool:
        """True only when the object root has actually left the table (not EE-near)."""
        need = float(dz if dz is not None else self.monitor.cfg.lift_confirm_dz)
        try:
            z = float(self.world.get(asset_id).pos[2])
        except KeyError:
            return False
        rest_z = self.monitor.rest_z(asset_id, self.world.table_z(fallback=0.30))
        return z > rest_z + need

    def _maybe_inject_drop_mass(self) -> None:
        if not self._wants_drop_test() or self.compiled is None:
            return
        pending = set(self._drop_test_pending_assets())
        if not pending:
            return
        holds = [
            lock
            for lock in self.monitor.lifted_holds()
            if lock.asset_id in pending and self._object_z_lifted(lock.asset_id)
        ]
        if self._lift_start_tick is None:
            if not holds:
                return
            wait = self._drop_wait_frames()
            self._lift_start_tick = self.tick_count
            lock = holds[0]
            try:
                lift_z = self.world.get(lock.asset_id).pos[2]
            except KeyError:
                lift_z = lock.object_z
            print(
                f"[TaskRuntime] drop-test {self.compiled.asset_id}.{self.compiled.atom}: "
                f"lifted {lock.asset_id} tick={self.tick_count} z={lift_z:.3f}, "
                f"will set mass {self.cfg.drop_mass:.1f} kg in {wait} frames"
            )
            return
        if not holds:
            self._lift_start_tick = None
            print("[TaskRuntime] drop-test: object no longer lifted, wait to lift again")
            return
        elapsed = self.tick_count - self._lift_start_tick
        if elapsed < self._drop_wait_frames():
            return
        aid = holds[0].asset_id
        if aid in self._drop_mass_injected or aid in self._drop_test_done:
            return
        if not self._object_z_lifted(aid):
            return
        self._remember_original_mass(aid)
        if self._write_mass(aid, self.cfg.drop_mass, f"dump after {elapsed} frames"):
            self._drop_mass_injected.add(aid)
            self._heavy_assets.add(aid)
            self._drop_inject_tick = self.tick_count

    def _bottle_ids(self) -> list[str]:
        ids = []
        for aid in self.world.knowledge_map:
            a_type = str(self.compiler.id_to_type.get(aid, "")).lower()
            if a_type == "bottle" or "bottle" in str(aid).lower():
                ids.append(aid)
        return ids

    def _quat_to_R(self, quat_wxyz) -> tuple[tuple[float, float, float], ...]:
        w, x, y, z = [float(v) for v in quat_wxyz]
        return (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        )

    def _inertia_long_axis(self, asset_id: str) -> tuple[float, float, float] | None:
        """Smallest principal inertia is the bottle long axis (slender body)."""
        try:
            asset, _ = self.world.resolve_asset(asset_id)
            view = getattr(asset, "root_physx_view", None)
            if view is None or not hasattr(view, "get_inertias"):
                return None
            raw = view.get_inertias()
            idx = min(self.cfg.env_id, int(raw.shape[0]) - 1)
            I = [float(v) for v in raw[idx].reshape(-1).tolist()[:9]]
            if len(I) < 9:
                return None
            diag = (abs(I[0]), abs(I[4]), abs(I[8]))
            amin = min(diag)
            amax = max(diag)
            if amax <= 1e-12 or amin > 0.85 * amax:
                return None
            axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            axis = axes[diag.index(amin)]
            print(
                f"[TaskRuntime] {asset_id} long-axis={axis} "
                f"inertia_diag=({diag[0]:.5f},{diag[1]:.5f},{diag[2]:.5f})"
            )
            return axis
        except Exception as exc:
            print(f"[TaskRuntime] inertia long-axis failed for {asset_id}: {exc}")
            return None

    def _bottle_long_axis_body(self, asset_id: str) -> tuple[float, float, float]:
        """Body-frame long axis. Independent of whether the bottle was imported standing or lying."""
        if asset_id in self._bottle_long_axis:
            return self._bottle_long_axis[asset_id]
        axis = self._inertia_long_axis(asset_id)
        if axis is None:
            axis = (0.0, 0.0, 1.0)
            print(f"[TaskRuntime] {asset_id} long-axis fallback mesh +Z")
        self._bottle_long_axis[asset_id] = axis
        return axis

    def _tilt_deg(self, asset_id: str, quat_wxyz) -> float:
        """0 deg = long axis vertical (standing), 90 deg = long axis on the table (fallen)."""
        ux, uy, uz = self._bottle_long_axis_body(asset_id)
        row2 = self._quat_to_R(quat_wxyz)[2]
        up_z = abs(row2[0] * ux + row2[1] * uy + row2[2] * uz)
        up_z = max(0.0, min(1.0, up_z))
        return math.degrees(math.acos(up_z))

    def _is_fallen_pose(self, asset_id: str, quat_wxyz) -> bool:
        return self._tilt_deg(asset_id, quat_wxyz) >= self.cfg.fallen_min_deg

    def _asset_settled(self, snap) -> bool:
        lin = snap.lin_vel or (0.0, 0.0, 0.0)
        ang = snap.ang_vel or (0.0, 0.0, 0.0)
        speed = math.sqrt(lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2)
        spin = math.sqrt(ang[0] ** 2 + ang[1] ** 2 + ang[2] ** 2)
        return speed < 0.08 and spin < 0.8

    def _is_holding(self, asset_id: str) -> bool:
        return any(lock.asset_id == asset_id and lock.lifted for lock in self.monitor.lifted_holds())

    def _current_atom_is_adjust(self, asset_id: str) -> bool:
        if self.compiled is None:
            return False
        atom = str(self.compiled.atom or "").lower()
        return "adjust" in atom and self.compiled.asset_id == asset_id

    def _bottle_is_fallen(self, asset_id: str, need_settle: bool) -> bool:
        try:
            snap = self.world.get(asset_id)
        except KeyError:
            return False
        if not self._is_fallen_pose(asset_id, snap.quat_wxyz):
            return False
        if need_settle and not self._asset_settled(snap):
            return False
        return True

    def _in_release_phase(self) -> bool:
        """True after this atom has already closed once and is now opening (placing)."""
        if self.compiled is None:
            return False
        wps = self.compiled.waypoints
        if not wps:
            return False
        i = min(max(self._seq.step, 0), len(wps) - 1)
        if str(wps[i].get("gripper", "open")) != "open":
            return False
        return any(str(wp.get("gripper")) == "close" for wp in wps[: i + 1])

    def _fallen_bottle_ignore_after(self) -> int | None:
        last = None
        for i, step in enumerate(self.atomic_plan):
            if "into_2" in str(step.get("atom") or "").lower():
                last = i
        if last is not None:
            return last
        after = int(self.cfg.ignore_fallen_bottle_after_index)
        return after if after >= 0 else None

    def _ignore_fallen_bottle(self) -> bool:
        after = self._fallen_bottle_ignore_after()
        if after is None:
            return False
        if self.atom_index > after:
            return True
        if self.compiled is not None and int(self.compiled.start_index) > after:
            return True
        return False

    def _dismiss_fallen_bottle(self, reason: str) -> None:
        aid = self._pending_adjust
        if aid or reason:
            print(f"[TaskRuntime] ignore fallen bottle ({reason}); continue plan")
        self._pending_adjust = None
        self._tilt_streak.clear()

    def _knocked_bottle_id(self, during_atom: bool = True) -> str | None:
        if not self.cfg.bottle_watch_enabled:
            return None
        if self._ignore_fallen_bottle():
            return None
        if self.tick_count < self.cfg.knock_settle_ticks:
            return None
        if self._recovery_adjust is not None:
            return None
        for aid in self._bottle_ids():
            if during_atom and self._current_atom_is_adjust(aid) and not self._in_release_phase():
                self._tilt_streak[aid] = 0
                continue
            if self._is_holding(aid):
                self._tilt_streak[aid] = 0
                continue
            if during_atom:
                fallen = self._bottle_is_fallen(aid, need_settle=True)
                if fallen:
                    self._tilt_streak[aid] = self._tilt_streak.get(aid, 0) + 1
                else:
                    self._tilt_streak[aid] = 0
                if self._tilt_streak.get(aid, 0) >= self.cfg.knock_confirm_frames:
                    self._tilt_streak[aid] = 0
                    return aid
            elif self._bottle_is_fallen(aid, need_settle=False):
                return aid
        return None

    def _note_fallen_bottle(self, aid: str) -> None:
        if self._pending_adjust == aid:
            return
        cur = f"{self.compiled.asset_id}.{self.compiled.atom}" if self.compiled else "?"
        print(f"[TaskRuntime] {aid} knocked over during {cur}; will stand it up")
        self._pending_adjust = aid

    def _bottle_debug(self, asset_id: str) -> str:
        try:
            snap = self.world.get(asset_id)
            tilt = self._tilt_deg(asset_id, snap.quat_wxyz)
            return f"tilt={tilt:.1f} deg z={snap.pos[2]:.3f}"
        except KeyError:
            return "missing"

    def _tick_place_watch(self) -> TickStatus:
        fallen = self._knocked_bottle_id(during_atom=False)
        if fallen:
            self._place_watch_until = None
            print(
                f"[TaskRuntime] {fallen} fell after place ({self._bottle_debug(fallen)}); "
                "adjust before next atom"
            )
            return self._begin_adjust_recovery(fallen)
        if self._place_watch_until is not None and self.tick_count >= self._place_watch_until:
            self._place_watch_until = None
            ids = self._bottle_ids()
            extra = self._bottle_debug(ids[0]) if ids else ""
            print(f"[TaskRuntime] place-watch done, bottle still standing {extra}")
            return self._continue_to_paused(reason="place_watch_ok")
        return "running"

    def _continue_to_paused(self, reason: str) -> TickStatus:
        paused = self._paused_atom_index
        self._paused_atom_index = None
        self._place_watch_until = None
        self.retry_count = 0
        if paused is not None:
            self.atom_index = paused
        print(f"[TaskRuntime] {reason}; continue atom={self.atom_index}")
        if self.atom_index >= len(self.atomic_plan):
            self.finished = True
            self.compiled = None
            return "finished"
        if not self._recompile_current(reason=reason):
            return "failed" if self.failed else "finished"
        return "running"

    def _ee_pose7(self, arm_name: str) -> list[float] | None:
        robot = self.robots.get(arm_name)
        if robot is None:
            return None
        try:
            ids, _ = robot.find_bodies(self.cfg.ee_link)
            bid = int(ids[0])
        except Exception:
            return None
        idx = min(self.cfg.env_id, int(robot.data.body_state_w.shape[0]) - 1)
        s = robot.data.body_state_w[idx, bid]
        return [float(s[i].item()) for i in range(7)]

    def _remap_executor(self, waypoints: list[dict], arm_name: str) -> list[dict]:
        ex = "simultaneous_left" if arm_name == "robot_left" else "simultaneous_right"
        out = []
        for wp in waypoints:
            item = dict(wp)
            item["executor"] = ex
            out.append(item)
        return out

    def _idle_continue_waypoints(self, idle: str) -> list[dict]:
        """Idle arm keeps working during bottle adjust: finish an in-progress
        grasp if the gripper still has close waypoints, otherwise run the next
        plan atom(s) for this arm (hamburg_2, hamburg_3, ...)."""
        out: list[dict] = []
        arm = self.left_arm if idle == "robot_left" else self.right_arm
        self._consume_open_phase_atoms(arm)
        already: set[int] = set(self._consumed_plan_indices)
        if arm.tasks and 0 <= int(arm.step_idx) < len(arm.tasks):
            remaining = [dict(wp) for wp in arm.tasks[int(arm.step_idx) :]]
            if remaining:
                print(
                    f"[TaskRuntime] idle {idle} keeps leftover queue "
                    f"{arm.step_idx}/{len(arm.tasks)} n={len(remaining)}"
                )
                out.extend(self._remap_executor(remaining, idle))
                for wp in remaining:
                    aidx = wp.get("atom_index")
                    if isinstance(aidx, int) and aidx >= 0:
                        already.add(aidx)

        search_from = (
            self._paused_atom_index if self._paused_atom_index is not None else self.atom_index
        )

        for i in range(int(search_from), len(self.atomic_plan)):
            if i in already:
                continue
            step = self.atomic_plan[i]
            try:
                wps = self.compiler.compile_step(
                    step, self.scene, i, use_live_poses=self.cfg.live_pose_enabled
                )
            except Exception as exc:
                print(f"[TaskRuntime] idle compile failed atom={i}: {exc}")
                continue
            if not wps:
                continue
            mine = [wp for wp in wps if self._waypoint_arm(wp) == idle]
            if not mine:
                continue
            print(
                f"[TaskRuntime] idle {idle} runs next atom {i} "
                f"{step.get('asset')}.{step.get('atom')} n={len(mine)}"
            )
            out.extend(self._remap_executor(mine, idle))
            already.add(i)
            if len(out) >= 8:
                break

        for wp in out:
            wp.pop("sync_id", None)
        print(f"[TaskRuntime] idle {idle} queued {len(out)} waypoints")
        return out

    def _mark_idle_plan_consumed(self) -> None:
        """After adjust, skip plan atoms the idle arm already finished."""
        idle = self._adjust_idle_arm
        if not idle:
            return
        arm = self.left_arm if idle == "robot_left" else self.right_arm
        if not arm.tasks or int(arm.step_idx) < len(arm.tasks):
            return
        for wp in arm.tasks:
            idx = wp.get("atom_index")
            if not isinstance(idx, int) or idx < 0:
                continue
            atom = str(self.atomic_plan[idx].get("atom") or wp.get("atom") or "").lower() if idx < len(self.atomic_plan) else str(wp.get("atom") or "").lower()
            if "stack_bottom" in atom or "stack_middle" in atom:
                continue
            self._consumed_plan_indices.add(idx)
            print(
                f"[TaskRuntime] consumed atom {idx} during adjust "
                f"({wp.get('grasp_asset')}.{wp.get('atom')})"
            )

    def _is_bottle_adjust_atom(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None:
            return False
        if "adjust" not in str(compiled.atom or "").lower():
            return False
        if compiled.grouped_assets and len(compiled.grouped_assets) > 1:
            return False
        a_type = str(self.compiler.id_to_type.get(compiled.asset_id, "") or "").lower()
        name = str(compiled.asset_id or "").lower()
        return a_type == "bottle" or "bottle" in name

    def _promote_adjust_dual_arm(
        self, waypoints: list[dict], attach_idle: bool = True
    ) -> tuple[list[dict], str, str]:
        """Remap bottle.adjust so split_pose_pack feeds both arms.

        The unused arm does not park: it keeps the current leftover queue or
        the next plan atom that uses that arm.
        """
        active = self._waypoint_arm(waypoints[0]) or "robot_right"
        idle = "robot_left" if active == "robot_right" else "robot_right"
        sim_ex = "simultaneous_left" if active == "robot_left" else "simultaneous_right"
        adjust_wps = []
        for wp in waypoints:
            if self._waypoint_arm(wp) not in (active, None):
                continue
            item = dict(wp)
            item["executor"] = sim_ex
            item.pop("sync_id", None)
            adjust_wps.append(item)
        if attach_idle:
            idle_wps = self._idle_continue_waypoints(idle)
            if not idle_wps:
                print(
                    f"[TaskRuntime] idle {idle} got 0 plan waypoints during adjust"
                )
            adjust_wps = adjust_wps + idle_wps
        return adjust_wps, active, idle

    def _compile_adjust(self, asset_id: str) -> bool:
        self.world.refresh()
        try:
            waypoints = self.compiler.compile_step(
                {"asset": asset_id, "atom": "adjust", "params": {}},
                self.scene,
                atom_index=-1,
                use_live_poses=self.cfg.live_pose_enabled,
            )
        except Exception as exc:
            print(f"[TaskRuntime] recovery adjust compile failed for {asset_id}: {exc}")
            return False
        if not waypoints:
            print(f"[TaskRuntime] recovery adjust empty waypoints for {asset_id}")
            return False
        keep_idle = None
        if self._adjust_retry > 0 and self._adjust_idle_arm:
            prev = (
                self.left_arm
                if self._adjust_idle_arm == "robot_left"
                else self.right_arm
            )
            keep_idle = copy.copy(prev)
        pack, active, idle = self._promote_adjust_dual_arm(
            waypoints, attach_idle=(self._adjust_retry == 0)
        )
        extra = []
        for wp in pack:
            aid = wp.get("grasp_asset")
            if aid and aid != asset_id and aid not in extra:
                extra.append(str(aid))
        self.compiled = CompiledAtom(
            start_index=self.atom_index,
            end_index=self.atom_index,
            asset_id=asset_id,
            atom="adjust",
            waypoints=pack,
            simultaneous=True,
            grouped_assets=[asset_id] + extra,
        )
        self._mark_interaction_points(pack)
        self._adjust_idle_arm = idle
        self._reset_executors()
        left, right = split_pose_pack(pack)
        self.left_arm.tasks = left
        self.right_arm.tasks = right
        if keep_idle is not None:
            if idle == "robot_left":
                self.left_arm = keep_idle
            else:
                self.right_arm = keep_idle
        self._remember_rest_heights(self.compiled)
        self._restore_grasp_mass(reason="recovery_adjust", asset_ids=[asset_id])
        print(
            f"[TaskRuntime] recovery adjust {asset_id} "
            f"try={self._adjust_retry}/{self.cfg.max_adjust_retries} n={len(pack)} "
            f"active={active} idle={idle} L={len(self.left_arm.tasks)} "
            f"R={len(self.right_arm.tasks)}"
        )
        self._log_compiled_waypoints(self.compiled)
        return True

    def _open_grippers(self, arm_names: list[str] | None = None) -> None:
        names = arm_names if arm_names is not None else list(self.robots.keys())
        for arm_name in names:
            robot = self.robots.get(arm_name)
            if robot is None:
                continue
            robot.set_joint_position_target(
                self.finger_open[arm_name], joint_ids=self.finger_indices[arm_name]
            )

    def _begin_adjust_recovery(self, asset_id: str) -> TickStatus:
        if self._should_defer_adjust():
            self._pending_adjust = asset_id
            print(
                f"[TaskRuntime] finish stack_bottom "
                f"before adjust {asset_id}"
            )
            return "running"
        self._pending_adjust = None
        if self._paused_atom_index is None:
            self._paused_atom_index = self.atom_index
        self._consume_open_phase_atoms(self.left_arm)
        self._consume_open_phase_atoms(self.right_arm)
        self._recovery_adjust = asset_id
        self._adjust_retry = 0
        if not self._compile_adjust(asset_id):
            return self._resume_after_adjust()
        active = self._recovery_active_arm()
        self._open_grippers([active] if active else None)
        return "running"

    def _resume_after_adjust(self) -> TickStatus:
        aid = self._recovery_adjust
        paused = self._paused_atom_index
        self._recovery_adjust = None
        self._paused_atom_index = None
        self._adjust_retry = 0
        self.retry_count = 0
        if paused is not None:
            self.atom_index = paused
        print(f"[TaskRuntime] resume atom={self.atom_index} after adjusting {aid}")
        if self.atom_index >= len(self.atomic_plan):
            self.finished = True
            self.compiled = None
            return "finished"
        if not self._recompile_current(reason="resume_after_adjust"):
            return "failed" if self.failed else "finished"
        return "running"

    def _recovery_active_arm(self) -> str | None:
        if self._recovery_adjust is None:
            return None
        if self._adjust_idle_arm == "robot_left":
            return "robot_right"
        if self._adjust_idle_arm == "robot_right":
            return "robot_left"
        return "robot_right"

    def _stack_bottom_not_finished(self) -> bool:
        """True until stack_bottom's last waypoint has actually been passed."""
        for i, step in enumerate(self.atomic_plan):
            if "stack_bottom" not in str(step.get("atom") or "").lower():
                continue
            if i not in self._consumed_plan_indices:
                return True
        return False

    def _finish_or_retry_adjust(self) -> TickStatus:
        aid = self._recovery_adjust
        if aid is None:
            return self._resume_after_adjust()
        try:
            snap = self.world.get(aid)
            tilt = self._tilt_deg(aid, snap.quat_wxyz)
            fallen = self._is_fallen_pose(aid, snap.quat_wxyz)
        except KeyError:
            tilt = 90.0
            fallen = True
        if not fallen:
            print(f"[TaskRuntime] {aid} standing tilt={tilt:.1f} deg")
            self._mark_idle_plan_consumed()
            return self._resume_after_adjust()
        self._adjust_retry += 1
        print(
            f"[TaskRuntime] {aid} still tilted {tilt:.1f} deg, "
            f"adjust retry {self._adjust_retry}/{self.cfg.max_adjust_retries}"
        )
        if self._adjust_retry > self.cfg.max_adjust_retries:
            print(f"[TaskRuntime] give up adjusting {aid}")
            self._mark_idle_plan_consumed()
            return self._resume_after_adjust()
        if not self._compile_adjust(aid):
            return self._resume_after_adjust()
        return "running"

    def _waypoint_arm(self, wp: dict) -> str | None:
        ex = str(wp.get("executor", "robot_right"))
        if ex in ("robot_left", "simultaneous_left") or "left" in ex:
            return "robot_left"
        if ex in ("robot_right", "simultaneous_right") or "right" in ex:
            return "robot_right"
        return "robot_right"

    def _is_in_drop_test_hold(self, arm_name: str) -> bool:
        """Dump-mass window only. Place waypoints after that are not a drop."""
        if self.compiled is None or not self._is_drop_test_atom():
            return False
        if self.compiled.simultaneous:
            arm = self.left_arm if arm_name == "robot_left" else self.right_arm
            return self._should_freeze_arm_for_drop_test(arm)
        return self._should_freeze_seq_for_drop_test()

    def _adjust_grasp_missed(self) -> GraspEvent:
        """If the adjust arm already closed and the bottle never lifted, retry now."""
        if not self.cfg.grasp_check_enabled:
            return GraspEvent.ok()
        aid = self._recovery_adjust
        if not aid:
            return GraspEvent.ok()
        active = self._recovery_active_arm() or "robot_right"
        arm = self.left_arm if active == "robot_left" else self.right_arm
        did_close = any(
            str(wp.get("gripper")) == "close" and i < int(arm.step_idx)
            for i, wp in enumerate(arm.tasks)
        )
        still_closing = (
            0 <= int(arm.step_idx) < len(arm.tasks)
            and str(arm.tasks[int(arm.step_idx)].get("gripper")) == "close"
        )
        if not did_close:
            return GraspEvent.ok()
        if still_closing:
            return GraspEvent.ok()
        remaining = arm.tasks[max(int(arm.step_idx), 0) :]
        if remaining and all(str(wp.get("gripper") or "open") == "open" for wp in remaining):
            # Close is done; remaining opens are standing the bottle down, not a miss.
            return GraspEvent.ok()
        if self.monitor.ever_lifted(aid) or self._object_z_lifted(aid):
            return GraspEvent.ok()
        try:
            snap = self.world.get(aid)
            z = float(snap.pos[2])
        except KeyError:
            z = 0.0
        rest_z = self.monitor.rest_z(aid, self.world.table_z(fallback=0.30))
        if z > rest_z + self.monitor.cfg.lift_confirm_dz:
            return GraspEvent.ok()
        print(
            f"[TaskRuntime] adjust {aid} closed but never lifted; "
            "retry now, do not finish the remaining waypoints"
        )
        return GraspEvent(
            dropped=True,
            reason="grasp_failed",
            asset_id=aid,
            arm_name=active,
            object_z=z,
            table_z=rest_z,
        )

    def _same_atom_remaining(self, arm_name: str) -> list[dict]:
        """Waypoints still ahead on this arm that belong to the current object/atom."""
        if self.compiled is None:
            return []
        if self.compiled.simultaneous:
            arm = self.left_arm if arm_name == "robot_left" else self.right_arm
            remaining = arm.tasks[max(int(arm.step_idx), 0) :]
            if not remaining:
                return []
            cur = remaining[0]
            cur_idx = cur.get("atom_index")
            cur_aid = cur.get("grasp_asset")

            def _same_atom(wp: dict) -> bool:
                if cur_idx is not None and wp.get("atom_index") is not None:
                    return wp.get("atom_index") == cur_idx
                return wp.get("grasp_asset") == cur_aid

            return [wp for wp in remaining if _same_atom(wp)]
        wps = self.compiled.waypoints
        i = max(int(self._seq.step), 0)
        return [
            wp
            for wp in wps[i:]
            if self._waypoint_arm(wp) == arm_name
        ]

    def _is_last_close_then_open(self, arm_name: str) -> bool:
        """True on the final close of a pick-then-place, or already opening to leave.

        Hamburg: above-close, down-close, above-open. After the last close the
        object is on the table and the arm flies to the next burger — not a miss.
        """
        remaining = self._same_atom_remaining(arm_name)
        if not remaining:
            return False
        later_open = any(str(wp.get("gripper")) == "open" for wp in remaining)
        closes_after = any(str(wp.get("gripper")) == "close" for wp in remaining[1:])
        return later_open and not closes_after

    def _is_planned_release(self, arm_name: str, gripper_cmd: str) -> bool:
        """True when this arm is opening, or a later waypoint is a planned open.

        Bowl/can gravity test must finish first: later-open is not a place until
        dump-mass has run (otherwise we never watch the lift and never dump).
        After the cycle, putting the object down is not a drop.
        """
        if gripper_cmd != "close":
            return True
        if self.compiled is None:
            return False
        aid = self._active_grasp(arm_name, gripper_cmd)
        if (
            aid
            and self._is_drop_test_target(aid)
            and aid not in self._drop_test_done
        ):
            return False
        if self._is_last_close_then_open(arm_name):
            return True
        remaining = self._same_atom_remaining(arm_name)
        later_open = any(str(wp.get("gripper")) == "open" for wp in remaining)
        if (
            aid
            and not self.monitor.ever_lifted(aid)
            and aid not in self._drop_test_done
            and not self._object_z_lifted(aid)
        ):
            return False
        return later_open

    def _right_gripper_closed(self) -> bool:
        """True when the right gripper is commanded closed.

        Bottle adjust only waits on the right arm (bowls/can_2). The left arm
        may keep holding hamburg.
        """
        arm = self.right_arm
        if str(getattr(arm, "gripper_cmd", None) or "open") == "close":
            return True
        if arm.tasks and int(arm.step_idx) >= len(arm.tasks):
            if str(arm.tasks[-1].get("gripper") or "open") == "close":
                return True
        if self.compiled is not None and not self.compiled.simultaneous:
            if self._seq_arm_name() == "robot_right":
                if str(getattr(self._seq, "gripper_cmd", None) or "open") == "close":
                    return True
        return False

    def _any_gripper_holding(self) -> bool:
        """True if either arm is still closed on an object (work still in the hand)."""
        if self._right_gripper_closed():
            return True
        for arm in (self.left_arm,):
            if str(getattr(arm, "gripper_cmd", None) or "open") == "close":
                return True
            if arm.tasks and int(arm.step_idx) >= len(arm.tasks):
                if str(arm.tasks[-1].get("gripper") or "open") == "close":
                    return True
        if str(getattr(self._seq, "gripper_cmd", None) or "open") == "close":
            return True
        return False

    def _just_finished_open_atom(self) -> bool:
        for arm in (self.left_arm, self.right_arm):
            for aidx in self._just_finished_atom_indices(arm):
                if self._atom_last_gripper_open(arm, aidx):
                    return True
        return False

    def _just_finished_stack_atom(self) -> bool:
        """True when bowl stack_bottom/stack_middle just finished (last gripper open)."""
        for arm in (self.left_arm, self.right_arm):
            for aidx in self._just_finished_atom_indices(arm):
                if aidx >= len(self.atomic_plan):
                    continue
                atom = str(self.atomic_plan[aidx].get("atom") or "").lower()
                if "stack_bottom" not in atom and "stack_middle" not in atom:
                    continue
                if self._atom_last_gripper_open(arm, aidx):
                    return True
        return False

    def _right_current_atom_name(self) -> str:
        arm = self.right_arm
        if not arm.tasks:
            return ""
        idx = max(int(arm.step_idx), 0)
        if idx >= len(arm.tasks):
            return ""
        wp = arm.tasks[idx]
        aidx = wp.get("atom_index")
        if isinstance(aidx, int) and 0 <= aidx < len(self.atomic_plan):
            return str(self.atomic_plan[aidx].get("atom") or "").lower()
        return str(wp.get("atom") or "").lower()

    def _should_defer_adjust(self) -> bool:
        """Only stack_bottom is protected: finish it before adjust.

        stack_middle and later right-arm atoms may be interrupted. Wait if the
        right gripper is closed (object still in hand).
        """
        if self._right_gripper_closed():
            return True
        if self._stack_bottom_not_finished():
            return True
        return "stack_bottom" in self._right_current_atom_name()

    def _maybe_adjust_after_open_atom(self) -> TickStatus | None:
        """Stand the bottle when the right gripper is open.

        Left arm is ignored. stack_bottom must finish first; other atoms may
        be interrupted.
        """
        if self._recovery_adjust is not None or self._ignore_fallen_bottle():
            return None
        aid = self._pending_adjust
        if not aid:
            return None
        if self._should_defer_adjust():
            return None
        print(f"[TaskRuntime] right open after stack_bottom; now adjust fallen {aid}")
        if self._paused_atom_index is None:
            self._paused_atom_index = self.atom_index
        return self._begin_adjust_recovery(aid)

    def _atom_ends_with_open(self) -> bool:
        """True when the atom finished with gripper(s) open, so a bottle may have been knocked."""
        if self.compiled is None:
            return False
        if self.compiled.simultaneous:
            cmds = []
            for arm in (self.left_arm, self.right_arm):
                if not arm.tasks:
                    continue
                cmds.append(str(arm.tasks[-1].get("gripper", "open")))
            return bool(cmds) and all(c == "open" for c in cmds)
        if self.compiled.waypoints:
            return self.compiled.waypoints[-1].get("gripper") == "open"
        return str(self._seq.gripper_cmd or "") == "open"

    def _is_open_loop_food(self, asset_id: str | None) -> bool:
        """Mug / fries / cans / hamburg: grasp once, no fail-retry.

        Left-arm hamburg EE-to-object distance stays ~0.4 m because of the
        planner offset; treating that as a miss retries the whole mega-pack
        and hits max_atom_retries.
        """
        if not asset_id:
            return False
        if _is_can_asset(asset_id) or _is_hamburg_asset(asset_id):
            return True
        a_type = str(self.compiler.id_to_type.get(asset_id, "")).lower()
        name = str(asset_id).lower()
        if a_type in ("can", "hamburg"):
            return True
        return (
            a_type in ("mug", "frenchfries")
            or "mug" in name
            or "fries" in name
        )

    def _is_mug_atom(self, compiled: CompiledAtom | None = None) -> bool:
        """True when every asset in this pack is mug/fries (open-loop, no miss retry)."""
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None:
            return False
        ids = list(compiled.grouped_assets) or ([compiled.asset_id] if compiled.asset_id else [])
        return bool(ids) and all(self._is_open_loop_food(aid) for aid in ids)

    def _should_watch_bottle_after_atom(self) -> bool:
        """After a release or a push-like atom, wait and see if the bottle tipped."""
        if not self.cfg.bottle_watch_enabled:
            return False
        if self._ignore_fallen_bottle():
            return False
        if self.compiled is None or not self._bottle_ids():
            return False
        if self._atom_ends_with_open():
            return True
        return False

    def _is_simultaneous_pick(self, compiled: CompiledAtom | None = None) -> bool:
        """True for dual-arm pick packs (cans, skillet+bread, ...)."""
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None or not compiled.simultaneous:
            return False
        atom = str(compiled.atom or "").lower()
        return any(key in atom for key in ("pick", "grap", "grasp"))

    def _is_simultaneous_can_pick(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if not self._is_simultaneous_pick(compiled) or compiled is None:
            return False
        ids = list(compiled.grouped_assets) or ([compiled.asset_id] if compiled.asset_id else [])
        if not ids:
            return False
        return all(
            str(self.compiler.id_to_type.get(aid, "")).lower() == "can"
            or "can" in str(aid).lower()
            for aid in ids
        )

    def _is_skillet_bread_pack(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None:
            return False
        ids = list(compiled.grouped_assets) or ([compiled.asset_id] if compiled.asset_id else [])
        if not ids:
            return False
        def _kind(aid: str) -> str:
            a_type = str(self.compiler.id_to_type.get(aid, "")).lower()
            name = str(aid).lower()
            if a_type == "skillet" or "skillet" in name:
                return "skillet"
            if a_type == "bread" or "bread" in name:
                return "bread"
            return a_type or name
        return all(_kind(aid) in ("skillet", "bread") for aid in ids)

    def _is_place_atom(self, compiled: CompiledAtom | None = None) -> bool:
        """True for into/place/on/hang: opening the gripper is a planned put-down."""
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None:
            return False
        atom = str(compiled.atom or "").lower()
        if any(key in atom for key in ("into", "place", "hang", "on_")):
            return True
        if compiled.simultaneous:
            for step in self.atomic_plan[compiled.start_index : compiled.end_index]:
                if any(key in str(step.get("atom") or "").lower() for key in ("into", "place", "hang")):
                    return True
        return False

    def _is_pick_atom(self, compiled: CompiledAtom | None = None) -> bool:
        compiled = compiled if compiled is not None else self.compiled
        if compiled is None or self._is_skillet_bread_pack(compiled) or self._is_place_atom(compiled):
            return False
        atom = str(compiled.atom or "").lower()
        if any(key in atom for key in ("adjust", "pick", "grap", "grasp", "stack")):
            return True
        if compiled.simultaneous:
            for step in self.atomic_plan[compiled.start_index : compiled.end_index]:
                if any(key in str(step.get("atom") or "").lower() for key in ("pick", "grap", "grasp", "stack")):
                    return True
        return False

    def _lift_check_assets(self, asset_ids: list[str], *, ends_open: bool, atom: str = "") -> GraspEvent:
        """Fail immediately if a watched object never left the table."""
        atom_name = atom or (self.compiled.atom if self.compiled is not None else "")
        for asset_id in asset_ids:
            if not asset_id or self._is_open_loop_food(asset_id):
                continue
            try:
                snap = self.world.get(asset_id)
            except KeyError:
                print(f"[TaskRuntime] lift-check skipped, no snapshot for {asset_id}")
                continue
            rest_z = self.monitor.rest_z(asset_id, self.world.table_z(fallback=0.30))
            dz = 0.02 if _is_bowl_asset(asset_id) else self.monitor.cfg.lift_confirm_dz
            live_lifted = snap.pos[2] > rest_z + dz
            lifted_once = self.monitor.ever_lifted(asset_id)
            print(
                f"[TaskRuntime] lift-check {asset_id}.{atom_name} "
                f"lifted_once={lifted_once} live_z={snap.pos[2]:.3f} rest_z={rest_z:.3f} "
                f"live_lifted={live_lifted} ends_open={ends_open}"
            )
            if live_lifted or lifted_once or asset_id in self._drop_test_done:
                continue
            # Hamburg/bowl pick ends by placing then lifting open; that is not a miss.
            if ends_open or self._asset_has_planned_place(asset_id):
                continue
            return GraspEvent(
                dropped=True,
                reason="grasp_failed",
                asset_id=asset_id,
                object_z=snap.pos[2],
                table_z=rest_z,
            )
        return GraspEvent.ok()

    def _asset_has_planned_place(self, asset_id: str) -> bool:
        """True when this object's last waypoint in the current pack is a planned open."""
        if self.compiled is None or not asset_id:
            return False
        mine = [
            wp
            for wp in self.compiled.waypoints
            if str(wp.get("grasp_asset") or "") == str(asset_id)
        ]
        if not mine:
            return False
        return str(mine[-1].get("gripper") or "open") == "open"

    def _just_finished_atom_indices(self, arm: ArmExecState) -> list[int]:
        done_upto = int(arm.step_idx)
        last_idx: dict[int, int] = {}
        for i, wp in enumerate(arm.tasks):
            aidx = wp.get("atom_index")
            if isinstance(aidx, int) and aidx >= 0:
                last_idx[aidx] = i
        out = []
        for aidx, last_i in last_idx.items():
            if last_i < done_upto and aidx not in self._consumed_plan_indices:
                out.append(aidx)
        return out

    def _atom_last_gripper_open(self, arm: ArmExecState, atom_index: int) -> bool:
        for wp in reversed(arm.tasks):
            if wp.get("atom_index") == atom_index:
                return str(wp.get("gripper") or "open") == "open"
        return False

    def _check_just_finished_atoms(self, arm: ArmExecState) -> GraspEvent:
        """As soon as one atom in the mega-pack ends, retry that grasp if it missed."""
        if not self.cfg.grasp_check_enabled:
            return GraspEvent.ok()
        if self.compiled is None or not self._is_pick_atom():
            return GraspEvent.ok()
        for aidx in self._just_finished_atom_indices(arm):
            if aidx >= len(self.atomic_plan):
                continue
            step = self.atomic_plan[aidx]
            aid = str(step.get("asset") or "")
            if not aid or self._is_open_loop_food(aid):
                continue
            # Hamburg/bowl last waypoint is a planned place. Do not re-pick
            # hamburg_2 while flying to hamburg_3.
            if self._atom_last_gripper_open(arm, aidx) and (
                _is_hamburg_asset(aid) or _is_bowl_asset(aid) or self._asset_has_planned_place(aid)
            ):
                continue
            if self._is_drop_test_target(aid) and aid not in self._drop_test_done:
                if self.monitor.ever_lifted(aid):
                    continue
            miss = self._lift_check_assets(
                [aid],
                ends_open=self._atom_last_gripper_open(arm, aidx),
                atom=str(step.get("atom") or ""),
            )
            if miss.dropped:
                print(
                    f"[TaskRuntime] grasp missed on {aid}.{step.get('atom')} "
                    f"(atom {aidx}); retry now, do not wait for the rest of the pack"
                )
                return miss
        return GraspEvent.ok()

    def _require_lift_before_done(self) -> GraspEvent:
        """Pick-like atoms are not done if the object never left the table.

        A planned final open is a release: success if we already lifted once,
        even though the object is back on the table.
        Dual-can: only judge after both arms finished the higher lift.
        """
        if not self.cfg.grasp_check_enabled:
            return GraspEvent.ok()
        if self.compiled is None or not self._is_pick_atom():
            return GraspEvent.ok()
        ids = []
        for aid in list(self.compiled.grasp_assets) or [self.compiled.asset_id]:
            if aid and not self._is_open_loop_food(aid) and aid not in ids:
                ids.append(aid)
        if not ids:
            return GraspEvent.ok()
        return self._lift_check_assets(ids, ends_open=self._atom_ends_with_open())

    def _arm_grasp_asset(self, arm: ArmExecState) -> str | None:
        for wp in arm.tasks:
            aid = wp.get("grasp_asset")
            if aid:
                return str(aid)
        return None

    def _arm_current_asset(self, arm: ArmExecState) -> str | None:
        if 0 <= arm.step_idx < len(arm.tasks):
            aid = arm.tasks[arm.step_idx].get("grasp_asset")
            return str(aid) if aid else None
        if arm.tasks:
            aid = arm.tasks[-1].get("grasp_asset")
            return str(aid) if aid else None
        return None

    def _keep_unaffected_arm(self, event: GraspEvent) -> ArmExecState | None:
        """Keep the arm that is not regrasping the failed object (independent dual-arm)."""
        if self.compiled is None or not self.compiled.simultaneous:
            return None
        failed = str(event.asset_id or "")
        for arm in (self.left_arm, self.right_arm):
            remaining = arm.tasks[max(arm.step_idx, 0) :]
            if not remaining and not arm.executed_any:
                continue
            raw = remaining[0].get("grasp_asset") if remaining else self._arm_current_asset(arm)
            current = str(raw) if raw else None
            if failed and (
                current == failed
                or any(str(wp.get("grasp_asset") or "") == failed for wp in remaining[:4])
            ):
                continue
            if str(arm.gripper_cmd or "open") != "close" and not arm.executed_any:
                continue
            return copy.copy(arm)
        kept = self._keep_open_loop_arm(event)
        if kept is not None and failed and self._arm_current_asset(kept) == failed:
            return None
        return kept

    def _keep_open_loop_arm(self, event: GraspEvent) -> ArmExecState | None:
        """Hamburg/fries grasp once: bowl miss must not reopen that gripper."""
        if self.compiled is None or not self.compiled.simultaneous:
            return None
        failed = str(event.asset_id or "")
        for arm in (self.left_arm, self.right_arm):
            aid = self._arm_grasp_asset(arm)
            if not aid or aid == failed or not self._is_open_loop_food(aid):
                continue
            if str(arm.gripper_cmd or "open") != "close" and not arm.executed_any:
                continue
            return copy.copy(arm)
        return None

    def _asset_place_leave_started(self, asset_id: str) -> bool:
        """True when this object is already placed and the arm is lifting away / on the next item."""
        if not asset_id or self.compiled is None:
            return False
        for i, step in enumerate(self.atomic_plan):
            if str(step.get("asset") or "") == str(asset_id) and i in self._consumed_plan_indices:
                return True
        for arm in (self.left_arm, self.right_arm):
            if not arm.tasks:
                continue
            idx = max(int(arm.step_idx), 0)
            last_i = None
            for i, wp in enumerate(arm.tasks):
                if str(wp.get("grasp_asset") or "") == str(asset_id):
                    last_i = i
            if last_i is None:
                continue
            if idx > last_i:
                return True
            remaining = [
                wp
                for wp in arm.tasks[idx:]
                if str(wp.get("grasp_asset") or "") == str(asset_id)
            ]
            if remaining and all(str(wp.get("gripper") or "open") == "open" for wp in remaining):
                return True
        return False

    def _abort_and_retry(self, event: GraspEvent) -> TickStatus:
        self.last_event = event
        tag = "GRASP_FAILED" if event.reason == "grasp_failed" else "DROP"
        if (
            event.reason == "grasp_failed"
            and event.asset_id
            and (
                _is_hamburg_asset(event.asset_id) or _is_bowl_asset(event.asset_id)
            )
            and self._asset_place_leave_started(event.asset_id)
        ):
            print(
                f"[TaskRuntime] ignore {tag} {event.asset_id}: "
                "already placed, flying to the next target"
            )
            return "running"
        if self._recovery_adjust is not None:
            print(
                f"[TaskRuntime] {tag} during adjust {self._recovery_adjust} "
                f"{event.reason} z={event.object_z:.3f}; retry adjust now"
            )
            active = self._recovery_active_arm()
            self._open_grippers([active] if active else None)
            return self._finish_or_retry_adjust()
        if event.reason != "drop_test":
            self.retry_count += 1
        print(
            f"[TaskRuntime] {tag} {event.reason} asset={event.asset_id} arm={event.arm_name} "
            f"rel={event.rel_disp:.3f} z={event.object_z:.3f} retry={self.retry_count}/{self.cfg.max_atom_retries}"
        )
        if self.retry_count > self.cfg.max_atom_retries:
            print("[TaskRuntime] max retries exceeded")
            self.failed = True
            return "failed"
        self._pending_adjust = None
        keep = self._keep_unaffected_arm(event)
        open_arms = list(self.robots.keys())
        if keep is not None:
            open_arms = [n for n in open_arms if n != keep.name]
            print(
                f"[TaskRuntime] keep {keep.name} {self._arm_current_asset(keep)} "
                "(unaffected arm, no regrasp)"
            )
        self._open_grippers(open_arms)
        self._restore_grasp_mass(reason="regrasp", asset_ids=[event.asset_id] if event.asset_id else None)
        if not self._recompile_current(reason=f"drop:{event.reason}"):
            return "failed"
        if keep is not None:
            if keep.name == "robot_left":
                self.left_arm = keep
            else:
                self.right_arm = keep
        return "dropped"

    def _active_grasp(self, arm_name: str, gripper_cmd: str) -> str | None:
        if gripper_cmd != "close" or self.compiled is None:
            return None
        if self.compiled.simultaneous:
            arm = self.left_arm if arm_name == "robot_left" else self.right_arm
            tasks = arm.tasks
            if 0 <= arm.step_idx < len(tasks):
                return tasks[arm.step_idx].get("grasp_asset")
            # Finished arm: keep watching ITS last object (can_2), never the
            # pack's asset_id (can_1). Otherwise the idle closed gripper is
            # scored against the other can still sitting on the table.
            if tasks:
                return tasks[-1].get("grasp_asset")
            return None
        wp = self.compiled.waypoints[self._seq.step] if self._seq.step < len(self.compiled.waypoints) else None
        if wp is not None:
            return wp.get("grasp_asset")
        return self.compiled.asset_id

    def _current_close_goal_xy(self) -> tuple[float, float] | None:
        if self.compiled is None:
            return None
        pose = None
        gripper = "open"
        if self.compiled.simultaneous:
            for arm in (self.left_arm, self.right_arm):
                if 0 <= arm.step_idx < len(arm.tasks):
                    task = arm.tasks[arm.step_idx]
                    if str(task.get("gripper") or "open") == "close":
                        pose = task.get("pose")
                        break
        elif self.compiled.waypoints:
            i = min(max(self._seq.step, 0), len(self.compiled.waypoints) - 1)
            wp = self.compiled.waypoints[i]
            gripper = str(wp.get("gripper") or "open")
            pose = wp.get("pose") if gripper == "close" else None
        if not pose or len(pose) < 2:
            return None
        return (float(pose[0]), float(pose[1]))

    def _arm_close_goal_xy(self, arm: ArmExecState) -> tuple[float, float] | None:
        """Current close waypoint only. A finished arm is holding, not carrying."""
        if not (0 <= arm.step_idx < len(arm.tasks)):
            return None
        task = arm.tasks[arm.step_idx]
        if str(task.get("gripper") or "open") != "close":
            return None
        pose = task.get("pose")
        if not pose or len(pose) < 2:
            return None
        return (float(pose[0]), float(pose[1]))

    def _carry_miss_event(self, asset_id: str, goal_xy: tuple[float, float], arm_name: str | None = None) -> GraspEvent:
        try:
            snap = self.world.get(asset_id)
        except KeyError:
            return GraspEvent.ok()
        rest_z = self.monitor.rest_z(asset_id, self.world.table_z(fallback=0.30))
        if self.monitor.ever_lifted(asset_id) or snap.pos[2] > rest_z + self.monitor.cfg.lift_confirm_dz:
            return GraspEvent.ok()
        dx = goal_xy[0] - snap.pos[0]
        dy = goal_xy[1] - snap.pos[1]
        if (dx * dx + dy * dy) ** 0.5 < 0.22:
            return GraspEvent.ok()
        print(
            f"[TaskRuntime] carry without lift {asset_id}.{self.compiled.atom if self.compiled else ''} "
            f"goal_xy={[round(goal_xy[0], 3), round(goal_xy[1], 3)]} "
            f"obj_xy={[round(snap.pos[0], 3), round(snap.pos[1], 3)]} "
            f"z={snap.pos[2]:.3f} rest_z={rest_z:.3f}"
        )
        return GraspEvent(
            dropped=True,
            reason="grasp_failed",
            asset_id=asset_id,
            arm_name=arm_name,
            object_z=snap.pos[2],
            table_z=rest_z,
        )

    def _carry_without_lift_event(self) -> GraspEvent:
        """Closed gripper flying to a place pose while the object is still on the table."""
        if self.compiled is None or not self._is_pick_atom():
            self._carry_miss_frames = 0
            return GraspEvent.ok()
        if self.compiled.simultaneous:
            hit = GraspEvent.ok()
            for arm_name, arm in (("robot_left", self.left_arm), ("robot_right", self.right_arm)):
                if not (0 <= arm.step_idx < len(arm.tasks)):
                    continue
                cmd = str(arm.gripper_cmd or "open")
                aid = self._active_grasp(arm_name, cmd)
                if (
                    not aid
                    or self._is_open_loop_food(aid)
                    or _is_bowl_asset(aid)
                    or _is_hamburg_asset(aid)
                ):
                    continue
                cur = arm.tasks[arm.step_idx]
                if str(cur.get("grasp_asset") or "") != str(aid):
                    continue
                if self._is_planned_release(arm_name, cmd) and (
                    self.monitor.ever_lifted(aid) or aid in self._drop_test_done
                ):
                    continue
                goal_xy = self._arm_close_goal_xy(arm)
                if goal_xy is None:
                    continue
                ev = self._carry_miss_event(aid, goal_xy, arm_name)
                if ev.dropped:
                    hit = ev
                    break
            if hit.dropped:
                self._carry_miss_frames += 1
                if self._carry_miss_frames >= 12:
                    return hit
                return GraspEvent.ok()
            self._carry_miss_frames = 0
            return GraspEvent.ok()
        asset_id = self.compiled.asset_id
        if (
            self._is_open_loop_food(asset_id)
            or _is_bowl_asset(asset_id)
            or _is_hamburg_asset(asset_id)
        ):
            self._carry_miss_frames = 0
            return GraspEvent.ok()
        goal_xy = self._current_close_goal_xy()
        if goal_xy is None:
            self._carry_miss_frames = 0
            return GraspEvent.ok()
        ev = self._carry_miss_event(asset_id, goal_xy)
        if ev.dropped:
            self._carry_miss_frames += 1
            if self._carry_miss_frames >= 12:
                return ev
            return GraspEvent.ok()
        self._carry_miss_frames = 0
        return GraspEvent.ok()

    def _check_drops(self) -> GraspEvent:
        if not self.cfg.grasp_check_enabled:
            return GraspEvent.ok()
        if self.compiled is None:
            return GraspEvent.ok()
        # Only watch pick/stack/adjust. into/on/hang close the gripper to place
        # something already in hand; the named asset is the container, not the grasp.
        if not self._is_pick_atom():
            return GraspEvent.ok()
        if self.compiled.simultaneous:
            pairs = (
                ("robot_left", self.left_arm.gripper_cmd),
                ("robot_right", self.right_arm.gripper_cmd),
            )
        else:
            pairs = ((self._seq_arm_name(), self._seq.gripper_cmd),)
        for arm_name, cmd in pairs:
            if arm_name is None:
                continue
            aid = self._active_grasp(arm_name, cmd)
            if self._is_open_loop_food(aid):
                continue
            if aid and self._asset_place_leave_started(aid):
                continue
            releasing = self._is_planned_release(arm_name, cmd)
            # Place then lift-away, or already open flying to the next target:
            # the object is supposed to be on the table. Do not require ever_lifted
            # (a short lift can miss the 12-frame mark and then look like a miss).
            if (
                cmd != "close"
                or not aid
                or (
                    releasing
                    and (
                        self.monitor.ever_lifted(aid)
                        or aid in self._drop_test_done
                        or self._is_last_close_then_open(arm_name)
                    )
                )
            ):
                continue
            event = self.monitor.update(
                world=self.world,
                arm_name=arm_name,
                gripper_cmd=cmd,
                grasp_asset=aid,
                releasing=releasing,
            )
            if event.dropped:
                if (
                    event.reason == "grasp_failed"
                    and event.asset_id
                    and self._object_z_lifted(event.asset_id)
                ):
                    print(
                        f"[TaskRuntime] ignore grasp_failed {event.asset_id}: "
                        "object is already in the air, do not open the gripper"
                    )
                    continue
                return event
        return self._carry_without_lift_event()

    def _seq_arm_name(self) -> str | None:
        if self.compiled is None or not self.compiled.waypoints:
            return None
        i = min(max(self._seq.step, 0), len(self.compiled.waypoints) - 1)
        ex = self.compiled.waypoints[i].get("executor", "robot_right")
        if ex == "robot_left":
            return "robot_left"
        if ex == "robot_right":
            return "robot_right"
        if "left" in str(ex):
            return "robot_left"
        if "right" in str(ex):
            return "robot_right"
        return "robot_right"

    def _reached(
        self,
        arm_name: str,
        pose: list[float],
        ori_threshold: float | None = None,
        pos_threshold: float | None = None,
    ) -> bool:
        fn = self.reach_fns.get(arm_name)
        if fn is not None:
            kwargs = {}
            if ori_threshold is not None:
                kwargs["ori_threshold"] = ori_threshold
            if pos_threshold is not None:
                kwargs["pos_threshold"] = pos_threshold
            return bool(fn(scene=self.scene, goal_pose=pose, **kwargs))
        robot = self.robots[arm_name]
        try:
            ids, _ = robot.find_bodies(self.cfg.ee_link)
            bid = int(ids[0])
        except Exception:
            return True
        idx = min(self.cfg.env_id, int(robot.data.body_state_w.shape[0]) - 1)
        state = robot.data.body_state_w[idx, bid]
        cur = state[0:3]
        tgt = torch.tensor(pose[:3], device=cur.device, dtype=cur.dtype)
        limit_pos = self.cfg.reach_threshold if pos_threshold is None else float(pos_threshold)
        if torch.norm(cur - tgt).item() >= limit_pos:
            return False
        if len(pose) < 7:
            return True
        qwxyz = [float(state[3 + i].item()) for i in range(4)]
        dot = abs(sum(qwxyz[i] * float(pose[3 + i]) for i in range(4)))
        ang = 2.0 * math.acos(min(1.0, dot))
        limit = self.cfg.ori_threshold if ori_threshold is None else ori_threshold
        return ang < float(limit)

    def _tick_sequential(self) -> TickStatus:
        assert self.compiled is not None
        pack = self.compiled.waypoints
        if self._seq.step >= len(pack):
            if not self._seq.executed_any:
                print("[TaskRuntime] atom failed: every sequential waypoint plan failed")
                self.failed = True
                return "failed"
            if self._should_hold_atom_for_drop_test():
                self._atom_hold_frames += 1
                if self._atom_hold_frames > int(self.cfg.drop_hold_max_frames):
                    print(
                        f"[TaskRuntime] drop-test hold timeout for "
                        f"{self.compiled.asset_id}.{self.compiled.atom}, finish atom"
                    )
                    return "atom_done"
                self._hold_last_grasp_pose()
                return "running"
            return "atom_done"
        if self._should_freeze_seq_for_drop_test():
            self._hold_last_grasp_pose()
            return "running"
        task = pack[self._seq.step]
        arm_name = self._seq_arm_name() or "robot_right"
        robot = self.robots[arm_name]
        planner = self.planners[arm_name]
        self._seq.gripper_cmd = task.get("gripper", "open")
        self._seq.pose = task["pose"]
        pose = list(task["pose"])

        if self._seq.joint_pos_des is None:
            atom_tasks = []
            atom_key = _atom_key(task)
            shown_key = None
            if task.get("marker_per_waypoint"):
                atom_tasks = [task]
                shown_key = ("point", atom_key, self._seq.step)
            else:
                for later in pack[self._seq.step :]:
                    if _atom_key(later) != atom_key:
                        break
                    atom_tasks.append(later)
            show_atom_trajectories(
                planner,
                arm_name,
                atom_tasks,
                self.q_starts[arm_name],
                self.scene,
                self.link_list,
                self._plan_safe,
                goal_fn=build_goal_pose_sequential,
                exclusive=True,
                shown_key=shown_key,
            )
            apply_table_obstacle(planner, task, self.scene)
            use_oa = task.get("use_obstacle_avoidance", True)
            if use_oa:
                planner.enable_link_collision(enable_collision_links=self.link_list)
            else:
                planner.disable_link_collision(disable_collision_links=self.link_list)
            goal = build_goal_pose_sequential(planner=planner, arm_name2=arm_name, pose_7d=pose)
            ori_tol = task.get("ori_tol_deg")
            pos_tol = task.get("position_tolerance")
            if pos_tol is not None:
                self._seq.joint_pos_des = plan_pose_pos_tol(
                    planner, self.q_starts[arm_name], goal, pos_tol
                )
            elif ori_tol is None:
                self._seq.joint_pos_des = self._plan_safe(planner, self.q_starts[arm_name], goal)
            else:
                self._seq.joint_pos_des = plan_pose_ori_tol(
                    planner, self.q_starts[arm_name], goal, ori_tol
                )
            self._seq.need_hold_this_step = self._seq.gripper_cmd == "close" and self._seq.prev_gripper_cmd != "close"
            self._seq.hold_counter = 0
            self._seq.waypoint_idx = 0
            if self._seq.joint_pos_des is None:
                self._seq.plan_fail_count += 1
                print(
                    f"[TaskRuntime] plan failed waypoint {self._seq.step} "
                    f"{self.compiled.asset_id}.{self.compiled.atom} "
                    f"arm={arm_name} world_xyz={[round(float(v), 3) for v in pose[:3]]} "
                    f"try={self._seq.plan_fail_count}"
                )
                if self._seq.plan_fail_count >= 8:
                    first_close = (
                        self._seq.gripper_cmd == "close"
                        and self._seq.prev_gripper_cmd != "close"
                    )
                    if first_close:
                        print(
                            f"[TaskRuntime] keep retrying grasp-close waypoint {self._seq.step} "
                            f"({self.compiled.asset_id}.{self.compiled.atom}); not skipping"
                        )
                        self._seq.plan_fail_count = 0
                    else:
                        print(f"[TaskRuntime] skip waypoint {self._seq.step} after plan failures")
                        self._seq.step += 1
                        hide_finished_atom(planner, arm_name, pack, self._seq.step)
                        self._seq.plan_fail_count = 0
                return "running"
            self._seq.plan_fail_count = 0
            self._seq.executed_any = True
            print(
                f"[TaskRuntime] planned waypoint {self._seq.step} "
                f"{self.compiled.asset_id}.{self.compiled.atom} "
                f"arm={arm_name} traj={int(self._seq.joint_pos_des.shape[0])} "
                f"world_xyz={[round(float(v), 3) for v in pose[:3]]} "
                f"{self._err_txt(arm_name, pose)}"
            )

        traj = self._seq.joint_pos_des
        if self._seq.waypoint_idx < traj.shape[0]:
            q_way = traj[self._seq.waypoint_idx].reshape(-1)
            self._set_arm_q(arm_name, q_way)
            if self._seq.need_hold_this_step and self._seq.waypoint_idx == 0 and self._seq.hold_counter < self.cfg.hold_frames:
                self._seq.hold_counter += 1
            else:
                self._seq.waypoint_idx += 1
            self._seq.reach_wait = 0
        elif self._reached(
            arm_name,
            pose,
            None if task.get("ori_tol_deg") is None else math.radians(float(task["ori_tol_deg"])),
            None if task.get("reach_threshold") is None else float(task["reach_threshold"]),
        ):
            last_q = traj[-1]
            dof = len(planner.joint_names)
            print(
                f"[TaskRuntime] reached waypoint {self._seq.step}/{len(pack)} "
                f"{self.compiled.asset_id}.{self.compiled.atom} "
                f"arm={arm_name} gripper={self._seq.gripper_cmd} {self._err_txt(arm_name, pose)}"
            )
            self.q_starts[arm_name] = JointState.from_position(
                last_q[:dof].unsqueeze(0),
                joint_names=planner.joint_names,
            )
            self._seq.step += 1
            hide_finished_atom(planner, arm_name, pack, self._seq.step)
            self._seq.joint_pos_des = None
            self._seq.prev_gripper_cmd = self._seq.gripper_cmd
            self._seq.reach_wait = 0
        else:
            self._seq.reach_wait += 1
            last_q = traj[-1]
            self._set_arm_q(arm_name, last_q)
            if self._seq.reach_wait == 1 or self._seq.reach_wait % 50 == 0:
                print(
                    f"[TaskRuntime] waiting to reach waypoint {self._seq.step} "
                    f"{self.compiled.asset_id}.{self.compiled.atom} "
                    f"frames={self._seq.reach_wait}/{self.cfg.reach_wait_frames} "
                    f"{self._err_txt(arm_name, pose)}"
                )
            if self._seq.reach_wait >= int(self.cfg.reach_wait_frames):
                print(
                    f"[TaskRuntime] skip waypoint {self._seq.step} after reach timeout "
                    f"({self.compiled.asset_id}.{self.compiled.atom})"
                )
                dof = len(planner.joint_names)
                self.q_starts[arm_name] = JointState.from_position(
                    last_q[:dof].unsqueeze(0),
                    joint_names=planner.joint_names,
                )
                self._seq.step += 1
                hide_finished_atom(planner, arm_name, pack, self._seq.step)
                self._seq.joint_pos_des = None
                self._seq.prev_gripper_cmd = self._seq.gripper_cmd
                self._seq.reach_wait = 0

        if self._seq.gripper_cmd == "close":
            robot.set_joint_position_target(self.finger_close[arm_name], joint_ids=self.finger_indices[arm_name])
        else:
            robot.set_joint_position_target(self.finger_open[arm_name], joint_ids=self.finger_indices[arm_name])

        if self._seq.step >= len(pack) and self._seq.joint_pos_des is None:
            if not self._seq.executed_any:
                print("[TaskRuntime] atom failed: every sequential waypoint plan failed")
                self.failed = True
                return "failed"
            if self._should_hold_atom_for_drop_test():
                self._atom_hold_frames += 1
                if self._atom_hold_frames > int(self.cfg.drop_hold_max_frames):
                    print(
                        f"[TaskRuntime] drop-test hold timeout for "
                        f"{self.compiled.asset_id}.{self.compiled.atom}, finish atom"
                    )
                    return "atom_done"
                self._hold_last_grasp_pose()
                return "running"
            return "atom_done"
        return "running"

    def _tick_simultaneous(self) -> TickStatus:
        assert self.compiled is not None
        self._mark_arm_finished_atoms(self.left_arm)
        self._mark_arm_finished_atoms(self.right_arm)
        started = self._maybe_adjust_after_open_atom()
        if started is not None:
            return started
        if self._hold_finished_arm(self.left_arm, "robot_left"):
            pass
        elif self._should_freeze_arm_for_drop_test(self.left_arm):
            self._hold_one_arm_close("robot_left")
        else:
            self.q_starts["robot_left"], _ = tick_one_arm(
                arm=self.left_arm,
                scene=self.scene,
                planner=self.planners["robot_left"],
                robot=self.robots["robot_left"],
                q_start=self.q_starts["robot_left"],
                move_to_pose_fn=self._plan_safe,
                reach_fn=self.reach_fns.get(
                    "robot_left",
                    lambda scene, goal_pose, ori_threshold=None: self._reached(
                        "robot_left", goal_pose, ori_threshold
                    ),
                ),
                link_list=self.link_list,
                finger_indices=self.finger_indices["robot_left"],
                finger_open_target=self.finger_open["robot_left"],
                finger_close_target=self.finger_close["robot_left"],
                hold_frames=self.cfg.hold_frames,
                arm_name="robot_left",
                count2=self.tick_count,
                sync_ctx=self.sync_ctx,
                reach_wait_frames=self.cfg.reach_wait_frames,
            )
        if self._hold_finished_arm(self.right_arm, "robot_right"):
            pass
        elif self._should_freeze_arm_for_drop_test(self.right_arm):
            self._hold_one_arm_close("robot_right")
        else:
            self.q_starts["robot_right"], _ = tick_one_arm(
                arm=self.right_arm,
                scene=self.scene,
                planner=self.planners["robot_right"],
                robot=self.robots["robot_right"],
                q_start=self.q_starts["robot_right"],
                move_to_pose_fn=self._plan_safe,
                reach_fn=self.reach_fns.get(
                    "robot_right",
                    lambda scene, goal_pose, ori_threshold=None: self._reached(
                        "robot_right", goal_pose, ori_threshold
                    ),
                ),
                link_list=self.link_list,
                finger_indices=self.finger_indices["robot_right"],
                finger_open_target=self.finger_open["robot_right"],
                finger_close_target=self.finger_close["robot_right"],
                hold_frames=self.cfg.hold_frames,
                arm_name="robot_right",
                count2=self.tick_count,
                sync_ctx=self.sync_ctx,
                reach_wait_frames=self.cfg.reach_wait_frames,
            )
        miss = self._check_just_finished_atoms(self.left_arm)
        if not miss.dropped:
            miss = self._check_just_finished_atoms(self.right_arm)
        if not miss.dropped:
            miss = self._adjust_grasp_missed()
        if miss.dropped:
            return self._abort_and_retry(miss)
        started = self._maybe_adjust_after_open_atom()
        if started is not None:
            return started
        self._mark_arm_finished_atoms(self.left_arm)
        self._mark_arm_finished_atoms(self.right_arm)
        left_done = self.left_arm.step_idx >= len(self.left_arm.tasks)
        right_done = self.right_arm.step_idx >= len(self.right_arm.tasks)
        # Retry standing the bottle as soon as the adjust arm finishes. Do not
        # wait for the other arm's leftover hamburg/bowl waypoints.
        if self._recovery_adjust is not None:
            active = self._recovery_active_arm()
            active_arm = self.left_arm if active == "robot_left" else self.right_arm
            if active_arm.tasks and active_arm.step_idx >= len(active_arm.tasks):
                if self._arm_executed(active_arm) or active_arm.step_idx > 0:
                    return "atom_done"
        if left_done and right_done:
            if not (self._arm_executed(self.left_arm) or self._arm_executed(self.right_arm)):
                print("[TaskRuntime] atom failed: every simultaneous waypoint plan failed")
                self.failed = True
                return "failed"
            if self._should_hold_atom_for_drop_test():
                self._hold_last_grasp_pose()
                return "running"
            return "atom_done"
        return "running"

    def _arm_executed(self, arm) -> bool:
        if getattr(arm, "executed_any", False):
            return True
        # Older ArmExecState has no executed_any; finishing the queue means a plan ran.
        return bool(arm.tasks) and arm.step_idx >= len(arm.tasks)

    def tick(self) -> TickStatus:
        self.tick_count += 1
        applied = self.world.apply_pending()
        if applied:
            print(f"[TaskRuntime] applied {applied} pending world mutation(s)")
        self.world.refresh()
        self._maybe_log_exec()
        self._maybe_inject_drop_mass()
        dump_event = self._maybe_finish_survived_drop_test()
        if dump_event.dropped:
            return self._abort_and_retry(dump_event)

        if self.failed:
            return "failed"
        if self.finished or (self.atom_index >= len(self.atomic_plan) and self._place_watch_until is None and self._recovery_adjust is None):
            self.finished = True
            return "finished"

        if self._place_watch_until is not None:
            if self._ignore_fallen_bottle():
                self._place_watch_until = None
                return self._continue_to_paused(reason="skip_bottle_watch_after_into_2")
            return self._tick_place_watch()

        if self._recovery_adjust is None and not self._ignore_fallen_bottle():
            fallen = self._knocked_bottle_id(during_atom=True)
            if fallen is not None and not self._current_atom_is_adjust(fallen):
                self._note_fallen_bottle(fallen)

        if self.compiled is None:
            if not self._recompile_current(reason="missing"):
                return "failed" if self.failed else "finished"

        if self.tick_count % self.cfg.plan_every_n != 0:
            event = self._check_drops()
            if event.dropped:
                return self._abort_and_retry(event)
            if self.compiled is not None and self.compiled.simultaneous:
                self._hold_finished_arm(self.left_arm, "robot_left")
                self._hold_finished_arm(self.right_arm, "robot_right")
            if self._should_hold_atom_for_drop_test():
                exhausted = (
                    self.compiled.simultaneous
                    and self.left_arm.step_idx >= len(self.left_arm.tasks)
                    and self.right_arm.step_idx >= len(self.right_arm.tasks)
                ) or (
                    (not self.compiled.simultaneous)
                    and self._seq.step >= len(self.compiled.waypoints)
                )
                if exhausted:
                    self._hold_last_grasp_pose()
            started = self._maybe_adjust_after_open_atom()
            if started is not None:
                return started
            return "running"

        event = self._check_drops()
        if event.dropped:
            return self._abort_and_retry(event)

        status = self._tick_simultaneous() if self.compiled.simultaneous else self._tick_sequential()
        if status == "running" and not self.compiled.simultaneous:
            started = self._maybe_adjust_after_open_atom()
            if started is not None:
                return started
        if status == "atom_done":
            if self._recovery_adjust is not None:
                print(
                    f"[TaskRuntime] recovery adjust done for {self._recovery_adjust}"
                )
                return self._finish_or_retry_adjust()
            miss = self._require_lift_before_done()
            fallen_id = self._pending_adjust or self._knocked_bottle_id(during_atom=False)
            if miss.dropped and miss.asset_id != fallen_id:
                return self._abort_and_retry(miss)
            if self._should_hold_atom_for_drop_test():
                self._hold_last_grasp_pose()
                return "running"
            print(
                f"[TaskRuntime] atom done [{self.compiled.start_index}:{self.compiled.end_index}] "
                f"{self.compiled.asset_id}.{self.compiled.atom} grasp_ok"
            )
            self._restore_grasp_mass(
                reason="atom_done",
                asset_ids=list(self.compiled.grasp_assets),
            )
            self.atom_index = self.compiled.end_index
            self.retry_count = 0
            self.monitor.reset()
            if self._ignore_fallen_bottle() and (fallen_id or self._pending_adjust):
                self._dismiss_fallen_bottle(
                    f"after {self.compiled.asset_id}.{self.compiled.atom}"
                )
                fallen_id = None
            if fallen_id and not self._should_defer_adjust():
                self._pending_adjust = fallen_id
                self._paused_atom_index = self.atom_index
                print(
                    f"[TaskRuntime] atom ended; now adjust fallen {fallen_id}"
                )
                return self._begin_adjust_recovery(fallen_id)
            if fallen_id:
                self._pending_adjust = fallen_id
                print(
                    f"[TaskRuntime] {fallen_id} still down after close-ending "
                    f"{self.compiled.asset_id}.{self.compiled.atom}; "
                    "defer adjust until an open-ending atom finishes"
                )
            if self._should_watch_bottle_after_atom():
                self._paused_atom_index = self.atom_index
                self._place_watch_until = self.tick_count + self.cfg.place_watch_frames
                print(
                    f"[TaskRuntime] watching bottle for {self.cfg.place_watch_frames} frames "
                    f"after {self.compiled.asset_id}.{self.compiled.atom}"
                )
                return "running"
            if self.atom_index >= len(self.atomic_plan):
                if self._pending_adjust and not self._ignore_fallen_bottle():
                    self._paused_atom_index = self.atom_index
                    print(
                        f"[TaskRuntime] plan done; adjust fallen {self._pending_adjust}"
                    )
                    return self._begin_adjust_recovery(self._pending_adjust)
                if self._pending_adjust:
                    self._dismiss_fallen_bottle("plan done")
                self.finished = True
                self.compiled = None
                return "finished"
            self._recompile_current(reason="next_atom")
            return "atom_done"
        return status
