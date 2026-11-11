"""Detect failed grasps, gripper-object slip, and objects falling back onto the table."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.world_state import WorldState


def _is_bowl_asset(asset_id: str | None) -> bool:
    if not asset_id:
        return False
    return "bowl" in str(asset_id).lower()


def _is_hamburg_asset(asset_id: str | None) -> bool:
    if not asset_id:
        return False
    return "hamburg" in str(asset_id).lower()


def _is_can_asset(asset_id: str | None) -> bool:
    if not asset_id:
        return False
    name = str(asset_id).lower()
    return name.startswith("can_") or "_can" in name or name.endswith("_can") or name == "can"


def _is_bread_asset(asset_id: str | None) -> bool:
    if not asset_id:
        return False
    name = str(asset_id).lower()
    if "basket" in name:
        return False
    return "bread" in name


@dataclass
class GraspMonitorCfg:
    table_z_margin: float = 0.04
    lift_confirm_dz: float = 0.02
    grasp_near_dist: float = 0.16
    settle_frames: int = 8
    hold_stable_frames: int = 12
    drop_confirm_frames: int = 8
    # After a stable lift, the object leaving the hand shows up as a growing
    # EE-to-object distance. World-frame (obj-ee) vectors are NOT used: they
    # change whenever the arm rotates even if the grasp is rigid.
    slip_dist_margin: float = 0.10
    failed_lift_ee_dz: float = 0.08
    failed_lift_obj_dz: float = 0.02
    failed_lift_frames: int = 20
    # Closed gripper then moved away, object still on the table.
    miss_extra_dist: float = 0.05


@dataclass
class GraspEvent:
    dropped: bool
    reason: str = ""
    asset_id: str | None = None
    arm_name: str | None = None
    rel_disp: float = 0.0
    object_z: float = 0.0
    table_z: float = 0.0

    @staticmethod
    def ok() -> "GraspEvent":
        return GraspEvent(dropped=False)

    @property
    def grasp_failed(self) -> bool:
        return self.dropped and self.reason == "grasp_failed"


@dataclass
class _HoldLock:
    asset_id: str
    arm_name: str
    object_z: float
    ee_z: float
    rest_z: float
    locked_dist: float | None = None
    lift_ee_xy: tuple[float, float] | None = None
    lifted: bool = False
    lifted_frames: int = 0
    frames: int = 0
    slip_streak: int = 0
    table_streak: int = 0
    fail_streak: int = 0


class GraspMonitor:
    """
    Hold tracker.

    After the gripper closes near the object:
      * grasp_failed   — never successfully picked up (arm left / rose, object stayed)
      * relative_slip  — lifted, then EE-to-object distance grew
      * back_on_table  — lifted, then z returned to the table
    All three must persist for `drop_confirm_frames` before firing.
    A planned release (`releasing=True`) suppresses slip while still holding
    and also `back_on_table` when putting the object down. A close that
    never lifts is still `grasp_failed` (until the last close-then-open place).
    """

    def __init__(self, cfg: GraspMonitorCfg | None = None):
        self.cfg = cfg or GraspMonitorCfg()
        self._locks: dict[str, _HoldLock] = {}
        self._pending: dict[str, dict] = {}
        self._away_frames: dict[str, int] = {}
        self._rest_z: dict[str, float] = {}
        self._lifted_assets: set[str] = set()

    def reset(self, arm_name: str | None = None) -> None:
        if arm_name is None:
            self._locks.clear()
            self._pending.clear()
            self._away_frames.clear()
            # Keep _lifted_assets across recompile so a planned place is not a miss.
            return
        self._locks.pop(arm_name, None)
        self._pending.pop(arm_name, None)
        self._away_frames.pop(arm_name, None)

    def mark_lifted(self, asset_id: str) -> None:
        if asset_id:
            self._lifted_assets.add(str(asset_id))

    def remember_rest_z(self, asset_id: str, z: float) -> None:
        """Keep the lowest seen z so a mid-air retry cannot raise the table height."""
        z = float(z)
        prev = self._rest_z.get(asset_id)
        if prev is None or z < prev:
            self._rest_z[asset_id] = z

    def rest_z(self, asset_id: str, fallback: float) -> float:
        return float(self._rest_z.get(asset_id, fallback))

    def ever_lifted(self, asset_id: str) -> bool:
        if asset_id in self._lifted_assets:
            return True
        need = self.cfg.hold_stable_frames
        return any(
            lock.asset_id == asset_id and lock.lifted and lock.lifted_frames >= need
            for lock in self._locks.values()
        )

    def lifted_holds(self) -> list[_HoldLock]:
        return [lock for lock in self._locks.values() if lock.lifted]

    def closing_holds(self) -> list[_HoldLock]:
        """Holds that have already closed near the object, whether or not it has lifted."""
        return list(self._locks.values())

    def _dist(self, a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    def _emit(self, lock: _HoldLock, reason: str, dist: float, obj_z: float) -> GraspEvent:
        return GraspEvent(
            dropped=True,
            reason=reason,
            asset_id=lock.asset_id,
            arm_name=lock.arm_name,
            rel_disp=dist,
            object_z=obj_z,
            table_z=lock.rest_z,
        )

    def update(
        self,
        world: WorldState,
        arm_name: str,
        gripper_cmd: str,
        grasp_asset: str | None,
        releasing: bool = False,
    ) -> GraspEvent:
        if not grasp_asset or gripper_cmd != "close":
            self.reset(arm_name)
            return GraspEvent.ok()

        try:
            snap = world.get(grasp_asset)
        except KeyError:
            return GraspEvent.ok()
        ee = world.ee_pos(arm_name)
        if ee is None:
            return GraspEvent.ok()

        obj = snap.pos
        rest_z = self._rest_z.get(grasp_asset, world.table_z(fallback=0.30))
        dist = self._dist(obj, ee)
        on_table = obj[2] <= rest_z + self.cfg.table_z_margin
        lifted_now = obj[2] > rest_z + self.cfg.lift_confirm_dz
        # Left-arm planning offset is about (0, -0.4, -0.2) → EE-to-object
        # distance stays ~0.45 even on a solid grasp. Never use that distance
        # to call a hamburg/bowl/bread miss; only object-z vs EE-z.
        offset_grasp = (
            _is_bowl_asset(grasp_asset)
            or _is_hamburg_asset(grasp_asset)
            or _is_bread_asset(grasp_asset)
        )
        near_dist = 0.55 if offset_grasp else self.cfg.grasp_near_dist

        lock = self._locks.get(arm_name)
        if lock is None or lock.asset_id != grasp_asset:
            pend = self._pending.get(arm_name)
            if pend is None or pend.get("asset_id") != grasp_asset:
                self._pending[arm_name] = {"asset_id": grasp_asset, "near_frames": 0}
                pend = self._pending[arm_name]
            if offset_grasp or dist <= near_dist:
                pend["near_frames"] += 1
                self._away_frames[arm_name] = 0
            elif on_table and (not lifted_now) and dist > (near_dist + self.cfg.miss_extra_dist):
                away = self._away_frames.get(arm_name, 0) + 1
                self._away_frames[arm_name] = away
                if away >= (self.cfg.failed_lift_frames + self.cfg.drop_confirm_frames):
                    return GraspEvent(
                        dropped=True,
                        reason="grasp_failed",
                        asset_id=grasp_asset,
                        arm_name=arm_name,
                        rel_disp=dist,
                        object_z=obj[2],
                        table_z=rest_z,
                    )
            if pend["near_frames"] < self.cfg.settle_frames:
                return GraspEvent.ok()
            self._locks[arm_name] = _HoldLock(
                asset_id=grasp_asset,
                arm_name=arm_name,
                object_z=obj[2],
                ee_z=ee[2],
                rest_z=rest_z,
                lifted=lifted_now,
                lifted_frames=1 if lifted_now else 0,
                lift_ee_xy=(ee[0], ee[1]) if lifted_now else None,
            )
            self._pending.pop(arm_name, None)
            self._away_frames.pop(arm_name, None)
            return GraspEvent.ok()

        lock.frames += 1
        if lifted_now:
            lock.lifted = True
            lock.lifted_frames += 1
            if lock.lifted_frames >= self.cfg.hold_stable_frames:
                self._lifted_assets.add(lock.asset_id)
            if lock.lift_ee_xy is None:
                lock.lift_ee_xy = (ee[0], ee[1])
        else:
            lock.lifted_frames = 0
            if lock.locked_dist is None:
                lock.lifted = False

        if lock.lifted and lock.locked_dist is None:
            if lock.lifted_frames < self.cfg.hold_stable_frames:
                return GraspEvent(
                    dropped=False,
                    asset_id=grasp_asset,
                    arm_name=arm_name,
                    rel_disp=dist,
                    object_z=obj[2],
                    table_z=lock.rest_z,
                )
            lock.locked_dist = dist
            lock.lift_ee_xy = (ee[0], ee[1])
            return GraspEvent(
                dropped=False,
                asset_id=grasp_asset,
                arm_name=arm_name,
                rel_disp=0.0,
                object_z=obj[2],
                table_z=lock.rest_z,
            )

        slip = (
            (not offset_grasp)
            and lock.lifted
            and lock.locked_dist is not None
            and dist > (lock.locked_dist + self.cfg.slip_dist_margin)
        )
        far = (not offset_grasp) and lock.lifted and dist > (near_dist + self.cfg.slip_dist_margin)
        back = lock.lifted and on_table
        arm_rose_without_object = (
            (not lock.lifted)
            and on_table
            and lock.frames >= self.cfg.failed_lift_frames
            and (ee[2] - lock.ee_z) >= self.cfg.failed_lift_ee_dz
            and (obj[2] - lock.object_z) < self.cfg.failed_lift_obj_dz
        )
        hand_left_without_object = (
            (not offset_grasp)
            and (not lock.lifted)
            and on_table
            and lock.frames >= self.cfg.failed_lift_frames
            and dist > (near_dist + self.cfg.miss_extra_dist)
        )
        failed = arm_rose_without_object or hand_left_without_object

        lock.slip_streak = lock.slip_streak + 1 if (slip or far) else 0
        lock.table_streak = lock.table_streak + 1 if back else 0
        lock.fail_streak = lock.fail_streak + 1 if failed else 0
        need = self.cfg.drop_confirm_frames

        # Planned place: in-hand motion, or putting the object down then leaving,
        # must not look like slip / back_on_table / grasp_failed.
        still_holding = (
            lock.lifted
            and not on_table
            and dist <= (near_dist + self.cfg.miss_extra_dist)
            and (lock.locked_dist is None or dist <= lock.locked_dist + self.cfg.slip_dist_margin)
        )
        if releasing and (still_holding or (lock.lifted and on_table)):
            return GraspEvent(
                dropped=False,
                asset_id=grasp_asset,
                arm_name=arm_name,
                rel_disp=0.0 if lock.locked_dist is None else max(0.0, dist - lock.locked_dist),
                object_z=obj[2],
                table_z=lock.rest_z,
            )

        if lock.table_streak >= need:
            return self._emit(lock, "back_on_table", dist, obj[2])
        if lock.slip_streak >= need:
            return self._emit(lock, "relative_slip", dist - (lock.locked_dist or 0.0), obj[2])
        if lock.fail_streak >= need:
            return self._emit(lock, "grasp_failed", dist, obj[2])

        return GraspEvent(
            dropped=False,
            asset_id=grasp_asset,
            arm_name=arm_name,
            rel_disp=0.0 if lock.locked_dist is None else max(0.0, dist - lock.locked_dist),
            object_z=obj[2],
            table_z=lock.rest_z,
        )
