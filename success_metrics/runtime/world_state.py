"""Live read/write access to every simulation asset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch


def _as_float_list(values: Iterable[float], n: int) -> list[float]:
    out = [float(v) for v in values]
    if len(out) != n:
        raise ValueError(f"expected {n} values, got {len(out)}: {out}")
    return out


def _tensor_on(ref: torch.Tensor, data: Iterable[float], shape: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(list(data), device=ref.device, dtype=ref.dtype).reshape(shape)


@dataclass
class AssetSnapshot:
    asset_id: str
    scene_key: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    lin_vel: tuple[float, float, float] | None = None
    ang_vel: tuple[float, float, float] | None = None
    joint_pos: list[float] | None = None
    joint_names: list[str] | None = None
    mass: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "scene_key": self.scene_key,
            "pos": list(self.pos),
            "quat_wxyz": list(self.quat_wxyz),
            "lin_vel": None if self.lin_vel is None else list(self.lin_vel),
            "ang_vel": None if self.ang_vel is None else list(self.ang_vel),
            "joint_pos": None if self.joint_pos is None else list(self.joint_pos),
            "joint_names": None if self.joint_names is None else list(self.joint_names),
            "mass": self.mass,
        }


@dataclass
class _PendingWrite:
    kind: str
    asset_id: str
    env_id: int
    payload: dict[str, Any]


class WorldState:
    """
    Single place to query and mutate scene assets.

    Typical use inside the physics loop::

        world.refresh()
        snap = world.get("bottle_1")
        world.set_pose("bottle_1", pos=(0.5, 0.0, 0.35))   # applied immediately
        world.queue_set_pose("block_1", pos=(0.6, 0.1, 0.32))  # applied on next apply_pending()
    """

    def __init__(
        self,
        scene: Any,
        knowledge_map: dict[str, Any],
        env_id: int = 0,
        table_asset_id: str | None = None,
    ):
        self.scene = scene
        self.knowledge_map = knowledge_map
        self.env_id = int(env_id)
        self.table_asset_id = table_asset_id or self._infer_table_id()
        self._pending: list[_PendingWrite] = []
        self._cache: dict[str, AssetSnapshot] = {}
        self._ee_cache: dict[str, tuple[float, float, float]] = {}

    def restore_import_poses(self, env_id: int | None = None) -> list[str]:
        """Teleport task assets back to the pose used when they were imported."""
        env_id = self.env_id if env_id is None else int(env_id)
        restored: list[str] = []
        for asset_id, obj in self.knowledge_map.items():
            poses = getattr(obj, "import_poses", None) or []
            if not poses:
                continue
            idx = min(env_id, len(poses) - 1)
            pos, quat = poses[idx]
            try:
                self.set_pose(asset_id, pos=pos, quat_wxyz=quat, env_id=env_id, zero_velocity=True)
                restored.append(asset_id)
            except Exception as exc:
                print(f"[WorldState] restore import pose skipped {asset_id}: {exc}")
        if restored:
            print(f"[WorldState] restored import poses: {restored}")
        return restored

    def _infer_table_id(self) -> str | None:
        for aid in self.knowledge_map:
            if "table" in str(aid).lower():
                return aid
        return None

    def _dedup(self, items: list[str]) -> list[str]:
        out, seen = [], set()
        for it in items:
            if not it or it in seen:
                continue
            seen.add(it)
            out.append(it)
        return out

    def resolve_asset(self, asset_id: str, env_id: int | None = None) -> tuple[Any, str]:
        env_id = self.env_id if env_id is None else int(env_id)
        obj = self.knowledge_map.get(asset_id)
        scene_key = getattr(obj, "scene_key", "") if obj is not None else ""
        base_key = re.sub(r"__env\d+$", "", scene_key) if isinstance(scene_key, str) else ""
        a_type = str(asset_id).rsplit("_", 1)[0] if "_" in str(asset_id) else str(asset_id)
        candidates = self._dedup(
            [
                f"{scene_key}__env{env_id}" if scene_key else "",
                scene_key,
                f"{base_key}__env{env_id}" if base_key else "",
                base_key,
                f"my_{a_type}_{asset_id}__env{env_id}",
                f"my_{asset_id}__env{env_id}",
                f"my_{a_type}_{asset_id}",
                f"my_{asset_id}",
                f"{asset_id}__env{env_id}",
                asset_id,
            ]
        )
        for name in candidates:
            try:
                return self.scene[name], name
            except Exception:
                continue
        raise KeyError(f"cannot resolve asset_id={asset_id}, tried={candidates}")

    def _env_index(self, tensor: torch.Tensor, env_id: int) -> int:
        return min(int(env_id), int(tensor.shape[0]) - 1)

    def _read_pose(self, asset: Any, env_id: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        data = asset.data
        if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
            idx = self._env_index(data.root_pos_w, env_id)
            p = data.root_pos_w[idx]
            q = data.root_quat_w[idx]
            return (
                (float(p[0].item()), float(p[1].item()), float(p[2].item())),
                (float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item())),
            )
        if hasattr(data, "root_state_w"):
            idx = self._env_index(data.root_state_w, env_id)
            s = data.root_state_w[idx]
            return (
                (float(s[0].item()), float(s[1].item()), float(s[2].item())),
                (float(s[3].item()), float(s[4].item()), float(s[5].item()), float(s[6].item())),
            )
        raise RuntimeError("asset has neither root_pos_w/root_quat_w nor root_state_w")

    def _read_vel(self, asset: Any, env_id: int) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        data = asset.data
        if hasattr(data, "root_lin_vel_w") and hasattr(data, "root_ang_vel_w"):
            idx = self._env_index(data.root_lin_vel_w, env_id)
            lv = data.root_lin_vel_w[idx]
            av = data.root_ang_vel_w[idx]
            return (
                (float(lv[0].item()), float(lv[1].item()), float(lv[2].item())),
                (float(av[0].item()), float(av[1].item()), float(av[2].item())),
            )
        if hasattr(data, "root_state_w") and data.root_state_w.shape[-1] >= 13:
            idx = self._env_index(data.root_state_w, env_id)
            s = data.root_state_w[idx]
            return (
                (float(s[7].item()), float(s[8].item()), float(s[9].item())),
                (float(s[10].item()), float(s[11].item()), float(s[12].item())),
            )
        return None, None

    def _read_joints(self, asset: Any, env_id: int) -> tuple[list[float] | None, list[str] | None]:
        data = getattr(asset, "data", None)
        if data is None or not hasattr(data, "joint_pos") or data.joint_pos is None:
            return None, None
        if data.joint_pos.numel() == 0:
            return None, None
        idx = self._env_index(data.joint_pos, env_id)
        names = list(getattr(data, "joint_names", []) or [])
        vals = [float(v) for v in data.joint_pos[idx].detach().cpu().tolist()]
        return vals, names or None

    def _mass_view(self, asset: Any):
        view = getattr(asset, "root_physx_view", None)
        if view is None or not hasattr(view, "get_masses"):
            raise RuntimeError("asset has no root_physx_view.get_masses()")
        return view

    def _read_mass(self, asset: Any, env_id: int) -> float | None:
        try:
            masses = self._mass_view(asset).get_masses()
        except Exception:
            return None
        idx = self._env_index(masses, env_id)
        return float(masses[idx].reshape(-1).sum().item())

    def refresh(self, env_id: int | None = None) -> dict[str, AssetSnapshot]:
        """Re-read every registered asset. Call once per physics step."""
        env_id = self.env_id if env_id is None else int(env_id)
        self._cache = {}
        for asset_id in self.knowledge_map:
            try:
                asset, scene_key = self.resolve_asset(asset_id, env_id)
                pos, quat = self._read_pose(asset, env_id)
                lin_vel, ang_vel = self._read_vel(asset, env_id)
                joint_pos, joint_names = self._read_joints(asset, env_id)
                self._cache[asset_id] = AssetSnapshot(
                    asset_id=asset_id,
                    scene_key=scene_key,
                    pos=pos,
                    quat_wxyz=quat,
                    lin_vel=lin_vel,
                    ang_vel=ang_vel,
                    joint_pos=joint_pos,
                    joint_names=joint_names,
                    mass=self._read_mass(asset, env_id),
                )
            except Exception:
                continue
        self._refresh_ee(env_id)
        return self._cache

    def _body_pos(self, robot: Any, link_names: tuple[str, ...], env_id: int) -> tuple[float, float, float] | None:
        names = []
        if hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            names = list(robot.data.body_names)
        elif hasattr(robot, "body_names"):
            names = list(robot.body_names)
        bid = None
        for want in link_names:
            for i, name in enumerate(names):
                short = str(name).split("/")[-1]
                if short == want or str(name) == want:
                    bid = i
                    break
            if bid is None:
                try:
                    ids, _ = robot.find_bodies(want)
                    if ids is not None and len(ids) > 0:
                        bid = int(ids[0])
                except Exception:
                    bid = None
            if bid is not None:
                break
        if bid is None or not hasattr(robot, "data"):
            return None
        idx = self._env_index(robot.data.body_state_w, env_id)
        p = robot.data.body_state_w[idx, bid, 0:3]
        return (float(p[0].item()), float(p[1].item()), float(p[2].item()))

    def _refresh_ee(self, env_id: int) -> None:
        self._ee_cache = {}
        for arm_name, links in (
            ("robot_left", ("panda_hand", "fl_link6")),
            ("robot_right", ("panda_hand", "fr_link6")),
        ):
            try:
                robot = self.scene[arm_name]
            except Exception:
                continue
            pos = self._body_pos(robot, links, env_id)
            if pos is not None:
                self._ee_cache[arm_name] = pos

    def get(self, asset_id: str, refresh: bool = False) -> AssetSnapshot:
        if refresh or asset_id not in self._cache:
            self.refresh()
        if asset_id not in self._cache:
            raise KeyError(f"no snapshot for asset_id={asset_id}")
        return self._cache[asset_id]

    def get_all(self, refresh: bool = False) -> dict[str, AssetSnapshot]:
        if refresh or not self._cache:
            self.refresh()
        return dict(self._cache)

    def ee_pos(self, arm_name: str) -> tuple[float, float, float] | None:
        return self._ee_cache.get(arm_name)

    def table_z(self, fallback: float = 0.30) -> float:
        if self.table_asset_id and self.table_asset_id in self._cache:
            return float(self._cache[self.table_asset_id].pos[2])
        zs = [s.pos[2] for aid, s in self._cache.items() if "table" not in aid]
        if zs:
            return float(min(zs))
        return float(fallback)

    def _write_root_pose(self, asset: Any, env_id: int, pos: list[float], quat: list[float] | None) -> None:
        data = asset.data
        if hasattr(data, "root_state_w"):
            state = data.root_state_w.clone()
            idx = self._env_index(state, env_id)
            state[idx, 0:3] = _tensor_on(state, pos, (3,))
            if quat is not None:
                state[idx, 3:7] = _tensor_on(state, quat, (4,))
            asset.write_root_pose_to_sim(state[:, 0:7])
            zeros = torch.zeros_like(state[:, 7:13]) if state.shape[-1] >= 13 else torch.zeros(
                (state.shape[0], 6), device=state.device, dtype=state.dtype
            )
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(zeros)
            return
        if hasattr(data, "root_pos_w"):
            n = int(data.root_pos_w.shape[0])
            pose = torch.zeros((n, 7), device=data.root_pos_w.device, dtype=data.root_pos_w.dtype)
            pose[:, 0:3] = data.root_pos_w
            if hasattr(data, "root_quat_w"):
                pose[:, 3:7] = data.root_quat_w
            idx = min(env_id, n - 1)
            pose[idx, 0:3] = _tensor_on(pose, pos, (3,))
            if quat is not None:
                pose[idx, 3:7] = _tensor_on(pose, quat, (4,))
            asset.write_root_pose_to_sim(pose)
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=pose.device, dtype=pose.dtype))
            return
        raise RuntimeError("cannot write root pose: unsupported asset data layout")

    def _write_root_vel(self, asset: Any, env_id: int, lin_vel: list[float], ang_vel: list[float]) -> None:
        if not hasattr(asset, "write_root_velocity_to_sim"):
            raise RuntimeError("asset does not support write_root_velocity_to_sim")
        data = asset.data
        if hasattr(data, "root_state_w"):
            n = int(data.root_state_w.shape[0])
            device, dtype = data.root_state_w.device, data.root_state_w.dtype
        elif hasattr(data, "root_lin_vel_w"):
            n = int(data.root_lin_vel_w.shape[0])
            device, dtype = data.root_lin_vel_w.device, data.root_lin_vel_w.dtype
        else:
            n, device, dtype = 1, torch.device("cuda"), torch.float32
        vel = torch.zeros((n, 6), device=device, dtype=dtype)
        idx = min(env_id, n - 1)
        vel[idx, 0:3] = torch.tensor(lin_vel, device=device, dtype=dtype)
        vel[idx, 3:6] = torch.tensor(ang_vel, device=device, dtype=dtype)
        asset.write_root_velocity_to_sim(vel)

    def _write_joints(self, asset: Any, env_id: int, joint_pos: list[float], joint_vel: list[float] | None) -> None:
        if not hasattr(asset, "write_joint_state_to_sim"):
            raise RuntimeError("asset does not support write_joint_state_to_sim")
        data = asset.data
        cur = data.joint_pos
        idx = self._env_index(cur, env_id)
        pos = cur.clone()
        n = min(pos.shape[-1], len(joint_pos))
        pos[idx, :n] = torch.tensor(joint_pos[:n], device=pos.device, dtype=pos.dtype)
        if hasattr(data, "joint_vel") and data.joint_vel is not None:
            vel = data.joint_vel.clone()
        else:
            vel = torch.zeros_like(pos)
        if joint_vel is not None:
            m = min(vel.shape[-1], len(joint_vel))
            vel[idx, :m] = torch.tensor(joint_vel[:m], device=vel.device, dtype=vel.dtype)
        else:
            vel[idx].zero_()
        asset.write_joint_state_to_sim(pos, vel)

    def set_pose(
        self,
        asset_id: str,
        pos: Iterable[float],
        quat_wxyz: Iterable[float] | None = None,
        env_id: int | None = None,
        zero_velocity: bool = True,
    ) -> None:
        """Immediately teleport an asset. Quaternion is wxyz."""
        env_id = self.env_id if env_id is None else int(env_id)
        asset, _ = self.resolve_asset(asset_id, env_id)
        pos_l = _as_float_list(pos, 3)
        quat_l = _as_float_list(quat_wxyz, 4) if quat_wxyz is not None else None
        if quat_l is None:
            _, cur_q = self._read_pose(asset, env_id)
            quat_l = list(cur_q)
        self._write_root_pose(asset, env_id, pos_l, quat_l)
        if zero_velocity:
            self._write_root_vel(asset, env_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.refresh(env_id)

    def set_velocity(
        self,
        asset_id: str,
        lin_vel: Iterable[float] = (0.0, 0.0, 0.0),
        ang_vel: Iterable[float] = (0.0, 0.0, 0.0),
        env_id: int | None = None,
    ) -> None:
        env_id = self.env_id if env_id is None else int(env_id)
        asset, _ = self.resolve_asset(asset_id, env_id)
        self._write_root_vel(asset, env_id, _as_float_list(lin_vel, 3), _as_float_list(ang_vel, 3))

    def set_joints(
        self,
        asset_id: str,
        joint_pos: Iterable[float],
        joint_vel: Iterable[float] | None = None,
        env_id: int | None = None,
    ) -> None:
        env_id = self.env_id if env_id is None else int(env_id)
        asset, _ = self.resolve_asset(asset_id, env_id)
        jp = [float(v) for v in joint_pos]
        jv = None if joint_vel is None else [float(v) for v in joint_vel]
        self._write_joints(asset, env_id, jp, jv)
        self.refresh(env_id)

    def set_state(
        self,
        asset_id: str,
        pos: Iterable[float] | None = None,
        quat_wxyz: Iterable[float] | None = None,
        lin_vel: Iterable[float] | None = None,
        ang_vel: Iterable[float] | None = None,
        joint_pos: Iterable[float] | None = None,
        joint_vel: Iterable[float] | None = None,
        mass: float | None = None,
        env_id: int | None = None,
    ) -> None:
        """Partial update of any combination of pose / velocity / joints / mass."""
        env_id = self.env_id if env_id is None else int(env_id)
        if pos is not None or quat_wxyz is not None:
            if pos is None:
                snap = self.get(asset_id, refresh=True)
                pos = snap.pos
            self.set_pose(asset_id, pos=pos, quat_wxyz=quat_wxyz, env_id=env_id, zero_velocity=False)
        if lin_vel is not None or ang_vel is not None:
            self.set_velocity(
                asset_id,
                lin_vel=lin_vel if lin_vel is not None else (0.0, 0.0, 0.0),
                ang_vel=ang_vel if ang_vel is not None else (0.0, 0.0, 0.0),
                env_id=env_id,
            )
        if joint_pos is not None:
            self.set_joints(asset_id, joint_pos=joint_pos, joint_vel=joint_vel, env_id=env_id)
        if mass is not None:
            self.set_mass(asset_id, mass=float(mass), env_id=env_id)

    def get_mass(self, asset_id: str, env_id: int | None = None) -> float:
        env_id = self.env_id if env_id is None else int(env_id)
        asset, _ = self.resolve_asset(asset_id, env_id)
        mass = self._read_mass(asset, env_id)
        if mass is None:
            raise RuntimeError(f"cannot read mass for {asset_id}")
        return mass

    def set_mass(
        self,
        asset_id: str,
        mass: float,
        env_id: int | None = None,
        scale_inertia: bool = True,
    ) -> None:
        """Change an asset's mass during simulation. Safe to call while it is being grasped.

        Inertia is scaled by the same ratio so the object does not become physically inconsistent.
        """
        env_id = self.env_id if env_id is None else int(env_id)
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}")
        asset, _ = self.resolve_asset(asset_id, env_id)
        view = self._mass_view(asset)
        masses = view.get_masses().clone()
        idx = self._env_index(masses, env_id)
        cur = float(masses[idx].reshape(-1).sum().item())
        if cur <= 1e-8:
            raise RuntimeError(f"{asset_id} current mass is {cur}, cannot rescale")
        ratio = float(mass) / cur
        masses[idx] = masses[idx] * ratio
        indices = torch.tensor([idx], dtype=torch.int, device="cpu")
        view.set_masses(masses, indices)
        if scale_inertia and hasattr(view, "get_inertias") and hasattr(view, "set_inertias"):
            inertias = view.get_inertias().clone()
            inertias[idx] = inertias[idx] * ratio
            view.set_inertias(inertias, indices)
        print(f"[WorldState] set_mass {asset_id}: {cur:.4f} -> {mass:.4f} kg")
        self.refresh(env_id)

    def queue_set_mass(self, asset_id: str, mass: float, env_id: int | None = None, scale_inertia: bool = True) -> None:
        self._pending.append(
            _PendingWrite(
                kind="mass",
                asset_id=asset_id,
                env_id=self.env_id if env_id is None else int(env_id),
                payload={"mass": float(mass), "scale_inertia": bool(scale_inertia)},
            )
        )

    def queue_set_pose(self, asset_id: str, pos: Iterable[float], quat_wxyz: Iterable[float] | None = None, env_id: int | None = None) -> None:
        self._pending.append(
            _PendingWrite(
                kind="pose",
                asset_id=asset_id,
                env_id=self.env_id if env_id is None else int(env_id),
                payload={"pos": list(pos), "quat_wxyz": None if quat_wxyz is None else list(quat_wxyz)},
            )
        )

    def queue_set_state(self, asset_id: str, env_id: int | None = None, **kwargs: Any) -> None:
        self._pending.append(
            _PendingWrite(
                kind="state",
                asset_id=asset_id,
                env_id=self.env_id if env_id is None else int(env_id),
                payload=dict(kwargs),
            )
        )

    def apply_pending(self) -> int:
        """Apply queued mutations. Safe to call every tick."""
        n = len(self._pending)
        pending, self._pending = self._pending, []
        for cmd in pending:
            if cmd.kind == "pose":
                self.set_pose(cmd.asset_id, pos=cmd.payload["pos"], quat_wxyz=cmd.payload.get("quat_wxyz"), env_id=cmd.env_id)
            elif cmd.kind == "mass":
                self.set_mass(
                    cmd.asset_id,
                    mass=cmd.payload["mass"],
                    env_id=cmd.env_id,
                    scale_inertia=cmd.payload.get("scale_inertia", True),
                )
            else:
                self.set_state(cmd.asset_id, env_id=cmd.env_id, **cmd.payload)
        return n
