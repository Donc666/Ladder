"""Compile one atomic task using each asset's import-time pose."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.import_pose_scene import wrap_scene_with_import_poses


def _is_simultaneous_pack(waypoints: list[dict]) -> bool:
    return any(str(w.get("executor", "")).startswith("simultaneous") for w in waypoints)


def _simultaneous_arms(waypoints: list[dict]) -> tuple[bool, bool]:
    left = False
    right = False
    for wp in waypoints:
        ex = str(wp.get("executor", ""))
        if "left" in ex:
            left = True
        if "right" in ex:
            right = True
    return left, right


def _find_can_sync_indices(atomic_plan: list[dict]) -> tuple[int | None, int | None]:
    """Plan indices of can_1/left and can_2/right, or (None, None)."""
    left_i = None
    right_i = None
    for i, step in enumerate(atomic_plan):
        asset = str(step.get("asset") or "")
        atom = str(step.get("atom") or "").lower()
        if asset == "can_1" and "left" in atom:
            left_i = i
        if asset == "can_2" and "right" in atom:
            right_i = i
    return left_i, right_i


def _annotate(waypoints: list[dict], asset_id: str, atom: str, atom_index: int) -> list[dict]:
    out = []
    for wp in waypoints:
        item = dict(wp)
        item.setdefault("grasp_asset", asset_id)
        item.setdefault("atom", atom)
        item.setdefault("atom_index", atom_index)
        out.append(item)
    return out


@dataclass
class CompiledAtom:
    start_index: int
    end_index: int
    asset_id: str
    atom: str
    waypoints: list[dict]
    simultaneous: bool
    grouped_assets: list[str] = field(default_factory=list)

    @property
    def grasp_assets(self) -> list[str]:
        ids = []
        for wp in self.waypoints:
            aid = wp.get("grasp_asset")
            if aid and aid not in ids:
                ids.append(aid)
        return ids or [self.asset_id]


class AtomCompiler:
    def __init__(
        self,
        knowledge_map: dict[str, Any],
        id_to_type: dict[str, str],
        dispatch: dict[tuple[str, str], Any],
        allowed_atoms: dict[str, set[str]],
        env_id: int = 0,
        import_pose_lookup: dict | None = None,
        live_pose_enabled: bool = False,
    ):
        self.knowledge_map = knowledge_map
        self.id_to_type = id_to_type
        self.dispatch = dispatch
        self.allowed_atoms = allowed_atoms
        self.env_id = int(env_id)
        self.import_pose_lookup = import_pose_lookup
        self.live_pose_enabled = bool(live_pose_enabled)

    def compile_step(self, step: dict, scene: Any, atom_index: int, use_live_poses: bool = False) -> list[dict]:
        if "asset" not in step or "atom" not in step:
            raise ValueError(f"atomic step missing fields: {step}")
        asset_id = step["asset"]
        atom = step["atom"]
        params = dict(step.get("params") or {})
        params.setdefault("_env_id", self.env_id)
        if asset_id not in self.knowledge_map:
            raise ValueError(f"unknown asset id: {asset_id}")
        a_type = self.id_to_type[asset_id]
        if atom not in self.allowed_atoms.get(a_type, set()):
            raise ValueError(f"atom '{atom}' not allowed for type '{a_type}'")
        key = (a_type, atom)
        if key not in self.dispatch:
            raise ValueError(f"no dispatcher for {key}")
        obj = self.knowledge_map[asset_id]
        # Kitchen collisions shove objects after spawn. These grasps must use
        # the live sim pose, not the import-time wrap.
        live_types = ("bowl", "bread", "skillet", "frenchfries", "hamburg", "bell")
        if self.live_pose_enabled and (use_live_poses or a_type in live_types):
            planning_scene = scene
        else:
            planning_scene = wrap_scene_with_import_poses(
                scene, self.import_pose_lookup, knowledge_map=self.knowledge_map
            )
        seg = self.dispatch[key](obj, planning_scene, params)
        if seg is None:
            return []
        if not isinstance(seg, list):
            seg = [seg]
        return _annotate(seg, asset_id=asset_id, atom=atom, atom_index=atom_index)

    def _compile_independent_until_cans(
        self,
        atomic_plan: list[dict],
        start_index: int,
        scene: Any,
        *,
        use_live_poses: bool,
        skip: set[int],
        left_sync: int,
        right_sync: int,
    ) -> CompiledAtom:
        """Pack every left atom through can_1 and every right atom through can_2.

        Arms are concatenated independently (no 1:1 pairing), so left can fly
        through hamburg/can_1 while right flies through bowls/can_2.
        """
        pack: list[dict] = []
        grouped: list[str] = []
        left_ids: list[str] = []
        right_ids: list[str] = []
        end_index = max(left_sync, right_sync) + 1
        first = atomic_plan[start_index]
        for i in range(start_index, end_index):
            if i in skip:
                continue
            step = atomic_plan[i]
            seg = self.compile_step(step, scene, i, use_live_poses=use_live_poses)
            if not seg:
                continue
            if not _is_simultaneous_pack(seg):
                continue
            has_left, has_right = _simultaneous_arms(seg)
            take = False
            aid = str(step.get("asset") or "")
            if has_left and i <= left_sync:
                take = True
                if aid:
                    left_ids.append(aid)
            if has_right and i <= right_sync:
                take = True
                if aid:
                    right_ids.append(aid)
            if take:
                pack.extend(seg)
                if aid:
                    grouped.append(aid)
        print(
            f"[AtomCompiler] independent until cans "
            f"L={','.join(left_ids) or '-'} R={','.join(right_ids) or '-'}"
        )
        return CompiledAtom(
            start_index=start_index,
            end_index=end_index,
            asset_id=str(first.get("asset") or ""),
            atom=str(first.get("atom") or ""),
            waypoints=pack,
            simultaneous=True,
            grouped_assets=grouped,
        )

    def compile_group(
        self,
        atomic_plan: list[dict],
        start_index: int,
        scene: Any,
        use_live_poses: bool = False,
        skip_indices: set[int] | None = None,
    ) -> CompiledAtom:
        """
        Compile the atom at `start_index`. Consecutive simultaneous atoms are
        merged until both arms have work, then the next pair starts after this
        group so a fallen bottle can be stood up between pairs.
        """
        skip = skip_indices or set()
        while start_index < len(atomic_plan) and start_index in skip:
            start_index += 1
        if start_index >= len(atomic_plan):
            return CompiledAtom(
                start_index=start_index,
                end_index=start_index,
                asset_id="",
                atom="",
                waypoints=[],
                simultaneous=False,
                grouped_assets=[],
            )
        left_sync, right_sync = _find_can_sync_indices(atomic_plan)
        if (
            left_sync is not None
            and right_sync is not None
            and start_index <= min(left_sync, right_sync)
        ):
            return self._compile_independent_until_cans(
                atomic_plan,
                start_index,
                scene,
                use_live_poses=use_live_poses,
                skip=skip,
                left_sync=left_sync,
                right_sync=right_sync,
            )
        first = self.compile_step(
            atomic_plan[start_index], scene, start_index, use_live_poses=use_live_poses
        )
        asset_id = atomic_plan[start_index]["asset"]
        atom = atomic_plan[start_index]["atom"]
        grouped = [asset_id]
        end = start_index + 1
        if _is_simultaneous_pack(first):
            pack = list(first)
            has_left, has_right = _simultaneous_arms(pack)
            while end < len(atomic_plan):
                if has_left and has_right:
                    break
                if end in skip:
                    end += 1
                    continue
                nxt = self.compile_step(
                    atomic_plan[end], scene, end, use_live_poses=use_live_poses
                )
                if not _is_simultaneous_pack(nxt):
                    break
                nxt_left, nxt_right = _simultaneous_arms(nxt)
                # Keep one atom per arm so can dual-pick is not merged into fries/hamburg.
                if (nxt_left and has_left) or (nxt_right and has_right):
                    break
                pack.extend(nxt)
                grouped.append(atomic_plan[end]["asset"])
                end += 1
                has_left = has_left or nxt_left
                has_right = has_right or nxt_right
            return CompiledAtom(
                start_index=start_index,
                end_index=end,
                asset_id=asset_id,
                atom=atom,
                waypoints=pack,
                simultaneous=True,
                grouped_assets=grouped,
            )
        return CompiledAtom(
            start_index=start_index,
            end_index=end,
            asset_id=asset_id,
            atom=atom,
            waypoints=first,
            simultaneous=False,
            grouped_assets=grouped,
        )
