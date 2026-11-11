"""Serve import-time asset poses to atomic-task planners.

Kitchen collisions can shove objects after spawn. Drop detection still reads the
live simulation; waypoint compilation should keep using the pose from import2sim.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


ImportPose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
PoseLookup = dict[str, ImportPose]


def _as_pos(pos) -> tuple[float, float, float]:
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _as_quat(rot) -> tuple[float, float, float, float]:
    return (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))


def attach_import_poses(
    knowledge_map: Mapping[str, Any],
    assets_cfg_2d: list[list[Any]],
    workspace_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PoseLookup:
    """Cache each AssetCfg pose (plus workspace offset) onto knowledge objects."""
    ox, oy, oz = workspace_offset
    lookup: PoseLookup = {}
    for env_id, env_assets in enumerate(assets_cfg_2d):
        for cfg in env_assets:
            pos = (float(cfg.pos[0]) + ox, float(cfg.pos[1]) + oy, float(cfg.pos[2]) + oz)
            quat = _as_quat(cfg.rot)
            pose = (pos, quat)
            lookup[str(cfg.name)] = pose
            lookup[f"{cfg.name}__env{env_id}"] = pose

    for obj in knowledge_map.values():
        scene_key = getattr(obj, "scene_key", None)
        poses: list[ImportPose] = []
        if scene_key:
            for env_id in range(len(assets_cfg_2d)):
                key = f"{scene_key}__env{env_id}"
                if key in lookup:
                    poses.append(lookup[key])
                elif scene_key in lookup:
                    poses.append(lookup[scene_key])
        if not poses:
            spawn = getattr(obj, "spawn_pos", None)
            quat = getattr(obj, "import_quat_wxyz", None) or getattr(obj, "q_obj_world_wxyz", None)
            if spawn is not None and quat is not None:
                poses = [(_as_pos(spawn), _as_quat(quat))]
        obj.import_poses = poses
        if poses:
            obj.spawn_pos = poses[0][0]
            obj.import_quat_wxyz = poses[0][1]
    return lookup


def lookup_from_knowledge_map(knowledge_map: Mapping[str, Any]) -> PoseLookup:
    lookup: PoseLookup = {}
    for obj in knowledge_map.values():
        scene_key = getattr(obj, "scene_key", None)
        poses = getattr(obj, "import_poses", None) or []
        if not scene_key or not poses:
            continue
        lookup[scene_key] = poses[0]
        for env_id, pose in enumerate(poses):
            lookup[f"{scene_key}__env{env_id}"] = pose
    return lookup


class _ImportPoseData:
    def __init__(self, data: Any, pos: tuple[float, float, float], quat: tuple[float, float, float, float]):
        object.__setattr__(self, "_data", data)
        ref = data.root_pos_w if hasattr(data, "root_pos_w") else data.root_state_w
        n = int(ref.shape[0])
        pos_t = torch.tensor(pos, device=ref.device, dtype=ref.dtype).reshape(1, 3).expand(n, 3).clone()
        quat_t = torch.tensor(quat, device=ref.device, dtype=ref.dtype).reshape(1, 4).expand(n, 4).clone()
        object.__setattr__(self, "_root_pos_w", pos_t)
        object.__setattr__(self, "_root_quat_w", quat_t)

    @property
    def root_pos_w(self):
        return self._root_pos_w

    @property
    def root_quat_w(self):
        return self._root_quat_w

    @property
    def root_state_w(self):
        if not hasattr(self._data, "root_state_w"):
            raise AttributeError("root_state_w")
        state = self._data.root_state_w.clone()
        state[:, 0:3] = self._root_pos_w
        state[:, 3:7] = self._root_quat_w
        return state

    def __getattr__(self, name: str):
        return getattr(self._data, name)


class _ImportPoseAsset:
    def __init__(self, asset: Any, pos: tuple[float, float, float], quat: tuple[float, float, float, float]):
        object.__setattr__(self, "_asset", asset)
        object.__setattr__(self, "_pos", pos)
        object.__setattr__(self, "_quat", quat)

    @property
    def data(self):
        return _ImportPoseData(self._asset.data, self._pos, self._quat)

    def __getattr__(self, name: str):
        return getattr(self._asset, name)


class ImportPoseScene:
    """Scene proxy that reports import-time root pose during atom compilation."""

    def __init__(self, scene: Any, lookup: PoseLookup):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_lookup", lookup)

    def __getitem__(self, key):
        asset = self._scene[key]
        pose = self._lookup.get(key)
        if pose is None:
            return asset
        return _ImportPoseAsset(asset, pose[0], pose[1])

    def __getattr__(self, name: str):
        return getattr(self._scene, name)


def wrap_scene_with_import_poses(scene: Any, lookup: PoseLookup | None, knowledge_map: Mapping[str, Any] | None = None):
    if lookup is None and knowledge_map is not None:
        lookup = lookup_from_knowledge_map(knowledge_map)
    if not lookup:
        return scene
    if isinstance(scene, ImportPoseScene):
        return scene
    return ImportPoseScene(scene, lookup)
