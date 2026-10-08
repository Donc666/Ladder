from __future__ import annotations

from typing import Any, Callable


CheckerFn = Callable[[Any, dict[str, Any], int, dict[str, Any]], tuple[bool, dict[str, float]]]


def _resolve_scene_asset(scene: Any, obj: Any, env_id: int):
    """Resolve an asset instance in scene by trying common env-suffixed keys."""
    base = getattr(obj, "scene_key", None)
    if not isinstance(base, str) or not base:
        raise KeyError(f"invalid scene_key on object: {obj}")

    candidates = [
        f"{base}__env{env_id}",
        base,
        f"{base}__env{env_id}__env{env_id}",
    ]
    for key in candidates:
        try:
            return scene[key], key
        except Exception:
            continue
    raise KeyError(f"asset not found in scene, tried keys={candidates}")


def _get_root_pos_xyz(scene: Any, obj: Any, env_id: int) -> tuple[float, float, float]:
    asset, _ = _resolve_scene_asset(scene, obj, env_id)
    p = asset.data.root_pos_w[env_id]
    return float(p[0].item()), float(p[1].item()), float(p[2].item())


def _capture_initial_positions(scene: Any, knowledge_map: dict[str, Any], num_envs: int) -> dict[str, list[tuple[float, float, float] | None]]:
    """Capture initial root positions for every named asset and environment."""
    out: dict[str, list[tuple[float, float, float] | None]] = {}
    for asset_id, obj in knowledge_map.items():
        per_env: list[tuple[float, float, float] | None] = []
        for env_id in range(num_envs):
            try:
                per_env.append(_get_root_pos_xyz(scene, obj, env_id))
            except Exception:
                per_env.append(None)
        out[asset_id] = per_env
    return out


def _check_beat_block_hammer(scene: Any, knowledge_map: dict[str, Any], env_id: int, ctx: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    """Success if block moves enough in XY from its initial position."""
    block_obj = knowledge_map.get("block_1")
    if block_obj is None:
        return False, {"missing_block_1": 1.0}

    x, y, z = _get_root_pos_xyz(scene, block_obj, env_id)
    init_pos = ctx.get("initial_positions", {}).get("block_1", [None])[env_id]
    if init_pos is None:
        return False, {"missing_initial_block_1": 1.0}

    dx = x - float(init_pos[0])
    dy = y - float(init_pos[1])
    disp_xy = (dx * dx + dy * dy) ** 0.5
    # Tune this threshold for your setup if needed.
    success = disp_xy > 0.08
    return success, {"disp_xy": disp_xy, "block_z": z}


def _check_turn_switch(scene: Any, knowledge_map: dict[str, Any], env_id: int, ctx: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    """Success if switch articulation joint moved enough from initial value."""
    switch_obj = knowledge_map.get("switch_1")
    if switch_obj is None:
        return False, {"missing_switch_1": 1.0}

    switch_asset, _ = _resolve_scene_asset(scene, switch_obj, env_id)
    if not hasattr(switch_asset, "data") or not hasattr(switch_asset.data, "joint_pos"):
        return False, {"switch_no_joint_pos": 1.0}

    try:
        curr = float(switch_asset.data.joint_pos[env_id, 0].item())
    except Exception:
        return False, {"switch_joint_read_error": 1.0}

    init_joint = ctx.get("initial_joint_pos", {}).get("switch_1", [None])[env_id]
    if init_joint is None:
        return False, {"missing_initial_switch_joint": 1.0}

    moved = abs(curr - float(init_joint))
    success = moved > 0.05
    return success, {"switch_joint_delta": moved, "switch_joint_curr": curr}


SUCCESS_CHECKERS: dict[str, CheckerFn] = {
    "beat_block_hammer": _check_beat_block_hammer,
    "turn_switch": _check_turn_switch,
}


def build_success_context(task_name: str, scene: Any, knowledge_map: dict[str, Any], num_envs: int) -> dict[str, Any]:
    """Build reusable context for success checking across steps/frames."""
    ctx: dict[str, Any] = {"task_name": task_name, "num_envs": num_envs}
    ctx["initial_positions"] = _capture_initial_positions(scene, knowledge_map, num_envs)

    # Optional per-task cached signals
    if task_name == "turn_switch" and "switch_1" in knowledge_map:
        switch_obj = knowledge_map["switch_1"]
        initial_joint: list[float | None] = []
        for env_id in range(num_envs):
            try:
                switch_asset, _ = _resolve_scene_asset(scene, switch_obj, env_id)
                initial_joint.append(float(switch_asset.data.joint_pos[env_id, 0].item()))
            except Exception:
                initial_joint.append(None)
        ctx["initial_joint_pos"] = {"switch_1": initial_joint}

    return ctx


def evaluate_task_success(
    task_name: str,
    scene: Any,
    knowledge_map: dict[str, Any],
    env_id: int,
    ctx: dict[str, Any],
) -> tuple[bool | None, dict[str, float]]:
    """
    Evaluate success for one environment.
    Returns:
      - success=True/False when checker exists
      - success=None when task checker is not registered yet
    """
    checker = SUCCESS_CHECKERS.get(task_name)
    if checker is None:
        return None, {}
    try:
        return checker(scene, knowledge_map, env_id, ctx)
    except Exception:
        return False, {"checker_exception": 1.0}
