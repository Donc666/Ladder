import re
from typing import Any, Callable

import torch


def _log_check(task: str, env_id: int, **metrics: Any) -> None:
    parts = [f"[CHECK][{task}] env={env_id}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))
    if metrics.get("phase") == "summary":
        return
    try:
        from collect_data_utils import record_check_metrics

        record_check_metrics(metrics=metrics, task_name=task)
    except Exception:
        pass


def _dedup_keep_order(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for it in items:
        if not it or it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _resolve_scene_asset(scene: Any, knowledge_map: dict[str, object], asset_id: str, env_id: int):
    """Resolve scene asset instance by asset id and env id with tolerant naming."""
    obj = knowledge_map.get(asset_id)
    if obj is None:
        raise KeyError(f"asset_id not found in knowledge_map: {asset_id}")

    scene_key = getattr(obj, "scene_key", "")
    base_key = re.sub(r"__env\d+$", "", scene_key) if isinstance(scene_key, str) else ""
    candidates = _dedup_keep_order(
        [
            f"{scene_key}__env{env_id}" if scene_key else "",
            scene_key if scene_key else "",
            f"{base_key}__env{env_id}" if base_key else "",
            base_key if base_key else "",
            f"my_{asset_id}__env{env_id}",
            f"my_{asset_id}",
            f"{asset_id}__env{env_id}",
            asset_id,
        ]
    )
    for name in candidates:
        try:
            return scene[name], name
        except Exception:
            continue
    raise KeyError(f"cannot resolve scene asset for asset_id={asset_id}, tried={candidates}")


def _get_asset_root_pos(scene: Any, knowledge_map: dict[str, object], asset_id: str, env_id: int) -> torch.Tensor:
    asset, _ = _resolve_scene_asset(scene, knowledge_map, asset_id, env_id)
    if not hasattr(asset, "data") or not hasattr(asset.data, "root_pos_w"):
        raise RuntimeError(f"asset {asset_id} has no root_pos_w")
    pos = asset.data.root_pos_w
    idx = min(env_id, int(pos.shape[0]) - 1)
    return pos[idx, 0:3].clone().detach()


def _get_asset_joint_pos(scene: Any, knowledge_map: dict[str, object], asset_id: str, env_id: int) -> torch.Tensor:
    asset, _ = _resolve_scene_asset(scene, knowledge_map, asset_id, env_id)
    if not hasattr(asset, "data") or not hasattr(asset.data, "joint_pos"):
        raise RuntimeError(f"asset {asset_id} has no joint_pos")
    joint_pos = asset.data.joint_pos
    if joint_pos is None or joint_pos.numel() == 0:
        raise RuntimeError(f"asset {asset_id} has empty joint_pos")
    idx = min(env_id, int(joint_pos.shape[0]) - 1)
    return joint_pos[idx].clone().detach()


def _try_get_asset_root_pos(scene: Any, knowledge_map: dict[str, object], candidate_ids: list[str], env_id: int) -> torch.Tensor | None:
    for aid in candidate_ids:
        try:
            return _get_asset_root_pos(scene=scene, knowledge_map=knowledge_map, asset_id=aid, env_id=env_id)
        except Exception:
            continue
    return None


def _try_get_asset_joint_pos(scene: Any, knowledge_map: dict[str, object], candidate_ids: list[str], env_id: int) -> torch.Tensor | None:
    for aid in candidate_ids:
        try:
            return _get_asset_joint_pos(scene=scene, knowledge_map=knowledge_map, asset_id=aid, env_id=env_id)
        except Exception:
            continue
    return None


def _robot_body_names(robot: Any) -> list[str]:
    if hasattr(robot, "data") and hasattr(robot.data, "body_names"):
        return list(robot.data.body_names)
    if hasattr(robot, "body_names"):
        return list(robot.body_names)
    return []


def _resolve_robot(scene: Any, robot_key: str = "robot"):
    for key in (robot_key, "robot"):
        try:
            return scene[key]
        except Exception:
            continue
    return None


def _get_body_pos_by_name(robot: Any, link_name: str, env_id: int) -> torch.Tensor | None:
    if robot is None or not hasattr(robot, "data"):
        return None
    names = _robot_body_names(robot)
    bid = None
    for i, name in enumerate(names):
        short = str(name).split("/")[-1]
        if short == link_name or str(name) == link_name:
            bid = i
            break
    if bid is None:
        try:
            body_ids, _ = robot.find_bodies(link_name)
            if body_ids is None or len(body_ids) == 0:
                return None
            bid = int(body_ids[0])
        except Exception:
            return None
    idx = min(env_id, int(robot.data.body_state_w.shape[0]) - 1)
    return robot.data.body_state_w[idx, bid, 0:3].clone().detach()


def _get_robot_hand_pos(scene: Any, robot_key: str, env_id: int) -> torch.Tensor | None:
    robot = _resolve_robot(scene, robot_key)
    key = str(robot_key).lower()
    if "left" in key:
        links = ("fl_link6", "fr_link6")
    else:
        links = ("fr_link6", "fl_link6")
    for link in links:
        pos = _get_body_pos_by_name(robot, link, env_id)
        if pos is not None:
            return pos
    return None


def _finger_joint_ids(joint_names: list[str], robot_key: str) -> list[int]:
    """
    Resolve gripper joint indices for Franka (`*finger*`) and Aloha
    (`fl_joint7/8`, `fr_joint7/8`) on a shared dual-arm robot.
    """
    names = list(joint_names)
    finger_ids = [i for i, n in enumerate(names) if "finger" in n.lower()]
    if len(finger_ids) >= 2:
        return finger_ids

    key = str(robot_key).lower()
    if "left" in key:
        want = {"fl_joint7", "fl_joint8"}
    elif "right" in key:
        want = {"fr_joint7", "fr_joint8"}
    else:
        want = {"fl_joint7", "fl_joint8", "fr_joint7", "fr_joint8"}
    ids = [i for i, n in enumerate(names) if n in want]
    return ids


def _get_gripper_joint_vals(scene: Any, robot_key: str, env_id: int) -> torch.Tensor | None:
    robot = None
    for key in (robot_key, "robot"):
        try:
            robot = scene[key]
            break
        except Exception:
            continue
    if robot is None or not hasattr(robot, "data"):
        return None
    joint_names = list(robot.data.joint_names)
    finger_ids = _finger_joint_ids(joint_names, robot_key)
    if len(finger_ids) < 1:
        return None
    idx = min(env_id, int(robot.data.joint_pos.shape[0]) - 1)
    return robot.data.joint_pos[idx, finger_ids]


def _is_gripper_open(scene: Any, robot_key: str, env_id: int, open_thresh: float = 0.03) -> bool:
    vals = _get_gripper_joint_vals(scene, robot_key, env_id)
    if vals is None:
        return False
    return bool(float(vals.mean().item()) >= open_thresh)


def _is_gripper_closed(scene: Any, robot_key: str, env_id: int, close_thresh: float = 0.015) -> bool:
    vals = _get_gripper_joint_vals(scene, robot_key, env_id)
    if vals is None:
        return False
    return bool(float(vals.mean().item()) <= close_thresh)


def _contact_proxy_by_distance(
    scene: Any,
    knowledge_map: dict[str, object],
    asset_a: str,
    asset_b: str,
    env_id: int,
    max_center_distance: float = 0.10,
) -> bool:
    """
    Conservative fallback for 'in contact' when no contact API is exposed:
    use center distance as proxy.
    """
    pa = _get_asset_root_pos(scene=scene, knowledge_map=knowledge_map, asset_id=asset_a, env_id=env_id).detach().cpu()
    pb = _get_asset_root_pos(scene=scene, knowledge_map=knowledge_map, asset_id=asset_b, env_id=env_id).detach().cpu()
    center_dist = float(torch.linalg.norm(pa - pb).item())
    return center_dist <= max_center_distance


def _both_grippers_open(scene: Any, env_id: int) -> bool:
    return _is_gripper_open(scene, "robot_left", env_id) and _is_gripper_open(scene, "robot_right", env_id)


def _both_grippers_closed(scene: Any, env_id: int) -> bool:
    return _is_gripper_closed(scene, "robot_left", env_id) and _is_gripper_closed(scene, "robot_right", env_id)


def _any_gripper_open(scene: Any, env_id: int) -> bool:
    return _is_gripper_open(scene, "robot_left", env_id) or _is_gripper_open(scene, "robot_right", env_id)


def _get_asset_root_quat(scene: Any, knowledge_map: dict[str, object], asset_id: str, env_id: int) -> torch.Tensor:
    asset, _ = _resolve_scene_asset(scene, knowledge_map, asset_id, env_id)
    if not hasattr(asset, "data") or not hasattr(asset.data, "root_quat_w"):
        raise RuntimeError(f"asset {asset_id} has no root_quat_w")
    quat = asset.data.root_quat_w
    idx = min(env_id, int(quat.shape[0]) - 1)
    return quat[idx, 0:4].clone().detach()


def _quat_close_signed_equiv(cur: torch.Tensor, target: torch.Tensor, eps: float) -> bool:
    # q and -q represent the same orientation.
    e1 = torch.max(torch.abs(cur - target))
    e2 = torch.max(torch.abs(cur + target))
    return bool(min(float(e1.item()), float(e2.item())) <= eps)


def _infer_non_table_assets(knowledge_map: dict[str, object]) -> list[str]:
    out = []
    for k in knowledge_map.keys():
        if "table" in k:
            continue
        out.append(k)
    return out


def _check_adjust_bottle(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    pos = _try_get_asset_root_pos(scene, knowledge_map, ["bottle_1"], env_id)
    if pos is None:
        return False
    init_pos = ctx.get("initial_root_pos", {}).get(("bottle_1", env_id), None)
    if init_pos is None:
        _log_check("adjust_bottle", env_id, reason="missing_initial_bottle", success=False)
        return False
    delta = (pos - init_pos).detach().cpu()
    disp_xy = float(torch.linalg.norm(delta[0:2]).item())
    disp_3d = float(torch.linalg.norm(delta).item())
    # Local scene uses different world-height baseline than RoboTwin.
    # Use relative displacement criterion for robust success judgment.
    success = disp_xy > 0.02
    _log_check(
        "adjust_bottle",
        env_id,
        bottle_xyz=pos.detach().cpu().tolist(),
        init_xyz=init_pos.detach().cpu().tolist(),
        dxyz=delta.tolist(),
        disp_xy=f"{disp_xy:.4f}",
        disp_3d=f"{disp_3d:.4f}",
        threshold_xy="0.0200",
        success=success,
    )
    return success


def _check_blocks_ranking(scene: Any, knowledge_map: dict[str, object], env_id: int, ids: list[str], task_label: str) -> bool:
    p = [_try_get_asset_root_pos(scene, knowledge_map, [aid], env_id) for aid in ids]
    if any(v is None for v in p):
        _log_check(task_label, env_id, reason="missing_block", ids=ids, success=False)
        return False
    p0, p1, p2 = p
    d01 = torch.abs(p0[0:2] - p1[0:2]).detach().cpu()
    d12 = torch.abs(p1[0:2] - p2[0:2]).detach().cpu()
    near = bool((d01[0] < 0.13 and d01[1] < 0.03 and d12[0] < 0.13 and d12[1] < 0.03).item())
    x_ok = bool((p0[0] < p1[0]).item() and (p1[0] < p2[0]).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = near and x_ok and gripper_open
    _log_check(
        task_label,
        env_id,
        pos=[pp.detach().cpu().tolist() for pp in p],
        d01=d01.tolist(),
        d12=d12.tolist(),
        near=near,
        x_order=x_ok,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_blocks_ranking_rgb(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_blocks_ranking(
        scene,
        knowledge_map,
        env_id,
        ids=["block_red_1", "block_green_1", "block_blue_1"],
        task_label="blocks_ranking_RGB",
    )


def _check_blocks_ranking_size(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_blocks_ranking(
        scene,
        knowledge_map,
        env_id,
        ids=["block_small_1", "block_middle_1", "block_big_1"],
        task_label="blocks_ranking_size",
    )


def _check_dump_bin_bigbin(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    deskbin = _try_get_asset_root_pos(scene, knowledge_map, ["tabletrashbin_1", "dustbin_1"], env_id)
    g1 = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    g2 = _try_get_asset_root_pos(scene, knowledge_map, ["block_2"], env_id)
    g3 = _try_get_asset_root_pos(scene, knowledge_map, ["block_3"], env_id)
    if deskbin is None or g1 is None or g2 is None or g3 is None:
        return False
    deskbin_up = float(deskbin[2].item()) >= 1.0
    garbage_ok = all(0.13 <= float(g[2].item()) <= 0.25 for g in [g1, g2, g3])
    success = deskbin_up and garbage_ok
    _log_check(
        "dump_bin_bigbin",
        env_id,
        deskbin_xyz=deskbin.detach().cpu().tolist(),
        garbage_xyz=[g1.detach().cpu().tolist(), g2.detach().cpu().tolist(), g3.detach().cpu().tolist()],
        deskbin_up=deskbin_up,
        garbage_ok=garbage_ok,
        success=success,
    )
    return success


def _check_grab_roller(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    roller = _try_get_asset_root_pos(scene, knowledge_map, ["roller_1"], env_id)
    if roller is None:
        return False
    both_closed = _both_grippers_closed(scene, env_id)
    lifted = float(roller[2].item()) > 0.8
    success = both_closed and lifted
    _log_check("grab_roller", env_id, roller_xyz=roller.detach().cpu().tolist(), both_closed=both_closed, lifted=lifted, success=success)
    return success


def _check_handover_block(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    block = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    right_hand = _get_robot_hand_pos(scene, "robot_right", env_id)
    if block is None or right_hand is None:
        return False
    d = torch.abs(block - right_hand).detach().cpu()
    near = bool((d[0] < 0.03 and d[1] < 0.03 and d[2] < 0.04).item())
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    success = near and right_open
    _log_check("handover_block", env_id, block_xyz=block.detach().cpu().tolist(), right_hand_xyz=right_hand.detach().cpu().tolist(), delta=d.tolist(), right_open=right_open, success=success)
    return success


def _check_handover_bottle(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    bottle = _try_get_asset_root_pos(scene, knowledge_map, ["bottle_1"], env_id)
    right_hand = _get_robot_hand_pos(scene, "robot_right", env_id)
    if bottle is None or right_hand is None:
        return False
    d = torch.abs(bottle - right_hand).detach().cpu()
    near = bool((d[0] < 0.04 and d[1] < 0.04 and d[2] < 0.05).item())
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    success = near and right_open
    _log_check("handover_bottle", env_id, bottle_xyz=bottle.detach().cpu().tolist(), right_hand_xyz=right_hand.detach().cpu().tolist(), delta=d.tolist(), right_open=right_open, success=success)
    return success


def _check_handover_mic_or_hanging_mug(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    keys = set(knowledge_map.keys())
    if {"mug_1", "rack_1"}.issubset(keys):
        mug = _try_get_asset_root_pos(scene, knowledge_map, ["mug_1"], env_id)
        rack = _try_get_asset_root_pos(scene, knowledge_map, ["rack_1"], env_id)
        if mug is None or rack is None:
            return False
        dxy = torch.abs(mug[0:2] - rack[0:2]).detach().cpu()
        near_xy = bool((dxy[0] < 0.02 and dxy[1] < 0.02).item())
        z_ok = float(mug[2].item()) > 0.86
        right_open = _is_gripper_open(scene, "robot_right", env_id)
        success = near_xy and z_ok and right_open
        _log_check("hanging_mug", env_id, mug_xyz=mug.detach().cpu().tolist(), rack_xyz=rack.detach().cpu().tolist(), dxy=dxy.tolist(), z_ok=z_ok, right_open=right_open, success=success)
        return success
    mic = _try_get_asset_root_pos(scene, knowledge_map, ["microphone_1"], env_id)
    if mic is None:
        return False
    left_open = _is_gripper_open(scene, "robot_left", env_id)
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    left_closed = _is_gripper_closed(scene, "robot_left", env_id)
    right_closed = _is_gripper_closed(scene, "robot_right", env_id)
    one_side_holding = (left_closed and right_open) or (right_closed and left_open)
    high_enough = float(mic[2].item()) > 0.92
    success = one_side_holding and high_enough
    _log_check(
        "handover_microphone",
        env_id,
        mic_xyz=mic.detach().cpu().tolist(),
        left_open=left_open,
        right_open=right_open,
        left_closed=left_closed,
        right_closed=right_closed,
        high_enough=high_enough,
        success=success,
    )
    return success


def _check_lift_pot(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    pot = _try_get_asset_root_pos(scene, knowledge_map, ["kitchenpot_1"], env_id)
    if pot is None:
        return False
    init_pot = ctx.get("initial_root_pos", {}).get(("kitchenpot_1", env_id), None)
    if init_pot is None:
        _log_check("lift_pot", env_id, reason="missing_initial_pot", success=False)
        return False
    both_closed = _both_grippers_closed(scene, env_id)
    lift_dz = float(pot[2].item() - init_pot[2].item())
    dz_threshold = 0.03
    height_ok = lift_dz > dz_threshold
    peak_map = ctx.setdefault("peak_lift_dz", {})
    peak_key = ("kitchenpot_1", env_id)
    prev_peak = float(peak_map.get(peak_key, -1e9))
    peak_dz = max(prev_peak, lift_dz)
    peak_map[peak_key] = peak_dz
    peak_ok = peak_dz > dz_threshold
    # Lift task is considered successful once lifted high enough at any time
    # in the episode, regardless of the final gripper state at evaluation frame.
    success = peak_ok
    _log_check(
        "lift_pot",
        env_id,
        pot_xyz=pot.detach().cpu().tolist(),
        init_z=f"{float(init_pot[2].item()):.4f}",
        lift_dz=f"{lift_dz:.4f}",
        peak_lift_dz=f"{peak_dz:.4f}",
        dz_threshold=f"{dz_threshold:.4f}",
        both_closed=both_closed,
        height_ok=height_ok,
        peak_ok=peak_ok,
        success=success,
    )
    return success


def _check_move_can_pot(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    can = _try_get_asset_root_pos(scene, knowledge_map, ["can_1"], env_id)
    pot = _try_get_asset_root_pos(scene, knowledge_map, ["kitchenpot_1"], env_id)
    init_can = ctx.get("initial_root_pos", {}).get(("can_1", env_id), None)
    if can is None or pot is None or init_can is None:
        return False
    dx = float(can[0].item() - pot[0].item())
    dy = abs(float(can[1].item() - pot[1].item()))
    side_ok = abs(dx) < 0.2 and abs(dx) > 0.02
    y_ok = dy < 0.035
    low_enough = float(can[2].item()) <= float(init_can[2].item()) + 0.02
    gripper_open = _both_grippers_open(scene, env_id)
    success = side_ok and y_ok and low_enough and gripper_open
    _log_check("move_can_pot", env_id, can_xyz=can.detach().cpu().tolist(), pot_xyz=pot.detach().cpu().tolist(), dx=f"{dx:.4f}", dy=f"{dy:.4f}", side_ok=side_ok, y_ok=y_ok, low_enough=low_enough, gripper_open=gripper_open, success=success)
    return success


def _check_move_pillbottle_pad(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    obj = _try_get_asset_root_pos(scene, knowledge_map, ["pillbottle_1"], env_id)
    pad = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    init_obj = ctx.get("initial_root_pos", {}).get(("pillbottle_1", env_id), None)
    if obj is None or pad is None or init_obj is None:
        return False
    dxy = torch.abs(obj[0:2] - pad[0:2]).detach().cpu()
    xy_ok = bool((dxy[0] < 0.03 and dxy[1] < 0.03).item())
    z_ok = abs(float(obj[2].item()) - float(init_obj[2].item())) < 0.03
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and z_ok and gripper_open
    _log_check("move_pillbottle_pad", env_id, obj_xyz=obj.detach().cpu().tolist(), pad_xyz=pad.detach().cpu().tolist(), dxy=dxy.tolist(), z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_move_playingcard_away(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    card = _try_get_asset_root_pos(scene, knowledge_map, ["playingcards_1", "playingcard_1"], env_id)
    if card is None:
        return False
    edge = abs(float(card[0].item())) > 0.23
    gripper_open = _both_grippers_open(scene, env_id)
    success = edge and gripper_open
    _log_check("move_playingcard_away", env_id, card_xyz=card.detach().cpu().tolist(), edge=edge, gripper_open=gripper_open, success=success)
    return success


def _check_pick_bottle(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    bottle = _try_get_asset_root_pos(scene, knowledge_map, ["bottle_1"], env_id)
    if bottle is None:
        return False
    success = float(bottle[2].item()) > 0.89
    _log_check("pick_bottle", env_id, bottle_xyz=bottle.detach().cpu().tolist(), success=success)
    return success


def _check_pick_dual_bottles(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    b1 = _try_get_asset_root_pos(scene, knowledge_map, ["bottle_1"], env_id)
    b2 = _try_get_asset_root_pos(scene, knowledge_map, ["bottle_2"], env_id)
    if b1 is None or b2 is None:
        return False
    success = float(b1[2].item()) > 0.89 and float(b2[2].item()) > 0.89
    _log_check("pick_dual_bottles", env_id, bottle1_xyz=b1.detach().cpu().tolist(), bottle2_xyz=b2.detach().cpu().tolist(), success=success)
    return success


def _check_place_a2b(scene: Any, knowledge_map: dict[str, object], env_id: int, left: bool, task_label: str) -> bool:
    ids = _infer_non_table_assets(knowledge_map)
    if len(ids) < 2:
        return False
    obj = _try_get_asset_root_pos(scene, knowledge_map, [ids[0]], env_id)
    tgt = _try_get_asset_root_pos(scene, knowledge_map, [ids[1]], env_id)
    if obj is None or tgt is None:
        return False
    d = torch.abs(obj - tgt).detach().cpu()
    dist = float(torch.linalg.norm((obj - tgt)[0:2]).item())
    side_ok = float(obj[0].item()) < float(tgt[0].item()) if left else float(obj[0].item()) > float(tgt[0].item())
    y_ok = d[1] < 0.05
    dist_ok = 0.08 < dist < 0.2
    gripper_open = _both_grippers_open(scene, env_id)
    success = side_ok and y_ok and dist_ok and gripper_open
    _log_check(task_label, env_id, obj_id=ids[0], tgt_id=ids[1], obj_xyz=obj.detach().cpu().tolist(), tgt_xyz=tgt.detach().cpu().tolist(), dist_xy=f"{dist:.4f}", side_ok=side_ok, y_ok=bool(y_ok.item()), dist_ok=dist_ok, gripper_open=gripper_open, success=success)
    return success


def _check_place_a2b_left(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_place_a2b(scene, knowledge_map, env_id, left=True, task_label="place_A2B_left")


def _check_place_a2b_right(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_place_a2b(scene, knowledge_map, env_id, left=False, task_label="place_A2B_right")


def _check_place_bread_basket(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    bread = _try_get_asset_root_pos(scene, knowledge_map, ["bread_1"], env_id)
    basket = _try_get_asset_root_pos(scene, knowledge_map, ["breadbasket_1", "basket_1"], env_id)
    if bread is None or basket is None:
        return False
    dxy = torch.abs(bread[0:2] - basket[0:2]).detach().cpu()
    xy_ok = bool((dxy[0] < 0.05 and dxy[1] < 0.05).item())
    z_ok = float(bread[2].item()) > 0.73
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and z_ok and gripper_open
    _log_check("place_bread_basket", env_id, bread_xyz=bread.detach().cpu().tolist(), basket_xyz=basket.detach().cpu().tolist(), dxy=dxy.tolist(), z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_place_bread_skillet(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    bread = _try_get_asset_root_pos(scene, knowledge_map, ["bread_1"], env_id)
    skillet = _try_get_asset_root_pos(scene, knowledge_map, ["skillet_1"], env_id)
    if bread is None or skillet is None:
        return False
    dxy = torch.abs(bread[0:2] - skillet[0:2]).detach().cpu()
    xy_ok = bool((dxy[0] < 0.035 and dxy[1] < 0.035).item())
    z_ok = float(bread[2].item()) > 0.76 and float(skillet[2].item()) > 0.76
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and z_ok and gripper_open
    _log_check("place_bread_skillet", env_id, bread_xyz=bread.detach().cpu().tolist(), skillet_xyz=skillet.detach().cpu().tolist(), dxy=dxy.tolist(), z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_place_burger_fries(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    burger = _try_get_asset_root_pos(scene, knowledge_map, ["hamburg_1", "burger_1"], env_id)
    fries = _try_get_asset_root_pos(scene, knowledge_map, ["frenchfries_1", "fries_1"], env_id)
    tray = _try_get_asset_root_pos(scene, knowledge_map, ["tray_1"], env_id)
    if burger is None or fries is None or tray is None:
        return False
    db = float(torch.linalg.norm((burger - tray)[0:2]).item())
    df = float(torch.linalg.norm((fries - tray)[0:2]).item())
    near = db < 0.08 and df < 0.08
    gripper_open = _both_grippers_open(scene, env_id)
    success = near and gripper_open
    _log_check("place_burger_fries", env_id, burger_xyz=burger.detach().cpu().tolist(), fries_xyz=fries.detach().cpu().tolist(), tray_xyz=tray.detach().cpu().tolist(), dist_burger=f"{db:.4f}", dist_fries=f"{df:.4f}", gripper_open=gripper_open, success=success)
    return success


def _check_place_object_in_basket(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any], object_ids: list[str], task_label: str) -> bool:
    obj = _try_get_asset_root_pos(scene, knowledge_map, object_ids, env_id)
    basket = _try_get_asset_root_pos(scene, knowledge_map, ["basket_1"], env_id)
    if obj is None or basket is None:
        return False
    obj_init = None
    for oid in object_ids:
        obj_init = ctx.get("initial_root_pos", {}).get((oid, env_id), None)
        if obj_init is not None:
            break
    basket_init = ctx.get("initial_root_pos", {}).get(("basket_1", env_id), None)
    if obj_init is None or basket_init is None:
        return False
    basket_lift = float(basket[2].item()) - float(basket_init[2].item())
    obj_lift = float(obj[2].item()) - float(obj_init[2].item())
    dist = float(torch.sum(torch.abs(obj - basket)).item())
    contact_proxy = _contact_proxy_by_distance(scene, knowledge_map, object_ids[0], "basket_1", env_id, max_center_distance=0.15)
    success = basket_lift > 0.02 and obj_lift > 0.02 and dist < 0.15 and contact_proxy
    _log_check(task_label, env_id, obj_xyz=obj.detach().cpu().tolist(), basket_xyz=basket.detach().cpu().tolist(), basket_lift=f"{basket_lift:.4f}", obj_lift=f"{obj_lift:.4f}", l1_dist=f"{dist:.4f}", contact_proxy=contact_proxy, success=success)
    return success


def _check_place_can_basket(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_place_object_in_basket(scene, knowledge_map, env_id, ctx, ["can_1"], "place_can_basket")


def _check_place_object_basket(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    return _check_place_object_in_basket(scene, knowledge_map, env_id, ctx, ["toycar_1", "object_1", "rubikscube_1"], "place_object_basket")


def _check_place_cans_plasticbox(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    c1 = _try_get_asset_root_pos(scene, knowledge_map, ["can_1"], env_id)
    c2 = _try_get_asset_root_pos(scene, knowledge_map, ["can_2"], env_id)
    box = _try_get_asset_root_pos(scene, knowledge_map, ["plasticbox_1"], env_id)
    if c1 is None or c2 is None or box is None:
        return False
    d1 = float(torch.linalg.norm(c1 - box).item())
    d2 = float(torch.linalg.norm(c2 - box).item())
    near = d1 < 0.04 and d2 < 0.04
    gripper_open = _both_grippers_open(scene, env_id)
    success = near and gripper_open
    _log_check("place_cans_plasticbox", env_id, can1_xyz=c1.detach().cpu().tolist(), can2_xyz=c2.detach().cpu().tolist(), box_xyz=box.detach().cpu().tolist(), d1=f"{d1:.4f}", d2=f"{d2:.4f}", gripper_open=gripper_open, success=success)
    return success


def _check_place_container_plate(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    cont = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_1", "container_1"], env_id)
    plate = _try_get_asset_root_pos(scene, knowledge_map, ["plate_1"], env_id)
    if cont is None or plate is None:
        return False
    d = torch.abs(cont - plate).detach().cpu()
    close = bool((d[0] < 0.05 and d[1] < 0.05 and d[2] < 0.03).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = close and gripper_open
    _log_check("place_container_plate", env_id, container_xyz=cont.detach().cpu().tolist(), plate_xyz=plate.detach().cpu().tolist(), delta=d.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_place_dual_shoes(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    s1 = _try_get_asset_root_pos(scene, knowledge_map, ["shoe_1"], env_id)
    s2 = _try_get_asset_root_pos(scene, knowledge_map, ["shoe_2"], env_id)
    box = _try_get_asset_root_pos(scene, knowledge_map, ["shoebox_1"], env_id)
    if s1 is None or s2 is None or box is None:
        return False
    dxy1 = torch.abs(s1[0:2] - box[0:2]).detach().cpu()
    dxy2 = torch.abs(s2[0:2] - box[0:2]).detach().cpu()
    xy_ok = bool((dxy1[0] < 0.05 and dxy1[1] < 0.05 and dxy2[0] < 0.05 and dxy2[1] < 0.05).item())
    z_target = float(box[2].item()) + 0.01
    z_ok = abs(float(s1[2].item()) - z_target) < 0.03 and abs(float(s2[2].item()) - z_target) < 0.03
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and z_ok and gripper_open
    _log_check("place_dual_shoes", env_id, shoe1_xyz=s1.detach().cpu().tolist(), shoe2_xyz=s2.detach().cpu().tolist(), box_xyz=box.detach().cpu().tolist(), xy_ok=xy_ok, z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_place_empty_cup(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    cup = _try_get_asset_root_pos(scene, knowledge_map, ["cup_1"], env_id)
    coaster = _try_get_asset_root_pos(scene, knowledge_map, ["coaster_1"], env_id)
    if cup is None or coaster is None:
        return False
    dxy = cup[0:2] - coaster[0:2]
    xy_sq = float(torch.sum(dxy * dxy).item())
    xy_dist = xy_sq ** 0.5
    dz = abs(float(cup[2].item() - coaster[2].item()))
    left_vals = _get_gripper_joint_vals(scene, "robot_left", env_id)
    right_vals = _get_gripper_joint_vals(scene, "robot_right", env_id)
    left_open = _is_gripper_open(scene, "robot_left", env_id)
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    gripper_open = left_open and right_open
    xy_ok = xy_sq < (0.035 * 0.035)
    z_ok = dz < 0.015
    success = xy_ok and z_ok and gripper_open
    _log_check(
        "place_empty_cup",
        env_id,
        cup_xyz=cup.detach().cpu().tolist(),
        coaster_xyz=coaster.detach().cpu().tolist(),
        xy_dist=f"{xy_dist:.4f}",
        xy_sq=f"{xy_sq:.5f}",
        xy_ok=xy_ok,
        dz=f"{dz:.4f}",
        z_ok=z_ok,
        left_grip=None if left_vals is None else [round(float(v), 4) for v in left_vals.detach().cpu().tolist()],
        right_grip=None if right_vals is None else [round(float(v), 4) for v in right_vals.detach().cpu().tolist()],
        left_open=left_open,
        right_open=right_open,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_place_mouse_pad(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    mouse = _try_get_asset_root_pos(scene, knowledge_map, ["mouse_1"], env_id)
    pad = _try_get_asset_root_pos(scene, knowledge_map, ["block_1", "mousepad_1"], env_id)
    if mouse is None or pad is None:
        return False
    d = torch.abs(mouse - pad).detach().cpu()
    xy_ok = bool((d[0] < 0.015 and d[1] < 0.012).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and gripper_open
    _log_check("place_mouse_pad", env_id, mouse_xyz=mouse.detach().cpu().tolist(), pad_xyz=pad.detach().cpu().tolist(), delta=d.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_place_object_scale(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    obj = _try_get_asset_root_pos(scene, knowledge_map, ["stapler_1", "object_1"], env_id)
    scale = _try_get_asset_root_pos(scene, knowledge_map, ["electronicscale_1", "scale_1"], env_id)
    if obj is None or scale is None:
        return False
    dxy = float(torch.linalg.norm((obj - scale)[0:2]).item())
    z_ok = float(obj[2].item()) > float(scale[2].item()) - 0.01
    gripper_open = _any_gripper_open(scene, env_id)
    success = dxy < 0.035 and z_ok and gripper_open
    _log_check("place_object_scale", env_id, obj_xyz=obj.detach().cpu().tolist(), scale_xyz=scale.detach().cpu().tolist(), dxy=f"{dxy:.4f}", z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_place_object_stand(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    obj = _try_get_asset_root_pos(scene, knowledge_map, ["rubikscube_1", "object_1"], env_id)
    stand = _try_get_asset_root_pos(scene, knowledge_map, ["displaystand_1", "stand_1"], env_id)
    if obj is None or stand is None:
        return False
    dxy = torch.abs(obj[0:2] - stand[0:2]).detach().cpu()
    xy_ok = bool((dxy[0] < 0.03 and dxy[1] < 0.03).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and gripper_open
    _log_check("place_object_stand", env_id, obj_xyz=obj.detach().cpu().tolist(), stand_xyz=stand.detach().cpu().tolist(), dxy=dxy.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_place_phone_stand(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    phone = _try_get_asset_root_pos(scene, knowledge_map, ["phone_1"], env_id)
    stand = _try_get_asset_root_pos(scene, knowledge_map, ["phonestand_1", "stand_1"], env_id)
    if phone is None or stand is None:
        return False
    d = torch.abs(phone - stand).detach().cpu()
    close = bool((d[0] < 0.045 and d[1] < 0.04 and d[2] < 0.04).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = close and gripper_open
    _log_check("place_phone_stand", env_id, phone_xyz=phone.detach().cpu().tolist(), stand_xyz=stand.detach().cpu().tolist(), delta=d.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_place_remote(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    remote = _try_get_asset_root_pos(scene, knowledge_map, ["remote_1"], env_id)
    init_remote = ctx.get("initial_root_pos", {}).get(("remote_1", env_id), None)
    if remote is None or init_remote is None:
        return False
    moved = float(torch.linalg.norm(remote - init_remote).item()) > 0.05
    gripper_open = _both_grippers_open(scene, env_id)
    success = moved and gripper_open
    _log_check("place_remote", env_id, remote_xyz=remote.detach().cpu().tolist(), moved=moved, gripper_open=gripper_open, success=success)
    return success


def _check_place_shoe(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    shoe = _try_get_asset_root_pos(scene, knowledge_map, ["shoe_1"], env_id)
    init_shoe = ctx.get("initial_root_pos", {}).get(("shoe_1", env_id), None)
    if shoe is None or init_shoe is None:
        return False
    moved = float(torch.linalg.norm((shoe - init_shoe)[0:2]).item()) > 0.05
    z_ok = float(shoe[2].item()) > float(init_shoe[2].item()) + 0.01
    gripper_open = _both_grippers_open(scene, env_id)
    success = moved and z_ok and gripper_open
    _log_check("place_shoe", env_id, shoe_xyz=shoe.detach().cpu().tolist(), moved=moved, z_ok=z_ok, gripper_open=gripper_open, success=success)
    return success


def _check_put_bottles_dustbin(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    bottles = [
        _try_get_asset_root_pos(scene, knowledge_map, ["bottle_1"], env_id),
        _try_get_asset_root_pos(scene, knowledge_map, ["bottle_2"], env_id),
        _try_get_asset_root_pos(scene, knowledge_map, ["bottle_3"], env_id),
    ]
    if any(b is None for b in bottles):
        return False
    all_ok = True
    for b in bottles:
        x_ok = abs(float(b[0].item()) + 0.45) < 0.221
        y_ok = abs(float(b[1].item())) < 0.325
        z = float(b[2].item())
        z_ok = 0.2 < z < 0.7
        all_ok = all_ok and x_ok and y_ok and z_ok
    _log_check("put_bottles_dustbin", env_id, bottle_xyz=[b.detach().cpu().tolist() for b in bottles], success=all_ok)
    return all_ok


def _check_put_object_cabinet(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    obj = _try_get_asset_root_pos(scene, knowledge_map, ["rubikscube_1", "stapler_1", "object_1"], env_id)
    cabinet = _try_get_asset_root_pos(scene, knowledge_map, ["cabinet_1"], env_id)
    if obj is None or cabinet is None:
        return False
    init_obj = None
    for oid in ("rubikscube_1", "stapler_1", "object_1"):
        init_obj = ctx.get("initial_root_pos", {}).get((oid, env_id), None)
        if init_obj is not None:
            break
    if init_obj is None:
        return False
    lift = float(obj[2].item()) - float(init_obj[2].item())
    dxy = torch.abs(obj[0:2] - cabinet[0:2]).detach().cpu()
    lift_ok = 0.007 < lift < 0.12
    xy_ok = bool((dxy[0] < 0.05 and dxy[1] < 0.05).item())
    gripper_open = _any_gripper_open(scene, env_id)
    success = lift_ok and xy_ok and gripper_open
    _log_check("put_object_cabinet", env_id, obj_xyz=obj.detach().cpu().tolist(), cabinet_xyz=cabinet.detach().cpu().tolist(), lift=f"{lift:.4f}", dxy=dxy.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_rotate_qrcode(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    target = torch.tensor([0.707, 0.707, 0.0, 0.0], dtype=torch.float32)
    pos = _try_get_asset_root_pos(scene, knowledge_map, ["paymentsign_1", "qrcode_1"], env_id)
    quat = None
    for aid in ("paymentsign_1", "qrcode_1"):
        try:
            quat = _get_asset_root_quat(scene, knowledge_map, aid, env_id).detach().cpu().to(torch.float32)
            break
        except Exception:
            continue
    if pos is None or quat is None:
        return False
    ori_ok = _quat_close_signed_equiv(quat, target, eps=0.08)
    z = float(pos[2].item())
    # RoboTwin uses z < 0.75, but a fallen object (z=-80) would pass; require table height.
    z_ok = 0.70 <= z <= 0.82
    left_open = _is_gripper_open(scene, "robot_left", env_id)
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    gripper_open = left_open and right_open
    success = ori_ok and z_ok and gripper_open
    _log_check(
        "rotate_qrcode",
        env_id,
        obj_xyz=pos.detach().cpu().tolist(),
        quat=quat.tolist(),
        target_quat=target.tolist(),
        ori_ok=ori_ok,
        z=f"{z:.4f}",
        z_ok=z_ok,
        left_open=left_open,
        right_open=right_open,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_scan_object(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    scanner = _try_get_asset_root_pos(scene, knowledge_map, ["scanner_1"], env_id)
    obj = _try_get_asset_root_pos(scene, knowledge_map, ["teabox_1", "object_1"], env_id)
    if scanner is None or obj is None:
        return False
    d = torch.abs(scanner - obj).detach().cpu()
    near_axis = bool((d[0] < 0.025 and d[1] < 0.025).item())
    depth_ok = 0.0 < float(d[2].item()) < 0.07
    gripper_closed = _both_grippers_closed(scene, env_id)
    success = near_axis and depth_ok and gripper_closed
    _log_check("scan_object", env_id, scanner_xyz=scanner.detach().cpu().tolist(), obj_xyz=obj.detach().cpu().tolist(), delta=d.tolist(), gripper_closed=gripper_closed, success=success)
    return success


def _check_stamp_seal(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    seal = _try_get_asset_root_pos(scene, knowledge_map, ["seal_1"], env_id)
    target = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    if seal is None or target is None:
        return False
    dxy = torch.abs(seal[0:2] - target[0:2]).detach().cpu()
    xy_ok = bool((dxy[0] < 0.01 and dxy[1] < 0.01).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = xy_ok and gripper_open
    _log_check("stamp_seal", env_id, seal_xyz=seal.detach().cpu().tolist(), target_xyz=target.detach().cpu().tolist(), dxy=dxy.tolist(), gripper_open=gripper_open, success=success)
    return success


def _check_stack_blocks_two(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    b1 = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    b2 = _try_get_asset_root_pos(scene, knowledge_map, ["block_2"], env_id)
    if b1 is None or b2 is None:
        return False
    target = b1 + torch.tensor([0.0, 0.0, 0.05], dtype=b1.dtype, device=b1.device)
    d = torch.abs(b2 - target).detach().cpu()
    aligned = bool((d[0] < 0.025 and d[1] < 0.025 and d[2] < 0.012).item())
    gripper_open = _both_grippers_open(scene, env_id)
    success = aligned and gripper_open
    _log_check("stack_blocks_two", env_id, block1_xyz=b1.detach().cpu().tolist(), block2_xyz=b2.detach().cpu().tolist(), delta_to_target=d.tolist(), gripper_open=gripper_open, success=success)
    return success


def _build_generic_compare_metrics(
    scene: Any,
    knowledge_map: dict[str, object],
    env_id: int,
    ctx: dict[str, Any],
    max_assets: int = 8,
) -> list[dict[str, Any]]:
    """
    Build a compact, task-agnostic comparison snapshot for logging.
    This guarantees every task prints comparable success data.
    """
    rows: list[dict[str, Any]] = []
    initial_root = ctx.get("initial_root_pos", {})
    for asset_id in sorted(knowledge_map.keys()):
        if "table" in asset_id:
            continue
        try:
            cur = _get_asset_root_pos(scene=scene, knowledge_map=knowledge_map, asset_id=asset_id, env_id=env_id).detach().cpu()
        except Exception:
            continue
        row: dict[str, Any] = {
            "asset": asset_id,
            "xyz": [round(float(cur[0].item()), 4), round(float(cur[1].item()), 4), round(float(cur[2].item()), 4)],
        }
        init = initial_root.get((asset_id, env_id), None)
        if init is not None:
            delta = (cur - init.detach().cpu()).to(torch.float32)
            row["dxyz"] = [round(float(delta[0].item()), 4), round(float(delta[1].item()), 4), round(float(delta[2].item()), 4)]
            row["disp"] = round(float(torch.linalg.norm(delta).item()), 4)
        rows.append(row)
        if len(rows) >= max_assets:
            break
    return rows


def build_success_context(scene: Any, knowledge_map: dict[str, object], num_envs: int) -> dict[str, dict[tuple[str, int], torch.Tensor]]:
    ctx: dict[str, dict[tuple[str, int], torch.Tensor]] = {
        "initial_root_pos": {},
        "initial_joint_pos": {},
    }
    for asset_id in knowledge_map.keys():
        for env_id in range(num_envs):
            try:
                ctx["initial_root_pos"][(asset_id, env_id)] = _get_asset_root_pos(
                    scene=scene, knowledge_map=knowledge_map, asset_id=asset_id, env_id=env_id
                )
            except Exception:
                pass
            try:
                ctx["initial_joint_pos"][(asset_id, env_id)] = _get_asset_joint_pos(
                    scene=scene, knowledge_map=knowledge_map, asset_id=asset_id, env_id=env_id
                )
            except Exception:
                pass
    return ctx


def _check_beat_block_hammer(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    """
    RobotWin2-style criterion:
    1) hammer functional point XY is close to block functional point XY (eps=[0.02, 0.02])
    2) hammer and block are in contact.
    Notes:
    - We intentionally avoid RoboTwin object interfaces and use local sim state only.
    - Contact is approximated by center-distance proxy unless you later wire a real contact sensor checker.
    """
    hammer_xy = _get_asset_root_pos(
        scene=scene,
        knowledge_map=knowledge_map,
        asset_id="hammer_1",
        env_id=env_id,
    )[0:2].detach().cpu()
    block_xy = _get_asset_root_pos(
        scene=scene,
        knowledge_map=knowledge_map,
        asset_id="block_1",
        env_id=env_id,
    )[0:2].detach().cpu()

    eps = torch.tensor([0.1, 0.1], dtype=torch.float32)
    xy_delta = torch.abs(hammer_xy - block_xy)
    xy_aligned = bool(torch.all(xy_delta < eps).item())

    hammer_pos_full = _get_asset_root_pos(
        scene=scene, knowledge_map=knowledge_map, asset_id="hammer_1", env_id=env_id
    ).detach().cpu()
    block_pos_full = _get_asset_root_pos(
        scene=scene, knowledge_map=knowledge_map, asset_id="block_1", env_id=env_id
    ).detach().cpu()
    center_dist = float(torch.linalg.norm(hammer_pos_full - block_pos_full).item())

    in_contact = _contact_proxy_by_distance(
        scene=scene,
        knowledge_map=knowledge_map,
        asset_a="hammer_1",
        asset_b="block_1",
        env_id=env_id,
        max_center_distance=0.1,
    )
    success = xy_aligned and in_contact
    _log_check(
        "beat_block_hammer",
        env_id,
        hammer_xyz=hammer_pos_full.tolist(),
        block_xyz=block_pos_full.tolist(),
        xy_delta=xy_delta.tolist(),
        xy_eps=eps.tolist(),
        center_dist=f"{center_dist:.4f}",
        xy_aligned=xy_aligned,
        in_contact_proxy=in_contact,
        success=success,
    )
    return success


def _check_open_microwave(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    """
    RoboTwin open_microwave style:
      qpos[0] >= limit_upper * target
    Local approximation without qlimits:
      current joint increased from initial by a meaningful margin.
    """
    cur = _try_get_asset_joint_pos(scene, knowledge_map, ["microwave_1"], env_id)
    init = ctx.get("initial_joint_pos", {}).get(("microwave_1", env_id), None)
    if cur is None or init is None or cur.numel() == 0:
        _log_check("open_microwave", env_id, reason="joint_not_available", success=False)
        return False
    delta = float(cur[0].item() - init[0].item())
    success = delta >= 0.20
    _log_check("open_microwave", env_id, joint_delta=f"{delta:.4f}", threshold="0.2000", success=success)
    return success


def _check_turn_switch(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    """Success if switch joint changed from initial by a meaningful amount."""
    cur = _get_asset_joint_pos(scene=scene, knowledge_map=knowledge_map, asset_id="switch_1", env_id=env_id)
    init = ctx.get("initial_joint_pos", {}).get(("switch_1", env_id), None)
    if init is None or cur.numel() == 0:
        _log_check("turn_switch", env_id, reason="joint_not_available", success=False)
        return False
    delta = abs(float(cur[0].item() - init[0].item()))
    success = delta >= 0.03
    _log_check("turn_switch", env_id, joint_abs_delta=f"{delta:.4f}", threshold="0.0300", success=success)
    return success


def _check_open_laptop(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    """Success if laptop hinge joint opened relative to initial."""
    cur = _get_asset_joint_pos(scene=scene, knowledge_map=knowledge_map, asset_id="laptop_1", env_id=env_id)
    init = ctx.get("initial_joint_pos", {}).get(("laptop_1", env_id), None)
    if init is None or cur.numel() == 0:
        _log_check("open_laptop", env_id, reason="joint_not_available", success=False)
        return False
    delta = abs(float(cur[0].item() - init[0].item()))
    success = delta >= 0.15
    _log_check("open_laptop", env_id, joint_abs_delta=f"{delta:.4f}", threshold="0.1500", success=success)
    return success


def _check_click_like(
    scene: Any,
    knowledge_map: dict[str, object],
    env_id: int,
    object_ids: list[str],
    eps_xy: float,
    eps_z: float,
    task_label: str,
    hand_z_offset: float = 0.0,
) -> bool:
    obj_pos = _try_get_asset_root_pos(scene, knowledge_map, object_ids, env_id)
    if obj_pos is None:
        _log_check(task_label, env_id, reason="object_not_found", object_ids=object_ids, success=False)
        return False
    robot = _resolve_robot(scene, "robot")
    names = _robot_body_names(robot)
    arms = (
        ("robot_left", "fl_link6"),
        ("robot_right", "fr_link6"),
    )
    hand_candidates = []
    arm_debug = []
    for robot_key, link_name in arms:
        hp = _get_body_pos_by_name(robot, link_name, env_id)
        if hp is None:
            arm_debug.append(
                {
                    "arm": robot_key,
                    "link": link_name,
                    "status": "hand_missing",
                    "n_bodies": len(names),
                    "body_tail": names[-8:],
                }
            )
            continue
        hp_eval = hp.clone().detach()
        hp_eval[2] = hp_eval[2] - float(hand_z_offset)
        d = torch.abs(hp_eval - obj_pos)
        close = bool((d[0] < eps_xy).item() and (d[1] < eps_xy).item() and (d[2] < eps_z).item())
        closed = _is_gripper_closed(scene, robot_key=robot_key, env_id=env_id)
        hit = close and closed
        hand_candidates.append(hit)
        arm_debug.append(
            {
                "arm": robot_key,
                "link": link_name,
                "hand_xyz_raw": [round(float(v), 4) for v in hp.detach().cpu().tolist()],
                "hand_xyz_eval": [round(float(v), 4) for v in hp_eval.detach().cpu().tolist()],
                "obj_xyz": [round(float(v), 4) for v in obj_pos.detach().cpu().tolist()],
                "delta_xyz": [round(float(v), 4) for v in d.detach().cpu().tolist()],
                "close_xyz": close,
                "gripper_closed": closed,
                "hit": hit,
            }
        )
    success = any(hand_candidates)
    _log_check(
        task_label,
        env_id,
        eps_xy=eps_xy,
        eps_z=eps_z,
        hand_z_offset=hand_z_offset,
        arm_debug=arm_debug,
        success=success,
    )
    return success


def _check_click_bell(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    # RoboTwin uses XY eps~0.025 and Z eps~0.03 around contact point with closed gripper.
    return _check_click_like(
        scene=scene,
        knowledge_map=knowledge_map,
        env_id=env_id,
        object_ids=["bell_1"],
        eps_xy=0.03,
        eps_z=0.04,
        task_label="click_bell",
        hand_z_offset=0.11,
    )


def _check_click_alarmclock(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    # Support both local ids seen in tasks: clock_1 / alarmclock_1.
    return _check_click_like(
        scene=scene,
        knowledge_map=knowledge_map,
        env_id=env_id,
        object_ids=["clock_1", "alarmclock_1"],
        eps_xy=0.03,
        eps_z=0.04,
        task_label="click_alarmclock",
        hand_z_offset=0.11,
    )


def _check_press_stapler(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    # RoboTwin checks gripper contact positions near stapler contact point. Local approximation:
    # closed gripper + hand close to stapler.
    return _check_click_like(
        scene=scene,
        knowledge_map=knowledge_map,
        env_id=env_id,
        object_ids=["stapler_1"],
        eps_xy=0.04,
        eps_z=0.05,
        task_label="press_stapler",
        hand_z_offset=0.11,
    )


def _check_stack_blocks_three(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    p1 = _try_get_asset_root_pos(scene, knowledge_map, ["block_1"], env_id)
    p2 = _try_get_asset_root_pos(scene, knowledge_map, ["block_2"], env_id)
    p3 = _try_get_asset_root_pos(scene, knowledge_map, ["block_3"], env_id)
    if p1 is None or p2 is None or p3 is None:
        return False
    poses = sorted([p1, p2, p3], key=lambda t: float(t[2].item()))
    b1, b2, b3 = poses
    xy12 = torch.abs(b1[0:2] - b2[0:2])
    xy23 = torch.abs(b2[0:2] - b3[0:2])
    dz12 = float((b2[2] - b1[2]).item())
    dz23 = float((b3[2] - b2[2]).item())
    # Slightly looser than RoboTwin defaults to tolerate Aloha placement noise.
    aligned = bool(torch.all(xy12 < 0.04).item() and torch.all(xy23 < 0.04).item())
    stacked = (0.03 <= dz12 <= 0.08) and (0.03 <= dz23 <= 0.08)
    left_open = _is_gripper_open(scene, "robot_left", env_id)
    right_open = _is_gripper_open(scene, "robot_right", env_id)
    gripper_open = left_open and right_open
    success = aligned and stacked and gripper_open
    _log_check(
        "stack_blocks_three",
        env_id,
        block_xyz=[b1.tolist(), b2.tolist(), b3.tolist()],
        xy12=xy12.tolist(),
        xy23=xy23.tolist(),
        dz12=f"{dz12:.4f}",
        dz23=f"{dz23:.4f}",
        aligned=aligned,
        stacked=stacked,
        left_open=left_open,
        right_open=right_open,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_stack_bowls_two(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    p1 = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_1"], env_id)
    p2 = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_2"], env_id)
    if p1 is None or p2 is None:
        return False
    low, high = sorted([p1, p2], key=lambda t: float(t[2].item()))
    xy = torch.abs(low[0:2] - high[0:2])
    dz = float((high[2] - low[2]).item())
    aligned = bool(torch.all(xy < 0.04).item())
    stacked = 0.03 <= dz <= 0.08
    gripper_open = _is_gripper_open(scene, "robot_left", env_id) and _is_gripper_open(scene, "robot_right", env_id)
    success = aligned and stacked and gripper_open
    _log_check(
        "stack_bowls_two",
        env_id,
        bowl_xyz=[low.tolist(), high.tolist()],
        xy_delta=xy.tolist(),
        dz=f"{dz:.4f}",
        aligned=aligned,
        stacked=stacked,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_stack_bowls_three(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    p1 = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_1"], env_id)
    p2 = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_2"], env_id)
    p3 = _try_get_asset_root_pos(scene, knowledge_map, ["bowl_3"], env_id)
    if p1 is None or p2 is None or p3 is None:
        return False
    low, mid, high = sorted([p1, p2, p3], key=lambda t: float(t[2].item()))
    xy12 = torch.abs(low[0:2] - mid[0:2])
    xy23 = torch.abs(mid[0:2] - high[0:2])
    dz12 = float((mid[2] - low[2]).item())
    dz23 = float((high[2] - mid[2]).item())
    aligned = bool(torch.all(xy12 < 0.04).item() and torch.all(xy23 < 0.04).item())
    stacked = (0.025 <= dz12 <= 0.08) and (0.025 <= dz23 <= 0.08)
    gripper_open = _is_gripper_open(scene, "robot_left", env_id) and _is_gripper_open(scene, "robot_right", env_id)
    success = aligned and stacked and gripper_open
    _log_check(
        "stack_bowls_three",
        env_id,
        bowl_xyz=[low.tolist(), mid.tolist(), high.tolist()],
        xy12=xy12.tolist(),
        xy23=xy23.tolist(),
        dz12=f"{dz12:.4f}",
        dz23=f"{dz23:.4f}",
        aligned=aligned,
        stacked=stacked,
        gripper_open=gripper_open,
        success=success,
    )
    return success


def _check_stacks_bowls_three_alias(scene: Any, knowledge_map: dict[str, object], env_id: int, ctx: dict[str, Any]) -> bool:
    """
    Local task JSONs overload task_name='stacks_bowls_three' for both blocks and bowls tasks.
    Dispatch by available asset ids.
    """
    keys = set(knowledge_map.keys())
    if {"bowl_1", "bowl_2", "bowl_3"}.issubset(keys):
        _log_check("stacks_bowls_three", env_id, branch="stack_bowls_three")
        return _check_stack_bowls_three(scene, knowledge_map, env_id, ctx)
    if {"bowl_1", "bowl_2"}.issubset(keys):
        _log_check("stacks_bowls_three", env_id, branch="stack_bowls_two")
        return _check_stack_bowls_two(scene, knowledge_map, env_id, ctx)
    if {"block_1", "block_2", "block_3"}.issubset(keys):
        _log_check("stacks_bowls_three", env_id, branch="stack_blocks_three")
        return _check_stack_blocks_three(scene, knowledge_map, env_id, ctx)
    _log_check("stacks_bowls_three", env_id, branch="unknown", success=False)
    return False


TASK_SUCCESS_CHECKERS: dict[str, Callable[[Any, dict[str, object], int, dict[str, Any]], bool]] = {
    "adjust_bottle": _check_adjust_bottle,
    "beat_block_hammer": _check_beat_block_hammer,
    "blocks_ranking_RGB": _check_blocks_ranking_rgb,
    "blocks_ranking_size": _check_blocks_ranking_size,
    "dump_bin_bigbin": _check_dump_bin_bigbin,
    "grab_roller": _check_grab_roller,
    "grap_roller": _check_grab_roller,  # local typo
    "hand_over_block": _check_handover_block,
    "handover_block": _check_handover_block,
    "hand_over_bottle": _check_handover_bottle,  # no exact RoboTwin env; closest local handover rule
    "handover_microphonr": _check_handover_mic_or_hanging_mug,  # local typo used by multiple jsons
    "handover_mic": _check_handover_mic_or_hanging_mug,
    "hanging_mug": _check_handover_mic_or_hanging_mug,
    "lift_pot": _check_lift_pot,
    "move_can_pot": _check_move_can_pot,
    "move_pillbottle_pad": _check_move_pillbottle_pad,
    "move_playingcards_away": _check_move_playingcard_away,  # local plural form
    "move_playingcard_away": _check_move_playingcard_away,
    "pick_bottle": _check_pick_bottle,
    "pick_dual_bottles": _check_pick_dual_bottles,
    "place_A2B_left": _check_place_a2b_left,
    "place_A2B_right": _check_place_a2b_right,
    "place_bread_basket": _check_place_bread_basket,
    "place_bread_skillet": _check_place_bread_skillet,
    "place_burger_fries": _check_place_burger_fries,
    "place_can_basket": _check_place_can_basket,
    "place_cans_plasticbox": _check_place_cans_plasticbox,
    "place_container_plate": _check_place_container_plate,
    "place_dual_shoe": _check_place_dual_shoes,  # local singular typo
    "place_dual_shoes": _check_place_dual_shoes,
    "place_empty_cup": _check_place_empty_cup,
    "place_mouse_pad": _check_place_mouse_pad,
    "place_object_basket": _check_place_object_basket,
    "place_object_scale": _check_place_object_scale,
    "place_object_stand": _check_place_object_stand,
    "place_phone_stand": _check_place_phone_stand,
    "place_remote": _check_place_remote,  # no exact RoboTwin env; local fallback
    "place_shoe": _check_place_shoe,
    "put_bottles_dustbin": _check_put_bottles_dustbin,
    "put_object_cabinet": _check_put_object_cabinet,
    "Rotate_QRcode": _check_rotate_qrcode,  # local capitalization
    "rotate_qrcode": _check_rotate_qrcode,
    "scan_object": _check_scan_object,
    "stamp_seal": _check_stamp_seal,
    "stacks_blocks_three": _check_stack_blocks_three,  # local plural typo
    "stacks_block_two": _check_stack_blocks_two,  # local singular typo
    "stack_blocks_two": _check_stack_blocks_two,
    "turn_switch": _check_turn_switch,
    "open_laptop": _check_open_laptop,
    "open_microwave": _check_open_microwave,
    "tuen_switch": _check_open_microwave,  # local typo in task json
    "press_stapler": _check_press_stapler,
    "click_bell": _check_click_bell,
    "click_alarmclock": _check_click_alarmclock,
    "stack_blocks_three": _check_stack_blocks_three,
    "stack_bowls_two": _check_stack_bowls_two,
    "stack_bowls_three": _check_stack_bowls_three,
    "stacks_bowls_three": _check_stacks_bowls_three_alias,
}


def evaluate_task_success(
    task_name: str,
    scene: Any,
    knowledge_map: dict[str, object],
    env_id: int,
    ctx: dict[str, Any],
) -> bool:
    checker = TASK_SUCCESS_CHECKERS.get(task_name)
    compare_rows = _build_generic_compare_metrics(scene=scene, knowledge_map=knowledge_map, env_id=env_id, ctx=ctx)
    if checker is None:
        print(f"[WARN] No success checker registered for task '{task_name}'.")
        _log_check(task_name, env_id, phase="summary", reason="checker_missing", compare=compare_rows, success=False)
        return False
    try:
        success = bool(checker(scene=scene, knowledge_map=knowledge_map, env_id=env_id, ctx=ctx))
        _log_check(task_name, env_id, phase="summary", compare=compare_rows, success=success)
        return success
    except Exception as exc:
        print(f"[WARN] success checker failed for task={task_name}, env={env_id}: {exc}")
        _log_check(task_name, env_id, phase="summary", reason=f"checker_exception:{exc}", compare=compare_rows, success=False)
        return False
