import json
import os
import pickle
import re
import fcntl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation as R


_ACTIVE_COLLECTOR = None


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _copy_camera_rgb(camera, env_id: int) -> np.ndarray:
    """Copy annotator RGB off the GPU before the render thread reuses it."""
    rgb = camera.data.output["rgb"][env_id, ..., :3]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().contiguous().clone().cpu().numpy()
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _hconcat_keep_height(images: List[np.ndarray], gap: int = 16, bg: int = 16) -> np.ndarray:
    """Place RGB frames side-by-side, matching the tallest frame."""
    frames = [np.ascontiguousarray(img, dtype=np.uint8) for img in images if img is not None]
    if not frames:
        raise ValueError("no images to concatenate")
    if len(frames) == 1:
        return frames[0]
    max_h = max(int(img.shape[0]) for img in frames)
    padded = []
    for img in frames:
        h, w = int(img.shape[0]), int(img.shape[1])
        if h != max_h:
            new_w = max(1, int(round(w * (max_h / max(h, 1)))))
            img = np.ascontiguousarray(
                np.asarray(Image.fromarray(img).resize((new_w, max_h), Image.LANCZOS), dtype=np.uint8)
            )
        padded.append(img)
    spacer = np.full((max_h, int(gap), 3), int(bg), dtype=np.uint8)
    parts: List[np.ndarray] = []
    for i, img in enumerate(padded):
        if i:
            parts.append(spacer)
        parts.append(img)
    return np.concatenate(parts, axis=1)


def save_third_person_scene_image(
    scene,
    sim,
    dataset_root: str,
    task_name: str = "task",
    env_id: int = 0,
    cam_key: str = "side_camera",
    cam_keys: Optional[Tuple[str, ...]] = None,
    warmup_steps: int = 24,
) -> Path:
    """Render HD stills from the existing side/head cameras before the task loop.

    Saves each camera as its own PNG and one combined third-person panel.
    """
    keys = list(cam_keys) if cam_keys else [cam_key]
    ordered = []
    for name in ("side_camera", "head_camera"):
        if name in keys and name not in ordered:
            ordered.append(name)
    for name in keys:
        if name not in ordered:
            ordered.append(name)
    keys = ordered
    safe_name = re.sub(r"[^\w.\-]+", "_", str(task_name or "task")).strip("._") or "task"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(dataset_root) / "scene_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(sim.get_physics_dt())
    for _ in range(int(warmup_steps)):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    rgbs: List[np.ndarray] = []
    last_path = out_dir / f"{safe_name}_{stamp}.png"
    for key in keys:
        try:
            cam = scene[key]
        except Exception:
            print(f"[WARN] scene has no camera '{key}', skip preview", flush=True)
            continue
        rgb = _copy_camera_rgb(cam, env_id)
        rgbs.append(rgb)
        cam_path = out_dir / f"{safe_name}_{stamp}_{key}.png"
        Image.fromarray(rgb, mode="RGB").save(cam_path, format="PNG", optimize=False)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        print(f"[INFO] scene preview {key}: {cam_path} ({width}x{height})", flush=True)
        last_path = cam_path
    if not rgbs:
        raise RuntimeError(f"no preview cameras found among {keys}")
    combined = _hconcat_keep_height(rgbs)
    Image.fromarray(combined, mode="RGB").save(last_path := out_dir / f"{safe_name}_{stamp}.png", format="PNG", optimize=False)
    height, width = int(combined.shape[0]), int(combined.shape[1])
    print(f"[INFO] third-person scene image: {last_path} ({width}x{height})", flush=True)
    return last_path


def save_all_cameras_task_image(
    scene,
    dataset_root: str,
    task_name: str = "task",
    env_id: int = 0,
    tick: int | None = None,
) -> Path:
    """Save each task camera as its own PNG under a folder named after the task."""
    cam_order = ("view_camera", "side_camera", "head_camera", "hand_left_camera", "hand_right_camera")
    safe_name = re.sub(r"[^\w.\-]+", "_", str(task_name or "task")).strip("._") or "task"
    out_dir = Path(dataset_root) / "scene_preview" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    last_path = out_dir
    tick_tag = f"_t{int(tick)}" if tick is not None else ""
    tick_label = str(int(tick)) if tick is not None else "?"
    for key in cam_order:
        try:
            cam = scene[key]
        except Exception:
            print(f"[WARN] scene has no camera '{key}', skip snapshot", flush=True)
            continue
        rgb = _copy_camera_rgb(cam, env_id)
        out_path = out_dir / f"{key}{tick_tag}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG", optimize=False)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        print(f"[INFO] t={tick_label} camera snapshot {key}: {out_path} ({width}x{height})", flush=True)
        last_path = out_path
        saved += 1
    if saved == 0:
        raise RuntimeError("no cameras available for task snapshot")
    return last_path


def _safe_quat_wxyz(q, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat must have 4 elements, got shape={q.shape}")
    n = np.linalg.norm(q)
    if (not np.isfinite(n)) or n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = _safe_quat_wxyz(quat_wxyz)
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def _pose7_to_T(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    quat_wxyz = _safe_quat_wxyz(quat_wxyz)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz).astype(np.float32)
    T[:3, 3] = pos
    return T


# Isaac world (x,y) -> SAPIEN/RoboTwin world (-y, x). Same Z-up meters, +90° yaw.
_ISAAC_TO_SAPIEN_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_ISAAC_TO_SAPIEN_T = np.eye(4, dtype=np.float32)
_ISAAC_TO_SAPIEN_T[:3, :3] = _ISAAC_TO_SAPIEN_R
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_Q_YAW90_WXYZ = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)
_DEFAULT_GRIPPER_OPEN_M = 0.047


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _safe_quat_wxyz(q1)
    w2, x2, y2, z2 = _safe_quat_wxyz(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _canonicalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _safe_quat_wxyz(q)
    if q[0] < 0.0:
        q = -q
    return q


def _pose7_isaac_to_sapien(pose7: List[float]) -> List[float]:
    p = np.asarray(pose7[:3], dtype=np.float32).reshape(3)
    q = _safe_quat_wxyz(pose7[3:7])
    p2 = _ISAAC_TO_SAPIEN_R @ p
    q2 = _canonicalize_quat_wxyz(_quat_mul_wxyz(_Q_YAW90_WXYZ, q))
    return [float(p2[0]), float(p2[1]), float(p2[2]), float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])]


def _T_isaac_to_sapien(T_w_c: np.ndarray) -> np.ndarray:
    return (_ISAAC_TO_SAPIEN_T @ np.asarray(T_w_c, dtype=np.float32)).astype(np.float32)


def _gl_to_cv_extrinsic(T_w_c_gl: np.ndarray) -> np.ndarray:
    T_w_c_cv = np.asarray(T_w_c_gl, dtype=np.float32) @ _GL_TO_CV
    T_c_w_cv = np.linalg.inv(T_w_c_cv).astype(np.float32)
    return T_c_w_cv[:3, :].astype(np.float32)


def _gripper_width_to_unit(width_m: float, open_width_m: float = _DEFAULT_GRIPPER_OPEN_M) -> float:
    return float(np.clip(float(width_m) / max(float(open_width_m), 1e-6), 0.0, 1.0))


def _find_body_pose7(robot, body_name: str, env_id: int) -> List[float]:
    body_ids, _ = robot.find_bodies(body_name)
    b = _to_np(robot.data.body_state_w[env_id, int(body_ids[0])])  # [x,y,z,qw,qx,qy,qz,...]
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]), float(b[6])]


def _infer_arm_prefix(ee_body_name: str) -> Optional[str]:
    if isinstance(ee_body_name, str):
        if ee_body_name.startswith("fl_"):
            return "fl_"
        if ee_body_name.startswith("fr_"):
            return "fr_"
    return None


def _get_arm_joint_state(robot, env_id: int, arm_dof: int, arm_prefix: Optional[str]) -> List[float]:
    names = list(robot.data.joint_names)
    q = _to_np(robot.data.joint_pos[env_id]).astype(np.float32)
    name_to_i = {n: i for i, n in enumerate(names)}

    if arm_prefix in ("fl_", "fr_"):
        wanted = [f"{arm_prefix}joint{i}" for i in range(1, int(arm_dof) + 1)]
        idx = [name_to_i[n] for n in wanted if n in name_to_i]
    else:
        idx = [i for i, n in enumerate(names) if ("finger" not in n.lower() and "gripper" not in n.lower())][:arm_dof]

    v = q[idx]
    if len(v) < arm_dof:
        v = np.pad(v, (0, arm_dof - len(v)), mode="constant")
    return [float(x) for x in v.tolist()]


def _get_ee_joint_state(robot, env_id: int, arm_prefix: Optional[str]) -> List[float]:
    names = list(robot.data.joint_names)
    q = _to_np(robot.data.joint_pos[env_id]).astype(np.float32)

    if arm_prefix in ("fl_", "fr_"):
        target = [f"{arm_prefix}joint7", f"{arm_prefix}joint8"]
        idx = [i for i, n in enumerate(names) if n in target]
    else:
        idx = [i for i, n in enumerate(names) if ("finger" in n.lower() or "gripper" in n.lower())]

    if len(idx) == 0:
        return [0.0]
    return [float(np.mean(q[idx]))]


def _delta_pose7(cur: List[float], prev: Optional[List[float]]) -> List[float]:
    if prev is None:
        return [0.0] * 7
    return [float(cur[i] - prev[i]) for i in range(7)]


def _refresh_camera_pose(camera, env_id: int) -> None:
    """Isaac Lab skips pose updates unless update_latest_camera_pose=True."""
    try:
        ids = getattr(camera, "_ALL_INDICES", None)
        if ids is None:
            camera._update_poses([env_id])
        else:
            camera._update_poses(ids)
    except Exception:
        pass


def _camera_quat_opengl(camera, env_id: int) -> np.ndarray:
    data = camera.data
    if hasattr(data, "quat_w_opengl"):
        return _to_np(data.quat_w_opengl[env_id]).astype(np.float32)
    return _to_np(data.quat_w_world[env_id]).astype(np.float32)


def _camera_matrices(camera, env_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OpenCV K, SAPIEN-world cam2world_gl (4x4), and extrinsic_cv (3x4)."""
    _refresh_camera_pose(camera, env_id)
    K = _to_np(camera.data.intrinsic_matrices[env_id]).astype(np.float32)
    c_pos = _to_np(camera.data.pos_w[env_id]).astype(np.float32)
    c_quat = _camera_quat_opengl(camera, env_id)
    T_w_c_gl_isaac = _pose7_to_T(c_pos, c_quat).astype(np.float32)
    T_w_c_gl = _T_isaac_to_sapien(T_w_c_gl_isaac)
    extr_cv = _gl_to_cv_extrinsic(T_w_c_gl)
    return K, T_w_c_gl, extr_cv


def _encode_rgb_bytes(rgb_uint8: np.ndarray) -> bytes:
    # PIL on the same thread. imageio PNG writers can spawn threads that
    # trip Isaac/carb mutex asserts while the simulator is still alive.
    img = Image.fromarray(np.ascontiguousarray(rgb_uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _bytes_to_fixed_array(items: List[bytes]) -> np.ndarray:
    if len(items) == 0:
        return np.asarray([], dtype="S1")
    max_len = max(len(x) for x in items)
    return np.asarray(items, dtype=f"S{max_len}")


def _stack_or_empty(values: List[np.ndarray], shape_tail: Tuple[int, ...], dtype) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0, *shape_tail), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _squeeze_traj(x) -> np.ndarray:
    arr = _to_np(x).astype(np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _infer_arm_from_planner(planner) -> str:
    names = [str(n) for n in list(getattr(planner, "joint_names", []) or [])]
    if any(n.startswith("fl_") for n in names):
        return "left"
    if any(n.startswith("fr_") for n in names):
        return "right"
    return "left"


def _even(n: int, minimum: int = 2) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 0 else n + 1


def _even_hw_frame(img: np.ndarray) -> np.ndarray:
    """libx264 / yuv420p rejects odd width or height (e.g. 853x360)."""
    h, w = int(img.shape[0]), int(img.shape[1])
    eh, ew = h - (h % 2), w - (w % 2)
    if eh < 2 or ew < 2:
        raise ValueError(f"frame too small for libx264: {w}x{h}")
    if eh == h and ew == w:
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(img[:eh, :ew])


def _resize_rgb(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    ht, wt = int(hw[0]), int(hw[1])
    h, w = img.shape[:2]
    if (h, w) == (ht, wt):
        return img
    return np.ascontiguousarray(
        np.asarray(Image.fromarray(img).resize((wt, ht), Image.LANCZOS), dtype=np.uint8)
    )


def _tile_2x2(head: np.ndarray, third: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = head.shape[:2]
    third = _resize_rgb(third, (h, w))
    left = _resize_rgb(left, (h, w))
    right = _resize_rgb(right, (h, w))
    top = np.concatenate([head, third], axis=1)
    bot = np.concatenate([left, right], axis=1)
    return np.concatenate([top, bot], axis=0)


def _tile_third_left_stack_right(
    third: np.ndarray,
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Large third-person on the left; head / left-hand / right-hand stacked on the right."""
    th, tw = int(third.shape[0]), int(third.shape[1])
    stack_h = _even(max(th // 3, 2))
    canvas_h = stack_h * 3
    main_w = _even(max(int(round(canvas_h * (tw / max(th, 1)))), 2))
    sh, sw = int(head.shape[0]), int(head.shape[1])
    stack_w = _even(max(int(round(stack_h * (sw / max(sh, 1)))), 2))
    main = _resize_rgb(third, (canvas_h, main_w))
    col = np.concatenate(
        [
            _resize_rgb(head, (stack_h, stack_w)),
            _resize_rgb(left, (stack_h, stack_w)),
            _resize_rgb(right, (stack_h, stack_w)),
        ],
        axis=0,
    )
    mosaic = np.concatenate([main, col], axis=1)
    return _even_hw_frame(mosaic)


def _write_mp4(out_path: str, frames: list, fps: int) -> None:
    if not frames:
        return
    frames = [_even_hw_frame(f) for f in frames]
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    print(f"[INFO] writing mp4 {out_path} {w}x{h} n={len(frames)}", flush=True)
    save_fps = min(20, max(5, int(fps)))
    try:
        imageio.mimsave(
            out_path,
            frames,
            fps=save_fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=[
                "-threads", "1",
                "-pix_fmt", "yuv420p",
                "-crf", "16",
                "-preset", "medium",
            ],
        )
    except TypeError:
        imageio.mimsave(out_path, frames, fps=save_fps)


def next_episode_index(dataset_root: str) -> int:
    """Scan RoboTwin-style and legacy export names to avoid overwriting."""
    root = Path(dataset_root)
    patterns = [
        (root / "data", r"episode(\d+)\."),
        (root / "_traj_data", r"episode(\d+)\."),
        (root / "video", r"episode(\d+)\."),
        (root / "data" / "chunk-000", r"episode_(\d+)\."),
        (root / "videos" / "chunk-000" / "images.rgb.head", r"episode_(\d+)\."),
    ]
    max_idx = -1
    for folder, pat in patterns:
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            match = re.match(pat, file_path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def allocate_episode_index(dataset_root: str) -> int:
    """
    Atomically allocate the next episode index for concurrent writers.

    Uses an exclusive flock on meta/episode_alloc.lock and persists the
    latest issued index in meta/episode_counter.txt. Also respects existing
    on-disk episode* files so mixed/legacy data stays safe.
    """
    root = Path(dataset_root)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "_traj_data", "video"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    lock_path = meta / "episode_alloc.lock"
    counter_path = meta / "episode_counter.txt"
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            scanned = next_episode_index(str(root))
            counter_val = -1
            if counter_path.exists():
                raw = counter_path.read_text(encoding="utf-8").strip()
                if raw:
                    counter_val = int(raw)
            episode_index = max(scanned, counter_val + 1)
            counter_path.write_text(str(episode_index), encoding="utf-8")
            return int(episode_index)
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _append_episodes_jsonl(dataset_root: str, record: Dict[str, Any]) -> None:
    meta = Path(dataset_root) / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    episodes_path = meta / "episodes.jsonl"
    lock_path = meta / "episodes_jsonl.lock"
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            with open(episodes_path, "a", encoding="utf-8") as fp:
                fp.write(line)
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


_STAGE_CTX: Dict[str, str] = {"dataset_root": "", "task_name": ""}
_STAGE_INT_KEYS = (
    "episode_attempts",
    "plan_calls",
    "plan_ok",
    "plan_fail",
    "stall_fail",
    "reached_eval",
    "check_ok",
    "check_fail",
    "dump_ok",
    "dump_fail",
    "crash",
)


def set_stage_context(dataset_root: str, task_name: str = "") -> None:
    _STAGE_CTX["dataset_root"] = str(dataset_root or "")
    _STAGE_CTX["task_name"] = str(task_name or "")
    if _STAGE_CTX["dataset_root"]:
        os.environ.setdefault("DATASET_ROOT", _STAGE_CTX["dataset_root"])


def _stage_dataset_root(dataset_root: Optional[str] = None) -> str:
    return str(
        dataset_root
        or _STAGE_CTX.get("dataset_root")
        or os.environ.get("DATASET_ROOT")
        or ""
    )


def _empty_stage_bucket() -> Dict[str, Any]:
    bucket: Dict[str, Any] = {k: 0 for k in _STAGE_INT_KEYS}
    bucket["check_metrics"] = {}
    return bucket


def _empty_stage_stats() -> Dict[str, Any]:
    stats = _empty_stage_bucket()
    stats["by_task"] = {}
    return stats


def _ensure_stage_bucket(bucket: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = _empty_stage_bucket()
    if not isinstance(bucket, dict):
        return out
    for key in _STAGE_INT_KEYS:
        try:
            out[key] = int(bucket.get(key, 0) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    metrics = bucket.get("check_metrics")
    if isinstance(metrics, dict):
        cleaned: Dict[str, Dict[str, int]] = {}
        for name, val in metrics.items():
            if not isinstance(val, dict):
                continue
            try:
                cleaned[str(name)] = {
                    "n": int(val.get("n", 0) or 0),
                    "ok": int(val.get("ok", 0) or 0),
                }
            except (TypeError, ValueError):
                continue
        out["check_metrics"] = cleaned
    return out


def _normalize_stage_stats(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _ensure_stage_bucket(raw)
    stats["by_task"] = {}
    by_task = raw.get("by_task") if isinstance(raw, dict) else None
    if isinstance(by_task, dict):
        for name, bucket in by_task.items():
            stats["by_task"][str(name)] = _ensure_stage_bucket(bucket)
    return stats


def _inc_stage_bucket(
    bucket: Dict[str, Any],
    deltas: Dict[str, int],
    check_metrics: Optional[Dict[str, bool]] = None,
) -> None:
    for key, val in deltas.items():
        if key not in _STAGE_INT_KEYS:
            continue
        try:
            inc = int(val or 0)
        except (TypeError, ValueError):
            continue
        if inc:
            bucket[key] = int(bucket.get(key, 0)) + inc
    if not check_metrics:
        return
    metrics = bucket.setdefault("check_metrics", {})
    for name, ok in check_metrics.items():
        entry = metrics.setdefault(str(name), {"n": 0, "ok": 0})
        entry["n"] = int(entry.get("n", 0)) + 1
        entry["ok"] = int(entry.get("ok", 0)) + int(bool(ok))


def _stage_rate(ok: int, n: int) -> str:
    if n <= 0:
        return "n/a"
    return f"{int(ok)}/{int(n)} ({ok / n:.3f})"


def format_stage_stats_text(stats_or_root: Any) -> str:
    if isinstance(stats_or_root, dict):
        stats = _normalize_stage_stats(stats_or_root)
    else:
        stats = load_stage_stats(str(stats_or_root or ""))
    attempts = int(stats.get("episode_attempts", 0))
    lines = [
        "[STAGE] "
        f"attempts={attempts} "
        f"plan={_stage_rate(stats.get('plan_ok', 0), stats.get('plan_calls', 0))} "
        f"stall={int(stats.get('stall_fail', 0))} "
        f"eval={int(stats.get('reached_eval', 0))} "
        f"check={_stage_rate(stats.get('check_ok', 0), stats.get('reached_eval', 0))} "
        f"dump={_stage_rate(stats.get('dump_ok', 0), attempts)} "
        f"crash={int(stats.get('crash', 0))} "
        f"overall={_stage_rate(stats.get('dump_ok', 0), attempts)}"
    ]
    metrics = stats.get("check_metrics") or {}
    if metrics:
        parts = [
            f"{name}={_stage_rate(val.get('ok', 0), val.get('n', 0))}"
            for name, val in sorted(metrics.items())
            if isinstance(val, dict)
        ]
        if parts:
            lines.append("[STAGE][check] " + " ".join(parts))
    by_task = stats.get("by_task") or {}
    if len(by_task) > 1:
        for name, bucket in sorted(by_task.items()):
            t_attempts = int(bucket.get("episode_attempts", 0))
            lines.append(
                f"[STAGE][task={name}] "
                f"attempts={t_attempts} "
                f"plan={_stage_rate(bucket.get('plan_ok', 0), bucket.get('plan_calls', 0))} "
                f"stall={int(bucket.get('stall_fail', 0))} "
                f"check={_stage_rate(bucket.get('check_ok', 0), bucket.get('reached_eval', 0))} "
                f"overall={_stage_rate(bucket.get('dump_ok', 0), t_attempts)}"
            )
    return "\n".join(lines)


def load_stage_stats(dataset_root: Optional[str] = None) -> Dict[str, Any]:
    root = _stage_dataset_root(dataset_root)
    if not root:
        return _empty_stage_stats()
    path = Path(root) / "meta" / "stage_stats.json"
    if not path.exists():
        return _empty_stage_stats()
    try:
        return _normalize_stage_stats(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return _empty_stage_stats()


def record_stage_event(
    dataset_root: Optional[str] = None,
    task_name: Optional[str] = None,
    check_metrics: Optional[Dict[str, bool]] = None,
    **deltas: int,
) -> Dict[str, Any]:
    """Flock-safe cumulative collection-stage counters under DATASET_ROOT/meta."""
    root = _stage_dataset_root(dataset_root)
    if not root:
        return _empty_stage_stats()
    task = str(task_name or _STAGE_CTX.get("task_name") or "")
    meta = Path(root) / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    stats_path = meta / "stage_stats.json"
    lock_path = meta / "stage_stats.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            stats = load_stage_stats(root)
            _inc_stage_bucket(stats, deltas, check_metrics)
            if task:
                bucket = stats["by_task"].setdefault(task, _empty_stage_bucket())
                _inc_stage_bucket(bucket, deltas, check_metrics)
            stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return stats
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def record_check_metrics(
    metrics: Optional[Dict[str, Any]] = None,
    task_name: str = "",
    dataset_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Accumulate boolean CHECK sub-criteria (xy_ok, gripper_open, ...)."""
    flags: Dict[str, bool] = {}
    for key, val in (metrics or {}).items():
        if key in {"phase", "reason", "compare", "ids", "obj_id", "tgt_id"}:
            continue
        if isinstance(val, bool):
            flags[str(key)] = val
            continue
        type_name = type(val).__name__
        if type_name in {"bool_", "bool8"}:
            flags[str(key)] = bool(val)
    if not flags:
        return load_stage_stats(dataset_root)
    return record_stage_event(dataset_root=dataset_root, task_name=task_name, check_metrics=flags)


def print_stage_rates(stats_or_root: Optional[Any] = None) -> None:
    root = _stage_dataset_root(None if isinstance(stats_or_root, dict) else stats_or_root)
    payload = stats_or_root if isinstance(stats_or_root, dict) else (root or stats_or_root)
    text = format_stage_stats_text(payload if payload else root)
    if text:
        print(text, flush=True)


def record_curobo_plan(planner, result, arm_dof: int = 6, arm: Optional[str] = None):
    """Record one cuRobo interpolated plan into the active episode pkl buffers."""
    collector = _ACTIVE_COLLECTOR
    if collector is None or result is None:
        return
    try:
        traj = result.get_interpolated_plan()
    except Exception:
        return
    if traj is None or getattr(traj, "position", None) is None:
        return

    pos = _squeeze_traj(traj.position)
    if hasattr(traj, "velocity") and traj.velocity is not None:
        vel = _squeeze_traj(traj.velocity)
        if vel.shape[0] != pos.shape[0]:
            vel = np.zeros_like(pos)
            if pos.shape[0] > 1:
                vel[1:] = pos[1:] - pos[:-1]
    else:
        vel = np.zeros_like(pos)
        if pos.shape[0] > 1:
            vel[1:] = pos[1:] - pos[:-1]

    dof = min(int(arm_dof), int(pos.shape[-1]))
    if arm is None:
        arm = _infer_arm_from_planner(planner)
    collector.append_joint_path(
        arm,
        position=pos[:, :dof],
        velocity=vel[:, :dof],
        status="Success",
    )


@dataclass
class CollectorConfig:
    output_root: str
    task_text: str = "STACK_BLOCKS_THREE"
    task_index: int = 0
    fps: int = 30
    arm_dof: int = 6
    gripper_open_width: float = _DEFAULT_GRIPPER_OPEN_M


class DualArmParquetCollector:
    """
    Writes RoboTwin-style folders:
      data/episode{N}.hdf5
      _traj_data/episode{N}.pkl
    HDF5 matches RoboTwin / SAPIEN layout:
      endpose, joint_action, observation/head_camera, pointcloud
      (wrist / third-view RGB are not stored in hdf5)
    World poses are stored in the SAPIEN frame (Isaac XY rotated +90° yaw).
    """

    def __init__(self, cfg: CollectorConfig):
        global _ACTIVE_COLLECTOR
        self.cfg = cfg
        self.episode_lengths: List[int] = []
        _ACTIVE_COLLECTOR = self
        self._prepare_dirs()
        self._reset_episode_buffers()

    def _prepare_dirs(self):
        os.makedirs(os.path.join(self.cfg.output_root, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.output_root, "_traj_data"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.output_root, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.output_root, "meta"), exist_ok=True)

        tasks_path = os.path.join(self.cfg.output_root, "meta", "tasks.jsonl")
        if not os.path.exists(tasks_path):
            with open(tasks_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"task_index": self.cfg.task_index, "task": self.cfg.task_text}, ensure_ascii=False) + "\n")

    def _reset_episode_buffers(self):
        self.state_left_arm_joint_states: List[List[float]] = []
        self.state_left_ee_joint_states: List[List[float]] = []
        self.state_left_ee_poses: List[List[float]] = []
        self.state_left_tcp_poses: List[List[float]] = []
        self.state_left_delta_ee_poses: List[List[float]] = []

        self.state_right_arm_joint_states: List[List[float]] = []
        self.state_right_ee_joint_states: List[List[float]] = []
        self.state_right_ee_poses: List[List[float]] = []
        self.state_right_tcp_poses: List[List[float]] = []
        self.state_right_delta_ee_poses: List[List[float]] = []

        self.action_left_arm_joint_states: List[List[float]] = []
        self.action_left_ee_joint_states: List[List[float]] = []
        self.action_left_ee_poses: List[List[float]] = []
        self.action_left_tcp_poses: List[List[float]] = []
        self.action_left_delta_ee_poses: List[List[float]] = []

        self.action_right_arm_joint_states: List[List[float]] = []
        self.action_right_ee_joint_states: List[List[float]] = []
        self.action_right_ee_poses: List[List[float]] = []
        self.action_right_tcp_poses: List[List[float]] = []
        self.action_right_delta_ee_poses: List[List[float]] = []

        # Store raw RGB uint8 during the episode; PNG-encode only at dump time.
        self.cam_head_colors: List[np.ndarray] = []
        self.cam_head_intrinsic: List[np.ndarray] = []
        self.cam_head_cam2world_gl: List[np.ndarray] = []
        self.cam_head_extrinsic_cv: List[np.ndarray] = []
        self.cam_head_shape: Optional[Tuple[int, int, int]] = None

        self.prev_left_ee_pose: Optional[List[float]] = None
        self.prev_right_ee_pose: Optional[List[float]] = None

        self.left_joint_path: List[Dict[str, Any]] = []
        self.right_joint_path: List[Dict[str, Any]] = []

    def discard_episode(self):
        """Drop the current episode buffers without writing any files."""
        self._reset_episode_buffers()

    def append_joint_path(
        self,
        arm: str,
        position,
        velocity=None,
        status: str = "Success",
    ):
        pos = np.asarray(position, dtype=np.float32)
        if pos.ndim == 1:
            pos = pos[None, :]
        if velocity is None:
            vel = np.zeros_like(pos)
            if pos.shape[0] > 1:
                vel[1:] = pos[1:] - pos[:-1]
        else:
            vel = np.asarray(velocity, dtype=np.float32)
            if vel.ndim == 1:
                vel = vel[None, :]
            if vel.shape != pos.shape:
                vel = np.zeros_like(pos)
                if pos.shape[0] > 1:
                    vel[1:] = pos[1:] - pos[:-1]
        item = {
            "status": str(status),
            "position": pos.astype(np.float32),
            "velocity": vel.astype(np.float32),
        }
        if arm == "right":
            self.right_joint_path.append(item)
        else:
            self.left_joint_path.append(item)

    @staticmethod
    def _joints_to_path_segments(joints: np.ndarray) -> List[Dict[str, Any]]:
        if joints.size == 0:
            return []
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim == 1:
            joints = joints[None, :]
        if np.allclose(joints, joints[:1], atol=1e-4):
            return []
        vel = np.zeros_like(joints)
        if joints.shape[0] > 1:
            vel[1:] = joints[1:] - joints[:-1]
        return [{"status": "Success", "position": joints, "velocity": vel}]

    @staticmethod
    def _append_camera_sample(
        camera,
        env_id: int,
        colors_buf: List[np.ndarray],
        intr_buf: List[np.ndarray],
        cam2world_buf: List[np.ndarray],
        extr_cv_buf: List[np.ndarray],
        shape_holder_name: str,
        owner,
    ):
        K, T_gl, extr_cv = _camera_matrices(camera, env_id)
        rgb = _copy_camera_rgb(camera, env_id)
        setattr(owner, shape_holder_name, tuple(int(x) for x in rgb.shape))
        colors_buf.append(rgb)
        intr_buf.append(K)
        cam2world_buf.append(T_gl)
        extr_cv_buf.append(extr_cv)

    def append_frame_from_sim(
        self,
        scene,
        env_id: int,
        episode_index: int,
        frame_index: int,
        sim_time_s: float,
        robot_left_key: str = "robot_left",
        robot_right_key: str = "robot_right",
        cam_head_key: str = "head_camera",
        cam_hand_left_key: str = "hand_left_camera",
        cam_hand_right_key: str = "hand_right_camera",
        cam_front_key: str = "side_camera",
        left_ee_body: str = "panda_hand",
        right_ee_body: str = "panda_hand",
        left_tcp_body: Optional[str] = None,
        right_tcp_body: Optional[str] = None,
        left_armbase_body: Optional[str] = None,
        right_armbase_body: Optional[str] = None,
        robot_frame_pose7: Optional[List[float]] = None,
    ):
        del (
            episode_index,
            frame_index,
            sim_time_s,
            left_armbase_body,
            right_armbase_body,
            robot_frame_pose7,
            cam_hand_left_key,
            cam_hand_right_key,
            cam_front_key,
        )
        robot_left = scene[robot_left_key]
        robot_right = scene[robot_right_key]
        cam_head = scene[cam_head_key]

        left_prefix = _infer_arm_prefix(left_ee_body)
        right_prefix = _infer_arm_prefix(right_ee_body)

        left_arm = _get_arm_joint_state(robot_left, env_id, self.cfg.arm_dof, left_prefix)
        right_arm = _get_arm_joint_state(robot_right, env_id, self.cfg.arm_dof, right_prefix)
        left_grip = [
            _gripper_width_to_unit(_get_ee_joint_state(robot_left, env_id, left_prefix)[0], self.cfg.gripper_open_width)
        ]
        right_grip = [
            _gripper_width_to_unit(_get_ee_joint_state(robot_right, env_id, right_prefix)[0], self.cfg.gripper_open_width)
        ]

        left_ee_pose = _pose7_isaac_to_sapien(_find_body_pose7(robot_left, left_ee_body, env_id))
        right_ee_pose = _pose7_isaac_to_sapien(_find_body_pose7(robot_right, right_ee_body, env_id))
        left_tcp_pose = (
            _pose7_isaac_to_sapien(_find_body_pose7(robot_left, left_tcp_body, env_id)) if left_tcp_body else left_ee_pose
        )
        right_tcp_pose = (
            _pose7_isaac_to_sapien(_find_body_pose7(robot_right, right_tcp_body, env_id)) if right_tcp_body else right_ee_pose
        )

        left_delta = _delta_pose7(left_ee_pose, self.prev_left_ee_pose)
        right_delta = _delta_pose7(right_ee_pose, self.prev_right_ee_pose)
        self.prev_left_ee_pose = list(left_ee_pose)
        self.prev_right_ee_pose = list(right_ee_pose)

        self.state_left_arm_joint_states.append(left_arm)
        self.state_left_ee_joint_states.append(left_grip)
        self.state_left_ee_poses.append(left_ee_pose)
        self.state_left_tcp_poses.append(left_tcp_pose)
        self.state_left_delta_ee_poses.append(left_delta)

        self.state_right_arm_joint_states.append(right_arm)
        self.state_right_ee_joint_states.append(right_grip)
        self.state_right_ee_poses.append(right_ee_pose)
        self.state_right_tcp_poses.append(right_tcp_pose)
        self.state_right_delta_ee_poses.append(right_delta)

        self.action_left_arm_joint_states.append(left_arm)
        self.action_left_ee_joint_states.append(left_grip)
        self.action_left_ee_poses.append(left_ee_pose)
        self.action_left_tcp_poses.append(left_tcp_pose)
        self.action_left_delta_ee_poses.append(left_delta)

        self.action_right_arm_joint_states.append(right_arm)
        self.action_right_ee_joint_states.append(right_grip)
        self.action_right_ee_poses.append(right_ee_pose)
        self.action_right_tcp_poses.append(right_tcp_pose)
        self.action_right_delta_ee_poses.append(right_delta)

        self._append_camera_sample(
            cam_head,
            env_id,
            self.cam_head_colors,
            self.cam_head_intrinsic,
            self.cam_head_cam2world_gl,
            self.cam_head_extrinsic_cv,
            "cam_head_shape",
            self,
        )

    @staticmethod
    def _write_robotwin_camera(
        g_obs,
        name: str,
        colors: List[np.ndarray],
        intrinsics: List[np.ndarray],
        cam2world_gl: List[np.ndarray],
        extrinsic_cv: List[np.ndarray],
    ):
        g = g_obs.create_group(name)
        encoded = []
        n_img = len(colors)
        print(f"[INFO] encoding {name} rgb: {n_img} frames...", flush=True)
        for i, c in enumerate(colors):
            encoded.append(_encode_rgb_bytes(c))
            if n_img > 0 and ((i + 1) % 200 == 0 or (i + 1) == n_img):
                print(f"[INFO]   {name} png {i + 1}/{n_img}", flush=True)
        g.create_dataset("rgb", data=_bytes_to_fixed_array(encoded))
        g.create_dataset("intrinsic_cv", data=_stack_or_empty(intrinsics, (3, 3), np.float32))
        g.create_dataset("cam2world_gl", data=_stack_or_empty(cam2world_gl, (4, 4), np.float32))
        g.create_dataset("extrinsic_cv", data=_stack_or_empty(extrinsic_cv, (3, 4), np.float32))

    def dump_episode(self, episode_index: Optional[int] = None) -> Optional[int]:
        T = len(self.state_left_arm_joint_states)
        if T == 0:
            return None

        if episode_index is None:
            episode_index = allocate_episode_index(self.cfg.output_root)
        episode_index = int(episode_index)

        left_arm = np.asarray(self.state_left_arm_joint_states, dtype=np.float64)
        right_arm = np.asarray(self.state_right_arm_joint_states, dtype=np.float64)
        left_grip = np.asarray(self.state_left_ee_joint_states, dtype=np.float64).reshape(T)
        right_grip = np.asarray(self.state_right_ee_joint_states, dtype=np.float64).reshape(T)
        left_ee = np.asarray(self.state_left_ee_poses, dtype=np.float64)
        right_ee = np.asarray(self.state_right_ee_poses, dtype=np.float64)
        vector = np.concatenate(
            [left_arm, left_grip[:, None], right_arm, right_grip[:, None]],
            axis=1,
        )

        out_path = os.path.join(self.cfg.output_root, "data", f"episode{episode_index}.hdf5")
        with h5py.File(out_path, "w") as f:
            g_end = f.create_group("endpose")
            g_end.create_dataset("left_endpose", data=left_ee)
            g_end.create_dataset("right_endpose", data=right_ee)
            g_end.create_dataset("left_gripper", data=left_grip)
            g_end.create_dataset("right_gripper", data=right_grip)

            g_ja = f.create_group("joint_action")
            g_ja.create_dataset("left_arm", data=left_arm)
            g_ja.create_dataset("right_arm", data=right_arm)
            g_ja.create_dataset("left_gripper", data=left_grip)
            g_ja.create_dataset("right_gripper", data=right_grip)
            g_ja.create_dataset("vector", data=vector)

            g_obs = f.create_group("observation")
            self._write_robotwin_camera(
                g_obs,
                "head_camera",
                self.cam_head_colors,
                self.cam_head_intrinsic,
                self.cam_head_cam2world_gl,
                self.cam_head_extrinsic_cv,
            )
            f.create_dataset("pointcloud", data=np.zeros((T, 0), dtype=np.float64))

        left_path = list(self.left_joint_path)
        right_path = list(self.right_joint_path)
        if not left_path and not right_path:
            left_path = self._joints_to_path_segments(left_arm.astype(np.float32))
            right_path = self._joints_to_path_segments(right_arm.astype(np.float32))
        traj_path = os.path.join(self.cfg.output_root, "_traj_data", f"episode{episode_index}.pkl")
        with open(traj_path, "wb") as f:
            pickle.dump({"left_joint_path": left_path, "right_joint_path": right_path}, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.episode_lengths.append(T)
        _append_episodes_jsonl(
            self.cfg.output_root,
            {
                "episode_index": episode_index,
                "tasks": [self.cfg.task_text],
                "length": int(T),
                "path": f"data/episode{episode_index}.hdf5",
                "traj_path": f"_traj_data/episode{episode_index}.pkl",
                "video_path": f"video/episode{episode_index}.mp4",
            },
        )
        self._reset_episode_buffers()
        return episode_index

    def dump_info_json(self):
        info = {
            "codebase_version": "v2.1",
            "robot_type": "sim_only_dual_arm",
            "total_episodes": len(self.episode_lengths),
            "total_frames": int(sum(self.episode_lengths)),
            "fps": self.cfg.fps,
            "data_path": "data/episode{episode_index}.hdf5",
            "traj_path": "_traj_data/episode{episode_index}.pkl",
            "video_path": "video/episode{episode_index}.mp4",
            "format": "robotwin_aloha_agilex",
            "world_frame": "sapien_z_up_yaw90_from_isaac",
        }
        with open(os.path.join(self.cfg.output_root, "meta", "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


class EpisodeVideoWriter:
    def __init__(self, dataset_root: str, fps: int = 30):
        self.dataset_root = dataset_root
        self.fps = fps
        self.frames_head = []
        self.frames_hand_left = []
        self.frames_hand_right = []
        self.frames_side = []
        self._prepare_dirs()

    def _prepare_dirs(self):
        os.makedirs(os.path.join(self.dataset_root, "video"), exist_ok=True)

    def append_from_scene(
        self,
        scene,
        env_id: int = 0,
        cam_head_key: str = "head_camera",
        cam_hand_left_key: str = "hand_left_camera",
        cam_hand_right_key: str = "hand_right_camera",
        cam_side_key: str = "side_camera",
    ):
        cam_head = scene[cam_head_key]
        cam_left = scene[cam_hand_left_key]
        cam_right = scene[cam_hand_right_key]
        cam_side = scene[cam_side_key]

        head = _copy_camera_rgb(cam_head, env_id)
        left = _copy_camera_rgb(cam_left, env_id)
        right = _copy_camera_rgb(cam_right, env_id)
        side = _copy_camera_rgb(cam_side, env_id)

        self.frames_head.append(head)
        self.frames_hand_left.append(left)
        self.frames_hand_right.append(right)
        self.frames_side.append(side)

    def discard_episode(self):
        """Drop buffered frames without writing video."""
        self.frames_head.clear()
        self.frames_hand_left.clear()
        self.frames_hand_right.clear()
        self.frames_side.clear()

    def dump_episode(self, episode_index: int):
        n = min(
            len(self.frames_head),
            len(self.frames_hand_left),
            len(self.frames_hand_right),
            len(self.frames_side),
        )
        if n == 0:
            self.discard_episode()
            return

        stride = max(1, int(round(self.fps / 20.0)))
        idxs = list(range(0, n, stride))
        mosaic = [
            _tile_third_left_stack_right(
                self.frames_side[i],
                self.frames_head[i],
                self.frames_hand_left[i],
                self.frames_hand_right[i],
            )
            for i in idxs
        ]
        singles = {
            "third": [self.frames_side[i] for i in idxs],
            "head": [self.frames_head[i] for i in idxs],
            "hand_left": [self.frames_hand_left[i] for i in idxs],
            "hand_right": [self.frames_hand_right[i] for i in idxs],
        }
        video_dir = os.path.join(self.dataset_root, "video")
        ep = int(episode_index)
        try:
            _write_mp4(os.path.join(video_dir, f"episode{ep}.mp4"), mosaic, self.fps)
            for name, frames in singles.items():
                _write_mp4(os.path.join(video_dir, f"episode{ep}_{name}.mp4"), frames, self.fps)
        except Exception as exc:
            print(f"[WARN] video dump failed: {exc}")
        self.discard_episode()
