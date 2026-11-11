import json
import os
import pickle
import re
import fcntl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation as R


_ACTIVE_COLLECTOR = None

# openpi / π0.5 reads a LeRobot dataset. The model uses the head and two wrist
# cameras. The third-person view is not recorded. This robot is a dual Franka,
# so state and action stay 7 joints plus one gripper per arm, not Aloha's 14.
_PI05_CAMERAS = {
    "head": "observation.images.cam_high",
    "hand_left": "observation.images.cam_left_wrist",
    "hand_right": "observation.images.cam_right_wrist",
}
_LEROBOT_CHUNK = 1000


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


def render_head_grasp_annotation(
    image: np.ndarray,
    cam_pos,
    cam_quat_wxyz,
    intrinsic,
    objects: List[dict],
    grasps: List[dict],
    cam_rot=None,
) -> np.ndarray:
    """Draw collision boxes, grasp approach, and affordance on one head-camera frame.

    ``cam_rot`` columns are the camera axes in world: +X right, +Y down, +Z forward.
    ``cam_quat_wxyz`` is that same ROS orientation when ``cam_rot`` is omitted.
    """
    from PIL import ImageDraw

    canvas = np.ascontiguousarray(image[..., :3], dtype=np.uint8).copy()
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    pil = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil)
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    if cam_rot is None:
        rot_wc = _quat_wxyz_to_rotmat(np.asarray(cam_quat_wxyz, dtype=np.float64))
    else:
        rot_wc = np.asarray(cam_rot, dtype=np.float64).reshape(3, 3)

    def project(points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cam = (pts - cam_pos) @ rot_wc
        z = cam[:, 2]
        valid = z > 0.02
        uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        uv[valid, 0] = K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]
        uv[valid, 1] = K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]
        return uv, valid

    def _seg(a, b, color, width_px=2):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return
        draw.line([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))], fill=color, width=width_px)

    palette = (
        (80, 220, 120),
        (80, 170, 255),
        (255, 180, 60),
        (255, 90, 160),
        (180, 120, 255),
        (80, 230, 220),
    )
    edges = (
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for index, obj in enumerate(objects):
        mn = np.asarray(obj["mins"], dtype=np.float64)
        mx = np.asarray(obj["maxs"], dtype=np.float64)
        corners = np.array(
            [
                [mx[0] if i & 1 else mn[0], mx[1] if i & 2 else mn[1], mx[2] if i & 4 else mn[2]]
                for i in range(8)
            ],
            dtype=np.float64,
        )
        uv, valid = project(corners)
        color = palette[index % len(palette)]
        for i, j in edges:
            if valid[i] and valid[j]:
                _seg(uv[i], uv[j], color, 2)
        seen = uv[valid]
        if len(seen) >= 2:
            x0, y0 = seen.min(axis=0)
            x1, y1 = seen.max(axis=0)
            draw.rectangle([float(x0), float(y0), float(x1), float(y1)], outline=color, width=1)
            label = str(obj.get("name", "object"))
            draw.text((float(x0) + 2, max(0.0, float(y0) - 12)), label, fill=color)

    for grasp_i, grasp in enumerate(grasps):
        pos = np.asarray(grasp["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(grasp["quat"], dtype=np.float64).reshape(4)
        hand_rot = _quat_wxyz_to_rotmat(quat)
        approach = hand_rot[:, 2]
        fingers = hand_rot[:, 1]
        affordance = np.asarray(grasp.get("affordance", pos), dtype=np.float64).reshape(3)
        tail = pos - approach * 0.12
        jaw_a = pos - fingers * 0.035
        jaw_b = pos + fingers * 0.035
        pts, valid = project(np.stack([tail, pos, affordance, jaw_a, jaw_b], axis=0))
        if valid[0] and valid[1]:
            _seg(pts[0], pts[1], (255, 60, 60), 3)
            direction = pts[1] - pts[0]
            norm = float(np.linalg.norm(direction))
            if norm > 1.0:
                direction = direction / norm
                side = np.array([-direction[1], direction[0]])
                head = pts[1]
                left = head - direction * 10 + side * 5
                right = head - direction * 10 - side * 5
                draw.polygon(
                    [(float(head[0]), float(head[1])), (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))],
                    fill=(255, 60, 60),
                )
        if valid[3] and valid[4]:
            _seg(pts[3], pts[4], (255, 230, 80), 3)
        if valid[2]:
            x, y = float(pts[2, 0]), float(pts[2, 1])
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 40, 40), outline=(255, 255, 255))
            tag = f"{grasp.get('name', '')} {grasp.get('atom', '')} {grasp.get('arm', '')}".strip()
            draw.text((x + 8, y - 8 + grasp_i * 16), tag, fill=(255, 230, 80))

    draw.rectangle([0, 0, width, 22], fill=(0, 0, 0))
    draw.text((6, 4), "box=collision  red arrow=grasp +Z  yellow=fingers  red dot=affordance", fill=(255, 255, 255))
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


def _project_world(points, cam_pos, cam_rot, intrinsic):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = (pts - np.asarray(cam_pos, dtype=np.float64).reshape(3)) @ np.asarray(cam_rot, dtype=np.float64).reshape(3, 3)
    z = cam[:, 2]
    valid = z > 0.02
    K = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]
    return uv, valid


def _aabb_corners(mins, maxs):
    mn = np.asarray(mins, dtype=np.float64)
    mx = np.asarray(maxs, dtype=np.float64)
    return np.array(
        [[mx[0] if i & 1 else mn[0], mx[1] if i & 2 else mn[1], mx[2] if i & 4 else mn[2]] for i in range(8)],
        dtype=np.float64,
    )


def _screen_rect(obj, cam_pos, cam_rot, intrinsic):
    """Flat image rectangle around the object, not the 3D box edges."""
    pts = obj.get("points")
    if pts is None:
        pts = _aabb_corners(obj["mins"], obj["maxs"])
    uv, valid = _project_world(pts, cam_pos, cam_rot, intrinsic)
    seen = uv[valid]
    if len(seen) < 2:
        return None
    x0, y0 = seen.min(axis=0)
    x1, y1 = seen.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _draw_arrow(draw, start, end, color, width_px=3):
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return
    draw.line(
        [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))],
        fill=color,
        width=width_px,
    )
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 6.0:
        return
    direction = direction / norm
    side = np.array([-direction[1], direction[0]])
    head = np.asarray(end, dtype=np.float64)
    left = head - direction * 12 + side * 6
    right = head - direction * 12 - side * 6
    draw.polygon(
        [(float(head[0]), float(head[1])), (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))],
        fill=color,
    )


_TRACK_COLORS = (
    (80, 220, 120),
    (80, 170, 255),
    (255, 180, 60),
    (255, 90, 160),
    (180, 120, 255),
)


def _draw_grasp_marks(draw, grasps, cam_pos, cam_rot, intrinsic):
    """Red arrow is the grasp direction, yellow bar the jaw, red dot the affordance."""
    for grasp_i, grasp in enumerate(grasps or []):
        pos = np.asarray(grasp["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(grasp["quat"], dtype=np.float64).reshape(4)
        hand_rot = _quat_wxyz_to_rotmat(quat)
        approach = hand_rot[:, 2]
        fingers = hand_rot[:, 1]
        raw_affordance = grasp.get("affordance", pos)
        if raw_affordance is None:
            raw_affordance = pos
        affordance = np.asarray(raw_affordance, dtype=np.float64).reshape(3)
        tail = pos - approach * 0.12
        jaw_a = pos - fingers * 0.035
        jaw_b = pos + fingers * 0.035
        uv, valid = _project_world(np.stack([tail, pos, affordance, jaw_a, jaw_b], axis=0), cam_pos, cam_rot, intrinsic)
        if valid[0] and valid[1]:
            _draw_arrow(draw, uv[0], uv[1], (255, 60, 60), 3)
        if valid[3] and valid[4]:
            draw.line(
                [(float(uv[3, 0]), float(uv[3, 1])), (float(uv[4, 0]), float(uv[4, 1]))],
                fill=(255, 230, 80),
                width=3,
            )
        if valid[2] and grasp.get("affordance") is not None:
            x, y = float(uv[2, 0]), float(uv[2, 1])
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 40, 40), outline=(255, 255, 255))
            if grasp.get("label", True):
                tag = f"{grasp.get('name', '')} {grasp.get('atom', '')} {grasp.get('arm', '')}".strip()
                draw.text((x + 8, y - 8 + grasp_i * 16), tag, fill=(255, 230, 80))


def _annotate_object_boxes(image, objects, cam_pos, cam_rot, intrinsic, labels):
    """2D rectangle and category name for each object."""
    from PIL import ImageDraw

    pil = Image.fromarray(np.ascontiguousarray(image[..., :3], dtype=np.uint8).copy(), mode="RGB")
    draw = ImageDraw.Draw(pil)
    for index, obj in enumerate(objects):
        rect = _screen_rect(obj, cam_pos, cam_rot, intrinsic)
        if rect is None:
            continue
        color = _TRACK_COLORS[index % len(_TRACK_COLORS)]
        x0, y0, x1, y1 = rect
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = str(labels.get(obj["name"], obj["name"]))
        draw.text((x1 + 8, y0), label, fill=color)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


class ThirdViewObjectTracker:
    """Third-person boxes that follow objects, plus a first-frame trajectory still.

    The still is the first third-person frame: boxes at the start, arrows for the
    path each object takes afterwards. The video keeps a box on each object.
    """

    def __init__(self, dataset_root: str, task_name: str, id_to_type: dict | None = None):
        safe = re.sub(r"[^\w.\-]+", "_", str(task_name or "task")).strip("._") or "task"
        self.task_name = safe
        self.id_to_type = dict(id_to_type or {})
        self.out_dir = Path(dataset_root) / "annotation"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.first_frame = None
        self.first_objects = None
        self.tracks: dict[str, list] = {}
        self._writer = None
        self._grasp_writer = None
        self._video_tmp = self.out_dir / f".{safe}_third_track_{os.getpid()}.mp4"
        self._grasp_video_tmp = self.out_dir / f".{safe}_third_track_grasp_{os.getpid()}.mp4"
        self._n = 0
        self._cam_pos = None
        self._cam_rot = None
        self._intrinsic = None
        self.grasps = []
        self._grasp_keys = set()

    def _label(self, asset_id: str) -> str:
        kind = self.id_to_type.get(asset_id)
        if kind:
            return str(kind)
        return re.sub(r"_\d+$", "", str(asset_id))

    def _labels(self, objects) -> dict:
        return {obj["name"]: self._label(obj["name"]) for obj in objects}

    def _skip_object(self, obj) -> bool:
        name = str(obj.get("name", "")).lower()
        kind = self._label(obj.get("name", "")).lower()
        return "table" in name or kind == "table"

    def note_grasps(self, grasps):
        for grasp in grasps or []:
            pos = np.asarray(grasp["pos"], dtype=np.float64).reshape(3)
            key = (
                str(grasp.get("name", "")),
                str(grasp.get("atom", "")),
                str(grasp.get("arm", "")),
                round(float(pos[0]), 3),
                round(float(pos[1]), 3),
                round(float(pos[2]), 3),
            )
            if key in self._grasp_keys:
                continue
            self._grasp_keys.add(key)
            kept = dict(grasp)
            kept["pos"] = pos.copy()
            kept["quat"] = np.asarray(grasp["quat"], dtype=np.float64).reshape(4).copy()
            kept["affordance"] = np.asarray(grasp.get("affordance", pos), dtype=np.float64).reshape(3).copy()
            self.grasps.append(kept)

    def _open_writer(self, path):
        try:
            return imageio.get_writer(
                str(path),
                fps=20,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=2,
                ffmpeg_params=["-threads", "1", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium"],
            )
        except TypeError:
            return imageio.get_writer(str(path), fps=20)

    def observe(self, rgb, cam_pos, cam_rot, intrinsic, objects, grasps=None, live_hands=None):
        movable = [obj for obj in objects if not self._skip_object(obj)]
        if not movable:
            return
        frame = np.asarray(rgb[..., :3])
        if frame.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(frame)) <= 1.5 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if self.first_frame is None:
            if float(frame.mean()) < 2.0:
                self._dark_frames = getattr(self, "_dark_frames", 0) + 1
                if self._dark_frames == 1 or self._dark_frames % 200 == 0:
                    print(
                        "[WARN] third-view frame is black, annotation not started. "
                        "Launch with --enable_cameras.",
                        flush=True,
                    )
                return
            self.first_frame = frame.copy()
            self.first_objects = [dict(obj) for obj in movable]
            self._cam_pos = np.asarray(cam_pos, dtype=np.float64).copy()
            self._cam_rot = np.asarray(cam_rot, dtype=np.float64).copy()
            self._intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
        self.note_grasps(grasps)
        for obj in movable:
            center = (np.asarray(obj["mins"], dtype=np.float64) + np.asarray(obj["maxs"], dtype=np.float64)) * 0.5
            hist = self.tracks.setdefault(str(obj["name"]), [])
            if not hist or float(np.linalg.norm(center - hist[-1])) > 0.01:
                hist.append(center.copy())
        annotated = _annotate_object_boxes(frame, movable, cam_pos, cam_rot, intrinsic, self._labels(movable))
        annotated = _even_hw_frame(annotated)
        grasp_frame = annotated
        if live_hands:
            from PIL import ImageDraw

            pil = Image.fromarray(annotated, mode="RGB")
            draw = ImageDraw.Draw(pil)
            _draw_grasp_marks(draw, live_hands, cam_pos, cam_rot, intrinsic)
            grasp_frame = _even_hw_frame(np.ascontiguousarray(np.asarray(pil, dtype=np.uint8)))
        if self._writer is None:
            self._writer = self._open_writer(self._video_tmp)
            self._grasp_writer = self._open_writer(self._grasp_video_tmp)
        self._writer.append_data(annotated)
        self._grasp_writer.append_data(grasp_frame)
        self._n += 1

    def finish(self):
        cam_pos, cam_rot, intrinsic = self._cam_pos, self._cam_rot, self._intrinsic
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                print(f"[WARN] track video close failed: {exc}")
            self._writer = None
        if self._grasp_writer is not None:
            try:
                self._grasp_writer.close()
            except Exception as exc:
                print(f"[WARN] grasp video close failed: {exc}")
            self._grasp_writer = None
        video_path = self.out_dir / f"{self.task_name}_third_track.mp4"
        grasp_video_path = self.out_dir / f"{self.task_name}_third_track_grasp.mp4"
        if self._n > 0 and self._video_tmp.is_file():
            os.replace(self._video_tmp, video_path)
            print(f"[INFO] third-view track video {video_path} frames={self._n}", flush=True)
        else:
            self._video_tmp.unlink(missing_ok=True)
        if self._n > 0 and self._grasp_video_tmp.is_file():
            os.replace(self._grasp_video_tmp, grasp_video_path)
            print(f"[INFO] third-view grasp video {grasp_video_path} frames={self._n}", flush=True)
        else:
            self._grasp_video_tmp.unlink(missing_ok=True)
        if self.first_frame is None or not self.first_objects or cam_pos is None:
            print(
                f"[WARN] third-view annotation not saved under {self.out_dir}: "
                "no side-camera frame was recorded",
                flush=True,
            )
            return
        from PIL import ImageDraw

        still = _annotate_object_boxes(
            self.first_frame,
            self.first_objects,
            cam_pos,
            cam_rot,
            intrinsic,
            self._labels(self.first_objects),
        )
        pil = Image.fromarray(still, mode="RGB")
        draw = ImageDraw.Draw(pil)
        for index, obj in enumerate(self.first_objects):
            hist = self.tracks.get(str(obj["name"]), [])
            if len(hist) < 2:
                continue
            color = _TRACK_COLORS[index % len(_TRACK_COLORS)]
            uv, valid = _project_world(np.stack(hist, axis=0), cam_pos, cam_rot, intrinsic)
            pts = [uv[i] for i in range(len(hist)) if valid[i]]
            if len(pts) < 2:
                continue
            for a, b in zip(pts, pts[1:]):
                draw.line(
                    [(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))],
                    fill=color,
                    width=3,
                )
            last_arrow = pts[0]
            for pt in pts[1:]:
                if float(np.linalg.norm(pt - last_arrow)) >= 40.0 or pt is pts[-1]:
                    _draw_arrow(draw, last_arrow, pt, color, 3)
                    last_arrow = pt
        out = self.out_dir / f"{self.task_name}_third_first.png"
        Image.fromarray(np.ascontiguousarray(np.asarray(pil, dtype=np.uint8)), mode="RGB").save(out, format="PNG")
        print(f"[INFO] third-view first frame {out}", flush=True)
        if self.grasps:
            _draw_grasp_marks(draw, self.grasps, cam_pos, cam_rot, intrinsic)
            grasp_out = self.out_dir / f"{self.task_name}_third_first_grasp.png"
            Image.fromarray(np.ascontiguousarray(np.asarray(pil, dtype=np.uint8)), mode="RGB").save(grasp_out, format="PNG")
            print(f"[INFO] third-view grasp frame {grasp_out}", flush=True)


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


# Kept for the SAPIEN/RoboTwin yaw. Export does not apply it while this is False.
# Set True to rotate exported xyz/quat by +90° yaw before writing.
EXPORT_ISAAC_TO_SAPIEN = False

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


def _export_pose7(pose7: List[float]) -> List[float]:
    if EXPORT_ISAAC_TO_SAPIEN:
        return _pose7_isaac_to_sapien(pose7)
    return [float(x) for x in pose7[:7]]


def _env_origin_xyz(scene, env_id: int) -> np.ndarray:
    """Isaac world position of this env. A single env is the zero vector."""
    origins = getattr(scene, "env_origins", None)
    if origins is None:
        return np.zeros(3, dtype=np.float32)
    arr = _to_np(origins).astype(np.float32)
    if arr.ndim != 2 or env_id < 0 or env_id >= int(arr.shape[0]):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(arr[int(env_id), :3], dtype=np.float32).reshape(3)


def _pose7_minus_origin(pose7: List[float], origin_xyz: np.ndarray) -> List[float]:
    origin = np.asarray(origin_xyz, dtype=np.float32).reshape(3)
    return [
        float(pose7[0]) - float(origin[0]),
        float(pose7[1]) - float(origin[1]),
        float(pose7[2]) - float(origin[2]),
        float(pose7[3]),
        float(pose7[4]),
        float(pose7[5]),
        float(pose7[6]),
    ]


def _T_isaac_to_sapien(T_w_c: np.ndarray) -> np.ndarray:
    return (_ISAAC_TO_SAPIEN_T @ np.asarray(T_w_c, dtype=np.float32)).astype(np.float32)


def _export_cam_gl(T_isaac: np.ndarray) -> np.ndarray:
    if EXPORT_ISAAC_TO_SAPIEN:
        return _T_isaac_to_sapien(T_isaac)
    return np.asarray(T_isaac, dtype=np.float32)


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


def _camera_matrices(camera, env_id: int, origin_xyz=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OpenCV K, cam2world_gl (4x4), and extrinsic_cv (3x4).

    ``origin_xyz`` is this env's Isaac origin. Positions are relative to it.
    The SAPIEN +90° yaw is applied only when EXPORT_ISAAC_TO_SAPIEN is True.
    """
    _refresh_camera_pose(camera, env_id)
    K = _to_np(camera.data.intrinsic_matrices[env_id]).astype(np.float32)
    c_pos = _to_np(camera.data.pos_w[env_id]).astype(np.float32)
    if origin_xyz is not None:
        c_pos = c_pos - np.asarray(origin_xyz, dtype=np.float32).reshape(3)
    c_quat = _camera_quat_opengl(camera, env_id)
    T_w_c_gl_isaac = _pose7_to_T(c_pos, c_quat).astype(np.float32)
    T_w_c_gl = _export_cam_gl(T_w_c_gl_isaac)
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


def _tile_main_and_stack(main_frame: np.ndarray, *panels: np.ndarray) -> np.ndarray:
    """Large frame on the left, the other cameras stacked on the right."""
    th, tw = int(main_frame.shape[0]), int(main_frame.shape[1])
    n = max(len(panels), 1)
    stack_h = _even(max(th // n, 2))
    canvas_h = stack_h * n
    main_w = _even(max(int(round(canvas_h * (tw / max(th, 1)))), 2))
    ref = panels[0] if panels else main_frame
    sh, sw = int(ref.shape[0]), int(ref.shape[1])
    stack_w = _even(max(int(round(stack_h * (sw / max(sh, 1)))), 2))
    main = _resize_rgb(main_frame, (canvas_h, main_w))
    col = np.concatenate([_resize_rgb(panel, (stack_h, stack_w)) for panel in panels], axis=0)
    return _even_hw_frame(np.concatenate([main, col], axis=1))


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
        (root / "videos" / "chunk-000" / "observation.images.cam_high", r"episode_(\d+)\."),
    ]
    max_idx = -1
    for folder, pat in patterns:
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            match = re.match(pat, file_path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    data_root = root / "data"
    if data_root.exists():
        for file_path in data_root.glob("chunk-*/episode_*.parquet"):
            match = re.search(r"episode_(\d+)\.parquet$", file_path.name)
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


def _pi05_motor_names(arm_dof: int) -> List[str]:
    left = [f"left_joint_{i}" for i in range(1, int(arm_dof) + 1)] + ["left_gripper"]
    right = [f"right_joint_{i}" for i in range(1, int(arm_dof) + 1)] + ["right_gripper"]
    return left + right


def _lerobot_chunk(episode_index: int) -> int:
    return int(episode_index) // _LEROBOT_CHUNK


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parquet_episode_paths(dataset_root: Path) -> List[Tuple[int, Path]]:
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        return []
    found = []
    for file_path in data_root.glob("chunk-*/episode_*.parquet"):
        match = re.search(r"episode_(\d+)\.parquet$", file_path.name)
        if match:
            found.append((int(match.group(1)), file_path))
    return sorted(found)


# Task-name words that mean a paymentsign, a shoe box, or a plastic box.
_ASSET_NAME_TOKENS = {
    "paymentsign": ("qrcode", "qrcodes"),
    "shoebox": ("box", "boxes"),
    "plasticbox": ("box", "boxes"),
}


def _operated_name_tokens(task_json: Optional[dict]) -> set:
    """Name tokens for assets that an atomic step actually operates on."""
    tokens = set()
    if not isinstance(task_json, dict):
        return tokens
    id_to_type = {}
    for asset in task_json.get("assets") or []:
        if isinstance(asset, dict) and asset.get("id") and asset.get("type"):
            id_to_type[str(asset["id"])] = str(asset["type"]).lower()
    for step in task_json.get("atomic_plan") or []:
        if not isinstance(step, dict):
            continue
        asset_type = id_to_type.get(str(step.get("asset") or ""))
        if not asset_type:
            continue
        tokens.add(asset_type)
        if asset_type.endswith("y") and not asset_type.endswith("ay"):
            tokens.add(asset_type[:-1] + "ies")
        elif asset_type.endswith(("s", "x", "ch", "sh", "z")):
            tokens.add(asset_type + "es")
        else:
            tokens.add(asset_type + "s")
        tokens.update(_ASSET_NAME_TOKENS.get(asset_type, ()))
    return tokens


def green_object_task_text(task_name: str, task_json: Optional[dict] = None) -> str:
    """Prompt stored in tasks.jsonl.

    The raw task id is not stored. The word object, and every asset the plan
    operates on, is written as green object.
    """
    operated = _operated_name_tokens(task_json)
    generic = {"object", "objects"}
    parts = []
    for token in str(task_name or "").replace("-", "_").split("_"):
        word = token.strip().lower()
        if not word:
            continue
        if word == "a2b":
            phrase = "green object to green object"
        elif word in generic or word in operated:
            phrase = "green object"
        else:
            phrase = word
        if parts and parts[-1].endswith("green object") and phrase == "green object":
            parts.append("and")
        parts.append(phrase)
    return " ".join(parts)


def _task_index_for(dataset_root: Path, task_text: str) -> int:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    for row in tasks:
        if row.get("task") == task_text:
            return int(row["task_index"])
    task_index = len(tasks)
    tasks.append({"task_index": task_index, "task": task_text})
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tasks_path, "w", encoding="utf-8") as fp:
        for row in tasks:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return task_index


def _column_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def write_lerobot_meta(dataset_root: str, fps: int) -> None:
    """Rewrite meta so the folder is a LeRobot dataset of the parquet episodes."""
    root = Path(dataset_root)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    episodes = _parquet_episode_paths(root)
    task_rows = _read_jsonl(meta / "tasks.jsonl")
    task_by_index = {int(row["task_index"]): row.get("task", "") for row in task_rows}
    previous = {
        int(row["episode_index"]): row
        for row in _read_jsonl(meta / "episodes.jsonl")
        if "episode_index" in row and "length" in row and "tasks" in row
    }
    shapes = {}
    shape_path = meta / "video_shapes.json"
    if shape_path.is_file():
        try:
            shapes = json.loads(shape_path.read_text(encoding="utf-8")).get("cameras") or {}
        except json.JSONDecodeError:
            shapes = {}

    episode_rows = []
    stats_rows = []
    total_frames = 0
    state_dim = None
    for episode_index, path in episodes:
        table = pq.read_table(path, columns=["observation.state", "action", "task_index"])
        length = int(table.num_rows)
        state = np.stack(table.column("observation.state").to_pylist(), axis=0)
        action = np.stack(table.column("action").to_pylist(), axis=0)
        state_dim = int(state.shape[1])
        task_index = int(table.column("task_index")[0].as_py()) if length else 0
        if task_index in task_by_index:
            tasks = [task_by_index[task_index]]
        else:
            kept = previous.get(episode_index) or {}
            tasks = list(kept.get("tasks") or [""])
        episode_rows.append({
            "episode_index": episode_index,
            "tasks": tasks,
            "length": length,
        })
        stats_rows.append({
            "episode_index": episode_index,
            "stats": {
                "observation.state": _column_stats(state),
                "action": _column_stats(action),
            },
        })
        total_frames += length

    motors = _pi05_motor_names(max(int(state_dim or 2) // 2 - 1, 1))
    if state_dim is not None and len(motors) != state_dim:
        motors = [f"motor_{i}" for i in range(state_dim)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(motors)], "names": [motors]},
        "action": {"dtype": "float32", "shape": [len(motors)], "names": [motors]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key, shape in shapes.items():
        height = int(shape["height"])
        width = int(shape["width"])
        features[video_key] = {
            "dtype": "video",
            "shape": [3, height, width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": int(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    n_episodes = len(episode_rows)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "franka_dual",
        "total_episodes": n_episodes,
        "total_frames": int(total_frames),
        "total_tasks": len(task_rows),
        "total_videos": n_episodes * len(shapes),
        "total_chunks": (max((ep for ep, _ in episodes), default=0) // _LEROBOT_CHUNK) + 1 if episodes else 0,
        "chunks_size": _LEROBOT_CHUNK,
        "fps": int(fps),
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "action_convention": "absolute joint position; action[t] is state[t+1], and the last frame repeats",
        "state_order": motors,
        "gripper": "0 closed, 1 open",
    }
    lock_path = meta / "episodes_jsonl.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            with open(meta / "episodes.jsonl", "w", encoding="utf-8") as fp:
                for row in episode_rows:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(meta / "episodes_stats.jsonl", "w", encoding="utf-8") as fp:
                for row in stats_rows:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(meta / "info.json", "w", encoding="utf-8") as fp:
                json.dump(info, fp, ensure_ascii=False, indent=2)
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
    Writes a LeRobot v2 dataset for openpi / π0.5:
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.cam_*/episode_000000.mp4
      meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl

    observation.state and action are absolute joint positions:
    left 7 joints, left gripper, right 7 joints, right gripper.
    action[t] is state[t+1]; the last frame repeats state[t].
    Gripper is 0 when closed and 1 when open.
    Poses used elsewhere stay in the Isaac env-origin frame.
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
        origin_xyz=None,
    ):
        K, T_gl, extr_cv = _camera_matrices(camera, env_id, origin_xyz)
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

        origin = _env_origin_xyz(scene, env_id)

        def _local_pose(robot, body_name: str) -> List[float]:
            pose7 = _pose7_minus_origin(_find_body_pose7(robot, body_name, env_id), origin)
            return _export_pose7(pose7)

        left_ee_pose = _local_pose(robot_left, left_ee_body)
        right_ee_pose = _local_pose(robot_right, right_ee_body)
        left_tcp_pose = _local_pose(robot_left, left_tcp_body) if left_tcp_body else left_ee_pose
        right_tcp_pose = _local_pose(robot_right, right_tcp_body) if right_tcp_body else right_ee_pose

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
        del cam_head, origin

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

        left_arm = np.asarray(self.state_left_arm_joint_states, dtype=np.float32)
        right_arm = np.asarray(self.state_right_arm_joint_states, dtype=np.float32)
        left_grip = np.asarray(self.state_left_ee_joint_states, dtype=np.float32).reshape(T, 1)
        right_grip = np.asarray(self.state_right_ee_joint_states, dtype=np.float32).reshape(T, 1)
        state = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=1)
        action = np.concatenate([state[1:], state[-1:]], axis=0)

        root = Path(self.cfg.output_root)
        task_index = _task_index_for(root, self.cfg.task_text)
        prior_rows = 0
        for other_index, other_path in _parquet_episode_paths(root):
            if other_index < episode_index:
                prior_rows += int(pq.read_metadata(other_path).num_rows)
        chunk = _lerobot_chunk(episode_index)
        out_dir = root / "data" / f"chunk-{chunk:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{episode_index:06d}.parquet"
        dim = int(state.shape[1])
        vector_type = pa.list_(pa.float32(), dim)
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=vector_type),
            "action": pa.array(action.tolist(), type=vector_type),
            "timestamp": pa.array((np.arange(T, dtype=np.float64) / float(self.cfg.fps)).astype(np.float32)),
            "frame_index": pa.array(np.arange(T, dtype=np.int64)),
            "episode_index": pa.array(np.full(T, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(prior_rows, prior_rows + T, dtype=np.int64)),
            "task_index": pa.array(np.full(T, task_index, dtype=np.int64)),
        })
        pq.write_table(table, out_path)
        print(f"[INFO] lerobot episode {out_path} frames={T}", flush=True)

        left_path = list(self.left_joint_path)
        right_path = list(self.right_joint_path)
        if not left_path and not right_path:
            left_path = self._joints_to_path_segments(left_arm)
            right_path = self._joints_to_path_segments(right_arm)
        traj_dir = root / "_traj_data"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"episode{episode_index}.pkl"
        with open(traj_path, "wb") as f:
            pickle.dump({"left_joint_path": left_path, "right_joint_path": right_path}, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.episode_lengths.append(T)
        self._reset_episode_buffers()
        return episode_index

    def dump_info_json(self):
        write_lerobot_meta(self.cfg.output_root, fps=int(self.cfg.fps))


class EpisodeVideoWriter:
    """Write each camera frame to disk as the simulation steps.

    Training cameras are renamed into the LeRobot video tree. The mosaic stays
    in ``video/`` as a preview and is not a training feature.
    """

    _NAMES = {
        "mosaic": "episode{ep}.mp4",
    }

    def __init__(self, dataset_root: str, fps: int = 30, stream_prefix: str = ""):
        self.dataset_root = dataset_root
        self.fps = fps
        self._stream_prefix = stream_prefix
        self._writers: dict = {}
        self._paths: dict = {}
        self._shapes: dict = {}
        self._n = 0
        self._prepare_dirs()

    def _prepare_dirs(self):
        os.makedirs(os.path.join(self.dataset_root, "video"), exist_ok=True)

    def _save_fps(self) -> int:
        return max(1, int(self.fps))

    def _open_writer(self, key: str, frame: np.ndarray):
        video_dir = os.path.join(self.dataset_root, "video")
        path = os.path.join(video_dir, f".episode_stream_{os.getpid()}_{self._stream_prefix}{key}.mp4")
        save_fps = self._save_fps()
        h, w = int(frame.shape[0]), int(frame.shape[1])
        try:
            writer = imageio.get_writer(
                path,
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
            writer = imageio.get_writer(path, fps=save_fps)
        self._writers[key] = writer
        self._paths[key] = path
        self._shapes[key] = (h, w)
        print(f"[INFO] streaming video {path} {w}x{h} fps={save_fps}", flush=True)

    def append_from_scene(
        self,
        scene,
        env_id: int = 0,
        cam_head_key: str = "head_camera",
        cam_hand_left_key: str = "hand_left_camera",
        cam_hand_right_key: str = "hand_right_camera",
        cam_side_key: str = "side_camera",
    ):
        del cam_side_key
        head = _even_hw_frame(_copy_camera_rgb(scene[cam_head_key], env_id))
        left = _even_hw_frame(_copy_camera_rgb(scene[cam_hand_left_key], env_id))
        right = _even_hw_frame(_copy_camera_rgb(scene[cam_hand_right_key], env_id))
        frames = {
            "mosaic": _tile_main_and_stack(head, left, right),
            "head": head,
            "hand_left": left,
            "hand_right": right,
        }
        for key, frame in frames.items():
            if key not in self._writers:
                self._open_writer(key, frame)
            self._writers[key].append_data(frame)
        self._n += 1

    def _close_writers(self):
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:
                print(f"[WARN] video writer close failed: {exc}")
        self._writers.clear()

    def discard_episode(self):
        """Close the stream and delete the partial files."""
        self._close_writers()
        for path in self._paths.values():
            try:
                os.remove(path)
            except OSError:
                pass
        self._paths.clear()
        self._shapes.clear()
        self._n = 0

    def dump_episode(self, episode_index: int):
        self._close_writers()
        if self._n == 0:
            self.discard_episode()
            return
        ep = int(episode_index)
        chunk = _lerobot_chunk(ep)
        video_dir = os.path.join(self.dataset_root, "video")
        cameras = {}
        try:
            for key, src in list(self._paths.items()):
                if not src:
                    continue
                if key in _PI05_CAMERAS:
                    video_key = _PI05_CAMERAS[key]
                    dst_dir = os.path.join(
                        self.dataset_root, "videos", f"chunk-{chunk:03d}", video_key
                    )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f"episode_{ep:06d}.mp4")
                    height, width = self._shapes.get(key, (0, 0))
                    cameras[video_key] = {"height": int(height), "width": int(width)}
                else:
                    pattern = self._NAMES.get(key)
                    if pattern is None:
                        continue
                    dst = os.path.join(video_dir, pattern.format(ep=ep))
                os.replace(src, dst)
                print(f"[INFO] video {dst} frames={self._n}", flush=True)
            if cameras:
                meta = Path(self.dataset_root) / "meta"
                meta.mkdir(parents=True, exist_ok=True)
                shape_path = meta / "video_shapes.json"
                payload = {"fps": self._save_fps(), "cameras": {}}
                if shape_path.is_file():
                    try:
                        payload = json.loads(shape_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        payload = {"fps": self._save_fps(), "cameras": {}}
                payload["fps"] = self._save_fps()
                stored = payload.setdefault("cameras", {})
                stored.pop("observation.images.cam_low", None)
                stored.update(cameras)
                shape_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] video dump failed: {exc}")
        self._paths.clear()
        self._shapes.clear()
        self._n = 0
