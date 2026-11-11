#!/usr/bin/env python3
"""Collect the ten wooden-box tasks, one fresh Isaac process per episode.

env_0 is the scene with green markers and is stored under annotated/<task>.
env_1 is the same scene without those markers and is stored under plain/<task>.
Each episode also writes the last third-person frame to a separate preview folder.

Choose the task and the number of episodes at startup:

    python scripts/collect_wooden_box_tasks.py --task click_bell_and_rotate_qrcode --repeats 5

Omit --task to collect all ten wooden-box tasks. Repeat --task to collect several.

The simulation loop stops itself after --max_seconds and saves. If the process
is still alive after that plus --startup_grace, it is closed and the next
episode starts.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path("/home/jiacheng/anaconda3/envs/env_isaaclab/bin/python")
OUTPUT_ROOT = Path("/inspire/hdd/project/robot-dna/inspire_shared/mount/jc-data")

TASKS = (
    "scan_object",
    "stack_bowls_and_stack_blocks",
    "pick_bottle_and_shake_horizontally",
    "open_cabinet_and_put_object_in",
    "place_object_in_basket_and_pick",
    "place_bread_skillet",
    "place_shoe_in_box_and_place_cans_in_box",
    "place_a2b_right_and_grap_roller",
    "open_window_and_handover_microphone",
    "click_bell_and_rotate_qrcode",
)


def _abs(path: str) -> Path:
    folder = Path(path)
    return folder if folder.is_absolute() else PROJECT_ROOT / folder


def _next_episode_index(dataset_root: Path) -> int:
    highest = -1
    data_root = dataset_root / "data"
    if data_root.is_dir():
        for file_path in data_root.glob("chunk-*/episode_*.parquet"):
            stem = file_path.stem
            digits = stem.rsplit("_", 1)[-1]
            if digits.isdigit():
                highest = max(highest, int(digits))
    return highest + 1


def _episode_saved(dataset_root: Path, episode_index: int) -> bool:
    chunk = episode_index // 1000
    path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    return path.is_file()


def _failure_line(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    reason = ""
    for line in lines:
        text = line.strip()
        if text.startswith("RuntimeError:") or text.startswith("Error:"):
            reason = text
    return reason


def _run_once(cmd: list[str], timeout_s: float, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[CMD] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.write(f"[TIMEOUT] closing process group {proc.pid} after {timeout_s:.0f}s\n")
            log.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return proc.wait()
            try:
                return proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                return proc.wait()


def _task_json_exists(name: str) -> bool:
    root = PROJECT_ROOT / "tasks"
    return any(path.is_file() for path in root.rglob(f"{name}.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect the ten wooden-box tasks.")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task name to collect. Repeat for more than one. Default: all ten wooden-box tasks.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many episodes to keep for each selected task.")
    parser.add_argument("--max_seconds", type=int, default=300, help="Stop one episode after this many seconds.")
    parser.add_argument(
        "--startup_grace",
        type=int,
        default=300,
        help="Extra seconds allowed for Isaac to open and close around the episode limit.",
    )
    parser.add_argument("--annotated_root", default=str(OUTPUT_ROOT / "annotated"))
    parser.add_argument("--plain_root", default=str(OUTPUT_ROOT / "plain"))
    parser.add_argument("--preview_dir", default=str(OUTPUT_ROOT / "previews"))
    parser.add_argument("--python", default=str(ISAAC_PYTHON if ISAAC_PYTHON.is_file() else Path(sys.executable)))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    tasks = list(args.task) if args.task else list(TASKS)
    missing = [name for name in tasks if not _task_json_exists(name)]
    if missing:
        known = ", ".join(TASKS)
        raise SystemExit(f"no task json for: {', '.join(missing)}\nwooden-box tasks: {known}")
    print(f"[collect] tasks={', '.join(tasks)} repeats={args.repeats}", flush=True)
    annotated_root = _abs(args.annotated_root)
    plain_root = _abs(args.plain_root)
    preview = _abs(args.preview_dir)
    for folder in (annotated_root, plain_root, preview):
        folder.mkdir(parents=True, exist_ok=True)

    summary = []
    for task in tasks:
        annotated = annotated_root / task
        plain = plain_root / task
        annotated.mkdir(parents=True, exist_ok=True)
        plain.mkdir(parents=True, exist_ok=True)
        episode_index = max(_next_episode_index(annotated), _next_episode_index(plain))
        kept = 0
        attempts = 0
        attempt_limit = args.repeats * 2
        while kept < args.repeats and attempts < attempt_limit:
            attempts += 1
            log_path = OUTPUT_ROOT / "logs" / f"{task}_ep{episode_index:06d}_try{attempts}.log"
            cmd = [
                args.python,
                str(PROJECT_ROOT / "run_task.py"),
                "--task",
                task,
                "--num_envs",
                "2",
                "--record",
                "--enable_cameras",
                "--max_seconds",
                str(args.max_seconds),
                "--episode_index",
                str(episode_index),
                "--dataset_root",
                str(annotated),
                "--dataset_root_env1",
                str(plain),
                "--preview_dir",
                str(preview),
                "--headless",
            ]
            print(
                f"[collect] {task} episode {episode_index} "
                f"try {attempts} kept {kept}/{args.repeats}",
                flush=True,
            )
            code = _run_once(cmd, timeout_s=float(args.max_seconds + args.startup_grace), log_path=log_path)
            saved = _episode_saved(annotated, episode_index) and _episode_saved(plain, episode_index)
            reason = "" if saved else _failure_line(log_path)
            message = f"[collect] {task} episode {episode_index} exit={code} saved={saved}"
            if reason:
                message = f"{message} {reason}"
            print(message, flush=True)
            summary.append({
                "task": task,
                "episode_index": episode_index,
                "attempt": attempts,
                "exit_code": code,
                "saved": saved,
                "reason": reason,
            })
            if saved:
                kept += 1
                episode_index += 1
        if kept < args.repeats:
            print(f"[collect] {task} stopped after {attempts} tries with {kept} saved episodes", flush=True)

    manifest = OUTPUT_ROOT / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as fp:
        for row in summary:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[collect] manifest {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
