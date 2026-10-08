#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import numpy as np
import trimesh
import coacd


def run_coacd_parts(mesh, threshold=0.05, max_convex_hull=24):
    """Compatible with multiple coacd return formats."""
    cmesh = coacd.Mesh(mesh.vertices, mesh.faces) if hasattr(coacd, "Mesh") else mesh
    parts = coacd.run_coacd(
        cmesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode="auto",
    )

    out = []
    for p in parts:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            v, f = p
        elif isinstance(p, dict):
            if "vertices" in p and "faces" in p:
                v, f = p["vertices"], p["faces"]
            elif "V" in p and "F" in p:
                v, f = p["V"], p["F"]
            else:
                raise RuntimeError(f"Unknown CoACD dict keys: {p.keys()}")
        elif hasattr(p, "vertices") and hasattr(p, "faces"):
            v, f = p.vertices, p.faces
        else:
            raise RuntimeError(f"Unknown CoACD part format: {type(p)}")
        pm = trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f), process=False)
        out.append(pm)

    if not out:
        raise RuntimeError("CoACD returned zero parts.")
    return out


def write_urdf(link_name: str, urdf_path: Path, part_rel_paths):
    collisions = []
    for rel in part_rel_paths:
        collisions.append(
f"""    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{rel}" scale="1 1 1"/>
      </geometry>
    </collision>"""
        )

    collisions_text = "\n".join(collisions)
    urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="{link_name}_robot">
  <link name="{link_name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../meshes/visual.obj" scale="1 1 1"/>
      </geometry>
    </visual>

{collisions_text}
  </link>
</robot>
"""
    urdf_path.write_text(urdf, encoding="utf-8")


def copy_visual_assets(src_mesh_dir: Path, dst_mesh_dir: Path):
    # 必要文件 visual.obj
    visual_obj = src_mesh_dir / "visual.obj"
    if not visual_obj.exists():
        raise FileNotFoundError(f"missing {visual_obj}")
    shutil.copy2(visual_obj, dst_mesh_dir / "visual.obj")

    # 可选材质文件
    for pat in ("*.mtl", "material_0.*", "material.*"):
        for f in src_mesh_dir.glob(pat):
            if f.is_file():
                shutil.copy2(f, dst_mesh_dir / f.name)


def process_one_base(base_dir: Path, out_base_dir: Path, threshold: float, max_convex_hull: int, link_name: str):
    src_mesh = base_dir / "meshes"
    src_urdf = base_dir / "urdf"
    collision_obj = src_mesh / "collision.obj"

    if not src_mesh.is_dir() or not src_urdf.is_dir():
        raise RuntimeError("missing meshes/ or urdf/")
    if not collision_obj.exists():
        raise RuntimeError("missing meshes/collision.obj")

    dst_mesh = out_base_dir / "meshes"
    dst_urdf = out_base_dir / "urdf"
    dst_mesh.mkdir(parents=True, exist_ok=True)
    dst_urdf.mkdir(parents=True, exist_ok=True)

    # 1) 复制 visual 相关文件
    copy_visual_assets(src_mesh, dst_mesh)

    # 2) 复制原始 collision.obj（可选，保留）
    shutil.copy2(collision_obj, dst_mesh / "collision.obj")

    # 3) 清理旧 part_*.obj
    for old in dst_mesh.glob("part_*.obj"):
        old.unlink(missing_ok=True)

    # 4) CoACD 分解
    mesh = trimesh.load(str(collision_obj), force="mesh")
    if mesh.is_empty:
        raise RuntimeError("collision.obj is empty")

    parts = run_coacd_parts(mesh, threshold=threshold, max_convex_hull=max_convex_hull)

    part_rel_paths = []
    for i, pm in enumerate(parts):
        name = f"part_{i}.obj"
        pm.export(dst_mesh / name)
        part_rel_paths.append(f"../meshes/{name}")

    # 5) 写 URDF
    write_urdf(link_name=link_name, urdf_path=dst_urdf / "mobility.urdf", part_rel_paths=part_rel_paths)

    return len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="例如: ./assets/002_bowl")
    ap.add_argument("--output-root", default="", help="输出根目录；留空表示原地覆盖(base目录内更新)")
    ap.add_argument("--link-name", default="baseLink")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--max-convex-hull", type=int, default=24)
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    in_place = (args.output_root.strip() == "")
    if in_place:
        output_root = input_root
    else:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    base_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("base")])
    if not base_dirs:
        print(f"[WARN] no base* dirs under {input_root}")
        return

    ok, fail = 0, 0
    for b in base_dirs:
        try:
            out_b = b if in_place else (output_root / b.name)
            n = process_one_base(
                base_dir=b,
                out_base_dir=out_b,
                threshold=args.threshold,
                max_convex_hull=args.max_convex_hull,
                link_name=args.link_name,
            )
            print(f"[OK] {b.name}: {n} convex parts")
            ok += 1
        except Exception as e:
            print(f"[ERR] {b.name}: {e}")
            fail += 1

    print(f"\nDone. total={len(base_dirs)} ok={ok} fail={fail}")


if __name__ == "__main__":
    main()