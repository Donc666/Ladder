#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch convert object folders with multiple GLBs to URDF assets.

Input structure (per object dir):
  <input-root>/<object-name>/
    visual/*.glb
    collision/*.glb

Output structure:
  <output-root>/<object-name>/<variant-name>/
    meshes/
      visual.obj
      collision.obj
    urdf/
      mobility.urdf

Pairing strategy:
1) exact stem match first (visual/base0.glb <-> collision/base0.glb)
2) unmatched visuals:
   - if only one collision glb exists, reuse that one
   - else pair by sorted index with remaining collisions
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List, Tuple
import trimesh


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def load_glb_baked_mesh(glb_path: Path) -> trimesh.Trimesh:
    """Load GLB and bake scene transforms into a single mesh."""
    scene_or_mesh = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        mesh = scene_or_mesh.dump(concatenate=True)  # bake transforms
    if mesh.is_empty:
        raise RuntimeError(f"Empty mesh: {glb_path}")
    return mesh


def sanitize_extents(extents, min_axis=1e-4):
    x, y, z = [abs(float(v)) for v in extents]
    return [max(x, min_axis), max(y, min_axis), max(z, min_axis)]


def inertia_from_box(mass: float, extents_xyz):
    x, y, z = extents_xyz
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    return ixx, iyy, izz


def write_urdf(
    urdf_path: Path,
    robot_name: str,
    visual_rel: str,
    collision_rel: str,
    scale: float,
    mass: float,
    inertia_xyz: Tuple[float, float, float],
):
    ixx, iyy, izz = inertia_xyz
    urdf_text = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="{robot_name}">
  <link name="baseLink">


    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_rel}" scale="{scale:.6f} {scale:.6f} {scale:.6f}"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_rel}" scale="{scale:.6f} {scale:.6f} {scale:.6f}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.write_text(urdf_text, encoding="utf-8")


def pair_glbs(visual_glbs: List[Path], collision_glbs: List[Path]) -> List[Tuple[Path, Path, str]]:
    """Return list of (visual_glb, collision_glb, variant_name)."""
    pairs: List[Tuple[Path, Path, str]] = []

    v_map = {p.stem: p for p in visual_glbs}
    c_map = {p.stem: p for p in collision_glbs}

    used_v = set()
    used_c = set()

    # 1) exact stem match
    for stem in sorted(set(v_map.keys()) & set(c_map.keys())):
        pairs.append((v_map[stem], c_map[stem], stem))
        used_v.add(stem)
        used_c.add(stem)

    remaining_v = [p for p in visual_glbs if p.stem not in used_v]
    remaining_c = [p for p in collision_glbs if p.stem not in used_c]

    # 2) fallback cases
    if remaining_v:
        if len(collision_glbs) == 1:
            # reuse single collision for all unmatched visuals
            c = collision_glbs[0]
            for v in remaining_v:
                variant = f"{v.stem}__col_{c.stem}"
                pairs.append((v, c, variant))
        elif remaining_c:
            # pair by sorted index for remaining
            n = min(len(remaining_v), len(remaining_c))
            for i in range(n):
                v = remaining_v[i]
                c = remaining_c[i]
                variant = f"{v.stem}__col_{c.stem}"
                pairs.append((v, c, variant))

    return pairs


def convert_pair(
    obj_name: str,
    visual_glb: Path,
    collision_glb: Path,
    variant: str,
    out_root: Path,
    mass: float,
    scale: float,
    min_extent: float,
):
    variant = safe_name(variant)
    out_dir = out_root / obj_name / variant
    mesh_dir = out_dir / "meshes"
    urdf_dir = out_dir / "urdf"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir.mkdir(parents=True, exist_ok=True)

    visual_mesh = load_glb_baked_mesh(visual_glb)
    collision_mesh = load_glb_baked_mesh(collision_glb)

    visual_obj = mesh_dir / "visual.obj"
    collision_obj = mesh_dir / "collision.obj"
    visual_mesh.export(visual_obj)
    collision_mesh.export(collision_obj)

    extents = sanitize_extents(collision_mesh.extents, min_axis=min_extent)
    scaled_extents = [e * scale for e in extents]
    inertia = inertia_from_box(mass, scaled_extents)

    urdf_path = urdf_dir / "mobility.urdf"
    write_urdf(
        urdf_path=urdf_path,
        robot_name=f"{safe_name(obj_name)}_{variant}",
        visual_rel="../meshes/visual.obj",
        collision_rel="../meshes/collision.obj",
        scale=scale,
        mass=mass,
        inertia_xyz=inertia,
    )

    print(
        f"[OK] {obj_name}/{variant}\n"
        f"     visual={visual_glb.name}, collision={collision_glb.name}\n"
        f"     collision_extents={collision_mesh.extents}, urdf={urdf_path}"
    )


def process_object_dir(obj_dir: Path, out_root: Path, mass: float, scale: float, min_extent: float):
    visual_dir = obj_dir / "visual"
    collision_dir = obj_dir / "collision"
    if not visual_dir.is_dir() or not collision_dir.is_dir():
        return 0

    visual_glbs = sorted(visual_dir.glob("*.glb"))
    collision_glbs = sorted(collision_dir.glob("*.glb"))

    if not visual_glbs or not collision_glbs:
        print(f"[SKIP] {obj_dir.name}: missing visual/collision glb files")
        return 0

    pairs = pair_glbs(visual_glbs, collision_glbs)
    if not pairs:
        print(f"[SKIP] {obj_dir.name}: no valid visual/collision pairs")
        return 0

    ok = 0
    for v, c, variant in pairs:
        try:
            convert_pair(
                obj_name=obj_dir.name,
                visual_glb=v,
                collision_glb=c,
                variant=variant,
                out_root=out_root,
                mass=mass,
                scale=scale,
                min_extent=min_extent,
            )
            ok += 1
        except Exception as e:
            print(f"[ERR] {obj_dir.name}/{variant}: {e}")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Root of object folders")
    parser.add_argument("--output-root", required=True, help="Output root")
    parser.add_argument("--mass", type=float, default=0.10, help="Mass for URDF inertial")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform mesh scale in URDF")
    parser.add_argument("--min-extent", type=float, default=1e-4, help="Min axis extent clamp")
    parser.add_argument("--recursive", action="store_true", help="Recursively find object dirs")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        candidates = [p for p in input_root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in input_root.iterdir() if p.is_dir()]

    total_objects = 0
    total_variants = 0

    for obj_dir in sorted(candidates):
        if (obj_dir / "visual").is_dir() and (obj_dir / "collision").is_dir():
            total_objects += 1
            total_variants += process_object_dir(
                obj_dir=obj_dir,
                out_root=output_root,
                mass=args.mass,
                scale=args.scale,
                min_extent=args.min_extent,
            )

    print(f"\nDone: objects={total_objects}, converted_variants={total_variants}")


if __name__ == "__main__":
    main()