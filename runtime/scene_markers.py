"""Green interaction cubes and planned end-effector lines in the Isaac stage.

Cubes and curves are visual only: no rigid body and no collision API, so they
are not physics obstacles and are not scanned into the cuRobo world.
"""

from __future__ import annotations

import numpy as np
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
from curobo.types import JointState
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils import bind_visual_material

_CUBE = 0.03
_CUBE_Y = 0.05
_DEDUP_M = 0.005
_MAX_LINE_PTS = 120


class ScenePlanMarkers:
    def __init__(self, origin=(0.0, 0.0, 0.0)):
        cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/InteractCubes",
            markers={
                "cube": sim_utils.CuboidCfg(
                    size=(_CUBE, _CUBE_Y, _CUBE),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    collision_props=None,
                    rigid_props=None,
                ),
            },
        )
        self._cubes = VisualizationMarkers(cfg)
        self._cubes.set_visibility(False)
        self._stage = self._cubes.stage
        self._ribbon = _make_ribbon_mesh(self._stage)
        self._arm_verts: dict[str, list] = {}
        self._arm_faces: dict[str, list] = {}
        self._arm_cubes: dict[str, list[np.ndarray]] = {}
        self._shown_atom: dict[str, object] = {}
        self._origin = np.asarray(origin, dtype=np.float64).reshape(3)
        print("[INFO] green interaction cubes and plan lines are visual only, no collision")

    def add_waypoints(self, waypoints: list[dict]) -> None:
        """Contacts are drawn with the atom that is executing, not ahead of time."""
        return

    def show_atom(self, arm_name: str, atom_key, polylines: list[np.ndarray], cube_xyzs: list, *, exclusive: bool = False) -> None:
        """Replace this arm's previous atom with the whole trajectory of the new one.

        A one-arm atom clears the other arm, so into_1 is gone before into_2 is drawn.
        """
        if self._shown_atom.get(arm_name) == atom_key and not exclusive:
            return
        if exclusive:
            for other in list(self._arm_verts):
                if other != arm_name:
                    self._arm_verts.pop(other, None)
                    self._arm_faces.pop(other, None)
                    self._arm_cubes.pop(other, None)
                    self._shown_atom.pop(other, None)
        verts: list[tuple[float, float, float]] = []
        faces: list[int] = []
        for pts in polylines:
            if pts is None or len(pts) < 2:
                continue
            pts = np.asarray(pts, dtype=np.float64) + self._origin
            for a, b in zip(pts[:-1], pts[1:]):
                _append_box(verts, faces, a, b, half=0.008)
        self._arm_verts[arm_name] = verts
        self._arm_faces[arm_name] = faces
        cubes = []
        for xyz in cube_xyzs:
            cubes.append((np.asarray(xyz, dtype=np.float64).reshape(-1)[:3] + self._origin).astype(np.float32))
        self._arm_cubes[arm_name] = cubes
        self._shown_atom[arm_name] = atom_key
        self._rebuild_ribbon()
        self._show_cubes()
        print(f"[INFO] atom markers {arm_name} segments={len(polylines)} cubes={len(cubes)}")

    def clear_arm(self, arm_name: str) -> None:
        """Drop this arm's line once its atom has finished."""
        had = arm_name in self._shown_atom or arm_name in self._arm_verts
        self._arm_verts.pop(arm_name, None)
        self._arm_faces.pop(arm_name, None)
        self._arm_cubes.pop(arm_name, None)
        self._shown_atom.pop(arm_name, None)
        if not had:
            return
        self._rebuild_ribbon()
        self._show_cubes()

    def _rebuild_ribbon(self) -> None:
        verts: list[tuple[float, float, float]] = []
        faces: list[int] = []
        for arm_name in self._arm_verts:
            base = len(verts)
            verts.extend(self._arm_verts[arm_name])
            faces.extend(index + base for index in self._arm_faces.get(arm_name, []))
        _write_ribbon(self._ribbon, verts, faces)

    def _show_cubes(self) -> None:
        cubes = [xyz for group in self._arm_cubes.values() for xyz in group]
        if not cubes:
            self._cubes.set_visibility(False)
            return
        self._cubes.set_visibility(True)
        self._cubes.visualize(translations=np.stack(cubes, axis=0))


_CUBE_EXTEND_M = 0.07
_LINE_EXTEND_M = 0.08


def _tool_z_axis(pose: list[float]) -> np.ndarray | None:
    """Unit local z of a wxyz end-effector pose, in the world frame."""
    if pose is None or len(pose) < 7:
        return None
    w, x, y, z = (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
    axis = np.array(
        [
            2.0 * (w * y + x * z),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return None
    return axis / norm


def extend_cubes_along_tool_z(polylines: list[np.ndarray], poses: list) -> list[list[float]]:
    """Move each cube 0.07 m along tool z and bridge any leftover gap with a straight segment."""
    cubes = []
    for pose in poses:
        if not pose or len(pose) < 3:
            continue
        start = np.asarray(pose[:3], dtype=np.float64)
        axis = _tool_z_axis(pose)
        end = start if axis is None else start + _CUBE_EXTEND_M * axis
        cubes.append([float(end[0]), float(end[1]), float(end[2])])
        anchor = None
        best_d = 0.05
        for pts in polylines:
            if pts is None or len(pts) < 1:
                continue
            last = np.asarray(pts[-1], dtype=np.float64).reshape(-1)[:3]
            dist = float(np.linalg.norm(last - end))
            if dist < best_d:
                anchor = last
                best_d = dist
        if anchor is not None and float(np.linalg.norm(end - anchor)) > 1e-4:
            polylines.append(np.stack([anchor, end], axis=0))
    return cubes


def _gripper_asset_poses(waypoints: list[dict]) -> list[list[float]]:
    """Poses where the gripper meets or releases an asset.

    Hover, lift, and retreat waypoints are skipped. A cube is kept only at the
    last open pose before the gripper closes, and the last closed pose before
    it opens, and only inside one atom on one arm. A click never changes the
    gripper, so its cube is the lowest pose, where the hand meets the object.
    """
    groups: list[list[dict]] = []
    key = None
    bucket: list[dict] = []
    for wp in waypoints:
        k = (wp.get("atom_index"), wp.get("executor"), wp.get("grasp_asset"), wp.get("atom"))
        if k != key:
            if bucket:
                groups.append(bucket)
            bucket = [wp]
            key = k
        else:
            bucket.append(wp)
    if bucket:
        groups.append(bucket)
    poses = []
    for group in groups:
        grips = [str(wp.get("gripper") or "open") for wp in group]
        changed = False
        for i in range(1, len(group)):
            if grips[i] == grips[i - 1]:
                continue
            changed = True
            pose = group[i - 1].get("pose")
            if pose and len(pose) >= 3:
                poses.append([float(v) for v in pose[:7]])
        if changed or str(group[0].get("atom") or "").lower() != "click":
            continue
        lowest = None
        lowest_z = None
        for wp in group:
            pose = wp.get("pose")
            if not pose or len(pose) < 3:
                continue
            z = float(pose[2])
            if lowest_z is None or z < lowest_z:
                lowest_z = z
                lowest = pose
        if lowest is not None:
            poses.append([float(v) for v in lowest[:7]])
    return poses


def _ee_world_points(planner, joint_traj) -> np.ndarray | None:
    q = joint_traj
    if not isinstance(q, torch.Tensor):
        q = torch.as_tensor(q, dtype=torch.float32)
    q = q.detach()
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.ndim != 2 or q.shape[0] < 2:
        return None
    names = list(getattr(planner, "joint_names", []) or [])
    if names and q.shape[1] > len(names):
        q = q[:, : len(names)]
    n = int(q.shape[0])
    stride = max(1, n // _MAX_LINE_PTS)
    sampled = q[::stride]
    if not torch.equal(sampled[-1], q[-1]):
        sampled = torch.cat([sampled, q[-1:]], dim=0)
    try:
        js = JointState.from_position(sampled.contiguous(), joint_names=names or None)
        state = planner.compute_kinematics(js)
    except Exception as exc:
        print(f"[WARN] plan FK with joint names failed ({exc}); retry without names")
        js = JointState.from_position(sampled.contiguous())
        state = planner.compute_kinematics(js)
    pos = state.tool_poses.position
    quat = state.tool_poses.quaternion
    pos = _squeeze_tool_rows(pos, 3)
    quat = _squeeze_tool_rows(quat, 4)
    offset = getattr(planner, "base_offset", (0.0, 0.0, 0.0))
    off = torch.tensor(offset, device=pos.device, dtype=pos.dtype)
    world = (pos + off).detach().cpu().numpy()
    world = _shift_rows_along_tool_z(world, quat.detach().cpu().numpy(), _LINE_EXTEND_M)
    return np.asarray(world, dtype=np.float64)


def _squeeze_tool_rows(tensor, width: int):
    if tensor.ndim == 4:
        tensor = tensor[:, 0, 0, :]
    elif tensor.ndim == 3:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor[:, 0, :]
    else:
        tensor = tensor.reshape(-1, width)
    return tensor


def _shift_rows_along_tool_z(pos: np.ndarray, quat_wxyz: np.ndarray, distance: float) -> np.ndarray:
    """Translate each point along that sample's end-effector +z."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    if quat_wxyz.shape[0] != pos.shape[0]:
        return pos
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    axis = np.stack(
        [
            2.0 * (w * y + x * z),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=1,
    )
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    ok = norms[:, 0] > 1e-8
    axis[ok] = axis[ok] / norms[ok]
    axis[~ok] = 0.0
    return pos + float(distance) * axis


def _make_ribbon_mesh(stage):
    root = "/World/Visuals/PlanLines"
    UsdGeom.Xform.Define(stage, root)
    mesh = UsdGeom.Mesh.Define(stage, root + "/ribbon")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    color = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    color.Set([(0.0, 1.0, 0.0)])
    mat_path = root + "/material"
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), roughness=0.2)
    mat.func(mat_path, mat)
    bind_visual_material(root + "/ribbon", mat_path, stage=stage)
    return mesh


def _write_ribbon(mesh, verts, faces) -> None:
    if not verts:
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([]))
        mesh.GetFaceVertexCountsAttr().Set([])
        mesh.GetFaceVertexIndicesAttr().Set([])
        return
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in verts]))
    mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
    mesh.GetFaceVertexIndicesAttr().Set(faces)


def _append_box(verts, faces, a, b, half: float) -> None:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length < 1e-4:
        return
    direction = delta / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(direction[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    side = np.cross(direction, helper)
    side = side / np.linalg.norm(side)
    up = np.cross(side, direction)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    base = len(verts)
    for end in (a, b):
        for su in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            corner = end + side * (half * su[0]) + up * (half * su[1])
            verts.append((float(corner[0]), float(corner[1]), float(corner[2])))
    quads = (
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
        (0, 2, 3, 1),
        (4, 6, 7, 5),
    )
    for i0, i1, i2, i3 in quads:
        faces.extend((base + i0, base + i1, base + i2, base + i0, base + i2, base + i3))
