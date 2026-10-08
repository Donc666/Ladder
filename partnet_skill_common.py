"""Shared PartNet + concept-template skill helpers (same pipeline as window)."""

from __future__ import annotations

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

from my_utils import *
from my_utils import (
    _urdf_xyz_to_sim_yzx,
    _xyz,
    _rpy,
    _T_from_pos_quat_wxyz,
    _pos_quat_from_T,
    _quat_wxyz_to_rotmat,
)
from set_interface_try import *

CONCEPT_PATHS: dict[str, str] = {}
YAW: dict[str, float] = {}
CONCEPT_Z_DROP = 0.28


@dataclass
class SkillSpec:
    type_name: str
    category: str
    grasp_keywords: list[str]
    motion: str  # prismatic | revolute | lift
    scale: tuple[float, float, float] = (0.30, 0.30, 0.30)
    spawn_xy: tuple[float, float] = (0.70, 0.0)
    spawn_z: float = 0.45  # real and concept share this; concept z = spawn_z - concept_z_drop
    concept_xy: tuple[float, float] = (1000.0, 1000.0)
    concept_z_drop: float = 0.28
    concept_yaw_offset: float = 90.0  # concept z-yaw minus real yaw (door/window=90, oven=180)
    spawn_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extra XYZ Euler before yaw

    @property
    def concept_spawn_z(self) -> float:
        return float(self.spawn_z) - float(self.concept_z_drop)
    fraction: float = 0.5
    slide_scale: float = 0.35
    max_deg: float = 25.0
    approach_z: float = -0.25
    approach_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world offset of the pre-grasp point
    approach_match_handle_z: bool = False  # pre-grasp uses the handle world z
    stop_at_grasp: bool = False  # approach and grasp only; no arc, slide, or retreat
    grasp_z: float = -0.10
    retreat_z: float = -0.25
    fix_base: bool = True
    joint_stiffness: float = 0.0
    joint_damping: float = 1.0
    extra_hand_ry: float = -90.0
    extra_hand_rx: float = -90.0
    extra_hand_rz: float = 0.0  # local EE Z, after ry/rx (oven handle = 90)
    extra_hand_local_ry: float = 0.0  # final EE local Y, after ry/rx/rz
    sign: float = 1.0  # -1 closes / reverses along the joint axis
    init_open_fraction: float = 0.0  # spawn already open, then close along the joint
    skip_joint_keywords: tuple[str, ...] = ()
    surface_clearance: float = 0.0  # push grasp/arc off the panel along the door normal
    upright_hinge: bool = False  # standing door: force world-Z hinge after mapping
    horizontal_hinge: bool = False  # oven-style drop door: keep a world-horizontal hinge, XY-map like microwave
    grasp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arc_radius_shrink: float = 0.0  # pull EE toward the hinge in the swing plane
    grasp_index: int = 0  # 0 = best handle, 1 = second door, ...


def spawn_rot_wxyz(spec: SkillSpec, yaw_deg: float) -> tuple[float, float, float, float]:
    """Object XYZ Euler, then world-Z yaw. Used by real and concept imports."""
    rx, ry, rz = tuple(spec.spawn_rpy_deg)
    r = R.from_euler("xyz", (float(rx), float(ry), float(rz)), degrees=True)
    r = R.from_euler("z", float(yaw_deg), degrees=True) * r
    return xyzw_to_wxyz(r.as_quat())


def _closed_open_limits(lo: float, hi: float) -> tuple[float, float]:
    if lo > hi:
        lo, hi = hi, lo
    if lo - 1e-6 <= 0.0 <= hi + 1e-6:
        opened = hi if abs(hi) >= abs(lo) else lo
        if abs(opened) < 1e-8:
            opened = hi
        return 0.0, opened
    if abs(lo) <= abs(hi):
        return lo, hi
    return hi, lo


def _is_skipped_joint(name: str, parent: str, child: str, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return False
    blob = f"{name} {parent} {child}".lower()
    return any(k.lower() in blob for k in keywords)


def posed_joint_pos(urdf_path: str, open_fraction: float, skip_keywords: tuple[str, ...] = ()) -> dict[str, float]:
    """Joint positions at `open_fraction` from closed toward open. Handle joints are left at default."""
    pos = safe_default_joint_pos(urdf_path)
    frac = float(open_fraction)
    if frac <= 1e-6:
        return pos
    frac = float(np.clip(frac, 0.05, 1.0))
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return pos
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "prismatic", "continuous"}:
            continue
        name = joint.attrib.get("name") or ""
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if _is_skipped_joint(name, parent, child, skip_keywords):
            continue
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        try:
            lo = float(limit.attrib.get("lower", "0"))
            hi = float(limit.attrib.get("upper", "0"))
        except (TypeError, ValueError):
            continue
        closed, opened = _closed_open_limits(lo, hi)
        pos[name] = float(closed + frac * (opened - closed))
    return pos


def safe_default_joint_pos(urdf_path: str) -> dict[str, float]:
    pos: dict[str, float] = {}
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return pos
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "prismatic"}:
            continue
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        try:
            lo = float(limit.attrib.get("lower", "0"))
            hi = float(limit.attrib.get("upper", "0"))
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        if lo - 1e-6 <= 0.0 <= hi + 1e-6:
            continue
        pos[name] = float(min(hi, max(lo, 0.0)))
    return pos


def _urdf_root_link(urdf_path: str) -> str:
    root = ET.parse(urdf_path).getroot()
    links = [el.attrib.get("name", "") for el in root.findall("link")]
    children = set()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            children.add(child.attrib.get("link", ""))
    for name in links:
        if name and name not in children:
            return name
    return links[0] if links else "world"


def fk_link_world_pos(
    urdf_path: str,
    target_link: str,
    joint_pos: dict[str, float],
    root_pos_w,
    root_quat_wxyz,
    scale=(1.0, 1.0, 1.0),
):
    """Handle pose at commanded joints, without waiting for PhysX to open the door."""
    xml = ET.parse(urdf_path).getroot()
    by_child = {}
    for joint in xml.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        by_child[child.attrib.get("link", "")] = {
            "name": joint.attrib.get("name", ""),
            "type": (joint.attrib.get("type") or "fixed").lower(),
            "parent": parent.attrib.get("link", ""),
            "xyz": _xyz(None if origin is None else origin.attrib.get("xyz")),
            "rpy": _rpy(None if origin is None else origin.attrib.get("rpy")),
            "axis": _xyz(None if axis is None else axis.attrib.get("xyz")),
        }
    root_link = _urdf_root_link(urdf_path)
    chain = []
    cur = target_link
    seen = set()
    while cur and cur != root_link and cur not in seen:
        seen.add(cur)
        joint = by_child.get(cur)
        if joint is None:
            break
        chain.append(joint)
        cur = joint["parent"]
    T = _T_from_pos_quat_wxyz(np.asarray(root_pos_w, dtype=np.float64), np.asarray(root_quat_wxyz, dtype=np.float64))
    s = np.asarray(scale, dtype=np.float64)
    for joint in reversed(chain):
        xyz = np.asarray(joint["xyz"], dtype=np.float64) * s
        rpy = np.asarray(joint["rpy"], dtype=np.float64)
        T_j = np.eye(4, dtype=np.float64)
        T_j[:3, 3] = xyz
        T_j[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T = T @ T_j
        q = float(joint_pos.get(joint["name"], 0.0))
        axis = np.asarray(joint["axis"], dtype=np.float64)
        n = np.linalg.norm(axis)
        if n > 1e-8:
            axis = axis / n
        if joint["type"] in {"revolute", "continuous"} and abs(q) > 1e-10:
            Tr = np.eye(4, dtype=np.float64)
            Tr[:3, :3] = R.from_rotvec(axis * q).as_matrix()
            T = T @ Tr
        elif joint["type"] == "prismatic" and abs(q) > 1e-10:
            Tp = np.eye(4, dtype=np.float64)
            Tp[:3, 3] = axis * q * (float(s[0]) if s.size else 1.0)
            T = T @ Tp
    pos, quat = _pos_quat_from_T(T)
    return pos, quat


def joint_limits(urdf_path: str, joint_name: str, default: float = 0.3):
    lo, hi = 0.0, float(default)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return lo, hi
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        limit = joint.find("limit")
        if limit is None:
            return lo, hi
        try:
            lo = float(limit.attrib.get("lower", "0"))
            hi = float(limit.attrib.get("upper", default))
            if lo > hi:
                lo, hi = hi, lo
        except (TypeError, ValueError):
            return lo, hi
        return lo, hi
    return lo, hi


def open_signed_distance(urdf_path: str, joint_name: str, fraction: float, scale: float) -> float:
    lo, hi = joint_limits(urdf_path, joint_name)
    frac = float(np.clip(fraction, 0.05, 1.0))
    span = hi - lo
    if abs(lo) <= abs(hi):
        delta = frac * span
    else:
        delta = -frac * span
    return float(delta) * float(scale)


def urdf_link_names(urdf_path: str) -> list[str]:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return []
    return [el.attrib.get("name", "") for el in root.findall("link") if el.attrib.get("name")]


def pick_fallback_link(asset, preferred: str | None, urdf_path: str | None = None):
    reject = ("button", "knob", "controller", "switch", "tray", "burner")
    names: list[str] = []
    seen: set[str] = set()
    if urdf_path:
        for name in urdf_link_names(urdf_path):
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    for name in asset.body_names:
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if preferred:
        if preferred in names or "handle" in preferred.lower():
            return preferred
    for name in names:
        low = name.lower()
        if low in {"base", "world"} or any(r in low for r in reject):
            continue
        if "door" in low:
            return name
    for name in names:
        low = name.lower()
        if low.startswith("link_") and "base" not in low:
            return name
    for name in names:
        low = name.lower()
        if low not in {"base", "world"} and not any(r in low for r in reject):
            return name
    raise RuntimeError(f"No grasp link in body_names={list(asset.body_names)}")


def urdf_ancestor_links(urdf_path: str, start_link: str) -> list[str]:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return [start_link]
    child_to_parent = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        child_to_parent[child.attrib.get("link", "")] = parent.attrib.get("link", "")
    chain = []
    cur = start_link
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = child_to_parent.get(cur)
    return chain


def _grasp_link_score(name: str, urdf_path: str, keys: list[str]) -> tuple:
    """Lower is better: door handles first, knobs/buttons last."""
    low = name.lower()
    chain = " ".join(urdf_ancestor_links(urdf_path, name)).lower()
    if any(s in low for s in ("button", "knob", "controller", "switch")):
        return (9, low)
    score = 5
    if any(k in low for k in keys):
        score -= 3
    if "handle" in low:
        score -= 2
    if "door" in chain:
        score -= 2
    if "sunken" in chain:
        score += 1
    if "cuboidal_door" in chain or "rotation_door" in chain:
        score -= 1
    return (score, low)


def pick_grasp_link(asset, urdf_path: str, keywords: list[str], index: int = 0) -> str:
    """Prefer URDF handle links even if PhysX merged them into the door body."""
    keys = [k.lower() for k in keywords if k]
    skip = ("_base", "world")
    reject = tuple(
        r for r in ("button", "knob", "controller", "switch", "tray", "burner") if r not in keys
    )
    hits = []
    seen: set[str] = set()
    for name in urdf_link_names(urdf_path) + list(asset.body_names):
        if not name or name in seen:
            continue
        seen.add(name)
        low = name.lower()
        if any(low.endswith(s) for s in skip) or low in {"base", "world"}:
            continue
        if any(r in low for r in reject):
            continue
        if any(k in low for k in keys):
            hits.append(name)
    if hits:
        hits.sort(key=lambda n: _grasp_link_score(n, urdf_path, keys))
        idx = int(max(0, min(int(index), len(hits) - 1)))
        return hits[idx]
    preferred = find_semantic_link(Path(urdf_path).parent, keys)
    return pick_fallback_link(asset, preferred, urdf_path)


def _joint_origin_axis(joint, yzx: bool = False):
    origin_elem = joint.find("origin")
    axis_elem = joint.find("axis")
    origin_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if origin_elem is not None and origin_elem.attrib.get("xyz"):
        origin_xyz = np.array([float(x) for x in origin_elem.attrib["xyz"].split()], dtype=np.float64)
    axis_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis_elem is not None and axis_elem.attrib.get("xyz"):
        axis_xyz = np.array([float(x) for x in axis_elem.attrib["xyz"].split()], dtype=np.float64)
    if yzx:
        origin_xyz = _urdf_xyz_to_sim_yzx(origin_xyz)
        axis_xyz = _urdf_xyz_to_sim_yzx(axis_xyz)
    n = np.linalg.norm(axis_xyz)
    if n > 1e-8:
        axis_xyz = axis_xyz / n
    return origin_xyz, axis_xyz


def pick_frame_hinge(urdf_path: str, skip_keywords: tuple[str, ...] = (), yzx: bool = True):
    """Revolute attached to the static frame (live door hinge), not a child-leaf joint."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None, None, None, None, None, None
    semantics = parse_partnet_semantics(Path(urdf_path).parent)
    frame_links = {
        link
        for link, kind, part in semantics
        if "door_frame" in part or kind == "static"
    }
    door_links = {
        link
        for link, kind, part in semantics
        if "rotation_door" in part or (kind == "hinge" and "door" in part)
    }
    ranked = []
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "continuous"}:
            continue
        name = joint.attrib.get("name", "")
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if _is_skipped_joint(name, parent, child, skip_keywords):
            continue
        if door_links and child not in door_links and parent not in frame_links:
            continue
        limit = joint.find("limit")
        lo = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
        hi = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
        closed, opened = _closed_open_limits(lo, hi)
        span = abs(opened - closed)
        frame_parent = parent in frame_links or parent.lower() in {"base", "world", "link_0"}
        origin_xyz, axis_xyz = _joint_origin_axis(joint, yzx=yzx)
        # prefer frame-parented, then larger span, then more vertical local axis
        vert = abs(float(axis_xyz[2]))
        ranked.append((0 if frame_parent else 1, -span, -vert, origin_xyz, axis_xyz, parent, child, name, jtype))
    if not ranked:
        return largest_revolute_joint(urdf_path, skip_keywords)
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    _fp, _sp, _v, origin_xyz, axis_xyz, parent, child, name, jtype = ranked[0]
    return origin_xyz, axis_xyz, parent, child, name, jtype


def largest_revolute_joint(urdf_path: str, skip_keywords: tuple[str, ...] = ()):
    """Primary door-like revolute: largest span, skipping handle/knob joints."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None, None, None, None, None, None
    best = None
    best_span = -1.0
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "continuous"}:
            continue
        name = joint.attrib.get("name", "")
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if _is_skipped_joint(name, parent, child, skip_keywords):
            continue
        limit = joint.find("limit")
        lo = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
        hi = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            continue
        closed, opened = _closed_open_limits(lo, hi)
        span = abs(opened - closed)
        if span <= best_span:
            continue
        origin_xyz, axis_xyz = _joint_origin_axis(joint)
        best = (origin_xyz, axis_xyz, parent, child, name, jtype)
        best_span = span
    if best is None:
        return None, None, None, None, None, None
    return best


def semantic_links_for_keywords(urdf_path: str, keywords: list[str]) -> set[str]:
    keys = [k.lower() for k in keywords if k]
    if not keys:
        return set()
    hits: set[str] = set()
    for link, kind, part in parse_partnet_semantics(Path(urdf_path).parent):
        blob = f"{link} {kind} {part}".lower()
        if any(k in blob for k in keys):
            hits.add(link)
    return hits


def pick_movable_joint(
    urdf_path: str,
    types: tuple[str, ...] = ("revolute", "continuous", "prismatic"),
    skip_keywords: tuple[str, ...] = (),
    keywords: list[str] | None = None,
    yzx: bool = False,
):
    """Largest-span movable joint, preferring semantics / grasp keywords."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None, None, None, None, None, None
    allowed = {t.lower() for t in types}
    sem_links = semantic_links_for_keywords(urdf_path, keywords or [])
    ranked = []
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in allowed:
            continue
        name = joint.attrib.get("name", "")
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if _is_skipped_joint(name, parent, child, skip_keywords):
            continue
        limit = joint.find("limit")
        try:
            lo = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
            hi = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        span = abs(hi - lo)
        origin_xyz, axis_xyz = _joint_origin_axis(joint, yzx=yzx)
        sem_hit = bool(sem_links) and (child in sem_links or parent in sem_links)
        ranked.append((0 if sem_hit else 1, -span, origin_xyz, axis_xyz, parent, child, name, jtype))
    if not ranked:
        return None, None, None, None, None, None
    ranked.sort(key=lambda r: (r[0], r[1]))
    _sem, _span, origin_xyz, axis_xyz, parent, child, name, jtype = ranked[0]
    return origin_xyz, axis_xyz, parent, child, name, jtype


def _match_body_name(asset, link_name: str | None) -> str | None:
    if not link_name:
        return None
    names = list(asset.body_names)
    if link_name in names:
        return link_name
    low = link_name.lower()
    for name in names:
        if name.lower() == low:
            return name
    skip = {"base", "world"}
    for name in names:
        nlow = name.lower()
        if nlow in skip:
            continue
        if nlow.endswith(low) or low.endswith(nlow):
            return name
    return None


def _rotation_sign_toward(axis_dir, pivot, p_from, p_to) -> float:
    """+1 if a positive rotation around axis_dir takes p_from toward p_to."""
    u = np.asarray(axis_dir, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-8:
        return 0.0
    u = u / n
    a = np.asarray(p_from, dtype=np.float64) - np.asarray(pivot, dtype=np.float64)
    b = np.asarray(p_to, dtype=np.float64) - np.asarray(pivot, dtype=np.float64)
    a = a - np.dot(a, u) * u
    b = b - np.dot(b, u) * u
    if np.linalg.norm(a) < 1e-4 or np.linalg.norm(b) < 1e-4:
        return 0.0
    s = float(np.dot(u, np.cross(a, b)))
    if abs(s) < 1e-8:
        return 0.0
    return 1.0 if s > 0.0 else -1.0


def _rotate_around_axis(pivot, axis_dir, point, angle_rad):
    c = np.asarray(pivot, dtype=np.float64)
    u = np.asarray(axis_dir, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-8:
        return np.asarray(point, dtype=np.float64).copy()
    u = u / n
    r0 = np.asarray(point, dtype=np.float64) - c
    th = float(angle_rad)
    r = (
        r0 * np.cos(th)
        + np.cross(u, r0) * np.sin(th)
        + u * np.dot(u, r0) * (1.0 - np.cos(th))
    )
    return c + r


def _inplane(point, pivot, axis_dir):
    u = np.asarray(axis_dir, dtype=np.float64)
    u = u / np.linalg.norm(u)
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(pivot, dtype=np.float64)
    return p - u * np.dot(p - c, u)


def hinge_from_open_closed(p_open, p_closed, axis_dir, delta_rad):
    """Recover a point on the hinge from one rigid point at open and closed."""
    u = np.asarray(axis_dir, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-8:
        return None
    u = u / n
    th = float(delta_rad)
    if abs(th) < np.deg2rad(8.0):
        return None
    a = np.asarray(p_open, dtype=np.float64)
    b = np.asarray(p_closed, dtype=np.float64)
    a_p = a - u * np.dot(a, u)
    b_p = b - u * np.dot(b, u)
    chord = b_p - a_p
    d = 0.5 * np.linalg.norm(chord)
    if d < 1e-3:
        return None
    perp = np.cross(u, chord)
    pn = np.linalg.norm(perp)
    if pn < 1e-8:
        return None
    perp = perp / pn
    h = d / np.tan(0.5 * abs(th))
    mid = 0.5 * (a_p + b_p)
    center = mid + np.sign(th) * perp * h
    height = 0.5 * (np.dot(a, u) + np.dot(b, u))
    return center + u * height


def fit_hinge_from_swing(p_open, p_closed, axis_dir, span_rad):
    """Pick the hinge + signed angle around axis_dir that maps p_open to p_closed."""
    span = abs(float(span_rad))
    best_c = None
    best_th = 0.0
    best_err = 1e9
    for th in (span, -span):
        center = hinge_from_open_closed(p_open, p_closed, axis_dir, th)
        if center is None:
            continue
        pred = _rotate_around_axis(center, axis_dir, p_open, th)
        err = float(np.linalg.norm(pred - np.asarray(p_closed, dtype=np.float64)))
        if err < best_err:
            best_err = err
            best_c = center
            best_th = float(th)
    if best_c is None or best_err > 0.20:
        return None, 0.0, best_err
    return best_c, best_th, best_err


def _refresh_articulation(asset):
    try:
        asset.write_data_to_sim()
    except Exception:
        pass
    try:
        asset.update(0.0)
    except Exception:
        pass


def _read_link_pos(asset, link_name: str, env_id: int):
    pos, _ = link_parent_pose(asset, link_name, env_id)
    return np.asarray(pos, dtype=np.float64)


def _read_joint_q(asset, joint_name: str, env_id: int) -> float | None:
    if asset is None or not joint_name:
        return None
    names = list(getattr(asset, "joint_names", []) or [])
    if joint_name not in names:
        return None
    jids, _ = asset.find_joints(joint_name)
    if len(jids) == 0:
        return None
    return float(asset.data.joint_pos[env_id, int(jids[0])].detach().cpu())


def sample_link_at_joints(asset, joint_pos: dict[str, float], env_id: int, link_name: str):
    apply_joint_pos(asset, joint_pos, env_id)
    _refresh_articulation(asset)
    _refresh_articulation(asset)
    return _read_link_pos(asset, link_name, env_id)


def close_arc_from_current(pivot, axis_dir, p0, delta_rad, step_deg: float = 4.0):
    """Waypoints from the current point on the hinge circle toward closed (skip start)."""
    span = abs(float(np.degrees(delta_rad)))
    if span < 1.0:
        return []
    n = max(1, int(round(span / float(step_deg))))
    pts = []
    for i in range(1, n + 1):
        p = _rotate_around_axis(pivot, axis_dir, p0, delta_rad * (i / n))
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    return pts


def xy_in_root_frame(asset, world_xyz, env_id: int):
    """XY of a world point in the asset root yaw frame (so concept vs real can be compared)."""
    root = asset.data.root_pos_w[env_id].detach().cpu().numpy()
    quat = asset.data.root_quat_w[env_id].detach().cpu().numpy()
    rot = _quat_wxyz_to_rotmat(np.asarray(quat, dtype=np.float64))
    local = rot.T @ (np.asarray(world_xyz, dtype=np.float64) - np.asarray(root, dtype=np.float64))
    return local[:2]


def detect_xy_mirror(concept_xy, real_xy) -> tuple[float, float]:
    """Which root-frame XY axes to flip so concept open side lands on the real open side."""
    mx, my = 1.0, 1.0
    if concept_xy is None or real_xy is None:
        return mx, my
    if abs(concept_xy[0]) > 0.02 and abs(real_xy[0]) > 0.02 and concept_xy[0] * real_xy[0] < 0.0:
        mx = -1.0
    if abs(concept_xy[1]) > 0.02 and abs(real_xy[1]) > 0.02 and concept_xy[1] * real_xy[1] < 0.0:
        my = -1.0
    return mx, my


def map_concept_point_to_real(
    point, concept_root, concept_quat, real_root, real_quat, mirror_xy=(1.0, 1.0), keep_z: bool = True
):
    """Map a concept-world point into the real object: local XY + optional mirror.

    Z stays the concept world height (same as the old XY-only map) unless keep_z is False.
    """
    rc = _quat_wxyz_to_rotmat(np.asarray(concept_quat, dtype=np.float64))
    rr = _quat_wxyz_to_rotmat(np.asarray(real_quat, dtype=np.float64))
    src = np.asarray(point, dtype=np.float64)
    local = rc.T @ (src - np.asarray(concept_root, dtype=np.float64))
    local = local.copy()
    local[0] *= float(mirror_xy[0])
    local[1] *= float(mirror_xy[1])
    mapped = np.asarray(real_root, dtype=np.float64) + rr @ local
    if keep_z:
        mapped[2] = float(src[2])
    return mapped


def map_concept_dir_to_real(direction, concept_quat, real_quat, mirror_xy=(1.0, 1.0)):
    rc = _quat_wxyz_to_rotmat(np.asarray(concept_quat, dtype=np.float64))
    rr = _quat_wxyz_to_rotmat(np.asarray(real_quat, dtype=np.float64))
    local = rc.T @ np.asarray(direction, dtype=np.float64)
    local = local.copy()
    local[0] *= float(mirror_xy[0])
    local[1] *= float(mirror_xy[1])
    mapped = rr @ local
    n = np.linalg.norm(mapped)
    if n > 1e-8:
        mapped = mapped / n
    return mapped


def reflect_open_joint_pos(urdf_path: str, joint_pos: dict[str, float], skip_keywords=()):
    """Reflect hinge qs around closed so the leaf swings the other way, clamped to limits."""
    out = dict(joint_pos)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return out
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in {"revolute", "continuous"}:
            continue
        name = joint.attrib.get("name") or ""
        if name not in out:
            continue
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if _is_skipped_joint(name, parent, child, skip_keywords):
            continue
        limit = joint.find("limit")
        lo = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
        hi = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
        if lo > hi:
            lo, hi = hi, lo
        closed, _opened = _closed_open_limits(lo, hi)
        q_flip = 2.0 * closed - float(out[name])
        out[name] = float(np.clip(q_flip, lo, hi))
    return out


def _leaf_open_xy(asset, leaf_name: str | None, env_id: int):
    moving = _match_body_name(asset, leaf_name)
    if not moving:
        return None
    return xy_in_root_frame(asset, _read_link_pos(asset, moving, env_id), env_id)


def link_parent_pose(asset, link_name: str, env_id: int):
    resolved = _match_body_name(asset, link_name)
    if resolved is not None:
        pid, _ = asset.find_bodies(resolved)
        pstate = asset.data.body_state_w[env_id, int(pid[0])]
        return (
            pstate[0:3].detach().cpu().numpy(),
            pstate[3:7].detach().cpu().numpy(),
        )
    return (
        asset.data.root_pos_w[env_id].detach().cpu().numpy(),
        asset.data.root_quat_w[env_id].detach().cpu().numpy(),
    )


def movable_joint_from_link(
    urdf_path: str,
    start_link: str,
    types=("prismatic", "revolute", "continuous"),
    skip_keywords: tuple[str, ...] = (),
):
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None, None, None, None, None, None
    allowed = {t.lower() for t in types}
    child_to_joint = {}
    for joint in root.findall("joint"):
        child_elem = joint.find("child")
        if child_elem is None:
            continue
        child_to_joint[child_elem.attrib.get("link", "")] = joint
    cur = start_link
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        joint = child_to_joint.get(cur)
        if joint is None:
            break
        jtype = (joint.attrib.get("type") or "").lower()
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        if jtype in allowed:
            jname = joint.attrib.get("name", "")
            if _is_skipped_joint(jname, parent, cur, skip_keywords):
                cur = parent
                continue
            origin_xyz, axis_xyz = _joint_origin_axis(joint)
            return origin_xyz, axis_xyz, parent, cur, jname, jtype
        cur = parent
    return get_movable_joint_from_urdf(urdf_path, types=types)


def scene_asset(scene, scene_key: str, env_id: int):
    for name in (f"{scene_key}__env{env_id}", scene_key):
        try:
            return scene[name]
        except (KeyError, AttributeError):
            continue
    return None


def hand_quat_wxyz(
    q_obj_world_wxyz,
    extra_ry: float,
    extra_rx: float,
    extra_rz: float = 0.0,
    extra_ry_local: float = 0.0,
):
    r_x = R.from_euler("x", 180, degrees=True)
    r_z = R.from_euler("z", 90, degrees=True)
    r_hand_obj = r_z * r_x
    r_obj_world = R.from_quat(wxyz_to_xyzw(q_obj_world_wxyz))
    r_hand_world = r_obj_world * r_hand_obj
    q = xyzw_to_wxyz(r_hand_world.as_quat())
    q = rotate_y(q, extra_ry)
    q = rotate_x(q, extra_rx)
    if abs(float(extra_rz)) > 1e-6:
        q, _ = rotate_z(q, extra_rz)
    if abs(float(extra_ry_local)) > 1e-6:
        q = rotate_y(q, extra_ry_local)
    return q


def pose7(xyz, q_wxyz):
    return [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        q_wxyz[0],
        q_wxyz[1],
        q_wxyz[2],
        q_wxyz[3],
    ]


def wp(xyz, q_wxyz, grip: str, executor: str):
    return {
        "pose": pose7(xyz, q_wxyz),
        "gripper": grip,
        "use_obstacle_avoidance": False,
        "executor": executor,
    }


def apply_joint_pos(asset, joint_pos: dict[str, float], env_id: int = 0):
    """Write named joint positions into a live articulation (keeps init-open from collapsing)."""
    if asset is None or not joint_pos:
        return
    try:
        names = list(getattr(asset, "joint_names", []))
        if not names:
            return
        device = asset.data.joint_pos.device
        dtype = asset.data.joint_pos.dtype
        env_ids = torch.tensor([int(env_id)], device=device, dtype=torch.long)
        applied = {}
        for jname, q in joint_pos.items():
            if jname not in names:
                continue
            jids, _ = asset.find_joints(jname)
            if len(jids) == 0:
                continue
            joint_ids = torch.as_tensor(jids, device=device, dtype=torch.long)
            val = torch.tensor([[float(q)]], device=device, dtype=dtype)
            asset.write_joint_position_to_sim(val, joint_ids=joint_ids, env_ids=env_ids)
            try:
                asset.set_joint_position_target(val, joint_ids=joint_ids, env_ids=env_ids)
            except TypeError:
                asset.set_joint_position_target(val, joint_ids=joint_ids)
            applied[jname] = float(q)
        if applied:
            print(f"[joint] applied {applied} available={names}")
        else:
            print(f"[WARN] no joint_pos matched asset joints={names} requested={list(joint_pos)}")
    except Exception as exc:
        print(f"[WARN] apply_joint_pos skipped: {exc}")


class PartNetConcept:
    spec: SkillSpec

    def __init__(self):
        self.scale = self.spec.scale
        self.type_name = self.spec.type_name

    def import2sim(self, urdf_path: str, instance_name: str, num_envs):
        spec = self.spec
        self.num_envs = num_envs
        yaw = float(YAW.setdefault(spec.type_name, float(np.random.uniform(-20, 20))))
        resolved, model_id = resolve_concept_urdf_path(
            source_urdf_path=urdf_path,
            concept_root=str(PARTNET_CONCEPT_ROOT),
            category=spec.category,
            allow_fallback=False,
        )
        if resolved is None:
            print(f"[WARN] {spec.type_name} model_id={model_id} has no concept template, use source URDF")
            resolved = urdf_path
        CONCEPT_PATHS[spec.type_name] = resolved
        self.concept_urdf_path = resolved
        rot_concept = spawn_rot_wxyz(spec, float(spec.concept_yaw_offset) + yaw)
        cx, cy = spec.concept_xy
        cz = spec.concept_spawn_z
        print(
            f"[{spec.type_name}_concept] paired {urdf_path} -> {resolved} "
            f"pos=({cx:.2f},{cy:.2f},{cz:.2f}) (real_z={spec.spawn_z:.2f} - drop={spec.concept_z_drop}) "
            f"yaw={yaw:+.1f} concept_yaw={float(spec.concept_yaw_offset)+yaw:+.1f}"
        )
        cfg = AssetCfg(
            name=f"my_concept_{spec.type_name}",
            urdf_path=resolved,
            pos=(cx, cy, cz),
            rot=rot_concept,
            scale=spec.scale,
            asset_kind="articulation",
            fix_base=True,
            stiffness=spec.joint_stiffness,
            damping=spec.joint_damping,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
            joint_pos=posed_joint_pos(resolved, spec.init_open_fraction, spec.skip_joint_keywords),
            disable_gravity=spec.init_open_fraction > 1e-6,
        )
        return [cfg]


class PartNetSkill:
    spec: SkillSpec

    def __init__(self):
        self.scale = self.spec.scale
        self.type_name = self.spec.type_name

    def import2sim(self, urdf_path: str, instance_name: str, num_envs: int):
        spec = self.spec
        self.num_envs = num_envs
        self.urdf_path = urdf_path
        self.scene_key = instance_name
        yaw = float(YAW.setdefault(spec.type_name, float(np.random.uniform(-20, 20))))
        self.q_obj_world_wxyz = spawn_rot_wxyz(spec, yaw)
        resolved, model_id = resolve_concept_urdf_path(
            source_urdf_path=urdf_path,
            concept_root=str(PARTNET_CONCEPT_ROOT),
            category=spec.category,
            allow_fallback=False,
        )
        self.concept_urdf_path = CONCEPT_PATHS.get(spec.type_name) or resolved or urdf_path
        joint_pos = posed_joint_pos(urdf_path, spec.init_open_fraction, spec.skip_joint_keywords)
        x1, y1 = spec.spawn_xy
        print(
            f"[{spec.type_name}] spawn PartNet urdf={urdf_path} model_id={model_id} "
            f"pos=({x1:.2f},{y1:.2f},{spec.spawn_z:.2f}) scale={spec.scale} "
            f"init_open={spec.init_open_fraction:.2f} joints={joint_pos}"
        )
        cfgs = []
        for _env_id in range(num_envs):
            cfgs.append(
                AssetCfg(
                    fix_base=spec.fix_base,
                    name=self.scene_key,
                    urdf_path=urdf_path,
                    pos=(x1, y1, spec.spawn_z),
                    rot=self.q_obj_world_wxyz,
                    scale=spec.scale,
                    asset_kind="articulation",
                    stiffness=spec.joint_stiffness,
                    damping=spec.joint_damping,
                    static_friction=0.8,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    compliant_contact_damping=0.0,
                    compliant_contact_stiffness=0.0,
                    joint_pos=joint_pos,
                    disable_gravity=spec.init_open_fraction > 1e-6,
                    roughness=0.35,
                    opacity=1.0,
                    force_usd_conversion=True,
                )
            )
        return cfgs

    def execute(self, scene, env_id: int = 0, fraction: float | None = None, spec_override: SkillSpec | None = None):
        spec = spec_override or self.spec
        asset = scene_asset(scene, self.scene_key, env_id)
        if asset is None:
            raise RuntimeError(f"{spec.type_name} asset not found for {self.scene_key}")
        concept_key = f"my_concept_{spec.type_name}__env{env_id}"
        try:
            my_concept = scene[concept_key]
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(
                f"{spec.type_name}_concept is not in the scene; add it to the task JSON"
            ) from exc
        concept_urdf = CONCEPT_PATHS.get(spec.type_name) or self.concept_urdf_path
        if not concept_urdf:
            raise RuntimeError(f"concept URDF missing for {spec.type_name}")

        if spec.init_open_fraction > 1e-6:
            open_real = posed_joint_pos(self.urdf_path, spec.init_open_fraction, spec.skip_joint_keywords)
            open_concept = posed_joint_pos(concept_urdf, spec.init_open_fraction, spec.skip_joint_keywords)
            apply_joint_pos(asset, open_real, env_id)
            apply_joint_pos(my_concept, open_concept, env_id)
            _refresh_articulation(asset)
            _refresh_articulation(my_concept)
        else:
            closed_real = posed_joint_pos(self.urdf_path, 0.0, spec.skip_joint_keywords)
            closed_concept = posed_joint_pos(concept_urdf, 0.0, spec.skip_joint_keywords)
            apply_joint_pos(asset, closed_real, env_id)
            apply_joint_pos(my_concept, closed_concept, env_id)
            open_concept = {}

        grasp_link = pick_grasp_link(my_concept, concept_urdf, spec.grasp_keywords, index=spec.grasp_index)
        merged = grasp_link not in list(my_concept.body_names)
        print(
            f"[{spec.type_name}] grasp link={grasp_link}"
            f"{' (URDF, PhysX-merged)' if merged else ''}"
            f" bodies={list(my_concept.body_names)}"
        )
        prefer_types = ("prismatic",) if spec.motion == "prismatic" else ("revolute", "continuous", "prismatic")
        concept_root = my_concept.data.root_pos_w[env_id].detach().cpu().numpy()
        concept_quat = my_concept.data.root_quat_w[env_id].detach().cpu().numpy()
        real_root = asset.data.root_pos_w[env_id].detach().cpu().numpy()
        real_quat = asset.data.root_quat_w[env_id].detach().cpu().numpy()
        use_fk = merged or (spec.init_open_fraction > 1e-6 and bool(open_concept))
        if use_fk:
            grasp_concept, _ = fk_link_world_pos(
                concept_urdf,
                grasp_link,
                open_concept if open_concept else {},
                concept_root,
                concept_quat,
                spec.scale,
            )
            print(
                f"[{spec.type_name}] handle from FK at open joints {open_concept} "
                f"-> ({grasp_concept[0]:.3f},{grasp_concept[1]:.3f},{grasp_concept[2]:.3f})"
            )
        else:
            grasp_concept, _ = get_link_pose_physics(my_concept, link_name=grasp_link, env_id=env_id)

        real_meta = pick_frame_hinge(self.urdf_path, spec.skip_joint_keywords, yzx=True)
        concept_meta = pick_frame_hinge(concept_urdf, spec.skip_joint_keywords, yzx=False)
        r_xy = _leaf_open_xy(asset, real_meta[3] if real_meta[0] is not None else None, env_id)
        c_xy = _leaf_open_xy(my_concept, concept_meta[3] if concept_meta[0] is not None else None, env_id)
        if c_xy is None:
            c_xy = xy_in_root_frame(my_concept, grasp_concept, env_id)
        if r_xy is None:
            r_xy = xy_in_root_frame(asset, real_root, env_id)
        mirror_xy = detect_xy_mirror(c_xy, r_xy)
        print(
            f"[{spec.type_name}] mirror map mx={mirror_xy[0]:+.0f} my={mirror_xy[1]:+.0f} "
            f"concept_leaf=({c_xy[0]:+.3f},{c_xy[1]:+.3f}) real_leaf=({r_xy[0]:+.3f},{r_xy[1]:+.3f})"
        )

        origin_local, axis_local, parent_link, child_link, joint_name, jtype = movable_joint_from_link(
            concept_urdf, grasp_link, types=prefer_types, skip_keywords=spec.skip_joint_keywords
        )
        if origin_local is None or axis_local is None:
            if concept_meta[0] is not None:
                origin_local, axis_local, parent_link, child_link, joint_name, jtype = concept_meta
            else:
                origin_local, axis_local, parent_link, child_link, joint_name, jtype = get_movable_joint_from_urdf(
                    concept_urdf, types=("prismatic", "revolute", "continuous")
                )
        joint_from_real = origin_local is None or axis_local is None
        joint_limit_urdf = concept_urdf
        if joint_from_real:
            origin_local, axis_local, parent_link, child_link, joint_name, jtype = pick_movable_joint(
                self.urdf_path,
                types=prefer_types,
                skip_keywords=spec.skip_joint_keywords,
                keywords=spec.grasp_keywords,
                yzx=True,
            )
            if origin_local is None or axis_local is None:
                origin_local, axis_local, parent_link, child_link, joint_name, jtype = pick_movable_joint(
                    self.urdf_path,
                    types=("prismatic", "revolute", "continuous"),
                    skip_keywords=spec.skip_joint_keywords,
                    keywords=spec.grasp_keywords,
                    yzx=True,
                )
            if origin_local is None or axis_local is None:
                raise RuntimeError(
                    f"No movable joint in concept {concept_urdf} or real {self.urdf_path}"
                )
            joint_limit_urdf = self.urdf_path
            print(
                f"[{spec.type_name}] concept has no DOF, use real joint={joint_name} "
                f"type={jtype} child={child_link}"
            )

        if joint_from_real:
            parent_pos_w, parent_quat_wxyz = link_parent_pose(asset, parent_link or "", env_id)
            axis_point_w, axis_dir = transform_joint_axis_to_world(
                parent_pos_w, parent_quat_wxyz, origin_local, axis_local
            )
            axis_dir = np.asarray(axis_dir, dtype=np.float64)
            n = np.linalg.norm(axis_dir)
            if n < 1e-8:
                raise RuntimeError(f"{spec.type_name} joint axis is degenerate")
            axis_dir = axis_dir / n
            lo, hi = joint_limits(joint_limit_urdf, joint_name)
            q_closed, q_opened = _closed_open_limits(lo, hi)
            swing_delta = None
            grasp_w = map_concept_point_to_real_xy(grasp_concept, concept_root, real_root)
            arc_sign = 1.0
            print(
                f"[{spec.type_name}] handle XY-only, real-joint axis "
                f"handle=({grasp_w[0]:.3f},{grasp_w[1]:.3f},{grasp_w[2]:.3f}) "
                f"pivot=({axis_point_w[0]:.3f},{axis_point_w[1]:.3f},{axis_point_w[2]:.3f}) "
                f"axis=({axis_dir[0]:.3f},{axis_dir[1]:.3f},{axis_dir[2]:.3f})"
            )
        else:
            parent_pos_w, parent_quat_wxyz = link_parent_pose(my_concept, parent_link or "", env_id)

            axis_point_c, axis_dir_c = transform_joint_axis_to_world(
                parent_pos_w, parent_quat_wxyz, origin_local, axis_local
            )
            axis_dir_c = np.asarray(axis_dir_c, dtype=np.float64)
            n = np.linalg.norm(axis_dir_c)
            if n < 1e-8:
                raise RuntimeError(f"{spec.type_name} joint axis is degenerate")
            axis_dir_c = axis_dir_c / n
            if spec.upright_hinge:
                axis_dir_c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            elif spec.horizontal_hinge:
                axis_dir_c = np.array(axis_dir_c, dtype=np.float64, copy=True)
                axis_dir_c[2] = 0.0
                hn = np.linalg.norm(axis_dir_c)
                axis_dir_c = (
                    axis_dir_c / hn if hn > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
                )

            lo, hi = joint_limits(joint_limit_urdf, joint_name)
            q_closed, q_opened = _closed_open_limits(lo, hi)
            swing_delta = None
            sample_link = grasp_link or child_link
            if spec.motion == "revolute" and spec.init_open_fraction > 1e-6 and sample_link:
                open_cmd = dict(open_concept) if open_concept else posed_joint_pos(
                    concept_urdf, spec.init_open_fraction, spec.skip_joint_keywords
                )
                closed_cmd = dict(open_cmd)
                closed_cmd[joint_name] = float(q_closed)
                p_open, _ = fk_link_world_pos(
                    concept_urdf, sample_link, open_cmd, concept_root, concept_quat, spec.scale
                )
                p_shut, _ = fk_link_world_pos(
                    concept_urdf, sample_link, closed_cmd, concept_root, concept_quat, spec.scale
                )
                recovered, swing_th, fit_err = fit_hinge_from_swing(
                    p_open, p_shut, axis_dir_c, q_closed - q_opened
                )
                chord = float(np.linalg.norm((p_shut - p_open) - axis_dir_c * np.dot(p_shut - p_open, axis_dir_c)))
                if recovered is not None and chord > 0.02:
                    axis_point_c = recovered
                    swing_delta = swing_th
                    print(
                        f"[{spec.type_name}] concept arc hinge leaf={sample_link} chord={chord:.3f} "
                        f"fit_err={fit_err:.3f} swing_deg={np.degrees(swing_th):+.1f}"
                    )
                else:
                    print(
                        f"[{spec.type_name}] concept swing sample failed chord={chord:.3f} "
                        f"fit_err={fit_err:.3f}, keep URDF origin"
                    )

            grasp_w = map_concept_point_to_real_xy(grasp_concept, concept_root, real_root)
            if spec.horizontal_hinge:
                axis_point_w = map_concept_point_to_real_xy(axis_point_c, concept_root, real_root)
                axis_dir = np.asarray(axis_dir_c, dtype=np.float64)
                arc_sign = 1.0
            else:
                axis_point_w = map_concept_point_to_real(
                    axis_point_c, concept_root, concept_quat, real_root, real_quat, mirror_xy
                )
                axis_dir = map_concept_dir_to_real(axis_dir_c, concept_quat, real_quat, mirror_xy)
                if spec.upright_hinge and abs(float(axis_dir[2])) < 0.7:
                    axis_dir = np.array([0.0, 0.0, 1.0 if axis_dir[2] >= 0.0 else -1.0], dtype=np.float64)
                arc_sign = -float(mirror_xy[0]) * float(mirror_xy[1])
            print(
                f"[{spec.type_name}] handle XY-only, "
                f"{'hinge XY+horizontal' if spec.horizontal_hinge else 'hinge mirrored'} "
                f"handle=({grasp_w[0]:.3f},{grasp_w[1]:.3f},{grasp_w[2]:.3f}) "
                f"pivot=({axis_point_w[0]:.3f},{axis_point_w[1]:.3f},{axis_point_w[2]:.3f}) "
                f"axis=({axis_dir[0]:.3f},{axis_dir[1]:.3f},{axis_dir[2]:.3f}) arc_sign={arc_sign:+.0f}"
            )

        x1, y1, z1 = float(grasp_w[0]), float(grasp_w[1]), float(grasp_w[2])
        q_hand = hand_quat_wxyz(
            self.q_obj_world_wxyz,
            spec.extra_hand_ry,
            spec.extra_hand_rx,
            spec.extra_hand_rz,
            spec.extra_hand_local_ry,
        )
        r_hand = R.from_quat(wxyz_to_xyzw(q_hand))
        grasp = np.array([x1, y1, z1], dtype=np.float64)
        handle_xyz = grasp + r_hand.apply([0.0, 0.0, spec.grasp_z])
        if spec.surface_clearance > 1e-6:
            axis_pt = np.asarray(axis_point_w, dtype=np.float64)
            radial = handle_xyz - axis_pt
            radial = radial - np.dot(radial, axis_dir) * axis_dir
            n_door = np.cross(axis_dir, radial)
            nn = np.linalg.norm(n_door)
            if nn > 1e-8:
                n_door = n_door / nn
                away = r_hand.apply([0.0, 0.0, -1.0])
                if np.dot(n_door, away) < 0.0:
                    n_door = -n_door
                handle_xyz = handle_xyz + n_door * float(spec.surface_clearance)
        handle_xyz = handle_xyz + np.array(spec.grasp_offset_xyz, dtype=np.float64)
        if spec.arc_radius_shrink > 1e-8:
            radial = _inplane(handle_xyz, axis_point_w, axis_dir) - np.asarray(axis_point_w, dtype=np.float64)
            n_rad = float(np.linalg.norm(radial))
            shrink = float(spec.arc_radius_shrink)
            if n_rad > shrink + 1e-8:
                handle_xyz = handle_xyz - radial * (shrink / n_rad)
        approach_xyz = handle_xyz + r_hand.apply([0.0, 0.0, spec.approach_z - spec.grasp_z])
        approach_xyz = approach_xyz + np.array(spec.approach_offset_xyz, dtype=np.float64)
        if spec.approach_match_handle_z:
            approach_xyz[2] = handle_xyz[2]
        json_frac = float(np.clip(fraction if fraction is not None else spec.fraction, 0.05, 1.0))
        frac = json_frac
        if spec.init_open_fraction > 1e-6 and spec.motion == "prismatic":
            frac = float(np.clip(spec.init_open_fraction * json_frac, 0.05, 1.0))

        motion_pts = []
        max_deg = 0
        jtype_l = (jtype or spec.motion).lower()
        if spec.stop_at_grasp:
            print(f"[{spec.type_name}] stop at grasp, skip joint motion")
        elif spec.motion == "lift":
            lift = handle_xyz + np.array([0.0, 0.0, 0.12], dtype=np.float64)
            motion_pts = [lift]
        elif spec.motion == "prismatic" or jtype_l == "prismatic":
            axis_dir = axis_dir * float(spec.sign)
            distance = open_signed_distance(joint_limit_urdf, joint_name, frac, self.scale[0]) * spec.slide_scale
            slide_pts = generate_prismatic_points(handle_xyz, axis_dir, distance, steps=4)
            motion_pts = slide_pts[1:] if len(slide_pts) > 1 else slide_pts
        else:
            q_now = _read_joint_q(my_concept, joint_name, env_id)
            if q_now is None:
                q_now = float(q_opened) if spec.init_open_fraction > 1e-6 else float(q_closed)
            if swing_delta is not None:
                delta_q = float(swing_delta) * json_frac * arc_sign
            elif spec.init_open_fraction > 1e-6:
                delta_q = (float(q_closed) - float(q_now)) * json_frac * arc_sign
            else:
                delta_q = float(spec.sign) * np.deg2rad(spec.max_deg * json_frac) * arc_sign
            close_deg = abs(float(np.degrees(delta_q)))
            close_deg = max(close_deg, float(spec.max_deg)) + 10.0
            step_deg = 5
            close_deg = max(step_deg, int(round(close_deg / step_deg) * step_deg))
            axis_close = np.asarray(axis_dir, dtype=np.float64)
            if float(delta_q) < 0.0:
                axis_close = -axis_close
            max_deg = close_deg
            radius = float(np.linalg.norm(_inplane(handle_xyz, axis_point_w, axis_dir) - np.asarray(axis_point_w)))
            print(
                f"[{spec.type_name}] continue on mirrored arc q_now={q_now:.3f} -> q_closed={q_closed:.3f} "
                f"delta_deg={np.degrees(delta_q):+.1f} max_deg={close_deg} step_deg={step_deg} radius={radius:.3f}"
            )
            arc_points = generate_handle_arc_points_about_axis(
                axis_point=axis_point_w,
                axis_dir=axis_close,
                handle_pos0=np.array(handle_xyz, dtype=np.float64),
                max_deg=close_deg,
                step_deg=step_deg,
            )
            motion_pts = arc_points[1:] if len(arc_points) > 1 else arc_points

        last = np.asarray(motion_pts[-1] if motion_pts else handle_xyz, dtype=np.float64)
        executor = "robot_left" if y1 > 0 else "robot_right"
        print(
            f"[{spec.type_name}] grasp=({x1:.3f},{y1:.3f},{z1:.3f}) "
            f"ee0=({approach_xyz[0]:.3f},{approach_xyz[1]:.3f},{approach_xyz[2]:.3f}) "
            f"handle=({handle_xyz[0]:.3f},{handle_xyz[1]:.3f},{handle_xyz[2]:.3f}) "
            f"joint={joint_name} type={jtype} n_move={len(motion_pts)} "
            f"mirror=({mirror_xy[0]:+.0f},{mirror_xy[1]:+.0f}) max_close_deg={max_deg} executor={executor}"
        )
        pack = [
            wp(approach_xyz, q_hand, "open", executor),
            wp(handle_xyz, q_hand, "open", executor),
            *[wp(p, q_hand, "close", executor) for p in motion_pts],
        ]
        if (not spec.stop_at_grasp) and abs(float(spec.retreat_z)) > 1e-6:
            retreat_xyz = last + r_hand.apply([0.0, 0.0, spec.retreat_z])
            pack.append(wp(retreat_xyz, q_hand, "open", executor))
        return pack
