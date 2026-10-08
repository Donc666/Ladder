from pathlib import Path
import json
import os
import numpy as np 
import random
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import re
from curobo.types import GoalToolPose, JointState
import torch


def live_root_pose_wxyz(asset, env_id=0):
    """Live root pose of an Isaac Lab rigid asset: (x, y, z), quat wxyz."""
    data = asset.data
    idx = int(env_id)
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        idx = min(idx, int(data.root_pos_w.shape[0]) - 1)
        p = data.root_pos_w[idx]
        q = data.root_quat_w[idx]
    elif hasattr(data, "root_state_w"):
        idx = min(idx, int(data.root_state_w.shape[0]) - 1)
        s = data.root_state_w[idx]
        p, q = s[0:3], s[3:7]
    else:
        raise RuntimeError("asset has neither root_pos_w/root_quat_w nor root_state_w")
    pos = (float(p[0].item()), float(p[1].item()), float(p[2].item()))
    quat = np.array(
        [float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item())],
        dtype=float,
    )
    return pos, quat


_FOOD_PLACE_Y = 0.7
_FOOD_PLACE_DX = 0.15
target_pos_x2 = 0.3
_food_place_x_by_key: dict[str, float] = {}


def next_food_place_xy(instance_key: str) -> tuple[float, float]:
    """Shared place x for hamburg/frenchfries. Each new item uses target_pos_x2, then +0.15."""
    global target_pos_x2
    key = str(instance_key)
    if key not in _food_place_x_by_key:
        _food_place_x_by_key[key] = float(target_pos_x2)
        print(
            f"[food-place] {key} target_pos_x2={target_pos_x2:.3f} "
            f"xy=[{target_pos_x2:.3f},{_FOOD_PLACE_Y:.3f}]"
        )
        target_pos_x2 = float(target_pos_x2) + _FOOD_PLACE_DX
    return float(_food_place_x_by_key[key]), float(_FOOD_PLACE_Y)


def food_pick_place_waypoints(
    asset,
    env_id,
    instance_key: str,
    executor: str | None = None,
    place: bool = True,
):
    """Live-pose grasp (and optional place at the next shared fries/hamburg slot)."""
    (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
    print(f"[food] live pose {instance_key} xyz={[round(x, 3), round(y, 3), round(z, 3)]}")
    q_xyzw = [float(q_live[1]), float(q_live[2]), float(q_live[3]), float(q_live[0])]
    r_hand_world = R.from_quat(q_xyzw) * R.from_euler("y", 90, degrees=True) * R.from_euler("x", 90, degrees=True)
    q_xyzw_h = r_hand_world.as_quat()
    q = [float(q_xyzw_h[3]), float(q_xyzw_h[0]), float(q_xyzw_h[1]), float(q_xyzw_h[2])]
    if executor is None:
        executor = "robot_left" if y > 0 else "robot_right"

    def _wp(xyz, grip):
        return {
            "pose": [float(xyz[0]), float(xyz[1]), float(xyz[2]), q[0], q[1], q[2], q[3]],
            "gripper": grip,
            "use_obstacle_avoidance": False,
            "executor": executor,
        }

    grasp = (x, y, z + 0.03)
    lift = (x, y, z + 0.32)
    wps = [
        _wp((x, y, z + 0.36), "open"),
        _wp(grasp, "open"),
        _wp(grasp, "close"),
        _wp(lift, "close"),
    ]
    if not place:
        return wps
    px, py = next_food_place_xy(instance_key)
    above = (px, py, z + 0.4)
    down = (px, py, z + 0.1)
    wps.extend(
        [
            _wp(above, "close"),
            _wp(down, "close"),
            _wp(above, "open"),
        ]
    )
    return wps


HANDLE_LINK_CANDIDATES = [
    "Trifold_Handle",
    "Cuboidal_Handle",
    "Trifold_Curve_Handle",
    "Curve_Handle",
    "Regular_drawer_id1",
    "Regular_drawer_id2"
    "Regular_drawer_id3"
]
DOOR_LINK_CANDIDATES = ["Sunken_Door", "Cuboidal_Door","Storagefurniture_body","Cylindrical_Connector","Cylindrical_Body"]




def get_handle_pose_physics(asset, scale_micwave, concept_urdf_path=None):
    # 1) 如果把手是独立 body，直接返回
    body_names = list(asset.body_names)
    print(f"the body list sadasdasdf:{body_names}")
    for handle_name in HANDLE_LINK_CANDIDATES:
        if handle_name in body_names:
            hid, _ = asset.find_bodies(handle_name)
            hstate = asset.data.body_state_w[0, int(hid[0])]
            pos = hstate[0:3].detach().cpu().numpy()
            quat_wxyz = hstate[3:7].detach().cpu().numpy()
            print(f"handle body matched: {handle_name}")
            return pos, quat_wxyz
    # 2) 否则用门的 physics 真值 + door->handle 的 joint origin 偏移
    found_door_name = None
    for door_name in DOOR_LINK_CANDIDATES:
        if door_name in body_names:
            found_door_name = door_name
            break
    if found_door_name is None:
        raise RuntimeError(f"未找到门link，候选: {DOOR_LINK_CANDIDATES}")
    did, _ = asset.find_bodies(found_door_name)
    dstate = asset.data.body_state_w[0, int(did[0])]
    door_pos = dstate[0:3].detach().cpu().numpy()
    door_quat_wxyz = dstate[3:7].detach().cpu().numpy()  # wxyz

    offset_from_urdf = None
    if concept_urdf_path:
        parsed_offset, parsed_door, parsed_handle = get_door_to_handle_offset_from_urdf(concept_urdf_path)
        if parsed_offset is not None:
            offset_from_urdf = parsed_offset * scale_micwave
            print(f"have a offest pose{parsed_offset}")
            print(
                f"door->handle offset from urdf: door={parsed_door}, handle={parsed_handle}, "
                f"offset={offset_from_urdf}"
            )

    # fallback to historical default when URDF pair is missing
    if offset_from_urdf is None:
        offset_from_urdf = np.array(
            [1.290500 * scale_micwave, -0.094500 * scale_micwave, 0.052500 * scale_micwave],
            dtype=np.float64,
        )
        print(f"use fallback door->handle offset: {offset_from_urdf}")

    qw, qx, qy, qz = door_quat_wxyz  
    R_world_door = R.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: xyzw
    handle_pos = door_pos + R_world_door @ offset_from_urdf
    handle_quat_wxyz = door_quat_wxyz.copy()  # 固定关节且rpy=0时与门同向
    return handle_pos, handle_quat_wxyz
def _xyz(s):
    if s is None:
        return np.zeros(3, dtype=np.float64)
    v = np.array([float(x) for x in s.split()], dtype=np.float64)
    return v if v.shape[0] == 3 else np.zeros(3, dtype=np.float64)
def _rpy(s):
    if s is None:
        return np.zeros(3, dtype=np.float64)
    v = np.array([float(x) for x in s.split()], dtype=np.float64)
    return v if v.shape[0] == 3 else np.zeros(3, dtype=np.float64)
def _scale_xyz(xyz, scale):
    s = np.asarray(scale, dtype=np.float64)
    if s.size == 1:
        return xyz * float(s.item())
    if s.size == 3:
        return xyz * s
    return xyz
def _T_from_pos_quat_wxyz(pos, quat_wxyz):
    w, x, y, z = quat_wxyz
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = pos
    return T
def _pos_quat_from_T(T):
    pos = T[:3, 3]
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return pos, q_wxyz
def get_link_pose_physics_or_urdf_chain(asset, link_name, urdf_path, scale, env_id=0):
    body_names = list(asset.body_names)
    # 1) 直接有独立 body 就直接读
    if link_name in body_names:
        ids, _ = asset.find_bodies(link_name)
        s = asset.data.body_state_w[env_id, int(ids[0])]
        return s[0:3].detach().cpu().numpy(), s[3:7].detach().cpu().numpy()
    # 2) 构建 child->joint 映射
    root = ET.parse(urdf_path).getroot()
    by_child = {}
    for j in root.findall("joint"):
        child = j.find("child")
        parent = j.find("parent")
        if child is None or parent is None:
            continue
        o = j.find("origin")
        a = j.find("axis")
        by_child[child.attrib["link"]] = {
            "name": j.attrib.get("name", ""),
            "type": j.attrib.get("type", "fixed"),
            "parent": parent.attrib["link"],
            "xyz": _xyz(None if o is None else o.attrib.get("xyz")),
            "rpy": _rpy(None if o is None else o.attrib.get("rpy")),
            "axis": _xyz(None if a is None else a.attrib.get("xyz")),
        }
    # 3) 找最近的“可读 body 祖先”
    cur = link_name
    chain = []  # parent->child joints, from anchor down to target
    anchor = None
    while cur in by_child:
        j = by_child[cur]
        chain.append(j)       # this joint maps parent -> cur
        cur = j["parent"]
        if cur in body_names:
            anchor = cur
            break
    if anchor is None:
        raise RuntimeError(f"找不到可用祖先body来推算 {link_name}，body_names={body_names}")
    # 4) 读祖先 body 世界位姿
    aid, _ = asset.find_bodies(anchor)
    astate = asset.data.body_state_w[env_id, int(aid[0])]
    T_world_anchor = _T_from_pos_quat_wxyz(
        astate[0:3].detach().cpu().numpy(),
        astate[3:7].detach().cpu().numpy(),
    )
    # 5) 沿链路累乘 T_anchor_target
    T_anchor_target = np.eye(4, dtype=np.float64)
    for j in reversed(chain):  # reversed: anchor->...->target
        T = np.eye(4, dtype=np.float64)
        xyz = _scale_xyz(j["xyz"], scale)
        rpy = j["rpy"]
        T[:3, 3] = xyz
        T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        jt = j["type"]
        if jt == "revolute":
            q = 0.0
            if j["name"] in list(asset.joint_names):
                jid = list(asset.joint_names).index(j["name"])
                q = float(asset.data.joint_pos[env_id, jid].item())
            Tr = np.eye(4); Tr[:3, :3] = R.from_rotvec(j["axis"] * q).as_matrix()
            T = T @ Tr
        elif jt == "prismatic":
            d = 0.0
            if j["name"] in list(asset.joint_names):
                jid = list(asset.joint_names).index(j["name"])
                d = float(asset.data.joint_pos[env_id, jid].item())
            Tp = np.eye(4); Tp[:3, 3] = _scale_xyz(j["axis"] * d, scale)
            T = T @ Tp
        T_anchor_target = T_anchor_target @ T
    T_world_target = T_world_anchor @ T_anchor_target
    return _pos_quat_from_T(T_world_target)


def get_first_available_handle_pose(
    asset,
    urdf_path: str,
    scale,
    env_id: int = 0,
    candidates: list[str] | None = None,
):
    if candidates is None:
        candidates = [
            "L_Shaped_Sidehandle",
            "Trifold_Sidehandle",
            "Cuboidal_Sidehandle",
            "Sidehandle",
        ]
    last_err = None
    for name in candidates:
        try:
            pos, quat = get_link_pose_physics_or_urdf_chain(
                asset=asset,
                link_name=name,
                urdf_path=urdf_path,
                scale=scale,
                env_id=env_id,
            )
            print(f"[INFO] handle matched: {name}")
            return pos, quat, name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"所有把手候选都失败: {candidates}, body_names={list(asset.body_names)}, last_err={last_err}"
    )
def get_link_pose_physics(asset, link_name: str, env_id: int = 0):
    """
    通用版：像 get_handle_pose_physics 一样，从物理状态直接取某个 link 的 world pose。
    返回: (pos_xyz, quat_wxyz)
    """
    body_names = list(asset.body_names)
    print(f"[DEBUG] body_names={body_names}")
    if link_name not in body_names:
        raise RuntimeError(f"未找到link: {link_name}, 可选: {body_names}")
    ids, _ = asset.find_bodies(link_name)
    if len(ids) == 0:
        raise RuntimeError(f"find_bodies 失败: {link_name}")
    bid = int(ids[0])
    state = asset.data.body_state_w[env_id, bid]
    pos = state[0:3].detach().cpu().numpy()
    quat_wxyz = state[3:7].detach().cpu().numpy()
    print(f"[INFO] link matched: {link_name}, pos={pos}, quat_wxyz={quat_wxyz}")
    return pos, quat_wxyz


def map_concept_point_to_real_xy(
    point_xyz: np.ndarray, concept_root_pos_w: np.ndarray, real_root_pos_w: np.ndarray
) -> np.ndarray:
    """
    Map a point from concept-world frame to real-world frame using XY translation only.
    Assumption (as discussed): same axes, same yaw, no fixed frame bias, only XY offset.
    """
    mapped = np.array(point_xyz, dtype=np.float64).copy()
    # real_root_pos_w[0] -= 1
    # real_root_pos_w[1] += 1
    mapped[0] += float(real_root_pos_w[0] - concept_root_pos_w[0])
    mapped[1] += float(real_root_pos_w[1] - concept_root_pos_w[1])
    return mapped

def get_revolute_hinge_by_child(
    urdf_path: str,
    child_link: str,
    convert_axis_urdf_to_sim=None,  # 可传你已有的 _urdf_xyz_to_sim_yzx
):
    """
    在 URDF 里按 child_link 精确查找 revolute 关节。
    返回: (origin_xyz, axis_xyz, parent_link, child_link, joint_name)
    找不到时返回: (None, None, None, None, None)
    """
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None, None, None
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None, None, None, None, None
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype != "revolute":
            continue
        child_elem = joint.find("child")
        parent_elem = joint.find("parent")
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")
        if child_elem is None or parent_elem is None:
            continue
        if child_elem.attrib.get("link", "") != child_link:
            continue
        # origin: 默认 [0,0,0]
        origin_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if origin_elem is not None and origin_elem.attrib.get("xyz"):
            origin_xyz = _parse_xyz_attr(origin_elem.attrib["xyz"])
        # axis: 默认 [1,0,0]
        axis_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis_elem is not None and axis_elem.attrib.get("xyz"):
            axis_xyz = _parse_xyz_attr(axis_elem.attrib["xyz"])
        if convert_axis_urdf_to_sim is not None:
            axis_xyz = convert_axis_urdf_to_sim(axis_xyz)
        n = np.linalg.norm(axis_xyz)
        if n > 1e-8:
            axis_xyz = axis_xyz / n
        parent_link = parent_elem.attrib.get("link", "")
        joint_name = joint.attrib.get("name", "")
        return origin_xyz, axis_xyz, parent_link, child_link, joint_name
    return None, None, None, None, None


def parse_partnet_semantics(model_dir: str | Path) -> list[tuple[str, str, str]]:
    """Parse semantics.txt rows: (link_name, joint_kind, part_name)."""
    path = Path(model_dir) / "semantics.txt"
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 3:
            continue
        rows.append((parts[0], parts[1].lower(), " ".join(parts[2:]).lower()))
    return rows


def find_semantic_link(model_dir: str | Path, keywords: list[str]) -> str | None:
    keys = [k.lower() for k in keywords if k]
    for link, _kind, part in parse_partnet_semantics(model_dir):
        if any(k in part or k in link.lower() for k in keys):
            return link
    return None


def get_movable_joint_from_urdf(
    urdf_path: str,
    child_link: str | None = None,
    types: tuple[str, ...] = ("revolute", "prismatic", "continuous"),
    child_keywords: list[str] | None = None,
):
    """Return (origin, axis, parent, child, joint_name, joint_type) for a movable joint."""
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None, None, None, None
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None, None, None, None, None, None
    allowed = {t.lower() for t in types}
    keys = [k.lower() for k in (child_keywords or [])]
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        if jtype not in allowed:
            continue
        child_elem = joint.find("child")
        parent_elem = joint.find("parent")
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")
        if child_elem is None or parent_elem is None:
            continue
        child_name = child_elem.attrib.get("link", "")
        if child_link and child_name != child_link:
            continue
        if keys and not any(k in child_name.lower() for k in keys):
            continue
        origin_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if origin_elem is not None and origin_elem.attrib.get("xyz"):
            origin_xyz = _parse_xyz_attr(origin_elem.attrib["xyz"])
        axis_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis_elem is not None and axis_elem.attrib.get("xyz"):
            axis_xyz = _parse_xyz_attr(axis_elem.attrib["xyz"])
        n = np.linalg.norm(axis_xyz)
        if n > 1e-8:
            axis_xyz = axis_xyz / n
        parent_name = parent_elem.attrib.get("link", "")
        joint_name = joint.attrib.get("name", "")
        return origin_xyz, axis_xyz, parent_name, child_name, joint_name, jtype
    return None, None, None, None, None, None


def generate_prismatic_points(start_pos, axis_dir, distance: float, steps: int = 6):
    p0 = np.asarray(start_pos, dtype=np.float64)
    u = np.asarray(axis_dir, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-8:
        raise ValueError("prismatic axis norm is too small")
    u = u / n
    pts = []
    for i in range(steps + 1):
        t = distance * (i / max(steps, 1))
        p = p0 + u * t
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    return pts


def first_urdf_in_category(root: Path, category: str) -> str | None:
    cat = root / category
    if not cat.is_dir():
        return None
    for model in sorted(p for p in cat.iterdir() if p.is_dir()):
        for name in (f"{category}.urdf", "mobility.urdf"):
            cand = model / name
            if cand.exists():
                return str(cand)
        urdfs = sorted(model.glob("*.urdf"))
        if urdfs:
            return str(urdfs[0])
    return None


def pick_partnet_mobility_urdf(
    category: str,
    root: Path | None = None,
    keywords: list[str] | None = None,
    require_concept: bool = False,
    require_concept_needles: list[str] | None = None,
) -> str | None:
    base = Path(root) if root is not None else PARTNET_MOBILITY_ROOT
    cat = base / category
    if not cat.is_dir():
        return None
    models = [p for p in cat.iterdir() if p.is_dir() and (p / "mobility.urdf").exists()]
    if keywords:
        matched = [p for p in models if find_semantic_link(p, keywords)]
        if matched:
            models = matched
    needles = [n.lower() for n in (require_concept_needles or []) if n]
    if require_concept or needles:
        if not _concept_category_exists(category):
            print(f"[INFO] PartNet {category}: no concept-template folder, skip pairing filter")
        else:
            paired = []
            for model in models:
                urdf = model / "mobility.urdf"
                concept_urdf, _mid = resolve_concept_urdf_path(
                    str(urdf), category=category, allow_fallback=False, verbose=False
                )
                if not concept_urdf:
                    continue
                if needles:
                    try:
                        text = Path(concept_urdf).read_text(encoding="utf-8", errors="ignore").lower()
                    except OSError:
                        continue
                    if not any(n in text for n in needles):
                        continue
                paired.append(model)
            if paired:
                models = paired
                extra = f" needles={needles}" if needles else ""
                print(f"[INFO] PartNet {category}: {len(models)} models with paired concept{extra}")
            else:
                print(
                    f"[WARN] no PartNet {category} model has a matching concept template"
                    + (f" containing {needles}" if needles else "")
                )
    if not models:
        return None
    chosen = _unseeded_choice(models)
    return str(chosen / "mobility.urdf")


def _unseeded_choice(seq):
    """Pick one item without following random.seed (Isaac/RL often freeze the module RNG)."""
    if not seq:
        raise IndexError("empty sequence")
    return random.SystemRandom().choice(seq)


def pick_concept_urdf_containing(category: str, needle: str, root: Path | None = None) -> str | None:
    """Randomly pick a concept-template URDF whose text contains needle."""
    base = Path(root) if root is not None else PARTNET_CONCEPT_ROOT
    cat = base / category
    if not cat.is_dir():
        return None
    key = needle.lower()
    hits = []
    for model in sorted(p for p in cat.iterdir() if p.is_dir()):
        for urdf in sorted(model.glob("*.urdf")):
            try:
                text = urdf.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if key in text.lower():
                hits.append(urdf)
    if not hits:
        return None
    chosen = _unseeded_choice(hits)
    print(f"[INFO] random {category} URDF ({needle}) {hits.index(chosen) + 1}/{len(hits)} -> {chosen}")
    return str(chosen)


def get_door_hinge_from_urdf(urdf_path: str):
    """
    Parse URDF and return hinge definition for the first matched door joint:
    - child link must be a door candidate
    - parent link can be any link (e.g. Cuboidal_Body/world)
    Returns: (origin_xyz, axis_xyz, parent_link, child_link) in parent frame.
    """
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None, None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        print(f"failed to parse urdf for door hinge: {urdf_path}, err={exc}")
        return None, None, None, None

    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        origin_elem = joint.find("origin")
        axis_elem = joint.find("axis")
        if parent_elem is None or child_elem is None or origin_elem is None:
            continue
        child_name = child_elem.attrib.get("link", "")
        if child_name not in DOOR_LINK_CANDIDATES:
            continue
        parent_name = parent_elem.attrib.get("link", "")
        xyz_text = origin_elem.attrib.get("xyz")
        if not xyz_text:
            continue
        try:
            origin_xyz = _parse_xyz_attr(xyz_text)
        except ValueError:
            continue
        # default axis for revolute door if missing
        axis_xyz = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if axis_elem is not None and axis_elem.attrib.get("xyz"):
            try:
                axis_xyz = _parse_xyz_attr(axis_elem.attrib.get("xyz"))
                axis_xyz = _urdf_xyz_to_sim_yzx(axis_xyz)
            except ValueError:
                pass
        norm = np.linalg.norm(axis_xyz)
        if norm > 1e-8:
            axis_xyz = axis_xyz / norm
        return origin_xyz, axis_xyz, parent_name, child_name
    return None, None, None, None

def _parse_xyz_attr(xyz_text: str):
    values = [float(v) for v in xyz_text.strip().split()]
    if len(values) != 3:
        raise ValueError(f"invalid xyz string: {xyz_text}")
    return np.array(values, dtype=np.float64)

def _urdf_xyz_to_sim_yzx(vec_xyz: np.ndarray) -> np.ndarray:
    """Convert URDF xyz ordering to simulator yzx ordering."""
    v = np.asarray(vec_xyz, dtype=np.float64)
    return np.array([v[2], v[0], v[1]], dtype=np.float64)

def transform_joint_axis_to_world(parent_pos_w, parent_quat_wxyz, joint_origin_parent, axis_parent):
    """
    Convert joint origin/axis from parent frame to world frame.
    Returns (axis_point_world, axis_dir_world).
    """
    rot_parent = _quat_wxyz_to_rotmat(np.asarray(parent_quat_wxyz, dtype=np.float64))
    p_parent = np.asarray(parent_pos_w, dtype=np.float64)
    axis_point_w = p_parent + rot_parent @ np.asarray(joint_origin_parent, dtype=np.float64)
    axis_dir_w = rot_parent @ np.asarray(axis_parent, dtype=np.float64)
    norm = np.linalg.norm(axis_dir_w)
    if norm > 1e-8:
        axis_dir_w = axis_dir_w / norm
    return axis_point_w, axis_dir_w

def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def generate_handle_arc_points_about_axis(
    axis_point,
    axis_dir,
    handle_pos0,
    max_deg=50,
    step_deg=10,
):
    """
    Generate handle trajectory by rotating initial handle position around an arbitrary axis
    using Rodrigues' rotation formula.
    """
    c = np.asarray(axis_point, dtype=np.float64)
    u = np.asarray(axis_dir, dtype=np.float64)
    p0 = np.asarray(handle_pos0, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-8:
        raise ValueError("axis_dir norm is too small")
    u = u / norm_u
    r0 = p0 - c

    pts = []
    for deg in range(0, max_deg + 1, step_deg):
        th = np.deg2rad(deg)
        r = (
            r0 * np.cos(th)
            + np.cross(u, r0) * np.sin(th)
            + u * np.dot(u, r0) * (1.0 - np.cos(th))
        )
        p = c + r
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    return pts

def plan_concat_segments(planner, q_start, arc_points, quat=[0.707, 0.0, 0.707, 0.0]):
    segs = []
    q_curr = q_start
    for i in range(len(arc_points) - 1):
        target = arc_points[i + 1]   # 从当前到下一个点
        goal_pose = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor([[[[target]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[quat]]]], device="cuda", dtype=torch.float32),
        )
        result = planner.plan_pose(goal_pose, q_curr)
        if result is None:
            print(f"[WARN] segment {i}->{i+1} planning failed")
            break
        traj = result.get_interpolated_plan().position.squeeze(0)  # [T, dof]
        joint_pos_des = traj[0]

        print(f"segment trajectory length: {joint_pos_des.shape[0]}")

        # 拼接时去掉后续段第一个点，避免重复
        if len(segs) > 0 and traj.shape[0] > 1:
            joint_pos_des = joint_pos_des[1:]
        segs.append(joint_pos_des)
        # 下一段起点 = 本段终点
        joint_each_time_pos_des = joint_pos_des[joint_pos_des.shape[0]-1]
        joint_each_time_pos_des[:7].unsqueeze(0)
        q_curr = JointState.from_position(
            joint_each_time_pos_des[:7].unsqueeze(0),
            joint_names=planner.joint_names,
        )
    if len(segs) == 0:
        return None
    return torch.cat(segs, dim=0)


def move_to_pose(planner,q_start,goal_pose):
    
    result = planner.plan_pose(goal_pose, q_start)
    
    if result == None:
        print("规划失败！")
        simulation_app.close()
        return
    # 获取插值后的轨迹
    traj = result.get_interpolated_plan()
    trajectory_positions = traj.position.squeeze(0)
    joint_pos_des = trajectory_positions[0]
    return joint_pos_des


def get_door_to_handle_offset_from_urdf(urdf_path: str):
    """
    Parse fixed/revolute/prismatic joint origins and return the first matched
    door->handle offset in URDF local frame.
    """
    path = Path(urdf_path)
    if not path.exists():
        return None, None, None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        print(f"failed to parse urdf for offset: {urdf_path}, err={exc}")
        return None, None, None

    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        origin_elem = joint.find("origin")
        if parent_elem is None or child_elem is None or origin_elem is None:
            continue
        parent_name = parent_elem.attrib.get("link", "")
        child_name = child_elem.attrib.get("link", "")
        if parent_name not in DOOR_LINK_CANDIDATES or child_name not in HANDLE_LINK_CANDIDATES:
            continue
        xyz_text = origin_elem.attrib.get("xyz")
        if not xyz_text:
            continue
        try:
            offset = _parse_xyz_attr(xyz_text)
        except ValueError:
            continue
        return offset, parent_name, child_name

    return None, None, None

def hand_rotate(current_quat,angle_deg:float):
    r_current = R.from_quat(current_quat)
    
    angle_rad1 = np.deg2rad(angle_deg)
    angle_rad2 = np.deg2rad(-90)
    

 #rotate around the z axis
    r_delta1 = R.from_euler('x', angle_rad1)
    r_delta2 = R.from_euler('y', angle_rad2)

    r_new = r_current * r_delta1 * r_delta2


    q_new = tuple(r_new.as_quat())
    q_new = xyzw_to_wxyz(q_new)
    
    return q_new,angle_deg

def rotate_z(current_quat,angle_deg:float):
    current_quat = wxyz_to_xyzw(current_quat)
    r_current = R.from_quat(current_quat)
    
    angle_rad = np.deg2rad(angle_deg)
    
    

 #rotate around the z axis
    r_delta = R.from_euler('z', angle_rad)
   

    r_new = r_current * r_delta


    q_new = tuple(r_new.as_quat())
    q_new = xyzw_to_wxyz(q_new)
    
    return q_new,angle_deg

def rotate_x(current_quat,angle_deg:float):
    current_quat = wxyz_to_xyzw(current_quat)
    r_current = R.from_quat(current_quat)
    
    angle_rad = np.deg2rad(angle_deg)
    
    

 #rotate around the x axis
    r_delta = R.from_euler('x', angle_rad)
   

    r_new = r_current * r_delta


    q_new = tuple(r_new.as_quat())
    q_new = xyzw_to_wxyz(q_new)
    
    return q_new

def xyzw_to_wxyz(q):
    x, y, z, w = q
    return (w, x, y, z)
def wxyz_to_xyzw(q):
    w, x, y, z = q
    return (x, y, z, w)


def live_root_pose_wxyz(asset, env_id: int = 0):
    """Current world-frame root pose from simulation (or an import-pose wrap)."""
    data = asset.data
    n = int(data.root_pos_w.shape[0])
    idx = min(max(int(env_id), 0), max(n - 1, 0))
    pos = data.root_pos_w[idx]
    if hasattr(data, "root_quat_w"):
        quat = data.root_quat_w[idx]
    else:
        quat = data.root_state_w[idx, 3:7]
    xyz = (float(pos[0].item()), float(pos[1].item()), float(pos[2].item()))
    qwxyz = tuple(float(quat[i].item()) for i in range(4))
    return xyz, qwxyz
def _norm_model_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _concept_category_exists(category: str, concept_root: str | Path | None = None) -> bool:
    root = Path(concept_root) if concept_root is not None else PARTNET_CONCEPT_ROOT
    cats = []
    alias = CONCEPT_CAT_ALIASES.get(category)
    if alias:
        cats.append(alias)
    cats.append(category)
    return any((root / cat).is_dir() for cat in cats)


def find_concept_model_dir(
    category: str, model_id: str, concept_root: str | Path | None = None
) -> Path | None:
    """Find concept-template folder whose name matches PartNet model_id.

    IDs may appear as UUID-with-hyphens, UUID-with-underscores (from robot name
    ``partnet_<id>``), or 32-hex strings. Trailing ``-0``/``-1`` variants are allowed.
    """
    if not model_id:
        return None
    root = Path(concept_root) if concept_root is not None else PARTNET_CONCEPT_ROOT
    cats = []
    alias = CONCEPT_CAT_ALIASES.get(category)
    if alias:
        cats.append(alias)
    cats.append(category)
    candidates = [model_id, model_id.replace("_", "-"), model_id.replace("-", "_")]
    for cat in cats:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for cand in candidates:
            d = cat_dir / cand
            if d.is_dir():
                return d
        key = _norm_model_id(model_id)
        if not key:
            continue
        hits = []
        for d in cat_dir.iterdir():
            if not d.is_dir():
                continue
            nk = _norm_model_id(d.name)
            nk_base = _norm_model_id(re.sub(r"-\d+$", "", d.name))
            if nk == key or nk_base == key:
                hits.append(d)
        if hits:
            hits.sort(key=lambda p: (len(p.name), p.name))
            return hits[0]
    return None


def _first_urdf_in_dir(model_dir: Path, category: str) -> str | None:
    for name in (f"{category}.urdf", "Window.urdf", "Microwave.urdf", "mobility.urdf"):
        cand = model_dir / name
        if cand.exists():
            return str(cand)
    urdfs = sorted(model_dir.glob("*.urdf"))
    return str(urdfs[0]) if urdfs else None


def resolve_concept_urdf_path(
    source_urdf_path: str,
    concept_root: str | None = None,
    category: str = "Refrigerator",
    allow_fallback: bool = True,
    verbose: bool = True,
) -> tuple[str | None, str | None]:
    """Resolve the concept URDF that 1:1-matches a PartNet-Mobility source URDF."""
    if concept_root is None:
        concept_root = str(PARTNET_CONCEPT_ROOT)
    model_id = extract_model_id_from_urdf(source_urdf_path)
    matched_dir = find_concept_model_dir(category, model_id or "", concept_root)
    if matched_dir is not None:
        urdf = _first_urdf_in_dir(matched_dir, category)
        if urdf:
            if verbose:
                print(f"[INFO] paired concept model_id={model_id} -> {urdf}")
            return urdf, model_id

    if not allow_fallback:
        if verbose:
            print(f"[WARN] no concept template for {category} model_id={model_id}")
        return None, model_id

    root = Path(concept_root)
    cats = []
    alias = CONCEPT_CAT_ALIASES.get(category)
    if alias:
        cats.append(alias)
    cats.append(category)
    for cat in cats:
        fallback = first_urdf_in_category(root, cat)
        if fallback:
            print(f"[WARN] concept model_id={model_id} missing, fallback {cat}: {fallback}")
            return fallback, model_id

    if source_urdf_path and Path(source_urdf_path).exists():
        print(f"[WARN] no concept URDF for {category}/{model_id}, use source {source_urdf_path}")
        return source_urdf_path, model_id
    return None, model_id

def random_rotate(current_quat,random_degree_range:float):
    r_current = R.from_quat(current_quat)

    # 2. 生成绕 Z 轴的随机角度 (弧度)
    angle_deg = np.random.uniform(-random_degree_range, random_degree_range)

    angle_rad = np.deg2rad(angle_deg)
    print(f"随机旋转角度: {angle_deg:.2f} 度")

    # 3. 创建增量旋转 (绕自身 Z 轴)
    r_delta = R.from_euler('z', angle_rad)

    # 4. 相对于自身坐标系旋转（右乘：current * delta）
    r_new = r_current * r_delta

    # 5. 获取结果
    q_new = tuple(r_new.as_quat())

    return q_new,angle_deg

def random_rotate_mic(current_quat,random_degree_range:float):
    r_current = R.from_quat(current_quat)

    # 2. 生成绕 Z 轴的随机角度 (弧度)
    angle_deg = np.random.uniform(-random_degree_range, random_degree_range)

    angle_rad = np.deg2rad(angle_deg)
    print(f"随机旋转角度: {angle_deg:.2f} 度")

    # 3. 创建增量旋转 (绕自身 Z 轴)
    r_delta = R.from_euler('x', angle_rad)

    # 4. 相对于自身坐标系旋转（右乘：current * delta）
    r_new = r_current * r_delta

    # 5. 获取结果
    q_new = tuple(r_new.as_quat())

    return q_new,angle_deg

def rotate_y(current_quat,angle_deg:float):
    current_quat = wxyz_to_xyzw(current_quat)
    r_current = R.from_quat(current_quat)
    
    angle_rad = np.deg2rad(angle_deg)
    
    

 #rotate around the z axis
    r_delta = R.from_euler('y', angle_rad)
   

    r_new = r_current * r_delta


    q_new = tuple(r_new.as_quat())
    q_new = xyzw_to_wxyz(q_new)
    
    return q_new


def rotate_end_effect(current_quat,random_degree):
    r_current = R.from_quat(current_quat)
    rad1 = np.deg2rad(random_degree)
    r_delta1 = R.from_euler('x', rad1)
    rad2 = np.deg2rad(-90)
    r_delta2 = R.from_euler('y', rad2)

    # 4. 相对于自身坐标系旋转（右乘：current * delta）
    r_new = r_current * r_delta1
    r_new = r_current * r_delta2

    # 5. 获取结果
    q_new = r_new.as_quat()
    q_new = xyzw_to_wxyz(q_new)

    return q_new


def extract_model_id_from_urdf(urdf_path: str) -> str | None:
    """Read URDF robot name and derive model_id."""
    path = Path(urdf_path)
    if not path.exists():
        print(f"[WARN] URDF 不存在: {urdf_path}")
        return None
    try:
        # Robot name is usually in first few lines.
        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:10])
    except OSError as exc:
        print(f"[WARN] 读取 URDF 失败: {urdf_path}, error={exc}")
        return None

    match = re.search(r'<robot\s+name="([^"]+)"', head)
    model_id = None
    if match:
        robot_name = match.group(1).strip()
        if robot_name.startswith("partnet_"):
            rest = robot_name[len("partnet_") :]
            parts = rest.split("_")
            # UUID written with underscores: 8_4_4_4_12
            if len(parts) >= 5 and all(parts[:5]) and len(parts[0]) == 8:
                model_id = "-".join(parts[:5])
                extra = "-".join(parts[5:])
                if extra:
                    model_id = f"{model_id}-{extra}"
            else:
                model_id = rest
        else:
            id_match = re.search(r"([0-9a-fA-F]{8,})", robot_name)
            model_id = id_match.group(1) if id_match else robot_name
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        try:
            meta_id = json.loads(meta_path.read_text(encoding="utf-8")).get("model_id")
            if meta_id:
                return str(meta_id)
        except Exception:
            pass
    if model_id:
        return model_id
    print(f"[WARN] 未在 URDF 中解析到 robot name: {urdf_path}")
    return None


def _as_project_root(path: Path | str | None = None) -> Path:
    """Directory of this repo, even if a star-import passed in a .py file path."""
    p = Path(__file__ if path is None else path).resolve()
    return p.parent if p.is_file() else p


PROJECT_ROOT = _as_project_root()
CONVERTED_ROOT = PROJECT_ROOT / "assets"
# Datasets live inside this repo. Override with env vars if you keep them elsewhere.
PARTNET_MOBILITY_ROOT = Path(os.environ.get("PARTNET_MOBILITY_ROOT", PROJECT_ROOT / "PartNetMobility"))
PARTNET_CONCEPT_ROOT = Path(os.environ.get("PARTNET_CONCEPT_ROOT", PROJECT_ROOT / "Partnet_Mobility_urdf"))

# Articulated types only. Rigid types (laptop, window, microwave, storagefurniture)
# resolve under assets/ via pick_random_urdf.
TYPE_TO_PARTNET_CAT = {
    "suitcase": "Suitcase",
    "toilet": "Toilet",
    "clock": "Clock",
    "oven": "Oven",
    "trashcan": "TrashCan",
    "display": "Display",
    "faucet": "Faucet",
    "foldingchair": "FoldingChair",
    "printer": "Printer",
    "kettle": "Kettle",
    "table_drawer": "Table",
    "partnet_door": "Door",
    "box": "Box",
    "coffeemachine": "CoffeeMachine",
    "stapler_lid": "Stapler",
    "dishwasher": "Dishwasher",
    "partnet_dispenser": "Dispenser",
    "partnet_fan": "Fan",
    "refrigerator": "Refrigerator",
    "lamp": "Lamp",
    "globe": "Globe",
    "washingmachine": "WashingMachine",
    "partnet_pot": "KitchenPot",
    "partnet_microwave": "Microwave",
    "partnet_switch": "Switch",
    "partnet_window": "Window",
    "partnet_laptop": "Laptop",
    "storage_drawer": "StorageFurniture",
}

# PartNet-Mobility category -> concept-template folder under Partnet_Mobility_urdf
CONCEPT_CAT_ALIASES = {
    "TrashCan": "Trashcan",
    "StorageFurniture": "Cabinet",
    "FoldingChair": "Chair",
    "WashingMachine": "Washingmachine",
    "KitchenPot": "kitchenpotPot",
}

TYPE_TO_SEMANTIC_KEYS = {
    "laptop": ["screen"],
    "suitcase": ["handle"],
    "window": ["window", "translation"],
    "toilet": ["lever", "lid"],
    "clock": ["hand"],
    "oven": ["door"],
    "trashcan": ["lid"],
    "display": ["screen"],
    "faucet": ["switch"],
    "foldingchair": ["seat"],
    "stapler_lid": ["cover", "lid"],
    "printer": ["button", "slider"],
    "kettle": ["handle"],
    "table": ["drawer"],
    "table_drawer": ["drawer"],
    "door": ["door"],
    "partnet_door": ["handle"],
    "box": ["lid", "cover"],
    "coffeemachine": ["knob"],
    "dishwasher": ["door", "button"],
    "partnet_dispenser": ["lid"],
    "partnet_fan": ["rotor"],
    "refrigerator": ["door"],
    "lamp": ["head", "bar", "button"],
    "globe": ["sphere"],
    "washingmachine": ["door"],
    "partnet_pot": ["lid"],
    "partnet_microwave": ["door"],
    "partnet_switch": ["toggle", "button", "switch"],
    "partnet_window": ["window", "translation"],
    "partnet_laptop": ["screen"],
    "storage_drawer": ["drawer"],
}

# When pairing a PartNet asset with its concept template, the concept URDF
# must contain at least one of these strings (e.g. handle link names).
TYPE_TO_CONCEPT_NEEDLES = {
    "window": ["handle"],
    "suitcase": ["handle"],
    "toilet": ["lever"],
    "clock": ["hand"],
    "oven": ["handle", "door"],
    "trashcan": ["cover", "lid"],
    "display": ["screen"],
    "faucet": ["switch"],
    "foldingchair": ["seat"],
    "stapler_lid": ["cover", "lid"],
    "printer": ["button", "slider"],
    "kettle": ["handle"],
    "table_drawer": ["drawer"],
    "partnet_door": ["handle"],
    "box": ["cover", "lid", "drawer"],
    "coffeemachine": ["knob", "lever", "button"],
    "laptop": ["screen"],
    "dishwasher": ["handle", "door", "button"],
    "partnet_dispenser": ["lid", "nozzle", "cover"],
    "partnet_fan": ["rotor"],
    "refrigerator": ["handle", "door"],
    "lamp": ["head", "bar", "button"],
    "globe": ["sphere"],
    "washingmachine": ["handle", "door"],
    "partnet_pot": ["cover", "lid"],
    "partnet_microwave": ["handle", "door"],
    "partnet_switch": ["switch", "toggle"],
    "partnet_window": ["handle", "window"],
    "partnet_laptop": ["screen"],
    "storage_drawer": ["drawer"],
}


def _norm_name(s: str) -> str:
    """Normalize: lowercase + keep only [a-z0-9]."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())
def _strip_prefix_num(name: str) -> str:
    """
    Strip leading numeric prefix and separators.
    Examples:
      '019_alarm-clock' -> 'alarm-clock'
      '12-remote'       -> 'remote'
      'bottle'          -> 'bottle'
    """
    return re.sub(r"^\d+[_-]*", "", name)
def _dir_norm_keys(dir_name: str) -> tuple[str, str]:
    """
    Return two normalized forms for a directory name:
      1) full normalized name
      2) normalized name after stripping numeric prefix
    """
    full_norm = _norm_name(dir_name)
    stripped_norm = _norm_name(_strip_prefix_num(dir_name))
    return full_norm, stripped_norm
def _match_dirs_by_exact_normalized(root: Path, query: str) -> list[Path]:
    """
    Strict equality matching after normalization.
    A dir matches if either:
      norm(dir_name) == norm(query)
      norm(strip_prefix_num(dir_name)) == norm(query)
    """
    nq = _norm_name(query)
    if not nq:
        return []
    matched = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        full_norm, stripped_norm = _dir_norm_keys(d.name)
        if full_norm == nq or stripped_norm == nq:
            matched.append(d)
    return matched
def _collect_urdf_files(chosen_dir: Path) -> list[Path]:
    urdf_dir = chosen_dir / "urdf"
    if urdf_dir.is_dir():
        files = sorted(urdf_dir.glob("*.urdf"))
    else:
        files = sorted(chosen_dir.rglob("*.urdf"))
    return files
def pick_random_urdf(
    asset_item: dict, root: Path = CONVERTED_ROOT, require_concept: bool = False
) -> str:
    """
    Pick one URDF by strict normalized equality.
    Priority:
      1) asset_item["asset_name"]
      2) asset_item["id"]
      3) asset_item["type"]
    Notes:
    - 'alarmclock' can match folder 'alarm-clock'
    - 'alarmclock' can match folder '019_alarm-clock'
    - Uses strict equality after normalization (NOT substring contains)
    """
    raw_type = str(asset_item.get("type") or "").strip()
    base_type = raw_type[:-8] if raw_type.endswith("_concept") else raw_type
    partnet_cat = TYPE_TO_PARTNET_CAT.get(base_type)
    if partnet_cat:
        needles = TYPE_TO_CONCEPT_NEEDLES.get(base_type) if require_concept else None
        partnet_urdf = pick_partnet_mobility_urdf(
            partnet_cat,
            keywords=TYPE_TO_SEMANTIC_KEYS.get(base_type),
            require_concept=require_concept,
            require_concept_needles=needles,
        )
        if partnet_urdf:
            print(f"[INFO] pick PartNet-Mobility URDF type={raw_type} -> {partnet_urdf}")
            return partnet_urdf
    if not root.exists():
        raise FileNotFoundError(f"converted_assets root not found: {root}")
    # Priority keys
    priority_fields = ("asset_name", "id", "type")
    tried = []
    candidate_dirs = []
    for field in priority_fields:
        v = asset_item.get(field)
        if not (isinstance(v, str) and v.strip()):
            continue
        query = v.strip()
        tried.append((field, query, _norm_name(query)))
        matches = _match_dirs_by_exact_normalized(root, query)
        if matches:
            candidate_dirs = matches
            break  # stop at first priority level with matches
    if not candidate_dirs:
        tried_text = ", ".join([f"{k}='{q}'(norm='{nq}')" for k, q, nq in tried]) or "none"
        raise FileNotFoundError(
            f"No asset directory matched by strict normalized equality for item={asset_item}. "
            f"Tried: {tried_text} under root={root}"
        )
    chosen_dir = random.choice(candidate_dirs)
    urdf_files = _collect_urdf_files(chosen_dir)
    if not urdf_files:
        raise FileNotFoundError(f"No .urdf found in matched directory: {chosen_dir}")
    return str(random.choice(urdf_files))