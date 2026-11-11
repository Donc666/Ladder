from my_utils import *
from set_interface_try import *
import random
import xml.etree.ElementTree as ET
from window_concept import random_degree,random_degree2
import window_concept as window_concept_mod


ZONES = [
    ((0.55, 0.70), (-0.15, 0.15)),
]


def sample_pos_by_instance(instance_name: str):
    m = re.search(r"(?:^|__)env(\d+)$", instance_name)
    if m:
        idx = int(m.group(1))
    else:
        m2 = re.search(r"_(\d+)$", instance_name)
        if not m2:
            raise ValueError(f"Cannot parse index from instance_name={instance_name}")
        idx = int(m2.group(1)) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y


def _joint_limits(urdf_path: str, joint_name: str, default: float = 0.3):
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


def _open_signed_distance(urdf_path: str, joint_name: str, fraction: float, scale: float) -> float:
    """Halfway-open travel along +joint-axis. Sign goes from the closed end (near 0)."""
    lo, hi = _joint_limits(urdf_path, joint_name)
    frac = float(np.clip(fraction, 0.05, 1.0))
    span = hi - lo
    if abs(lo) <= abs(hi):
        delta = frac * span
    else:
        delta = -frac * span
    return float(delta) * float(scale)


def _joints_at_open_fraction(urdf_path: str, base: dict[str, float], fraction: float) -> dict[str, float]:
    """Move each sliding/hinge joint from the closed end by ``fraction`` of its travel."""
    pos = dict(base)
    frac = float(np.clip(fraction, 0.05, 1.0))
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return pos
    for joint in root.findall("joint"):
        jtype = (joint.attrib.get("type") or "").lower()
        name = joint.attrib.get("name")
        if jtype not in {"revolute", "prismatic"} or not name:
            continue
        lo, hi = _joint_limits(urdf_path, name)
        span = hi - lo
        delta = frac * span if abs(lo) <= abs(hi) else -frac * span
        pos[name] = float(min(hi, max(lo, delta)))
    return pos


def _safe_default_joint_pos(urdf_path: str) -> dict[str, float]:
    """Isaac Lab rejects default 0 when 0 is outside URDF joint limits."""
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


def _pick_slider_link(asset, preferred: str | None):
    names = list(asset.body_names)
    if preferred and preferred in names:
        return preferred
    for name in names:
        low = name.lower()
        if low.startswith("link_") and "base" not in low:
            return name
    for name in names:
        if name.lower() not in {"base", "world"}:
            return name
    raise RuntimeError(f"No slider link in body_names={names}")


def _map_window_point(point_w, concept_root, real_root):
    """Copy a concept-world point onto the real window.

    The concept root is yawed and lowered so the mesh matches the PartNet
    window. The URDF frame origin still sits 0.25 m above that root, so the
    offset's extra height is removed before the point is placed on the real root.
    """
    offset = np.asarray(point_w, dtype=np.float64) - np.asarray(concept_root, dtype=np.float64)
    offset[2] -= float(window_concept_mod._CONCEPT_FRAME_LIFT)
    return np.asarray(real_root, dtype=np.float64) + offset


def _pick_handle_link(asset, urdf_path: str) -> str:
    names = list(asset.body_names)
    handle_names = [n for n in names if "handle" in n.lower() and not n.lower().endswith("_base")]
    if handle_names:
        prefer = [
            n
            for n in handle_names
            if n.lower() == "arched_handle" or n.lower().endswith("_handle") or n.lower().endswith("handle")
        ]
        return prefer[0] if prefer else handle_names[0]
    preferred = find_semantic_link(Path(urdf_path).parent, ["handle"])
    return _pick_slider_link(asset, preferred)


def _prismatic_joint_for_link(urdf_path: str, start_link: str):
    """Walk URDF parents from the handle until the sash prismatic joint."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None, None, None, None, None, None
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
        if jtype == "prismatic":
            origin_elem = joint.find("origin")
            axis_elem = joint.find("axis")
            origin_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            if origin_elem is not None and origin_elem.attrib.get("xyz"):
                origin_xyz = np.array([float(x) for x in origin_elem.attrib["xyz"].split()], dtype=np.float64)
            axis_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if axis_elem is not None and axis_elem.attrib.get("xyz"):
                axis_xyz = np.array([float(x) for x in axis_elem.attrib["xyz"].split()], dtype=np.float64)
            n = np.linalg.norm(axis_xyz)
            if n > 1e-8:
                axis_xyz = axis_xyz / n
            return origin_xyz, axis_xyz, parent, cur, joint.attrib.get("name", ""), jtype
        cur = parent
    return get_movable_joint_from_urdf(urdf_path, types=("prismatic", "revolute", "continuous"))


class window:
    def __init__(self):
        self.scale = (0.35, 0.35, 0.35)

    def import2sim(self, urdf_path: str, instance_name: str, num_envs: int):
        self.num_envs = num_envs
        self.urdf_path = urdf_path
        self.scene_key = instance_name
        # Real window stays here. The concept is parked at (1000, 1000).
        x1, y1, z1 = window_concept_mod.WINDOW_SPAWN_XYZ
        r_z = R.from_euler("z", float(random_degree), degrees=True)
        self.q_obj_world_wxyz = xyzw_to_wxyz(r_z.as_quat())
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=urdf_path,
            concept_root=str(PARTNET_CONCEPT_ROOT),
            category="Window",
            allow_fallback=False,
        )
        self.concept_urdf_path = window_concept_mod.CONCEPT_URDF_PATH or resolved_concept_urdf
        joint_pos = _safe_default_joint_pos(urdf_path)
        open_frac = getattr(self, "spawn_open_fraction", None)
        if open_frac:
            joint_pos = _joints_at_open_fraction(urdf_path, joint_pos, float(open_frac))
            print(f"[window] spawn open fraction={float(open_frac):.2f} joint_pos={joint_pos}")
        self.joint_pos = joint_pos
        if joint_pos:
            print(f"[window] default joint_pos within limits: {joint_pos}")
        print(
            f"[window] spawn PartNet urdf={urdf_path} model_id={model_id} "
            f"pos=({x1:.2f},{y1:.2f},{z1:.2f}) yaw={float(random_degree):.1f} scale={self.scale}"
        )
        cfgs = []
        for _env_id in range(num_envs):
            cfgs.append(
                AssetCfg(
                    fix_base=True,
                    name=self.scene_key,
                    urdf_path=urdf_path,
                    pos=(x1, y1, z1),
                    rot=self.q_obj_world_wxyz,
                    scale=self.scale,
                    asset_kind="articulation",
                    stiffness=0.0,
                    damping=1.0,
                    static_friction=0.8,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    compliant_contact_damping=0.0,
                    compliant_contact_stiffness=0.0,
                    joint_pos=joint_pos,
                    # diffuse_color=(0.25, 0.55, 0.85),
                    roughness=0.35,
                    opacity=1.0,
                    force_usd_conversion=True,
                )
            )
        return cfgs

    def open(self, scene, env_id: int = 0, fraction: float = 0.5):
        """Grasp the concept-handle pose mapped onto the PartNet window, then slide it open."""
        asset = None
        for name in (f"{self.scene_key}__env{env_id}", self.scene_key):
            try:
                asset = scene[name]
                break
            except (KeyError, AttributeError):
                continue
        if asset is None:
            raise RuntimeError(f"window asset not found for {self.scene_key} env={env_id}")
        try:
            my_concept = scene[f"my_concept_window__env{env_id}"]
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(
                "window_concept is not in the scene; add type=window_concept to the task JSON"
            ) from exc

        concept_urdf = window_concept_mod.CONCEPT_URDF_PATH or self.concept_urdf_path
        if not concept_urdf:
            raise RuntimeError("concept Window URDF path is missing")
        handle_link = _pick_handle_link(my_concept, concept_urdf)
        print(f"[window] concept handle link={handle_link} bodies={list(my_concept.body_names)}")
        origin_local, axis_local, parent_link, child_name, joint_name, jtype = _prismatic_joint_for_link(
            concept_urdf, handle_link
        )
        if origin_local is None or axis_local is None:
            raise RuntimeError(f"No movable window joint in {concept_urdf}")

        handle_pos_concept, _handle_quat = get_link_pose_physics(
            my_concept, link_name=handle_link, env_id=env_id
        )
        real_root_pos = asset.data.root_pos_w[env_id].detach().cpu().numpy()
        concept_root_pos = my_concept.data.root_pos_w[env_id].detach().cpu().numpy()
        handle_pos = np.asarray(
            map_concept_point_to_real_xy(handle_pos_concept, concept_root_pos, real_root_pos),
            dtype=np.float64,
        )
        handle_pos = handle_pos - planner_env_origin(scene, env_id)

        body_names = list(my_concept.body_names)
        if parent_link in body_names:
            pid, _ = my_concept.find_bodies(parent_link)
            pstate = my_concept.data.body_state_w[env_id, int(pid[0])]
            parent_pos_w = pstate[0:3].detach().cpu().numpy()
            parent_quat_wxyz = pstate[3:7].detach().cpu().numpy()
        else:
            parent_pos_w = my_concept.data.root_pos_w[env_id].detach().cpu().numpy()
            parent_quat_wxyz = my_concept.data.root_quat_w[env_id].detach().cpu().numpy()

        axis_point_concept, axis_dir_w = transform_joint_axis_to_world(
            parent_pos_w, parent_quat_wxyz, origin_local, axis_local
        )
        _axis_point_w = map_concept_point_to_real_xy(axis_point_concept, concept_root_pos, real_root_pos)
        axis_dir_real = np.asarray(axis_dir_w, dtype=np.float64)
        n = np.linalg.norm(axis_dir_real)
        if n < 1e-8:
            raise RuntimeError("Window joint axis is degenerate")
        axis_dir_real = axis_dir_real / n

        x1, y1, z1 = float(handle_pos[0]), float(handle_pos[1]), float(handle_pos[2])
        print(f"[window] mapped handle pose: ({x1:.3f}, {y1:.3f}, {z1:.3f}) joint={joint_name} type={jtype}")
        r_x = R.from_euler("x", 180, degrees=True)
        r_z = R.from_euler("z", 90, degrees=True)
        r_hand_obj = r_z * r_x
        r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
        r_hand_world = r_obj_world * r_hand_obj
        q_hand_world_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())
        q_hand_world_wxyz = rotate_y(q_hand_world_wxyz,-90)
        q_hand_world_wxyz = rotate_x(q_hand_world_wxyz,-90)

        r_hand_final = R.from_quat(wxyz_to_xyzw(q_hand_world_wxyz))
        grasp = np.array([x1, y1, z1 - 0.01], dtype=np.float64) + r_hand_final.apply([0.0, 0.0, -0.08])
        frac = float(np.clip(fraction, 0.05, 1.0))
        signed = _open_signed_distance(concept_urdf, joint_name, frac, self.scale[0])
        distance = float(np.copysign(0.19, signed if abs(signed) > 1e-8 else -1.0))
        print(f"[window] slide distance={distance:.3f} m")
        slide_pts = generate_prismatic_points(grasp, axis_dir_real, distance, steps=4)
        open_pts = slide_pts[1:] if len(slide_pts) > 1 else slide_pts

        approach_xyz = grasp + r_hand_final.apply([0, 0.0, -0.25])
        handle_xyz = grasp
        pose_approach = [
            float(approach_xyz[0]),
            float(approach_xyz[1]),
            float(approach_xyz[2]),
            q_hand_world_wxyz[0],
            q_hand_world_wxyz[1],
            q_hand_world_wxyz[2],
            q_hand_world_wxyz[3],
        ]
        pose_handle = [
            float(handle_xyz[0]),
            float(handle_xyz[1]),
            float(handle_xyz[2]),
            q_hand_world_wxyz[0],
            q_hand_world_wxyz[1],
            q_hand_world_wxyz[2],
            q_hand_world_wxyz[3],
        ]
        open_poses = [
            [float(p[0]), float(p[1]), float(p[2]), q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            for p in open_pts
        ]
        last_open = np.asarray(open_pts[-1] if open_pts else grasp, dtype=np.float64)
        retreat_xyz = last_open + r_hand_final.apply([0, 0.0, -0.25])
        pose_retreat = [
            float(retreat_xyz[0]),
            float(retreat_xyz[1]),
            float(retreat_xyz[2]),
            q_hand_world_wxyz[0],
            q_hand_world_wxyz[1],
            q_hand_world_wxyz[2],
            q_hand_world_wxyz[3],
        ]
        executor = "robot_left" if y1 > 0 else "robot_right"
        print(f"[window] waypoints: approach -> handle -> {len(open_poses)} open pts -> retreat, executor={executor}")

        pose_pack = [
            {"pose": pose_approach, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            {"pose": pose_handle, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            *[{"pose": p, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor} for p in open_poses],
            # {"pose": pose_retreat, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
        ]
        return pose_pack

    def close(self, scene, env_id: int = 0, fraction: float = 0.5):
        """Slide the sash shut along the open path, with the left arm."""
        pack = self.open(scene, env_id, fraction=fraction)
        if not pack:
            return pack
        closed = []
        for wp in reversed(pack):
            step = dict(wp)
            step["executor"] = "robot_left"
            closed.append(step)
        print(f"[window] close reverses open on robot_left, waypoints={len(closed)}")
        return closed
