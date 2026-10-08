from my_utils import *
from set_interface_try import *
import random
import xml.etree.ElementTree as ET

ZONES = [
    ((0.70, 0.80), (-0.15, 0.15)),
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


def _pick_door_hinge(urdf_path: str):
    """Pick the primary rotation_door revolute joint (leaf attached to the frame)."""
    path = Path(urdf_path)
    semantics = parse_partnet_semantics(path.parent)
    door_links = {
        link
        for link, kind, part in semantics
        if "rotation_door" in part or (kind == "hinge" and "door" in part)
    }
    frame_links = {
        link
        for link, kind, part in semantics
        if "door_frame" in part or kind == "static"
    }
    root = ET.parse(path).getroot()
    candidates = []
    for joint in root.findall("joint"):
        if (joint.attrib.get("type") or "").lower() != "revolute":
            continue
        child_el = joint.find("child")
        parent_el = joint.find("parent")
        if child_el is None or parent_el is None:
            continue
        child = child_el.attrib.get("link", "")
        parent = parent_el.attrib.get("link", "")
        if door_links and child not in door_links:
            continue
        origin = np.zeros(3, dtype=np.float64)
        origin_el = joint.find("origin")
        if origin_el is not None and origin_el.attrib.get("xyz"):
            origin = _parse_xyz_attr(origin_el.attrib["xyz"])
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_el = joint.find("axis")
        if axis_el is not None and axis_el.attrib.get("xyz"):
            axis = _parse_xyz_attr(axis_el.attrib["xyz"])
        n = np.linalg.norm(axis)
        if n > 1e-8:
            axis = axis / n
        lim = joint.find("limit")
        lower = float(lim.attrib.get("lower", 0.0)) if lim is not None else 0.0
        upper = float(lim.attrib.get("upper", 0.0)) if lim is not None else 0.0
        preferred = 0 if (parent in frame_links or parent in ("base", "world")) else 1
        candidates.append(
            (preferred, joint.attrib.get("name", ""), parent, child, origin, axis, lower, upper)
        )
    if not candidates:
        raise RuntimeError(f"No revolute door joint in {urdf_path}")
    candidates.sort(key=lambda x: (x[0], x[1]))
    _pref, name, parent, child, origin, axis, lower, upper = candidates[0]
    if abs(lower) < 1e-3:
        closed_q, open_q = lower, upper
    elif abs(upper) < 1e-3:
        closed_q, open_q = upper, lower
    elif abs(lower) <= abs(upper):
        closed_q, open_q = lower, upper
    else:
        closed_q, open_q = upper, lower
    return {
        "name": name,
        "parent": parent,
        "child": child,
        "origin": origin,
        "axis": axis,
        "lower": lower,
        "upper": upper,
        "closed_q": float(closed_q),
        "open_q": float(open_q),
    }


class door:
    def __init__(self):
        self.scale = (0.5, 0.5, 0.5)

    def import2sim(self, urdf_path: str, instance_name: str, num_envs: int):
        self.num_envs = num_envs
        self.urdf_path = urdf_path
        self.scene_key = instance_name
        self.hinge = _pick_door_hinge(urdf_path)
        x1, y1 = sample_pos_by_instance(instance_name)
        r_z = R.from_euler("z", 0.0, degrees=True)
        self.q_obj_world_wxyz = xyzw_to_wxyz(r_z.as_quat())
        print(
            f"[door] urdf={urdf_path} joint={self.hinge['name']} "
            f"child={self.hinge['child']} open={self.hinge['open_q']:.3f} "
            f"closed={self.hinge['closed_q']:.3f}"
        )
        cfgs = []
        for _env_id in range(num_envs):
            cfgs.append(
                AssetCfg(
                    fix_base=True,
                    name=self.scene_key,
                    urdf_path=urdf_path,
                    pos=(x1, y1, 0.35),
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
                    joint_pos={self.hinge["name"]: self.hinge["open_q"]},
                )
            )
        return cfgs

    def close(self, scene, env_id):
        pose_pack = []
        for envid in range(self.num_envs):
            instance_name = f"{self.scene_key}__env{envid}"
            door_asset = scene[instance_name]
            child = self.hinge["child"]
            door_pos, _door_quat = get_link_pose_physics(door_asset, link_name=child, env_id=envid)

            body_names = list(door_asset.body_names)
            parent = self.hinge["parent"]
            if parent in body_names:
                pid, _ = door_asset.find_bodies(parent)
                pstate = door_asset.data.body_state_w[envid, int(pid[0])]
                parent_pos_w = pstate[0:3].detach().cpu().numpy()
                parent_quat_wxyz = pstate[3:7].detach().cpu().numpy()
            else:
                parent_pos_w = door_asset.data.root_pos_w[envid].detach().cpu().numpy()
                parent_quat_wxyz = door_asset.data.root_quat_w[envid].detach().cpu().numpy()

            axis_point_w, axis_dir_w = transform_joint_axis_to_world(
                parent_pos_w,
                parent_quat_wxyz,
                self.hinge["origin"] * np.array(self.scale, dtype=np.float64),
                self.hinge["axis"],
            )
            if self.hinge["open_q"] > self.hinge["closed_q"]:
                axis_dir_w = -np.asarray(axis_dir_w, dtype=np.float64)

            radial = np.asarray(door_pos, dtype=np.float64) - np.asarray(axis_point_w, dtype=np.float64)
            radial_n = np.linalg.norm(radial)
            if radial_n < 0.04:
                grasp = np.asarray(door_pos, dtype=np.float64) + np.array([-0.08, 0.0, 0.0])
            else:
                grasp = np.asarray(axis_point_w, dtype=np.float64) + radial / radial_n * max(radial_n, 0.10)
            grasp[0] -= 0.03

            swing = abs(self.hinge["open_q"] - self.hinge["closed_q"])
            max_deg = int(np.clip(np.degrees(swing) * 0.55, 20.0, 70.0))
            arc_points = generate_handle_arc_points_about_axis(
                axis_point=axis_point_w,
                axis_dir=axis_dir_w,
                handle_pos0=grasp,
                max_deg=max_deg,
                step_deg=5,
            )

            r_x = R.from_euler("x", 180, degrees=True)
            r_z = R.from_euler("z", 90, degrees=True)
            r_hand_obj = r_z * r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_xyzw = r_hand_world.as_quat()
            q_wxyz = xyzw_to_wxyz(q_xyzw)

            x1, y1, z1 = float(grasp[0]), float(grasp[1]), float(grasp[2])
            approach = grasp - r_hand_world.apply([0.0, 0.0, 0.18])
            contact = grasp - r_hand_world.apply([0.0, 0.0, 0.04])
            executor = "robot_left" if y1 > 0 else "robot_right"
            arc_goal_poses = [
                [p[0], p[1], p[2], q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]]
                for p in arc_points
            ]
            pose_pack = [
                {
                    "pose": [float(approach[0]), float(approach[1]), float(approach[2]), q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]],
                    "gripper": "open",
                    "use_obstacle_avoidance": False,
                    "executor": executor,
                },
                {
                    "pose": [float(contact[0]), float(contact[1]), float(contact[2]), q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]],
                    "gripper": "open",
                    "use_obstacle_avoidance": False,
                    "executor": executor,
                },
                *[
                    {
                        "pose": p,
                        "gripper": "close",
                        "use_obstacle_avoidance": False,
                        "executor": executor,
                    }
                    for p in arc_goal_poses
                ],
            ]
            print(f"[door.close] grasp={[round(x1, 3), round(y1, 3), round(z1, 3)]} max_deg={max_deg} executor={executor}")
        return pose_pack
