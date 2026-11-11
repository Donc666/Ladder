import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

def _parse_xyz(s, default=(0, 0, 0)):
    if s is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in s.split()], dtype=np.float64)

def _parse_rpy(s, default=(0, 0, 0)):
    if s is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in s.split()], dtype=np.float64)

def _T_origin(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T

def _quat_wxyz_to_T(pos, quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T

def load_urdf_graph(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    child_to_joint = {}

    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]

        o = j.find("origin")
        xyz = _parse_xyz(o.attrib.get("xyz") if o is not None else None)
        rpy = _parse_rpy(o.attrib.get("rpy") if o is not None else None)

        a = j.find("axis")
        axis = _parse_xyz(a.attrib.get("xyz") if a is not None else None, default=(0, 0, 1))
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        joints[name] = {
            "name": name,
            "type": jtype,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }
        child_to_joint[child] = name

    return joints, child_to_joint

def build_chain_to_target(joints, child_to_joint, root_link, target_link):
    chain = []
    cur = target_link
    while cur != root_link:
        jname = child_to_joint[cur]
        j = joints[jname]
        chain.append(j)
        cur = j["parent"]
    chain.reverse()
    return chain

def fk_chain(chain, joint_values):
    T = np.eye(4, dtype=np.float64)
    for j in chain:
        Tj = _T_origin(j["xyz"], j["rpy"])
        q = float(joint_values.get(j["name"], 0.0))

        if j["type"] in ("revolute", "continuous"):
            Tr = np.eye(4, dtype=np.float64)
            Tr[:3, :3] = R.from_rotvec(j["axis"] * q).as_matrix()
            T = T @ Tj @ Tr
        elif j["type"] == "prismatic":
            Tp = np.eye(4, dtype=np.float64)
            Tp[:3, 3] = j["axis"] * q
            T = T @ Tj @ Tp
        else:  # fixed
            T = T @ Tj
    return T

def get_handle_pose_from_urdf_fk(microwave_asset, urdf_path, root_link="world", target_link="Trifold_Handle"):
    # 1) 读URDF图
    joints, child_to_joint = load_urdf_graph(urdf_path)
    chain = build_chain_to_target(joints, child_to_joint, root_link, target_link)

    # 2) 当前关节值（来自仿真）
    sim_joint_names = list(microwave_asset.data.joint_names)
    sim_joint_pos = microwave_asset.data.joint_pos[0].detach().cpu().numpy()
    joint_values = {name: float(sim_joint_pos[i]) for i, name in enumerate(sim_joint_names)}

    # 3) 机器人 spawn/root 在世界坐标系的位姿
    root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
    root_quat_wxyz = microwave_asset.data.root_quat_w[0].detach().cpu().numpy()
    T_spawn = _quat_wxyz_to_T(root_pos, root_quat_wxyz)

    # 4) 链式FK
    T_root_to_handle = fk_chain(chain, joint_values)
    T_world_handle = T_spawn @ T_root_to_handle

    pos = T_world_handle[:3, 3]
    quat_xyzw = R.from_matrix(T_world_handle[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return pos, quat_wxyz, T_world_handle