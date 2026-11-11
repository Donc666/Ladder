from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.40, 0.6), (0, 0.1)),  # zone 0

]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 



def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    m = re.search(r"(?:^|__)env(\d+)$", instance_name)
    if m:
        idx = int(m.group(1))
    else:
        m2 = re.search(r"_(\d+)$", instance_name)  # 兼容旧格式 xxx_1
        if not m2:
            raise ValueError(f"Cannot parse index from instance_name={instance_name}")
        idx = int(m2.group(1)) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y
class basket ():
    def __init__(self):
        self.scale = (0.18,0.18,0.18)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs:int
    ):
        self.scene_key = instance_name
        self.num_envs = num_envs
    
        cfgs = []
        self.scene_keys_by_env = {}
        for env_id in range(num_envs):
            self.angle_degree = float(np.random.uniform(0, 90))
            x1, y1 = sample_pos_by_instance(instance_name)  # 或你自己的分区函数
            r_z = R.from_euler("z", self.angle_degree, degrees=True)
            r_x = R.from_euler("x", 90, degrees=True)
            self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())
            name_env = f"{instance_name}__env{env_id}"
            instance_name =name_env
            prim_env = f"/World/envs/env_{env_id}/{instance_name}"
            cfgs.append(AssetCfg(
                name = self.scene_key,
                urdf_path=urdf_path,
                pos = (x1, y1, 0.35),
                rot = self.q_obj_world_wxyz,
                scale = self.scale,
                stiffness=0.0,
                damping=0.0,
                fix_base= False,
                # disable_gravity=True,
                # source_kind = "cuboid",
                cuboid_size= (0.05,0.05,0.05),
                cuboid_color = (1.0, 0.5, 0.2),
                asset_kind = "rigid",
                # diffuse_color=(1.0,0.0,0.0),
                # roughness=0.5,
                metallic=0.9,
                # opacity=1,
                static_friction=0.8,
                dynamic_friction= 0.8,
                restitution= 0,
                compliant_contact_damping=0,
                compliant_contact_stiffness=0
            ))
            self.scene_keys_by_env[env_id] = name_env

        return cfgs


    def _root_pose(self, scene, env_id):
        """This env's basket pose. import2sim overwrites the stored yaw per env."""
        self.actual_instance_name = f"{self.scene_key}__env{env_id}"
        asset = scene[self.actual_instance_name]
        pos = np.asarray(asset.data.root_pos_w[env_id].detach().cpu().numpy(), dtype=np.float64)
        quat = np.asarray(asset.data.root_quat_w[env_id].detach().cpu().numpy(), dtype=np.float64)
        return pos.reshape(3), quat.reshape(4)

    def _hand_rotation(self, root_quat_wxyz):
        r_hand_obj = R.from_euler("y", 90, degrees=True) * R.from_euler("x", 90, degrees=True)
        r_obj = R.from_quat(wxyz_to_xyzw(root_quat_wxyz))
        r_hand = r_obj * r_hand_obj
        q_wxyz = [float(v) for v in xyzw_to_wxyz(r_hand.as_quat())]
        return r_obj, r_hand, q_wxyz

    def _into(self, scene, env_id, along_hand_y):
        pos, quat = self._root_pose(scene, env_id)
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        _r_obj, r_hand, q = self._hand_rotation(quat)
        retreat = float(np.random.uniform(0.12, 0.18))
        if y > 0:
            # 4 cm along the basket's own wide axis. The opening is only ~10 cm on the narrow side.
            drop = pos - r_hand.apply([0.0, along_hand_y, 0.0])
            # Leave toward the table center. A fixed +0.3 m sends the left basket (y=0.4) out to y=0.7.
            retreat_x, retreat_y = x, y - retreat
        else:
            # Same ±4 cm along this basket's wide axis as the left arm. The offset
            # uses this env's root yaw, so it stays inside the opening. Retreat stays -X.
            drop = pos - r_hand.apply([0.0, along_hand_y, 0.0])
            retreat_x, retreat_y = x - retreat, y
        above = z + 0.50
        place_z = z + 0.36
        print(
            f"[basket] into drop=[{drop[0]:.3f}, {drop[1]:.3f}, {place_z:.3f}] "
            f"retreat=[{retreat_x:.3f}, {retreat_y:.3f}] root=[{x:.3f}, {y:.3f}, {z:.3f}]"
        )

        def waypoint(px, py, pz, gripper):
            return {
                "pose": [float(px), float(py), float(pz), *q],
                "gripper": gripper,
                "use_obstacle_avoidance": False,
                "executor": "robot_left" if y > 0 else "robot_right",
            }

        return [
            waypoint(drop[0], drop[1], above, "close"),
            waypoint(drop[0], drop[1], place_z, "close"),
            waypoint(drop[0], drop[1], above, "open"),
            waypoint(retreat_x, retreat_y, z + 0.45, "open"),
            waypoint(retreat_x, retreat_y, z + 0.45, "open"),
        ]

    def into_1(self, scene, env_id):
        return self._into(scene, env_id, 0.04)

    def into_2(self, scene, env_id):
        return self._into(scene, env_id, -0.04)

    def pick(self, scene, env_id):
        print("pick", self.scene_key, "obj_id=", id(self))
        pos, quat = self._root_pose(scene, env_id)
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        _r_obj, _r_hand, q = self._hand_rotation(quat)
        # Opening center. A 0.10–0.14 m shift lands on the frame; the rim is about 0.17 m out.
        gz = z + 0.30
        high = z + 0.50
        print(f"[basket] pick grasp=[{x:.3f}, {y:.3f}, {gz:.3f}] root=[{x:.3f}, {y:.3f}, {z:.3f}]")
        executor = "robot_left" if y > 0 else "robot_right"

        def waypoint(px, py, pz, gripper):
            return {
                "pose": [float(px), float(py), float(pz), *q],
                "gripper": gripper,
                "use_obstacle_avoidance": False,
                "executor": executor,
            }

        return [
            waypoint(x, y, high, "open"),
            waypoint(x, y, gz, "open"),
            waypoint(x, y, high, "close"),
            waypoint(x, y, high, "close"),
        ]
