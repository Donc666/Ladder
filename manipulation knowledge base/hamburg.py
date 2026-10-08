from my_utils import *
from set_interface_try import *
import random
ZONES = [
      ((0.67, 0.675), (0.14, 0.145)),  # zone 0
    ((0.75, 0.751), (0.07, 0.0755)),  # zone 0
((0.77, 0.771), (0.17, 0.175)), 
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
class hamburg ():
    def __init__(self):
        self.scale = (0.025,0.025,0.025)
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
            self.angle_degree = float(np.random.uniform(0, 359))
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


    def pick(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            return food_pick_place_waypoints(asset, env_id, self.scene_key, place=True)

    def pick_simultaneously_left(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            return food_pick_place_waypoints(
                asset, env_id, self.scene_key, executor="simultaneous_left", place=True
            )

    def pick_simultaneously_right(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            return food_pick_place_waypoints(
                asset, env_id, self.scene_key, executor="simultaneous_right", place=True
            )
