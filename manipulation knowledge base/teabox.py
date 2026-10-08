from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.40, 0.6), (0.3, 0.6)),  # zone 0

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
class teabox ():
    def __init__(self):
        self.scale = (0.03,0.03,0.03)
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
            angle_degree = float(np.random.uniform(40, 90))
            x1, y1 = sample_pos_by_instance(instance_name)  # 或你自己的分区函数
            r_z = R.from_euler("z", angle_degree, degrees=True)
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


    def pick_simultaneously_left(self,scene,env_id):
    
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    
            target_pos_x = x 
            target_pos_y = y 
            target_pos_z = z

            r_x = R.from_euler('x', 90, degrees=True)
            r_y = R.from_euler('y',90, degrees=True)
            r_hand_obj = r_y*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos_pre = pos_w - hand_to_world_rot.apply([0.0,0,0.2])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
    

            # q_hand_world_wxyz = rotate_y([0.707, 0, 0.707, 0],90)
            # q_hand_world_wxyz,_ = rotate_z(current_quat=list(q_hand_world_wxyz),angle_deg=90 + self.angle_degree )   

            hand_rot = rotate_y([0.707, 0, 0.707, 0],90)
            # q_hand_world_wxyz2= rotate_y([0.707, 0, 0.707, 0],90)
            rand = np.random.uniform(0,-0.5)
            pose1= [target_pos_x,target_pos_y,target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [target_pos_x,target_pos_y,target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    
        
    
    
            if y >0:
                executor = "robot_left"
            else:
                executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
              
    
            ]
            return pose
    def scan(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
        
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        


            r_x = R.from_euler('x', 90, degrees=True)
            
            # r_z = R.from_euler('z', 90, degrees=True)
            r_hand_obj = r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            q_hand_world_wxyz2,_ = rotate_z(list(q_hand_world_wxyz),90)
            q_hand_world_wxyz4 = rotate_y([0.707,0,0.707,0],90)

            q_hand_world_wxyz3 = rotate_x(list(q_hand_world_wxyz4),90)
            delta = r_hand_world.apply([0.0, -0.08, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            
            target_pos_x = 0.5
            target_pos_y = 0.2
            target_pos_z = 0.5

            pose1= [target_pos_x,target_pos_y,target_pos_z+0.2,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
            pose2= [target_pos_x,target_pos_y,target_pos_z+0.2,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]

        


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},

            ]
            return pose