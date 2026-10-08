from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.6, 0.61), (0.45, 0.455)),  # zone 0
    ((0.40, 0.7), (-0.3, -0.7)),  # zone 1
]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 

angle_degree = np.random.uniform(60,70)

def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    idx = int(instance_name.split("_")[-1]) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y
class skillet ():
    def __init__(self):
        self.scale = (0.15,0.15,0.15)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        self.num_envs = num_envs
        
        x1, y1 = sample_pos_by_instance(instance_name)

        
        r_z = R.from_euler('z', angle_degree, degrees=True)
        r_x = R.from_euler('x', 90, degrees=True)
        r_obj = r_z * r_x   
        # self.rot1 若是 wxyz，要先转
        q_obj_world_xyzw = r_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(q_obj_world_xyzw)
        # rot,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(-60.0,60.0)),current_quat=[1,0,0,0])
        # rot1,_ = rotate_z([1,0,0,0],40)
        
   

        self.scene_key = instance_name
        print("import", instance_name, "obj_id=", id(self))
        print("pick", self.scene_key, "obj_id=", id(self))

        cfg = AssetCfg(
            
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.35),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            mass  = 0.1,

            fix_base= False,
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
        )

        return [cfg]
    

    def pick_simultaneously_left(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
        
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[skillet] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            q_hand_world_wxyz2,_ = rotate_z(list(q_hand_world_wxyz),90)
            q_hand_world_wxyz3 = rotate_y([0.707,0,0.707,0],90)
            q_hand_world_wxyz3,_ = rotate_z(list(q_hand_world_wxyz3),-90)
            delta = r_hand_world.apply([0.0, -0.08, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            
            target_pos_x = 0.5
            target_pos_y =0.2
            target_pos_z = 0.6

            pose1= [pos[0],pos[1],pos[2]+0.3,q_hand_world_wxyz2[0], q_hand_world_wxyz2[1],q_hand_world_wxyz2[2],q_hand_world_wxyz2[3]]
            pose2 = [pos[0],pos[1],pos[2]+0.05,q_hand_world_wxyz2[0], q_hand_world_wxyz2[1],q_hand_world_wxyz2[2],q_hand_world_wxyz2[3]]
            pose3 = [pos[0],pos[1],pos[2]+0.4,q_hand_world_wxyz2[0], q_hand_world_wxyz2[1],q_hand_world_wxyz2[2],q_hand_world_wxyz2[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.1,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z+0.1,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
        


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left","sync_id": "sync"},

            ]
            return pose

    def place(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
        
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[skillet] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            q_hand_world_wxyz2,_ = rotate_z(list(q_hand_world_wxyz),90)
            q_hand_world_wxyz3 = rotate_y([0.707,0,0.707,0],90)
            q_hand_world_wxyz3,_ = rotate_z(list(q_hand_world_wxyz3),-90)
            delta = r_hand_world.apply([0.0, -0.08, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            
            target_pos_x = 0.5
            target_pos_y =0.2
            target_pos_z = 0.6


            pose1 = [target_pos_x+0.25,target_pos_y,target_pos_z+0.1,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
            pose2 = [target_pos_x+0.25,target_pos_y,target_pos_z-0.1,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
            pose3 = [target_pos_x+0.25,target_pos_y,target_pos_z+0.10,q_hand_world_wxyz3[0], q_hand_world_wxyz3[1],q_hand_world_wxyz3[2],q_hand_world_wxyz3[3]]
        


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose3,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},


            ]
            return pose
    def into(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
        
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[skillet] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            q_hand_world_wxyz2,_ = rotate_z(list(q_hand_world_wxyz),90)
            q_hand_world_wxyz4 = rotate_y([0.707,0,0.707,0],90)
            q_hand_world_wxyz4,_= rotate_z(list(q_hand_world_wxyz4),90)
            q_hand_world_wxyz3 = rotate_y(list(q_hand_world_wxyz4),-30)
            delta = r_hand_world.apply([0.0, -0.08, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            
            target_pos_x = 0.5
            target_pos_y = 0.2
            target_pos_z = 0.5

            pose1= [target_pos_x,target_pos_y-0.1,target_pos_z+0.35,q_hand_world_wxyz4[0], q_hand_world_wxyz4[1],q_hand_world_wxyz4[2],q_hand_world_wxyz4[3]]
            pose2 = [target_pos_x,target_pos_y-0.1,target_pos_z+0.25,q_hand_world_wxyz4[0], q_hand_world_wxyz4[1],q_hand_world_wxyz4[2],q_hand_world_wxyz4[3]]
            pose3 = [target_pos_x,target_pos_y-0.1,target_pos_z+0.35,q_hand_world_wxyz4[0], q_hand_world_wxyz4[1],q_hand_world_wxyz4[2],q_hand_world_wxyz4[3]]
            pose4 = [target_pos_x,target_pos_y-0.1,target_pos_z+0.35,q_hand_world_wxyz4[0], q_hand_world_wxyz4[1],q_hand_world_wxyz4[2],q_hand_world_wxyz4[3]]
        


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose3,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose4,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},

            ]
            return pose