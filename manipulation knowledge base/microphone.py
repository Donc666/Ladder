from my_utils import *
from set_interface_try import *
import random

ZONES = [
    ((0.4, 0.7), (0.1, 0.7)),  # zone 0
]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 



def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    idx = int(instance_name.split("_")[-1]) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y

class microphone ():
    def __init__(self):
        self.scale = (0.08,0.08,0.08)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        self.scene_key = instance_name

        self.angle_degree = np.random.uniform(30,150)
        r_y = R.from_euler('y', 90, degrees=True)
        r_z = R.from_euler('z', self.angle_degree, degrees=True)
        r_obj = r_z * r_y   
        q_obj_world_xyzw = r_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(q_obj_world_xyzw)
        #adjust
        
        # x1 = random.uniform(0.4, 0.8)
        # y1 = random.uniform(-0.5, 0.5)
         # adjust

        #others
        x1, y1 = sample_pos_by_instance(instance_name)
        # x1 = random.uniform(0.4, 0.8)
        # y1 = random.uniform(0.0, 0.5)
        # rot1= rotate_x(current_quat=[1,0,0,0],angle_deg=-90) #not adjust
        
        
        
        
        # rot = rotate_y(list(rot1),-90)




        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.3),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            asset_kind = "rigid",
            disable_gravity=True,
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






    def hand_over(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}" 
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]   
            
    
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            
            r_y = R.from_euler('y', 90, degrees=True)
            r_z = R.from_euler('z', 0, degrees=True)
            r_hand_obj = r_z * r_y    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos1 = pos_w + hand_to_world_rot.apply([-0.05,0,0.0])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            handover_rot_right,_ = rotate_z(list(handover_rot_right),180)
            x_range = (0,0.3)
            target_pos_x = 0.55
            target_pos_y = 0
            target_pos_z = 0.5

            pose1 = [pos1[0],pos1[1],z+0.2,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose2 = [pos1[0],pos1[1],z+0.11,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose3 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose5 = [target_pos_x,target_pos_y-0.3,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose6 = [target_pos_x,target_pos_y-0.17,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose7 = [target_pos_x,target_pos_y-0.17,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose8 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose9 = [target_pos_x,target_pos_y+0.1,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose10 = [target_pos_x,target_pos_y-0.3,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose11 = [target_pos_x,target_pos_y-0.3,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose9,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose10,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose11,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                

            ]
            return pose