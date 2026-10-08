from my_utils import *
from set_interface_try import *
import random

ZONES = [
    ((0.4, 0.45), (0.1, 0.15)),  # zone 0
    ((0.70, 0.85), (-0.7, -0.30)),  # zone 1
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

class roller ():
    def __init__(self):
        self.scale = (0.1,0.1,0.15)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        self.scene_key = instance_name

        self.angle_degree = np.random.uniform(90,91)
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
            pos = (x1, y1, 0.30),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
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
    
    def grap_simulataneously(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"  
            dispenser_asset = scene[self.actual_instance_name]
            root_pos = dispenser_asset.data.root_pos_w[env_id]
            root_quat = dispenser_asset.data.root_quat_w[env_id]
            self.q_obj_world_wxyz = [
                float(root_quat[0].item()),
                float(root_quat[1].item()),
                float(root_quat[2].item()),
                float(root_quat[3].item()),
            ]

            x = root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()


            target_pos_x = np.random.uniform(0.4,0.8)
            target_pos_y = np.random.uniform(-0.2,0.2)
            target_pos_z = z + 0.2 + 0.1 * np.random.uniform(-1,1)
            
            # r_x = R.from_euler('x', 180, degrees=True)
            # r_z = R.from_euler('z', self.angle_degree, degrees=True)
            # r_hand_obj = r_z * r_x    # 先z后x
            # # self.rot1 若是 wxyz，要先转
            # r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            # r_hand_world = r_obj_world * r_hand_obj
            # q_hand_world_xyzw = r_hand_world.as_quat()
            # hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            # pos_w = [x,y,z]
            # pos_pre = pos_w - hand_to_world_rot.apply([0.0,0,0.2])
            # pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            # q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
        
            


            # grasp_pose_pre= [pos_pre[0],pos_pre[1],pos_pre[2],self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            # grasp_pose = [x,y,z+0.06,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            # adjust_pose = [x,y,z+0.36,0.707, 0, 0.707, 0]
            # pose1 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]
            # pose2 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]
            # pose3 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]

        
            r_y = R.from_euler('y', 90, degrees=True)
            r_z = R.from_euler('z', 0, degrees=True)
            r_hand_obj = r_z * r_y    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos1 = pos_w + hand_to_world_rot.apply([-0.1,0,0.0])
            pos2 = pos_w + hand_to_world_rot.apply([0.1,0,0.0])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            pose1 = [pos1[0],pos1[1],z+0.2,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose2 = [pos1[0],pos1[1],z+0.08,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose3 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose4 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose5 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]

            pose6 = [pos2[0],pos2[1],z+0.2,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose7 = [pos2[0],pos2[1],z+0.08,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose8 = [pos2[0],pos2[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose9 = [pos2[0],pos2[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose10 = [pos2[0],pos2[1],z+0.3,q_hand_world_wxyz[0],q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            
            
            pose = [
                        {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                        {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                        {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left","sync_id": "pick_1"},
                        {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                        {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                        {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right","sync_id": "pick_1"},
                        {"pose":pose9,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        
                                                
                        ]

            
            

        
            return pose


