from my_utils import *
from set_interface_try import *
import random


ZONES = [
    ((0.40, 0.7), (0.3, 0.7)),  # zone left
    ((0.40, 0.7), (-0.3, -0.7)),  # zone right

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

class shoe ():
    def __init__(self):
        self.scale = (0.1,0.1,0.1)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        x1, y1 = sample_pos_by_instance(instance_name)


        self.angle_degree = np.random.uniform(0,359)
        r_z = R.from_euler("z", self.angle_degree, degrees=True)
        r_x = R.from_euler("x", 90, degrees=True)
        self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())
        
   

        self.scene_key = instance_name

        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.35),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
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
    

    def place(self,scene,env_id):
        asset = scene[self.scene_key]
        root_pos = asset.data.root_pos_w[env_id]      
 


        
        x= root_pos[0].item()-1
        y = root_pos[1].item()+1
        z = root_pos[2].item()
       

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)
        
        # self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angel_degree)
        self.hand_rot = rotate_y([0.707, 0, 0.707, 0],-90)
        self.hand_rot,_ = rotate_z(current_quat=list(self.hand_rot),angle_deg=self.angel_degree + 90)
        handover_rot = rotate_y([0.707, 0, 0.707, 0],-90)
        target_pos_x = 0.5
        target_pos_y = 0
        target_pos_z = z+0.2
        # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
        # handover_rot_right,_ = rotate_z(list(handover_rot_right),90)

        pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose2 = [x,y,z+0.01,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose3 = [x,y,z+0.25,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose4 = [target_pos_x,target_pos_y,target_pos_z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose5 = [target_pos_x,target_pos_y,target_pos_z-0.15,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]
        pose6 = [target_pos_x,target_pos_y,target_pos_z,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]
        pose7= [x,y,z+0.4,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose8= [x,y,z+0.4,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        if y < 0:
            executor = "robot_right"
        else:
            executor = "robot_left"
        pose = [
            {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
        ]
        return pose


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
                    dx = np.random.uniform(-0.03,0.03)
                    pos1 = pos_w - hand_to_world_rot.apply([dx,0,0.0])
                    pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
                    q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)  
                    
            
            
                    pose1= [pos1[0],pos1[1],target_pos_z+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose2 = [pos1[0],pos1[1],target_pos_z+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose3 = [pos1[0],pos1[1],target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose4 = [pos1[0],pos1[1],target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                
            
            
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
        
    def pick_simultaneously_right(self,scene,env_id):
            
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
                    dx = np.random.uniform(-0.03,0.03)
                    pos1 = pos_w - hand_to_world_rot.apply([dx,0,0.0])
                    pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
                    q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)    
                    
            
            
                    pose1= [pos1[0],pos1[1],target_pos_z+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose2 = [pos1[0],pos1[1],target_pos_z+0.06,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose3 = [pos1[0],pos1[1],target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    pose4 = [pos1[0],pos1[1],target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                
            
            
                    if y >0:
                        executor = "robot_left"
                    else:
                        executor = "robot_right"
                    pose = [
                        {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                        {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
            
                    ]
                    return pose