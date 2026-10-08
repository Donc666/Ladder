from my_utils import *
from set_interface_try import *
import random


class block_small ():
    def __init__(self):
        self.scale = (0.05,0.05,0.05)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        x1 = random.uniform(0.4, 0.8)
        y1 = random.uniform(-0.7,-0.4)



        angle_degree = float(np.random.uniform(40, 90))
        r_z = R.from_euler("z", angle_degree, degrees=True)
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
            source_kind = "cuboid",
            cuboid_size= (0.02,0.02,0.02),
            cuboid_color = (0.0,0.0,1.0),
            # asset_kind = "rigid",
            # diffuse_color=(1.0,0.0,0.0),
            # roughness=0.5,
            # metallic=0.9,
            # opacity=1,
            static_friction=0.8,
            dynamic_friction= 0.8,
            restitution= 0,
            compliant_contact_damping=0,
            compliant_contact_stiffness=0
        )

        return [cfg]
    
    def pick(self,scene,env_id):
        asset = scene[self.scene_key]
        root_pos = asset.data.root_pos_w[0]      
 


        
        x= root_pos[0].item()
        y = root_pos[1].item()
        z = root_pos[2].item()
       

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)

        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.angel_degree)
        self.hand_rot = rotate_z(list(self.hand_rot),90)

        grasp_pose_pre= [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        grasp_pose = [x,y,z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        adjust_pose = [x,y,z+0.1,0.707, 0, 0.707, 0]

        pose = [
            {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":adjust_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"}
        ]
        return pose
    def beat(self,scene,env_id):
        asset = scene[self.scene_key]
        root_pos = asset.data.root_pos_w[env_id]      
 


        x= root_pos[0].item()
        y = root_pos[1].item()
        z = root_pos[2].item()
       

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)
        
        self.hand_rot = rotate_x(list(self.hand_rot),np.random.uniform(()))
        

        beat_pose= [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        grasp_pose = [x,y,z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        adjust_pose = [x,y,z+0.1,0.707, 0, 0.707, 0]

        pose = [
            {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":adjust_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"}
        ]
        return pose
    def hand_over(self,scene,env_id):
        asset = scene[self.scene_key]
        root_pos = asset.data.root_pos_w[env_id]   
         
 
        print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

        
        x= root_pos[0].item()-1
        y = root_pos[1].item()+1
        z = root_pos[2].item()
        print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)
        
        self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angel_degree)
       

        handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
        handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
        handover_rot_right,_ = rotate_z(list(handover_rot_right),90)
        x_range = (0,0.3)
        target_pos_x1 = 0.55
        target_pos_y1 = 0
        target_pos_z = 0.5

        pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose2 = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose3 = [x,y,z+0.3,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose4 = [target_pos_x1,0,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
        pose5 = [target_pos_x1,-0.30,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
        pose6 = [target_pos_x1,-0.10,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
        pose7 = [target_pos_x1,0,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
        pose8 = [target_pos_x1,-0.40,0.65,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
        pose9 = [target_pos_x1,-0.30,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose10 = [target_pos_x1,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

        pose = [
            {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":pose9,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":pose10,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
        ]
        return pose
    
    def place_small(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        
            handover_rot = rotate_y([0.707, 0, 0.707, 0],90)

            target_pos_z = z+0.2
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
            target_pos_x =0.4
            target_pos_y =-0.2

            pose1= [x,y,z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [x,y,z+0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [x,y,z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z-0.12,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]
            pose6 = [target_pos_x,target_pos_y,target_pos_z,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]
            pose7= [x,y,z+0.2,handover_rot[0], handover_rot[1],handover_rot[2],handover_rot[3]]

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
            ]
            return pose
    
