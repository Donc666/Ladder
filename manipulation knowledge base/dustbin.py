from my_utils import *
from set_interface_try import *
import random


angle_degree = np.random.uniform(0,359)
class dustbin ():
    def __init__(self):
        self.scale = (0.3,0.05,0.2)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        x1 = 0.6
        y1 = 0.75



        # rot,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(0,90)),current_quat=[1,0,0,0])
        # rot1 = rotate_x(list(rot),-90)
        # rot1 = rotate_x([1,0,0,0],-50.0)

        r_z = R.from_euler('z', 0, degrees=True)
        r_x = R.from_euler('x', 90, degrees=True)
        r_obj = r_z * r_x   
        q_obj_world_xyzw = r_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(q_obj_world_xyzw)

        
   

        self.scene_key = instance_name

        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.15),
      
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            # source_kind = "cuboid",
            cuboid_size= (0.05,0.05,0.0),
            cuboid_color = (np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)),
            asset_kind = "rigid",
            # diffuse_color=(1.0,0.0,0.0),
            # roughness=0.5,
            metallic=np.random.uniform(0,1),
            # opacity=1,
            static_friction=np.random.uniform(0,1),
            dynamic_friction= np.random.uniform(0,1),
            restitution= 0,
            compliant_contact_damping=0,
            compliant_contact_stiffness=0
        )

        return [cfg]
    

    def into(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name ]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()

            # dl = np.random.uniform(-0.02,0.05)
            # x  +=  dl * np.sin(np.deg2rad(self.angel_degree))
            # y  +=  dl * np.cos(np.deg2rad(self.angel_degree))

            r_z = R.from_euler('y', 90, degrees=True)
            r_x = R.from_euler('z', np.random.uniform(30,55), degrees=True)
            r_hand_obj = r_x * r_z   # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            


            hand_standard_rot = rotate_y([0.707, 0, 0.707, 0],-90)
            target_pos_x = np.random.uniform(0.4,0.6)
            target_pos_y = y
            target_pos_z = z 


            pose1= [target_pos_x,target_pos_y,target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [target_pos_x,target_pos_y,target_pos_z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose6 = [target_pos_x,target_pos_y,target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            
            
            
            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                
            
                
            ]
            return pose


    def into_2(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()

            # dl = np.random.uniform(-0.02,0.05)
            # x  +=  dl * np.sin(np.deg2rad(self.angel_degree))
            # y  +=  dl * np.cos(np.deg2rad(self.angel_degree))

            r_z = R.from_euler('y', 90, degrees=True)
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x * r_z   # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            


            hand_standard_rot = rotate_y([0.707, 0, 0.707, 0],-90)
            target_pos_x = np.random.uniform(0.4,0.6)
            target_pos_y = y
            target_pos_z = z 


            pose1= [target_pos_x,target_pos_y,target_pos_z+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [target_pos_x,target_pos_y,target_pos_z+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]

            
            
            
            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
               
                
            
                
            ]
            return pose