from my_utils import *
from set_interface_try import *
import random

class hammer ():
    def __init__(self):
        self.scale = (0.09,0.09,0.09)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs       
        x1 = random.uniform(0.3, 0.8)
        y1 = random.uniform(0.3, 0.7)



        # rot,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(0,90)),current_quat=[1,0,0,0])
        # rot1 = rotate_x(list(rot),-90)
        # self.rot1 = rotate_x([1,0,0,0],180.0)
        self.random_rotate_deg = np.random.uniform(0,359)
        r_z = R.from_euler('z', self.random_rotate_deg, degrees=True)
        r_x = R.from_euler('x', 180, degrees=True)
        r_object_obj = r_x * r_z
        self.q_object_obj = r_object_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(self.q_object_obj)
        # self.rot1,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(-90,90)),current_quat=[1,0,0,0])
        # rot1 = [1,0,0,0]
        
   

        self.scene_key = instance_name

        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.37),
            disable_gravity = False, 
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            # source_kind = "cuboid",
            cuboid_size= (0.05,0.05,0.05),
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
    

    def pick(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()

            r_z = R.from_euler('z', 90, degrees=True)
            r_x = R.from_euler('x', 0, degrees=True)
            r_hand_obj = r_x * r_z   # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            # apply 要用 Rotation 对象（xyzw）
            delta = r_hand_world.apply([np.random.uniform(-0.03, 0.00), 0, 0])
            pos_beat = np.array([x, y, z], dtype=float) + delta
            
            



            
            # self.hand_rot = rotate_y([0.707, 0, 0.707, 0],-90)
            # self.hand_rot,_ = rotate_z(current_quat=list(self.hand_rot),angle_deg=self.random_rotate_deg + 90)

        

            pose1= [pos_beat[0],pos_beat[1],pos_beat[2]+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos_beat[0],pos_beat[1],pos_beat[2]+0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos_beat[0],pos_beat[1],pos_beat[2]+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos_beat[0],pos_beat[1],pos_beat[2]+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]

            
            
            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            
                
            ]
            return pose