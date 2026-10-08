from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.40, 0.5), (-0.1, 0)),  # zone 0

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
class playingcards ():
    def __init__(self):
        self.scale = (0.05,0.05,0.05)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str
    ):

        self.angle_degree = np.random.uniform(0,359)
        x1, y1 = sample_pos_by_instance(instance_name)

        
        r_z = R.from_euler('z', self.angle_degree, degrees=True)
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
        )

        return [cfg]


    def place_away(self,scene,env_id):
    
            
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.scene_key]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    
            target_pos_x = x 
            target_pos_y = y 
            target_pos_z = z
    

            q_hand_world_wxyz = rotate_y([0.707, 0, 0.707, 0],90)
            q_hand_world_wxyz,_ = rotate_z(current_quat=list(q_hand_world_wxyz),angle_deg=self.angle_degree)   
            
            # q_hand_world_wxyz2= rotate_y([0.707, 0, 0.707, 0],90)
            rand = np.random.uniform(0,-0.5)
            pose1= [target_pos_x,target_pos_y,target_pos_z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.05,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [target_pos_x,target_pos_y+rand,target_pos_z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,target_pos_y+rand,target_pos_z+0.05,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [target_pos_x,target_pos_y+rand,target_pos_z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose6 = [target_pos_x,target_pos_y+rand,target_pos_z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                    
        
    
    
            if y >0:
                executor = "robot_left"
            else:
                executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
    
            ]
            return pose