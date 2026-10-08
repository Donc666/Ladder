from my_utils import *
from set_interface_try import *
import random

class remote ():
    def __init__(self):
        self.scale = (0.6,0.6,0.6)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str
    ):
        
        
        x1 = random.uniform(0.4, 0.8)
        y1 = random.uniform(-0.6, 0.0)



        rot,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(-60.0,60.0)),current_quat=[1,0,0,0])
        rot1,_ = rotate_z([1,0,0,0],self.angel_degree)
        
   

        self.scene_key = instance_name

        cfg = AssetCfg(
            
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.34),
            rot = rot1,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            fix_base= False,
            # source_kind = "cuboid",
            cuboid_size= (0.05,0.05,0.05),
            cuboid_color = (1.0, 0.5, 0.2),
            # asset_kind = "articulation",
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
    

    def pick(self,scene,env_id):
        asset = scene[self.scene_key]
        root_pos = asset.data.root_pos_w[env_id]      
 


        
        x= root_pos[0].item()-1
        y = root_pos[1].item()+1
        z = root_pos[2].item()
        self.hand_rot = rotate_y([0.707, 0, 0.707, 0],-90)
        self.hand_rot,_ = rotate_z(current_quat=list(self.hand_rot),angle_deg=self.angel_degree )

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)
        
        self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angel_degree)

        handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
        target_pos_x = 0.5
        target_pos_y = 0
        target_pos_z = z+0.2
        # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
        # handover_rot_right,_ = rotate_z(list(handover_rot_right),90)

        pose1= [x,y-0.01,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose2 = [x,y-0.01,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose3 = [x,y-0.01,z+0.4,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose4 = [x,y+0.4,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose5 = [x,y+0.4,z+0.4,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        pose6 = [x,y+0.4,z+0.4,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

        if y < 0:
            executor = "robot_right"
        else:
            executor = "robot_left"
        pose = [
            {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},

        ]
        return pose