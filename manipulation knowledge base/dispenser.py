from my_utils import *
from set_interface_try import *
import random

class dispenser ():
    def __init__(self):
        self.scale_dispenser = (0.07,0.07,0.07)
        pass

    def import2sim(
        self,
        urdf_path: str,
    ):
        
        rot_real, self.random_degree = random_rotate(current_quat=[1, 0, 0, 0], random_degree_range=25.0)
        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)
        
        x_dispenser = random.uniform(0.6, 0.8)
        y_dispenser = random.uniform(-0.6, -0.4)
        # x_dispenser = 0.6
        # y_dispenser =  -0.5
        rot1,_ = random_rotate(current_quat=[1,0,0,0],random_degree_range=25.0)
        dispenser_cfg = AssetCfg(
            fix_base=False,
            name = "my_dispenser",
            urdf_path=urdf_path,
            pos = (x_dispenser, y_dispenser, 0.37),
            rot = rot1,
            scale = self.scale_dispenser,
            stiffness=0.0,
            damping=0.0,
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

        return [dispenser_cfg]
    
    def put(self,scene):
        dispenser_asset = scene["my_dispenser"]
        root_pos = dispenser_asset.data.root_pos_w[0]      

        x_dispenser = root_pos[0].item()
        y_dispenser = root_pos[1].item()
        z_dispenser = root_pos[2].item()

        height_range=(-0.05, 0.0)
        dz = np.random.uniform(*height_range)
        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)
        grasp_pose_pre= [x_dispenser-0.2,y_dispenser,z_dispenser + dz,0.707, 0, 0.707, 0]
        grasp_pose = [x_dispenser-0.06,y_dispenser,z_dispenser + dz,0.707, 0, 0.707, 0]

        put_pose = [
            {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
        ]
        return put_pose