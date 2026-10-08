from my_utils import *
from set_interface_try import *
import random

class storagefurniture ():
    def __init__(self):
        self.scale_storagefurniture = (0.3,0.3,0.3)
        pass

    def import2sim(
        self,
        urdf_path: str,
    ):
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "StorageFurniture"
        concept_urdf_fallback= str(PARTNET_CONCEPT_ROOT / "Microwave" / "87bae84777fe8b702bac1bcdfc2402d2" / "Microwave.urdf")


        rot_real, self.random_degree = random_rotate(current_quat=[1, 0, 0, 0], random_degree_range=25.0)
        rot_concept,_ = concept_rotate(current_quat=[0.707, 0, 0, -0.707], angle_deg=self.random_degree)
        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)

        x_micwave = random.uniform(0.7, 0.8)
        y_micwave = random.uniform(0.1, 0.2)
        
        storagefurniture_cfg = AssetCfg(
            fix_base=True,
            name="my_storagefurniture",
            urdf_path=urdf_path,
            pos=(x_micwave, y_micwave, 0.4),
            rot=rot_real,
            scale=self.scale_storagefurniture,
            stiffness=0.0,
            damping=0.0,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
        )
        # 3) concept urdf 
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=storagefurniture_cfg.urdf_path,
            concept_root=concept_root,
            category=category,
        )
        if resolved_concept_urdf is None:
            print(f"ssdafasdfasfasdfasdf[WARN] model_id={model_id} not found, fallback: {concept_urdf_fallback}")
            resolved_concept_urdf = str(PARTNET_CONCEPT_ROOT / "Microwave" / "12cb11dbb135ca9ac59350d819542ec7" / "Storagefurniture.urdf")
        else:
            print(f"i have found a urdfP:[INFO] model_id={model_id}, concept URDF: {resolved_concept_urdf}")
        #concept urdf will be used in the atomic task
        self.concept_urdf_path = resolved_concept_urdf
        
        concept_storagefurniture_cfg = AssetCfg(
            name="my_storagefurniture_micwave",
            urdf_path=resolved_concept_urdf,
            pos=(1000, 1000, 0.3),
            rot=rot_concept,
            scale=self.scale_storagefurniture,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
        )
        return [storagefurniture_cfg, concept_storagefurniture_cfg]

    
    def open(self,scene):
        microwave_asset = scene["my_storagefurniture"]
        my_concept_micwave = scene["my_storagefurniture_micwave"]
        real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
        concept_root_pos = my_concept_micwave.data.root_pos_w[0].detach().cpu().numpy()

        h_pos, h_quat = get_handle_pose_physics(
            my_concept_micwave,
            self.scale_storagefurniture,
            concept_urdf_path=self.concept_urdf_path,
        )
        h_pos_real = map_concept_point_to_real_xy(h_pos, concept_root_pos, real_root_pos)
        finger_length = 0.05
        x1 = h_pos_real[0] - finger_length
        y1 = h_pos_real[1]
        z1 = h_pos_real[2]

        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)
        self.hand_rot = rotate_z(self.hand_rot)
        grasp_pose_pre= [x1-0.35,y1,z1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        grasp_pose = [x1-0.15,y1+0.05,z1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
        after_grasp = [x1 -0.4 ,y1+0.05 ,z1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

        put_pose = [
            {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
            {"pose":after_grasp,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"}
        ]
        return put_pose