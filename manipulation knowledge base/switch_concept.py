from my_utils import *
from set_interface_try import *
import random

random_degree = random.uniform(-70,70)
class switch_concept ():
    def __init__(self,):
        self.scale = (0.06,0.06,0.06)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs 
    ):
        self.num_envs = num_envs
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "Switch"
        concept_urdf_fallback= str(PROJECT_ROOT / "assets/switch_concept/5a8c4b01d7cdc687243fd6c0bc6255e8/Switch.urdf")


        self.random_degree = random_degree
        rot_concept,_= rotate_z(current_quat=[1, 0, 0, 0], angle_deg=-90)
        rot_concept,_ = rotate_z(list(rot_concept),self.random_degree)
        # self.hand_rot = rotate_x(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)

        x_micwave = random.uniform(0.8, 0.9)
        y_micwave = random.uniform(0.1, 0.2)
        x_micwave = 0.85
        y_micwave = 0.4
        
        self.scene_key = instance_name

        # 3) concept urdf 
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=urdf_path,
            concept_root=concept_root,
            category=category,
        )
        if resolved_concept_urdf is None:
            print(f"[WARN] model_id={model_id} not found, fallback: {concept_urdf_fallback}")
            resolved_concept_urdf = concept_urdf_fallback
        else:
            print(f"[INFO] model_id={model_id}, concept URDF: {resolved_concept_urdf}")
        #concept urdf will be used in the atomic task
        self.concept_urdf_path = resolved_concept_urdf
        
        concept_laptop_cfg = AssetCfg(
            name="my_concept_switch",
            urdf_path=resolved_concept_urdf,
            pos=(1000, 1000, 0.5),
            rot=rot_concept,
            scale=self.scale,
            asset_kind = "articulation",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
            joint_pos = {"Standard_Base_to_FlipX_Switch.*": 0.14},
            # joint_vel = {"Standard_Base_to_FlipX_Switch.*": 0.0},
        )
        return  [concept_laptop_cfg]