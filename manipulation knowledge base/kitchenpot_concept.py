from my_utils import *
from set_interface_try import *
import random

random_degree = random.uniform(-20,20)
class kitchenpot_concept ():
    def __init__(self,):
        self.scale = (0.18,0.18,0.18)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs 
    ):
        self.num_envs = num_envs
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "KitchenPot"
        concept_urdf_fallback= str(PARTNET_CONCEPT_ROOT / "kitchenpotPot" / "1fde7ce0a20f1502f54e1b6f41fdd78a" / "KitchenPot.urdf")


        self.random_degree = random_degree
        rot_concept,_= rotate_z(current_quat=[1, 0, 0, 0], angle_deg=-90)
        rot_concept,_ = rotate_z(list(rot_concept),self.random_degree)
        # self.hand_rot = rotate_x(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)

        x_micwave = random.uniform(0.8, 0.9)
        y_micwave = random.uniform(0.1, 0.2)
        x_micwave = 0.85
        y_micwave = 0.4
        
        self.scene_key = instance_name
        micwave_cfg = AssetCfg(
            fix_base=True,
            name=self.scene_key,
            urdf_path=urdf_path,
            pos=(x_micwave, y_micwave, 0.2),
            rot=(1,0,0,0),
            scale=self.scale,
            asset_kind = "articulation",
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
            source_urdf_path=micwave_cfg.urdf_path,
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
        
        concept_pot_cfg = AssetCfg(
            name="my_concept_kitchenpot",
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
        )
        return  [concept_pot_cfg]



 