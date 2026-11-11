import random

from my_utils import *
from set_interface_try import *

# Local albedo maps already in the project. Missing files are skipped at import time.
# Set TABLE_TEXTURE_NAME to pin one look, or None to sample from TABLE_TEXTURE_POOL.
TABLE_TEXTURES = {
    "walnut": {
        "file": str(PROJECT_ROOT / "assets" / "table" / "textures" / "wood_walnut.jpg"),
        "roughness": 0.55,
    },
    "birch": {
        "file": str(
            PROJECT_ROOT
            / "Lightwheel_Kitchen"
            / "Collected_KitchenRoom"
            / "Kitchen_InsularShelf"
            / "texture"
            / "tfjbderc_4K_Albedo.jpg"
        ),
        "roughness": 0.50,
    },
    "cabinet_oak": {
        "file": str(
            PROJECT_ROOT
            / "Lightwheel_Kitchen"
            / "Collected_KitchenRoom"
            / "Kitchen_Cabinet001"
            / "texture"
            / "Kitchen_Cabinet001_BC001.jpg"
        ),
        "roughness": 0.58,
    },
    "honey_oak": {
        "file": str(PROJECT_ROOT / "Lightwheel_Bedroom" / "LW_Loft" / "textures" / "T_Wood001_BC.png"),
        "roughness": 0.55,
    },
    "ash": {
        "file": str(PROJECT_ROOT / "Lightwheel_Bedroom" / "LW_Loft" / "textures" / "T_Wood002_BC.png"),
        "roughness": 0.50,
    },
    "oak_veneer": {
        "file": str(
            PROJECT_ROOT
            / "Lightwheel_Toyroom"
            / "Assets"
            / "KidRoom_Table01"
            / "textures"
            / "T_WoodVeneerOak_BC.png"
        ),
        "roughness": 0.52,
    },
    "walnut_plank": {
        "file": str(PROJECT_ROOT / "Lightwheel_Bedroom" / "textures" / "T_Floor001_BC.png"),
        "roughness": 0.62,
    },
    "gray_oak": {
        "file": str(PROJECT_ROOT / "Lightwheel_Toyroom" / "Assets" / "textures" / "T_WoodenFloor_BC.jpg"),
        "roughness": 0.60,
    },
    "marble": {
        "file": str(
            PROJECT_ROOT
            / "Lightwheel_Bedroom"
            / "LW_Loft"
            / "Loft"
            / "Table038"
            / "texture"
            / "T_Table038_BC001.png"
        ),
        "roughness": 0.22,
    },
    "terrazzo": {
        "file": str(
            PROJECT_ROOT
            / "Lightwheel_Kitchen"
            / "Collected_KitchenRoom"
            / "Kitchen_InsularShelf"
            / "texture"
            / "Kitchen_InsularShelf_BC001.jpg"
        ),
        "roughness": 0.35,
    },
}
TABLE_TEXTURE_POOL = tuple(TABLE_TEXTURES)
TABLE_TEXTURE_NAME = None  # pin a name, or None to randomize each import


def available_table_textures() -> list[str]:
    names = []
    for name in TABLE_TEXTURE_POOL:
        path = Path(TABLE_TEXTURES[name]["file"])
        if path.exists():
            names.append(name)
    return names


def pick_table_texture() -> tuple[str | None, str | None, float]:
    if TABLE_TEXTURE_NAME is not None:
        spec = TABLE_TEXTURES.get(TABLE_TEXTURE_NAME)
        if spec and Path(spec["file"]).exists():
            return TABLE_TEXTURE_NAME, spec["file"], float(spec["roughness"])
        print(f"[Table] pinned texture missing: {TABLE_TEXTURE_NAME}")
        return None, None, 0.55
    names = available_table_textures()
    if not names:
        return None, None, 0.55
    name = random.SystemRandom().choice(names)
    spec = TABLE_TEXTURES[name]
    return name, spec["file"], float(spec["roughness"])


class table ():
    def __init__(self):
        # Calibrated to approximate RoboTwin2 table geometry:
        # tabletop size ~ (1.2, 0.7) and tabletop height ~ 0.74.
        #
        # Source URDF bbox (assets/table/20985/bounding_box.json):
        #   size_x ~= 1.797336, size_y ~= 0.446670, size_z ~= 1.110815
        # target scales:
        #   sx = 1.2 / 1.797336 ~= 0.6677
        #   sy = 0.7 / 0.446670 ~= 1.5672
        #   sz ~= 0.6663  (so table overall height ~= 0.74)
        self.scale_micwave = (1.0, 1.0, 0.5)
        # Set root z so scaled top surface is close to z=0.74.
        self.table_root_pos = (0.6, 0.0, 0.25)
        pass                                                                    



    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        tex_name, tex_file, tex_rough = pick_table_texture()
        print(f"[Table] texture={tex_name} file={tex_file}", flush=True)
        cfgs = []
        self.scene_keys_by_env = {}
        for env_id in range(num_envs):
        
            self.scene_key = instance_name
            x_table = 0.6
            y_table = 0
            cfgs.append(AssetCfg(
                fix_base=False,
                name = self.scene_key,
                urdf_path=urdf_path,
                pos = self.table_root_pos,
                rot = (1,0,0,0),
                scale = self.scale_micwave,
                stiffness=0.0,
                damping=0.0,
                texture_file=tex_file,
                roughness=tex_rough,
                static_friction=0.8,
                dynamic_friction= 0.8,
                restitution= 0,
                compliant_contact_damping=0,
                compliant_contact_stiffness=0
            ))

        return cfgs
        