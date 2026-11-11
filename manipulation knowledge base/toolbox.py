"""Plastic toolbox spawned from a USD that already has simplified collision.

Source: Synthesis AI plastic_toolbox001, collision simplified in
DRL-navigation-isaac (CC BY-NC 4.0). See assets/toolbox/LICENSES.md.
The mesh origin sits on the bottom, about 0.29 x 0.18 x 0.16 m.
"""

from my_utils import PROJECT_ROOT
from set_interface_try import AssetCfg

TOOLBOX_USD = str(PROJECT_ROOT / "assets" / "toolbox" / "toolbox_01.usd")


class toolbox:
    def __init__(self):
        # Left of the breads, on the same table height the breads use.
        self.spawn_pos = (0.28, 0.0, 0.35)
        self.scale = (1.0, 1.0, 1.0)

    def import2sim(self, urdf_path, instance_name: str, num_envs):
        self.scene_key = instance_name
        print("import", instance_name, "usd=", TOOLBOX_USD)
        cfg = AssetCfg(
            name=instance_name,
            source_kind="usd",
            usd_path=TOOLBOX_USD,
            pos=tuple(self.spawn_pos),
            rot=(1.0, 0.0, 0.0, 0.0),
            scale=tuple(self.scale),
            fix_base=False,
            asset_kind="rigid",
            mass=1.2,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        return [cfg for _ in range(num_envs)]
