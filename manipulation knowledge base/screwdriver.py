"""Screwdriver spawned next to the toolbox.

Mesh long axis is +Y and about 1.89 m, so scale 0.12 makes it roughly 23 cm.
"""

from set_interface_try import AssetCfg


class screwdriver:
    def __init__(self):
        self.spawn_pos = (0.70, 0.02, 0.38)
        self.scale = (0.12, 0.12, 0.12)

    def import2sim(self, urdf_path, instance_name: str, num_envs):
        self.scene_key = instance_name
        print("import", instance_name, "urdf=", urdf_path)
        cfg = AssetCfg(
            name=instance_name,
            urdf_path=urdf_path,
            pos=tuple(self.spawn_pos),
            rot=(1.0, 0.0, 0.0, 0.0),
            scale=tuple(self.scale),
            fix_base=False,
            asset_kind="rigid",
            mass=0.15,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        return [cfg for _ in range(num_envs)]
