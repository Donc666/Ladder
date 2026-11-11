"""Drill spawned next to the toolbox.

Mesh long axis is Z and about 1.9 m. A local X rotation of 90 degrees lays it
flat, and scale 0.11 makes the length about 21 cm.
"""

from set_interface_try import AssetCfg


class drill:
    def __init__(self):
        self.spawn_pos = (0.70, -0.16, 0.40)
        self.scale = (0.11, 0.11, 0.11)
        # wxyz, +90 deg about local X: long axis Z falls onto world -Y.
        self.rot = (0.70710678, 0.70710678, 0.0, 0.0)

    def import2sim(self, urdf_path, instance_name: str, num_envs):
        self.scene_key = instance_name
        print("import", instance_name, "urdf=", urdf_path)
        cfg = AssetCfg(
            name=instance_name,
            urdf_path=urdf_path,
            pos=tuple(self.spawn_pos),
            rot=tuple(self.rot),
            scale=tuple(self.scale),
            fix_base=False,
            asset_kind="rigid",
            mass=0.4,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        return [cfg for _ in range(num_envs)]
