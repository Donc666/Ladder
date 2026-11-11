"""Oven: open or close the door."""

from dataclasses import replace

from my_utils import rotate_z
from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec, YAW, hand_quat_wxyz, scene_asset

SPEC = SkillSpec(
    type_name="oven",
    category="Oven",
    grasp_keywords=["handle"],
    motion="revolute",
    scale=(0.50, 0.50, 0.50),
    spawn_xy=(0.65, -1.25),
    spawn_z=0.50,
    concept_xy=(1000.0, 1000.0),
    concept_z_drop=0.38,
    concept_yaw_offset=270.0,
    max_deg=35,
    sign=-1.0,
    init_open_fraction=0.5,
    skip_joint_keywords=("handle", "button", "tray"),
    horizontal_hinge=True,
    extra_hand_rz=90.0,
    approach_z=-0.40,
    grasp_z=-0.08,
    retreat_z=0.0,
    arc_radius_shrink=0.0,
    joint_stiffness=0.1,
    joint_damping=10.0,
)
# Open swings the other way from close, and plans the handle at the closed joint.
OPEN = replace(
    SPEC,
    init_open_fraction=0.0,
    sign=1.0,
    plan_from_commanded=True,
    max_deg=70,
    retreat_z=-0.20,
    grasp_z=-0.06,
)
# Close follows the open hinge and swing, starting from the live door angle.
# The pack is then lowered slightly in world z.
CLOSE = replace(OPEN, sign=-1.0, init_open_fraction=0.0, arc_radius_shrink=0.0)


def _pin_oven_pose() -> None:
    """Door faces +Y toward the right arm. The extra yaw is fixed at 15 deg."""
    YAW["oven"] = -90.0 + 15.0


class oven_concept(PartNetConcept):
    spec = SPEC

    def import2sim(self, urdf_path, instance_name, num_envs):
        _pin_oven_pose()
        return super().import2sim(urdf_path, instance_name, num_envs)


class oven(PartNetSkill):
    spec = SPEC

    def import2sim(self, urdf_path, instance_name, num_envs):
        _pin_oven_pose()
        cfgs = super().import2sim(urdf_path, instance_name, num_envs)
        for cfg in cfgs:
            x, y, z = cfg.pos
            cfg.pos = (x, y, float(z) - 0.05)
            self._real_spawn_z = float(cfg.pos[2])
            cfg.link_density = 2000.0
            cfg.collider_type = "convex_decomposition"
        return cfgs

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=OPEN)

    def close(self, scene, env_id=0, fraction=0.5):
        """Close along the open arc from the live door angle, a little lower in z."""
        pack = self.execute(scene, env_id, fraction, spec_override=CLOSE, live_state=True)
        if not pack:
            return pack
        for step in pack:
            pose = list(step["pose"])
            pose[2] = float(pose[2]) - 0.05
            step["pose"] = pose
        return pack

    def into(self, scene, env_id=0, fraction=0.5):
        """Place the held bread in through the door, which faces +Y."""
        asset = scene_asset(scene, self.scene_key, env_id)
        if asset is None:
            raise RuntimeError(f"oven asset not found for {self.scene_key}")
        root = asset.data.root_pos_w[env_id].detach().cpu().numpy()
        q = hand_quat_wxyz(
            self.q_obj_world_wxyz,
            self.spec.extra_hand_ry,
            self.spec.extra_hand_rx,
            self.spec.extra_hand_rz,
            self.spec.extra_hand_local_ry,
        )
        q, _ = rotate_z(q, 90.0)
        x, y, z = float(root[0]), float(root[1]), float(root[2])
        pre = [x, y + 0.45, z , q[0], q[1], q[2], q[3]]
        place = [x, y - 0.15, z , q[0], q[1], q[2], q[3]]
        back = [x, y + 0.45, z , q[0], q[1], q[2], q[3]]
        return [
            {"pose": pre, "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_right"},
            {"pose": place, "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_right"},
            {"pose": back, "gripper": "open", "use_obstacle_avoidance": False, "executor": "robot_right"},
        ]


ATOMS = {"open", "close", "into"}
