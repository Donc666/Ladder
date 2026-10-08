"""Oven: close the door."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="oven",
    category="Oven",
    grasp_keywords=["handle"],
    motion="revolute",
    scale=(0.50, 0.50, 0.50),
    spawn_xy=(0.95, 0.0),
    spawn_z=0.50,
    concept_xy=(0.95, 1.6),
    concept_z_drop=0.38,
    concept_yaw_offset=270.0,
    max_deg=35,
    sign=-1.0,
    init_open_fraction=0.5,
    skip_joint_keywords=("handle", "button", "tray"),
    horizontal_hinge=True,
    extra_hand_rz=90.0,
    grasp_z=-0.05,
    retreat_z=0.0,
    arc_radius_shrink=0.04,
)

class oven_concept(PartNetConcept):
    spec = SPEC


class oven(PartNetSkill):
    spec = SPEC

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"close"}
