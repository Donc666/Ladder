"""Washing machine: open the door."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="washingmachine",
    category="WashingMachine",
    grasp_keywords=["handle", "door"],
    motion="revolute",
    scale=(0.42, 0.42, 0.42),
    spawn_xy=(0.90, 0.0),
    spawn_z=0.42,
    concept_xy=(0.90, 1.7),
    max_deg=40,
    skip_joint_keywords=("button", "knob"),
)

class washingmachine_concept(PartNetConcept):
    spec = SPEC


class washingmachine(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"open"}
