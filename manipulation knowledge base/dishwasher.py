"""Dishwasher: open the door, or press the start button."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="dishwasher",
    category="Dishwasher",
    grasp_keywords=["handle", "door"],
    motion="revolute",
    scale=(0.45, 0.45, 0.45),
    spawn_xy=(0.90, 0.0),
    spawn_z=0.42,
    concept_xy=(0.90, 1.7),
    max_deg=40,
    skip_joint_keywords=("button", "knob", "tray"),
    upright_hinge=True,
)
BUTTON = replace(
    SPEC,
    grasp_keywords=["button"],
    motion="prismatic",
    skip_joint_keywords=("door", "handle", "tray"),
    max_deg=12,
)

class dishwasher_concept(PartNetConcept):
    spec = SPEC


class dishwasher(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def press(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=BUTTON)


ATOMS = {"open", "press"}
