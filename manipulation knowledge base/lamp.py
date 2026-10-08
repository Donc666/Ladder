"""Lamp: aim the head, or press the button."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="lamp",
    category="Lamp",
    grasp_keywords=["head", "bar"],
    motion="revolute",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.40,
    max_deg=30,
    concept_yaw_offset=0.0,
)
BUTTON = replace(SPEC, grasp_keywords=["button"], max_deg=20)

class lamp_concept(PartNetConcept):
    spec = SPEC


class lamp(PartNetSkill):
    spec = SPEC

    def aim(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def on(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=BUTTON)


ATOMS = {"aim", "on"}
