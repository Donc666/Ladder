"""Display: extend the screen, or tilt it."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="display",
    category="Display",
    grasp_keywords=["screen"],
    motion="prismatic",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.42,
)
TILT = replace(SPEC, motion="revolute", max_deg=30)

class display_concept(PartNetConcept):
    spec = SPEC


class display(PartNetSkill):
    spec = SPEC

    def extend(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def tilt(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=TILT)


ATOMS = {"extend", "tilt"}
