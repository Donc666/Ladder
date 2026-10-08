"""Clock: advance the hand."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="clock",
    category="Clock",
    grasp_keywords=["hand"],
    motion="revolute",
    scale=(0.18, 0.18, 0.18),
    spawn_z=0.42,
    max_deg=30,
    grasp_offset_xyz=(0.0, 0.0, 0.25),
)

class clock_concept(PartNetConcept):
    spec = SPEC


class clock(PartNetSkill):
    spec = SPEC

    def advance(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"advance"}
