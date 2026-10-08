"""Stapler lid: open the cover."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="stapler_lid",
    category="Stapler",
    grasp_keywords=["cover", "lid"],
    motion="revolute",
    scale=(0.20, 0.20, 0.20),
    spawn_z=0.38,
    max_deg=25,
)

class stapler_lid_concept(PartNetConcept):
    spec = SPEC


class stapler_lid(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"open"}
