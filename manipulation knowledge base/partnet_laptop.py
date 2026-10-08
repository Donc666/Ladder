"""PartNet laptop: open the screen."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_laptop",
    category="Laptop",
    grasp_keywords=["screen"],
    motion="revolute",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.38,
    max_deg=40,
)

class partnet_laptop_concept(PartNetConcept):
    spec = SPEC


class partnet_laptop(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"open"}
