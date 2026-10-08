"""Folding chair: unfold the seat."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="foldingchair",
    category="FoldingChair",
    grasp_keywords=["seat"],
    motion="revolute",
    scale=(0.40, 0.40, 0.40),
    spawn_z=0.46,
    max_deg=35,
)

class foldingchair_concept(PartNetConcept):
    spec = SPEC


class foldingchair(PartNetSkill):
    spec = SPEC

    def unfold(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"unfold"}
