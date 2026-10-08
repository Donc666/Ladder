"""PartNet pot: remove the lid."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_pot",
    category="KitchenPot",
    grasp_keywords=["cover", "lid"],
    motion="prismatic",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.38,
)

class partnet_pot_concept(PartNetConcept):
    spec = SPEC


class partnet_pot(PartNetSkill):
    spec = SPEC

    def remove(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"remove"}
