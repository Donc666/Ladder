"""Storage furniture: pull a drawer out."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="storage_drawer",
    category="StorageFurniture",
    grasp_keywords=["drawer"],
    motion="prismatic",
    scale=(0.30, 0.30, 0.30),
    spawn_z=0.40,
)

class storage_drawer_concept(PartNetConcept):
    spec = SPEC


class storage_drawer(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"open"}
