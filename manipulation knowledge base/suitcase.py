"""Suitcase: extend the handle."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="suitcase",
    category="Suitcase",
    grasp_keywords=["handle"],
    motion="prismatic",
    scale=(0.40, 0.40, 0.40),
    spawn_z=0.44,
    grasp_offset_xyz=(0.0, 0.0, 0.33),
    approach_offset_xyz=(-0.1, 0.0, 0.0),
    approach_match_handle_z=True,
)

class suitcase_concept(PartNetConcept):
    spec = SPEC


class suitcase(PartNetSkill):
    spec = SPEC

    def extend(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"extend"}
