"""PartNet fan: rotate the rotor."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_fan",
    category="Fan",
    grasp_keywords=["rotor"],
    motion="revolute",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.40,
    max_deg=40,
    concept_yaw_offset=0.0,
)

class partnet_fan_concept(PartNetConcept):
    spec = SPEC


class partnet_fan(PartNetSkill):
    spec = SPEC

    def rotate(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"rotate"}
