"""PartNet switch: turn the toggle on."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_switch",
    category="Switch",
    grasp_keywords=["switch", "toggle"],
    motion="revolute",
    scale=(0.18, 0.18, 0.18),
    spawn_z=0.42,
    max_deg=20,
)

class partnet_switch_concept(PartNetConcept):
    spec = SPEC


class partnet_switch(PartNetSkill):
    spec = SPEC

    def on(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"on"}
