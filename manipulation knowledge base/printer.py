"""Printer: press the button or slider."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="printer",
    category="Printer",
    grasp_keywords=["button", "slider"],
    motion="prismatic",
    scale=(0.35, 0.35, 0.35),
    spawn_z=0.40,
)

class printer_concept(PartNetConcept):
    spec = SPEC


class printer(PartNetSkill):
    spec = SPEC

    def on(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"on"}
