"""Coffee machine: turn the knob, lever, or button."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="coffeemachine",
    category="CoffeeMachine",
    grasp_keywords=["knob", "lever", "button"],
    motion="revolute",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.40,
    max_deg=25,
)

class coffeemachine_concept(PartNetConcept):
    spec = SPEC


class coffeemachine(PartNetSkill):
    spec = SPEC

    def on(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"on"}
