"""PartNet microwave: open the door."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_microwave",
    category="Microwave",
    grasp_keywords=["handle", "door"],
    motion="revolute",
    scale=(0.28, 0.28, 0.28),
    spawn_xy=(0.80, 0.0),
    spawn_z=0.42,
    concept_xy=(0.80, 1.6),
    max_deg=40,
    skip_joint_keywords=("button", "knob", "tray"),
    upright_hinge=True,
)

class partnet_microwave_concept(PartNetConcept):
    spec = SPEC


class partnet_microwave(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"open"}
