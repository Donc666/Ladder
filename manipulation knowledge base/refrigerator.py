"""Refrigerator: open one door, or the other door."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="refrigerator",
    category="Refrigerator",
    grasp_keywords=["handle", "door"],
    motion="revolute",
    scale=(0.45, 0.45, 0.45),
    spawn_xy=(0.90, 0.0),
    spawn_z=0.45,
    concept_xy=(0.90, 1.7),
    max_deg=40,
    skip_joint_keywords=("tray", "button"),
    upright_hinge=True,
)
OTHER = replace(SPEC, grasp_index=1)

class refrigerator_concept(PartNetConcept):
    spec = SPEC


class refrigerator(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def open_other(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=OTHER)


ATOMS = {"open", "open_other"}
