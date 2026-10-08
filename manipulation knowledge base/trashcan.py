"""Trash can: open or close the lid."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="trashcan",
    category="TrashCan",
    grasp_keywords=["cover", "lid"],
    motion="revolute",
    scale=(0.36, 0.36, 0.36),
    spawn_z=0.44,
    max_deg=40,
)
CLOSE = replace(SPEC, sign=-1.0, init_open_fraction=0.6)

class trashcan_concept(PartNetConcept):
    spec = SPEC


class trashcan(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=CLOSE)


ATOMS = {"open", "close"}
