"""PartNet dispenser: open or close the lid."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_dispenser",
    category="Dispenser",
    grasp_keywords=["lid", "nozzle", "cover"],
    motion="prismatic",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.40,
)
CLOSE = replace(SPEC, sign=-1.0, init_open_fraction=0.6)

class partnet_dispenser_concept(PartNetConcept):
    spec = SPEC


class partnet_dispenser(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=CLOSE)


ATOMS = {"open", "close"}
