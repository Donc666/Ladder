"""Faucet: turn the switch on or off."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="faucet",
    category="Faucet",
    grasp_keywords=["switch"],
    motion="revolute",
    scale=(0.25, 0.25, 0.25),
    spawn_z=0.40,
    max_deg=25,
)
OFF = replace(SPEC, sign=-1.0, init_open_fraction=0.6)

class faucet_concept(PartNetConcept):
    spec = SPEC


class faucet(PartNetSkill):
    spec = SPEC

    def on(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def off(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=OFF)


ATOMS = {"on", "off"}
