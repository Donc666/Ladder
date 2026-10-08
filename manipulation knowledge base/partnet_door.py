"""PartNet door: close from ajar, or swing open."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_door",
    category="Door",
    grasp_keywords=["handle"],
    motion="revolute",
    scale=(0.8, 0.8, 0.8),
    spawn_xy=(0.80, 0.0),
    spawn_z=0.90,
    concept_xy=(0.80, 1.8),
    concept_z_drop=0.73,
    max_deg=40,
    sign=-1.0,
    init_open_fraction=0.7,
    skip_joint_keywords=("handle",),
    grasp_z=-0.08,
    surface_clearance=0.04,
    upright_hinge=True,
    grasp_offset_xyz=(0.10, 0.0, 0.0),
)
OPEN = replace(SPEC, sign=1.0, init_open_fraction=0.0)

class partnet_door_concept(PartNetConcept):
    spec = SPEC


class partnet_door(PartNetSkill):
    spec = SPEC

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=OPEN)


ATOMS = {"close", "open"}
