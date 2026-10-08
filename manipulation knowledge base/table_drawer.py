"""Table drawer: close from a partly open pose, or pull open."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="table_drawer",
    category="Table",
    grasp_keywords=["drawer"],
    motion="prismatic",
    scale=(0.22, 0.22, 0.22),
    spawn_z=0.40,
    sign=-1.0,
    init_open_fraction=0.5,
)
OPEN = replace(SPEC, sign=1.0, init_open_fraction=0.0)

class table_drawer_concept(PartNetConcept):
    spec = SPEC


class table_drawer(PartNetSkill):
    spec = SPEC

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=OPEN)


ATOMS = {"close", "open"}
