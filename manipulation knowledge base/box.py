"""Box: open the lid, or pull the drawer."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="box",
    category="Box",
    grasp_keywords=["cover", "lid", "drawer"],
    motion="revolute",
    scale=(0.4, 0.4, 0.4),
    spawn_z=0.38,
    max_deg=30,
)
DRAWER = replace(SPEC, grasp_keywords=["drawer"], motion="prismatic", scale=(0.22, 0.22, 0.22), spawn_z=0.38)

class box_concept(PartNetConcept):
    spec = SPEC


class box(PartNetSkill):
    spec = SPEC

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=SPEC)

    def pull(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=DRAWER)


ATOMS = {"open", "pull"}
