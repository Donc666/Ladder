"""Kettle: lift by the handle, or open the lid."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="kettle",
    category="Kettle",
    grasp_keywords=["handle"],
    motion="lift",
    scale=(0.22, 0.22, 0.22),
    spawn_xy=(0.65, 0.0),
    spawn_z=0.38,
    fix_base=False,
    joint_stiffness=0.0,
    joint_damping=0.0,
)
LID = replace(SPEC, grasp_keywords=["cover", "lid"], motion="prismatic")
LIFT = replace(SPEC, grasp_offset_xyz=(0.0, 0.0, 0.25), extra_hand_local_ry=90.0)

class kettle_concept(PartNetConcept):
    spec = SPEC


class kettle(PartNetSkill):
    spec = SPEC

    def lift(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=LIFT)

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=LID)


ATOMS = {"lift", "open"}
