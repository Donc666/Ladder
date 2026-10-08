"""PartNet window: close the sash from a partly open pose."""

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="partnet_window",
    category="Window",
    grasp_keywords=["handle", "window"],
    motion="prismatic",
    scale=(0.35, 0.35, 0.35),
    spawn_xy=(0.80, 0.0),
    spawn_z=0.55,
    concept_xy=(0.80, 1.7),
    sign=-1.0,
    init_open_fraction=0.6,
    skip_joint_keywords=("handle",),
)

class partnet_window_concept(PartNetConcept):
    spec = SPEC


class partnet_window(PartNetSkill):
    spec = SPEC

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction)


ATOMS = {"close"}
