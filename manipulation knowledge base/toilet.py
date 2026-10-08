"""Toilet: flush the lever, open or close the lid."""

from dataclasses import replace

from partnet_skill_common import PartNetConcept, PartNetSkill, SkillSpec

SPEC = SkillSpec(
    type_name="toilet",
    category="Toilet",
    grasp_keywords=["lever"],
    motion="revolute",
    scale=(0.38, 0.38, 0.38),
    spawn_z=0.46,
    spawn_rpy_deg=(0.0, 0.0, 0.0),
    max_deg=20,
)
LID = replace(SPEC, grasp_keywords=["lid", "cover", "seat"], max_deg=40)
LID_CLOSE = replace(LID, sign=-1.0, init_open_fraction=0.7)
FLUSH = replace(
    SPEC,
    grasp_offset_xyz=(0.19, 0.08, 0.29),
    approach_offset_xyz=(-0.19, -0.08, 0.0),
    approach_match_handle_z=True,
    stop_at_grasp=True,
)

class toilet_concept(PartNetConcept):
    spec = SPEC


class toilet(PartNetSkill):
    spec = SPEC

    def flush(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=FLUSH)

    def open(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=LID)

    def close(self, scene, env_id=0, fraction=0.5):
        return self.execute(scene, env_id, fraction, spec_override=LID_CLOSE)


ATOMS = {"flush", "open", "close"}
