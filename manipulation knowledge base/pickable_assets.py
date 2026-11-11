"""Unused rigid assets, each with a toycar-style pick atom.

Folders under assets/ that no existing type name matches are registered here.
Object tasks sample about five of them at import time, so the grasped
categories are not the same fixed set on every run.
"""

import random
import re
import zlib

from my_utils import *
from set_interface_try import *

# Tabletop folders that were not matched by an existing type.
# Furniture racks and unnamed dumps are left out.
PICKABLE_TYPE_NAMES = (
    "flutedblock",
    "calculator",
    "cupwithliquid",
    "tissuebox",
    "chipstub",
    "shampoo",
    "tabletennis",
    "perfume",
    "oliveoil",
    "jamjar",
    "fork",
    "knife",
    "apple",
    "milkbox",
    "woodenbox",
    "book",
    "sandclock",
    "shampoo",
    "candlestick",
    "dumbbell",
    "teanet",
    "baguette",
    "smallspeaker",
    "markpen",
    "pencup",
    "battery",
    "msg",
    "soysauce",
    "vinegar",
    "steamer",
    "boxdrink",
    "vagetable",
    "remotecontrol",
    "smallshovel",
    "brush",
    "woodenmallet",
    "gong",
    "woodenblock",
    "waterer",
    "wineglass",
    "trophy",
    "notebook",
    "brushpen",
    "rest",
    "glue",
    "cleaner",
    "screen",
    "speaker",
    "milktea",
    "fruit",
    "board",
    "soap",
    "hydratingoil",
    "callbell",
    "coffeebox",
    "perfume",
    "keyboard",
    "whiteboarderaser",
    "toothpaste",
    "minichalkboard",
    "plant",
)

_PICK_ATOMS = {
    "pick",
    "pick_simultaneously_left",
    "pick_simultaneously_right",
    "pick_with_left_arm",
}

_ZONES = [
    ((0.45, 0.75), (-0.35, 0.35)),
]


def _sample_pos(instance_name: str):
    idx = 0
    tail = str(instance_name).split("_")[-1]
    if tail.isdigit():
        idx = int(tail) - 1
    xr, yr = _ZONES[idx % len(_ZONES)]
    return random.uniform(*xr), random.uniform(*yr)


def make_pickable(type_name: str, scale=(0.05, 0.05, 0.05), mass: float = 0.1, grasp_z: float = 0.0):
    """One rigid category whose pick waypoints match toycar.pick."""

    class Pickable:
        def __init__(self):
            self.scale = scale
            self.type_name = type_name

        def import2sim(self, urdf_path: str, instance_name: str, num_envs):
            self.num_envs = num_envs
            self.angle_degree = np.random.uniform(40, 90)
            x1, y1 = _sample_pos(instance_name)
            r_z = R.from_euler("z", self.angle_degree, degrees=True)
            r_x = R.from_euler("x", 90, degrees=True)
            self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())
            self.scene_key = instance_name
            cfg = AssetCfg(
                name=instance_name,
                urdf_path=urdf_path,
                pos=(x1, y1, 0.35),
                rot=self.q_obj_world_wxyz,
                scale=self.scale,
                mass=mass,
                stiffness=0.0,
                damping=0.0,
                fix_base=False,
                cuboid_size=(0.05, 0.05, 0.05),
                cuboid_color=(1.0, 0.5, 0.2),
                asset_kind="rigid",
                metallic=0.9,
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0,
                compliant_contact_damping=0,
                compliant_contact_stiffness=0,
            )
            return [cfg]

        def pick(self, scene, env_id):
            for envid in range(self.num_envs):
                self.actual_instance_name = f"{self.scene_key}__env{envid}"
                asset = scene[self.actual_instance_name]
                root_pos = asset.data.root_pos_w[env_id]
                x = root_pos[0].item()
                y = root_pos[1].item()
                z = root_pos[2].item()
                r_x = R.from_euler("x", 90, degrees=True)
                r_y = R.from_euler("y", 90, degrees=True)
                r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
                r_hand_world = r_obj_world * (r_y * r_x)
                q_hand_world_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())
                hand_to_world_rot = R.from_quat(r_hand_world.as_quat())
                pos = [x, y, z] - hand_to_world_rot.apply([-0.02, 0, 0.0])
                gz = z + grasp_z
                pose1 = [pos[0], pos[1], gz + 0.3, *q_hand_world_wxyz]
                pose2 = [pos[0], pos[1], gz + 0.05, *q_hand_world_wxyz]
                pose3 = [pos[0], pos[1], gz + 0.2, *q_hand_world_wxyz]
                pose4 = [pos[0], pos[1], gz + 0.2, *q_hand_world_wxyz]
                executor = "robot_left" if y > 0 else "robot_right"
                return [
                    {"pose": pose1, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose2, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose3, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose4, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                ]

        def right(self, scene, env_id):
            for envid in range(self.num_envs):
                self.actual_instance_name = f"{self.scene_key}__env{envid}"
                asset = scene[self.actual_instance_name]
                root_pos = asset.data.root_pos_w[env_id]
                x = root_pos[0].item()
                y = root_pos[1].item()
                z = root_pos[2].item() + grasp_z
                target_pos_y = y - np.random.uniform(0.05, 0.07)
                hand_rot = rotate_y([0.707, 0, 0.707, 0], 90)
                pose1 = [x, target_pos_y, z + 0.2, *hand_rot]
                pose2 = [x, target_pos_y, z + 0.1, *hand_rot]
                pose3 = [x, target_pos_y, z + 0.2, *hand_rot]
                pose4 = [x, target_pos_y, z + 0.2, *hand_rot]
                executor = getattr(self, "right_executor", None)
                if executor not in ("robot_left", "robot_right"):
                    executor = "robot_left" if y > 0 else "robot_right"
                return [
                    {"pose": pose1, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose2, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose3, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                    {"pose": pose4, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                ]

        def _pick_as(self, scene, env_id, executor: str):
            poses = self.pick(scene, env_id)
            for pose in poses:
                pose["executor"] = executor
            return poses

        def pick_simultaneously_left(self, scene, env_id):
            executor = getattr(self, "left_executor", None)
            if executor not in (
                "robot_left",
                "robot_right",
                "simultaneous_left",
                "simultaneous_right",
            ):
                executor = "robot_left"
            return self._pick_as(scene, env_id, executor)

        def pick_simultaneously_right(self, scene, env_id):
            return self._pick_as(scene, env_id, "robot_right")

        def pick_with_left_arm(self, scene, env_id):
            return self._pick_as(scene, env_id, "robot_left")

        def scan(self, scene, env_id):
            for envid in range(self.num_envs):
                self.actual_instance_name = f"{self.scene_key}__env{envid}"
                asset = scene[self.actual_instance_name]
                q_hand = rotate_x(list(rotate_y([0.707, 0, 0.707, 0], 90)), 90)
                pose1 = [0.5, 0.2, 0.7, *q_hand]
                pose2 = [0.5, 0.2, 0.7, *q_hand]
                return [
                    {"pose": pose1, "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_left"},
                    {"pose": pose2, "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_left"},
                ]

    Pickable.__name__ = type_name
    return Pickable


def _bind(cls):
    return lambda: cls()


def _dispatch(method: str):
    return lambda obj, sc, p: getattr(obj, method)(scene=sc, env_id=p.get("_env_id", 0))


_ATOM_METHODS = (
    "pick",
    "right",
    "pick_simultaneously_left",
    "pick_simultaneously_right",
    "pick_with_left_arm",
)

PICKABLE_FACTORY = {}
PICKABLE_ATOMS = {}
PICKABLE_DISPATCH = {}
# Shared size for the newly added pickables. 0.04 is 0.8 of the original 0.05.
# Olive oil, jam jar, the table-tennis ball, and soap keep their own scales.
_DEFAULT_PICKABLE_SCALE = (0.04, 0.04, 0.04)
_SCALES = {
    # 0.85 times the enlarged scale 0.5 (~15 cm bottle), not 0.85 times 0.05.
    "oliveoil": (0.425, 0.425, 0.425),
    # Same enlargement as olive oil: the mesh is ~0.25 m, so 0.04 would be ~1 cm.
    "soysauce": (0.425, 0.425, 0.425),
    # 0.7 times the original shared scale of 0.05.
    "jamjar": (0.035, 0.035, 0.035),
    # 0.8 times the original shared scale of 0.05.
    "tabletennis": (0.04, 0.04, 0.04),
    # 0.8 times the current default 0.04.
    "soap": (0.032, 0.032, 0.032),
    # Mesh is already about 7.5 cm. 0.06 shrank it to a few millimeters.
    "apple": (1.0, 1.0, 1.0),
    # A little above the shared 0.04 scale.
    "sandclock": (0.05, 0.05, 0.05),
}
_GRASP_Z = {
    "shampoo": 0.03,
    "perfume": 0.05,
    "oliveoil": 0.09,
    "jamjar": 0.03,
    "soysauce": 0.08,
    "sandclock": 0.05,
    "fruit": 0.03,
    "trophy": 0.05,
}
for _name in PICKABLE_TYPE_NAMES:
    _mass = 0.05 if _name == "tabletennis" else 0.1
    _scale = _SCALES.get(_name, _DEFAULT_PICKABLE_SCALE)
    PICKABLE_FACTORY[_name] = _bind(
        make_pickable(_name, scale=_scale, mass=_mass, grasp_z=_GRASP_Z.get(_name, 0.0))
    )
    PICKABLE_ATOMS[_name] = set(_ATOM_METHODS)
    for _atom in _ATOM_METHODS:
        PICKABLE_DISPATCH[(_name, _atom)] = _dispatch(_atom)


def candidate_groups(size: int = 5) -> list[list[str]]:
    names = list(PICKABLE_TYPE_NAMES)
    groups = []
    for start in range(0, len(names), size):
        group = names[start : start + size]
        if len(group) < size:
            for name in names:
                if name not in group:
                    group.append(name)
                if len(group) == size:
                    break
        groups.append(group)
    return groups


def pool_for_task(task_name: str, size: int = 5) -> list[str]:
    groups = candidate_groups(size)
    idx = zlib.crc32(str(task_name).encode("utf-8")) % len(groups)
    return list(groups[idx])


def _task_tokens(task_name: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", task_name.lower()) if tok}


def _type_named_in_task(type_name: str, tokens: set[str]) -> bool:
    """True when the task name already names this category, including a plural."""
    name = re.sub(r"[^a-z0-9]+", "", str(type_name).lower())
    if not name:
        return False
    plurals = {name, name + "s", name + "es"}
    if name.endswith("y"):
        plurals.add(name[:-1] + "ies")
    return bool(tokens & plurals)


def assign_pick_candidates(task: dict) -> None:
    """Vary the grasped category only for a task whose name says "object".

    place_bread_skillet names bread, so bread stays bread and the URDF is
    still drawn from the bread folder. A name like open_cabinet_and_put_object_in
    does not name the grasped category, so that slot is drawn from five
    candidates. Phone in place_phone_stand_and_place_object_scale stays a
    phone; the unnamed object on the scale is the one that varies.
    """
    task_name = str(task.get("task_name") or "")
    tokens = _task_tokens(task_name)
    if task_name == "place_a2b_right_and_grap_roller":
        # Cleaner, fruit, hourglass, soy sauce. One distinct type per slot.
        pool = ["cleaner", "fruit", "sandclock", "soysauce"]
        swappable = [
            asset
            for asset in task.get("assets", [])
            if isinstance(asset, dict) and asset.get("type") in ("toycar", "rubikscube")
        ]
        if len(swappable) <= len(pool):
            chosen = random.sample(pool, len(swappable))
        else:
            chosen = [pool[i % len(pool)] for i in range(len(swappable))]
            random.shuffle(chosen)
        print(f"[INFO] {task_name} pick candidates: {', '.join(pool)}")
        for asset, new_type in zip(swappable, chosen):
            old = asset.get("type")
            asset["type"] = new_type
            asset.pop("urdf_path", None)
            asset.pop("asset_name", None)
            if asset.get("id") == "toycar_1":
                asset["pos"] = [
                    random.uniform(0.55, 0.8),
                    random.uniform(0.05, 0.15),
                    0.35,
                ]
                print(f"[INFO]   toycar_1 pos: {asset['pos']}")
            if asset.get("id") == "rubikscube_1":
                y_band = random.choice(((0.2, 0.7), (-0.7, -0.2)))
                asset["pos"] = [
                    random.uniform(0.5, 0.8),
                    random.uniform(*y_band),
                    0.35,
                ]
                print(f"[INFO]   rubikscube_1 pos: {asset['pos']}")
            print(f"[INFO]   {asset.get('id')}: {old} -> {new_type}")
        # right() uses the same arm as the previous rubik pick.
        rubik = next(
            (a for a in swappable if a.get("id") == "rubikscube_1" and "pos" in a),
            None,
        )
        toycar = next((a for a in swappable if a.get("id") == "toycar_1"), None)
        if rubik is not None and toycar is not None:
            pick_arm = "robot_left" if float(rubik["pos"][1]) > 0 else "robot_right"
            toycar["right_executor"] = pick_arm
            print(f"[INFO]   rubik pick {pick_arm}; toycar right {toycar['right_executor']}")
        return
    if task_name == "place_object_in_basket_and_pick":
        pool = [
            name
            for name in pool_for_task(task_name)
            if name not in ("perfume", "oliveoil")
        ]
        swappable = [
            asset
            for asset in task.get("assets", [])
            if isinstance(asset, dict) and asset.get("type") in ("toycar", "rubikscube")
        ]
        if len(swappable) <= len(pool):
            chosen = random.sample(pool, len(swappable))
        else:
            chosen = [pool[i % len(pool)] for i in range(len(swappable))]
            random.shuffle(chosen)
        print(f"[INFO] {task_name} pick candidates: {', '.join(pool)}")
        for asset, new_type in zip(swappable, chosen):
            old = asset.get("type")
            asset["type"] = new_type
            asset.pop("urdf_path", None)
            asset.pop("asset_name", None)
            print(f"[INFO]   {asset.get('id')}: {old} -> {new_type}")
        return
    if task_name == "scan_object":
        # The scanned object is teabox_1. One of this task's five categories.
        pool = ["apple" if name == "oliveoil" else name for name in pool_for_task(task_name)]
        pool = [name for name in pool if name != "perfume"]
        print(f"[INFO] {task_name} pick candidates: {', '.join(pool)}")
        for asset in task.get("assets", []):
            if not isinstance(asset, dict) or asset.get("id") != "teabox_1":
                continue
            new_type = random.choice(pool)
            old = asset.get("type")
            asset["type"] = new_type
            asset.pop("urdf_path", None)
            asset.pop("asset_name", None)
            print(f"[INFO]   {asset.get('id')}: {old} -> {new_type}")
        return
    if "object" not in tokens:
        return
    assets = [a for a in task.get("assets", []) if isinstance(a, dict)]
    atoms_by_id: dict[str, set[str]] = {}
    for step in task.get("atomic_plan", []):
        if not isinstance(step, dict):
            continue
        aid = step.get("asset")
        if not aid:
            continue
        atoms_by_id.setdefault(aid, set()).add(str(step.get("atom") or ""))
    concept_bases = {
        str(asset.get("type"))[:-8]
        for asset in assets
        if str(asset.get("type", "")).endswith("_concept")
    }
    swappable = []
    for asset in assets:
        atoms = atoms_by_id.get(asset.get("id")) or set()
        if not atoms or not atoms <= _PICK_ATOMS:
            continue
        if _type_named_in_task(str(asset.get("type") or ""), tokens):
            continue
        # One body grasped by both arms, or an articulated pair, stays as authored.
        both_arms = "pick_simultaneously_left" in atoms and "pick_simultaneously_right" in atoms
        if both_arms or asset.get("type") in concept_bases:
            continue
        swappable.append(asset)
    if not swappable:
        return
    task_name = str(task.get("task_name") or "")
    pool = pool_for_task(task_name)
    if len(swappable) <= len(pool):
        chosen = random.sample(pool, len(swappable))
    else:
        chosen = [pool[i % len(pool)] for i in range(len(swappable))]
        random.shuffle(chosen)
    print(f"[INFO] {task_name} pick candidates: {', '.join(pool)}")
    for asset, new_type in zip(swappable, chosen):
        old = asset.get("type")
        asset["type"] = new_type
        asset.pop("urdf_path", None)
        asset.pop("asset_name", None)
        print(f"[INFO]   {asset.get('id')}: {old} -> {new_type}")
