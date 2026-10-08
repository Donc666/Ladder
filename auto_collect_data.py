import json
from dataclasses import dataclass
from typing import Any, Callable

# 按你项目实际路径改 import
import kb_path  # noqa: F401
from my_utils import *
from microwave import mircrowave
from dispenser import dispenser
from table import table
from bottle import bottle
from storagefurniture import storagefurniture
from laptop import laptop
from window import window
from window_concept import window_concept
from robogen_skills import ROBOGEN_FACTORY, ROBOGEN_ATOMS, ROBOGEN_DISPATCH
from hammer import hammer
from block import block
from bowl import bowl
from coaster import coaster
from bell import bell
from shoe import shoe
from switch import switch
from cabinet import cabinet
from door import door
from tray import tray
from alarmclock import alarmclock
from block_blue import block_blue
from block_red import block_red
from block_green import block_green
from block_big import block_big
from block_middle import block_middle
from block_small import block_small
from stapler import stapler
from fan import fan
from seal import seal
from remote import remote
from displaystand import displaystand
from dustbin import dustbin
from tabletrashbin import tabletrashbin
from set_interface_try import SceneBuilder
from runtime.import_pose_scene import attach_import_poses, wrap_scene_with_import_poses
from roller import roller
from microphone import microphone
from rack import rack
from mug import mug
from can import can
from kitchenpot import kitchenpot
from playingcards import playingcards
from mouse import mouse
from bread import bread
from toycar import toycar
from breadbasket import breadbasket
from basket import basket
from plasticbox import plasticbox
from microwave_concept import microwave_concept
from plate import plate
from shoebox import shoebox
from cup import cup
from frenchfries import frenchfries
from hamburg import hamburg
from skillet import skillet
from electronicscale import electronicscale
from rubikscube import rubikscube
from phone import phone
from phonestand import phonestand
from paymentsign import paymentsign
from scanner import scanner
from teabox import teabox
from seal import seal
from pillbottle import pillbottle
from laptop_concept import laptop_concept
from kitchenpot_concept import kitchenpot_concept
from switch_concept import switch_concept
from cabinet_concept import cabinet_concept

# TableTopSceneCfg / sim 这些用你当前 micwave_scene.py 里已有对象

# ---------- 1) 资产知识对象工厂 ----------
ASSET_FACTORY: dict[str, Callable[[], Any]] = {
    "microwave": lambda: mircrowave(),
    "dispenser": lambda: dispenser(),
    "table": lambda: table(),
    "bottle": lambda: bottle(),
    "storagefurniture": lambda: storagefurniture(),
    "laptop" : lambda: laptop(),
    "window" : lambda: window(),
    "window_concept" : lambda: window_concept(),
    "hammer" : lambda: hammer(),
    "block" : lambda: block(),
    "bowl" : lambda: bowl(),
    "coaster" : lambda: coaster(),
    "bell" : lambda: bell(),
    "shoe" : lambda: shoe(),
    "roller" : lambda: roller(),
    "can" : lambda: can(),
    "" : lambda: bell(),
    "microphone" : lambda: microphone(),
    "switch" : lambda: switch(),
    "tray" : lambda: tray(),
    "alarmclock" : lambda: alarmclock(),
    "block_green" : lambda: block_green(),
    "block_red" : lambda: block_red(),
    "block_blue" : lambda: block_blue(),
    "block_small" : lambda: block_small(),
    "block_middle" : lambda: block_middle(),
    "block_big" : lambda: block_big(),
    "stapler" : lambda: stapler(),
    "fan" : lambda: fan(),
    "seal" : lambda: seal(),
    "cabinet" : lambda: cabinet(),
    "door" : lambda: door(),
    "remote" : lambda: remote(),
    "displaystand" : lambda: displaystand(),
    "dustbin" : lambda: dustbin(),
    "tabletrashbin" : lambda: tabletrashbin(),
    "roller" : lambda: roller(),
    "microphone" : lambda: microphone(),
    "mug" : lambda: mug(),
    "rack" : lambda: rack(),
    "can" : lambda: can(),
    "kitchenpot" : lambda: kitchenpot(),
    "playingcards" : lambda: playingcards(),
    "mouse" : lambda: mouse(),
    "bread" : lambda: bread(),
    "toycar" : lambda: toycar(),
    "breadbasket" : lambda: breadbasket(),
    "basket" : lambda: basket(),
    "plasticbox" : lambda: plasticbox(),
    "microwave_concept" : lambda: microwave_concept(),
    "plate" : lambda: plate(),
    "shoebox" : lambda: shoebox(),
    "cup" : lambda: cup(),
    "frenchfries" : lambda: frenchfries(),
    "hamburg" : lambda: hamburg(),
    "tray" : lambda: tray(),
    "skillet" : lambda: skillet(),
    "electronicscale" : lambda: electronicscale(),
    "rubikscube" : lambda: rubikscube(),
    "phonestand" : lambda: phonestand(),
    "phone" : lambda: phone(),
    "paymentsign" : lambda: paymentsign(),
    "scanner" : lambda: scanner(),
    "teabox" : lambda: teabox(),
    "seal" : lambda: seal(),
    "pillbottle" : lambda: pillbottle(),
    "laptop_concept" : lambda: laptop_concept(),
    "kitchenpot_concept" : lambda: kitchenpot_concept(),
    "switch_concept" : lambda: switch_concept(),
    "cabinet_concept" : lambda: cabinet_concept(),
}

ASSET_FACTORY.update(ROBOGEN_FACTORY)
# ---------- 2) 每类资产允许的 atomic ----------
ALLOWED_ATOMS: dict[str, set[str]] = {
    "microwave": {"open", "right", "on", "into"},
    "dispenser": {"put"},
    "table": {},
    "bottle":{"adjust","adjust_simultaneously_left","adjust_simultaneously_right","pick","place","pick_with_left_arm","shake","shake_horizontally","hand_over_left2right","hand_over_right2left","pick_simultaneously_left","pick_simultaneously_right"},
    "storagefurniture":{"open"},
    "laptop":{"open"},
    "window":{"open"},
    "window_concept":{},
    "hammer":{"pick"},
    "block":{"beat","hand_over","stack_top","stack_middle","stack_bottom","pick","on","hand_over_left2right"},
    "bowl":{"pick","stack_bottom","stack_middle","stack_top","stack_bottom_simultaneously_right","stack_middle_simultaneously_right"},
    "coaster":{"on"},
    "bell":{"click"},
    "alarmclock":{"click"},
    "block_green":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_green"},
    "block_red":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_red"},
    "block_blue":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_blue"},
    "block_small":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_small"},
    "block_middle":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_middle"},
    "block_big":{"beat","hand_over","stack_top","stack_middle","stack_bottom","place_big"},
    "shoe":{"place","pick_simultaneously_left","pick_simultaneously_right"},
    "fan":{"pick"},
    "seal":{"place"},
    "cabinet":{"open","into"},
    "door":{"close"},
    "remote":{"pick"},
    "displaystand":{"on"},
    "dustbin":{"into","into_2"},
    "tabletrashbin":{"pick"},
    "roller":{"grap_simulataneously"},
    "microphone":{"hand_over"},
    "mug":{"pick","place"},
    "rack":{"hang"},
    "can":{"pick","pick_simultaneously_left","pick_simultaneously_right"},
    "kitchenpot":{"left","pick_simultaneously_left","pick_simultaneously_right"},
    "playingcards":{"place_away"},
    "mouse":{"pick","left"},
    "bread":{"pick","left","right","pick_simultaneously_right"},
    "stapler":{"pick","right","press"},
    "toycar":{"pick","right"},
    "breadbasket":{"into"},
    "basket":{"into_1","pick","into_2"},
    "plasticbox":{"into_1","into_2"},
    "microwave_concept":{},
    "plate":{"on"},
    "shoebox":{"into_1","into_2"},
    "cup":{"pick",},
    "frenchfries":{"pick","pick_simultaneously_left","pick_simultaneously_right"},
    "hamburg":{"pick","pick_simultaneously_left","pick_simultaneously_right"},
    "tray":{"into_1","into_2"},
    "skillet":{"pick_simultaneously_left","into"},
    "electronicscale":{"on"},
    "rubikscube":{"pick"},
    "phonestand":{"on"},
    "phone":{"pick"},
    "paymentsign":{"pick","rotate"},
    "scanner":{"pick_simultaneously_right","scan"},
    "teabox":{"pick_simultaneously_left","scan"},
    "seal":{"pick"},
    "switch":{"click"},
    "pillbottle":{"pick"},
    "laptop_concept":{},
    "pot_concept":{},
    "switch_concept":{},
    "cabinet_concept":{},
}

ALLOWED_ATOMS.update(ROBOGEN_ATOMS)
# ---------- 3) atomic 调度表（按 type + atom） ----------
DISPATCH: dict[tuple[str, str], Callable[[Any, Any, dict], Any]] = {
    ("microwave", "open"): lambda obj, sc, p: obj.open(
        scene=sc, env_id=p.get("_env_id", 0),is_clockwise=p.get("is_clockwise", True)
    ),
    ("microwave", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("microwave", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("microwave", "into"): lambda obj, sc, p: obj.into(scene=sc,env_id=p.get("_env_id", 0)),
    ("dispenser", "put"): lambda obj, sc, p: obj.put(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "adjust"): lambda obj, sc, p: obj.adjust(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "adjust_simultaneously_left"): lambda obj, sc, p: obj.adjust(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "adjust_simultaneously_right"): lambda obj, sc, p: obj.adjust(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "pick_with_left_arm"): lambda obj, sc, p: obj.pick_with_left_arm(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "shake"): lambda obj, sc, p: obj.shake(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "shake_horizontally"): lambda obj, sc, p: obj.shake_horizontally(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "place"): lambda obj, sc, p: obj.place(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "hand_over_right2left"): lambda obj, sc, p: obj.hand_over_right2left(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "hand_over_left2right"): lambda obj, sc, p: obj.hand_over_left2right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("bottle", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("storagefurniture", "open"): lambda obj, sc, p: obj.open(scene=sc,env_id=p.get("_env_id", 0)),
    ("laptop", "open"): lambda obj, sc, p: obj.open(scene=sc,env_id=p.get("_env_id", 0),is_clockwise=p.get("is_clockwise", True)),
    ("window", "open"): lambda obj, sc, p: obj.open(scene=sc, env_id=p.get("_env_id", 0), fraction=p.get("fraction", 0.5)),
    ("hammer", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "beat"): lambda obj, sc, p: obj.beat(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "hand_over"): lambda obj, sc, p: obj.hand_over(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "stack_top"): lambda obj, sc, p: obj.stack_top(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "stack_middle"): lambda obj, sc, p: obj.stack_middle(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "stack_bottom"): lambda obj, sc, p: obj.stack_bottom(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("block", "hand_over_left2right"): lambda obj, sc, p: obj.hand_over_left2right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "stack_bottom"): lambda obj, sc, p: obj.stack_bottom(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "stack_bottom_simultaneously_right"): lambda obj, sc, p: obj.stack_bottom_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "stack_middle"): lambda obj, sc, p: obj.stack_middle(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "stack_middle_simultaneously_right"): lambda obj, sc, p: obj.stack_middle_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bowl", "stack_top"): lambda obj, sc, p: obj.stack_top(scene=sc,env_id=p.get("_env_id", 0)),
    ("coaster", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("bell", "click"): lambda obj, sc, p: obj.click(scene=sc,env_id=p.get("_env_id", 0)),
    ("alarmclock", "click"): lambda obj, sc, p: obj.click(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_green", "place_green"): lambda obj, sc, p: obj.place_green(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_red", "place_red"): lambda obj, sc, p: obj.place_red(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_blue", "place_blue"): lambda obj, sc, p: obj.place_blue(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_small", "place_small"): lambda obj, sc, p: obj.place_small(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_middle", "place_middle"): lambda obj, sc, p: obj.place_middle(scene=sc,env_id=p.get("_env_id", 0)),
    ("block_big", "place_big"): lambda obj, sc, p: obj.place_big(scene=sc,env_id=p.get("_env_id", 0)),
    ("shoe", "place"): lambda obj, sc, p: obj.place(scene=sc,env_id=p.get("_env_id", 0)),
    ("shoe", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("shoe", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("fan", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("seal", "place"): lambda obj, sc, p: obj.place(scene=sc,env_id=p.get("_env_id", 0)),
    ("cabinet", "open"): lambda obj, sc, p: obj.open(scene=sc,env_id=p.get("_env_id", 0)),
    ("door", "close"): lambda obj, sc, p: obj.close(scene=sc,env_id=p.get("_env_id", 0)),
    ("cabinet", "into"): lambda obj, sc, p: obj.into(scene=sc,env_id=p.get("_env_id", 0)),
    ("remote", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("displaystand", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("dustbin", "into"): lambda obj, sc, p: obj.into(scene=sc,env_id=p.get("_env_id", 0)),
    ("dustbin", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("tabletrashbin", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("roller", "grap_simulataneously"): lambda obj, sc, p: obj.grap_simulataneously(scene=sc,env_id=p.get("_env_id", 0)),
    ("microphone", "hand_over"): lambda obj, sc, p: obj.hand_over(scene=sc,env_id=p.get("_env_id", 0)),
    ("rack", "hang"): lambda obj, sc, p: obj.hang(scene=sc,env_id=p.get("_env_id", 0)),
    ("mug", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("mug", "place"): lambda obj, sc, p: obj.place(scene=sc,env_id=p.get("_env_id", 0)),
    ("can", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("can", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("can", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("kitchenpot", "left"): lambda obj, sc, p: obj.left(scene=sc,env_id=p.get("_env_id", 0)),
    ("kitchenpot", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("kitchenpot", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("playingcards", "place_away"): lambda obj, sc, p: obj.place_away(scene=sc,env_id=p.get("_env_id", 0)),
    ("mouse", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("mouse", "left"): lambda obj, sc, p: obj.left(scene=sc,env_id=p.get("_env_id", 0)),
    ("mouse", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "left"): lambda obj, sc, p: obj.left(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "press"): lambda obj, sc, p: obj.press(scene=sc,env_id=p.get("_env_id", 0)),
    ("toycar", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("toycar", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("breadbasket", "into_1"): lambda obj, sc, p: obj.into_1(scene=sc,env_id=p.get("_env_id", 0)),
    ("breadbasket", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("basket", "into_1"): lambda obj, sc, p: obj.into_1(scene=sc,env_id=p.get("_env_id", 0)),
    ("basket", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("basket", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("plasticbox", "into_1"): lambda obj, sc, p: obj.into_1(scene=sc,env_id=p.get("_env_id", 0)),
    ("plasticbox", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("plate", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("shoebox", "into_1"): lambda obj, sc, p: obj.into_1(scene=sc,env_id=p.get("_env_id", 0)),
    ("shoebox", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("cup", "pick"): lambda obj, sc, p: obj.pick(scene=sc, env_id=p.get("_env_id", 0)),
    ("frenchfries", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("frenchfries", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("frenchfries", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("hamburg", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("hamburg", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("hamburg", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("tray", "into_1"): lambda obj, sc, p: obj.into_1(scene=sc,env_id=p.get("_env_id", 0)),
    ("tray", "into_2"): lambda obj, sc, p: obj.into_2(scene=sc,env_id=p.get("_env_id", 0)),
    ("skillet", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("skillet", "into"): lambda obj, sc, p: obj.into(scene=sc,env_id=p.get("_env_id", 0)),
    ("electronicscale", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("rubikscube", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("phonestand", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("phone", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("paymentsign", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("paymentsign", "rotate"): lambda obj, sc, p: obj.rotate(scene=sc,env_id=p.get("_env_id", 0)),
    ("scanner", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("scanner", "scan"): lambda obj, sc, p: obj.scan(scene=sc,env_id=p.get("_env_id", 0)),
    ("teabox", "pick_simultaneously_left"): lambda obj, sc, p: obj.pick_simultaneously_left(scene=sc,env_id=p.get("_env_id", 0)),
    ("teabox", "scan"): lambda obj, sc, p: obj.scan(scene=sc,env_id=p.get("_env_id", 0)),
    ("seal", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("switch", "click"): lambda obj, sc, p: obj.click(scene=sc,env_id=p.get("_env_id", 0)),
    ("pillbottle", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
}

DISPATCH.update(ROBOGEN_DISPATCH)

CUBOID_TASK_TYPES: set[str] = {
    "block",
    "block_red",
    "block_green",
    "block_blue",
    "block_small",
    "block_middle",
    "block_big",
}


@dataclass
class TaskContext:
    """Scene + high-level plan. Waypoints are compiled from import-time poses."""
    scene: Any
    knowledge_map: dict[str, Any]
    id_to_type: dict[str, str]
    atomic_plan: list[dict]
    task_name: str
    task_json: dict
    import_pose_lookup: dict = None
    apply_visual_textures: Any = None


def compile_atomic_step(
    step: dict,
    knowledge_map: dict[str, Any],
    id_to_type: dict[str, str],
    scene,
    env_id: int = 0,
    import_pose_lookup: dict | None = None,
):
    """Compile one atomic step against the asset's import-time pose."""
    if "asset" not in step or "atom" not in step:
        raise ValueError(f"atomic step missing required fields: {step}")
    asset_id = step["asset"]
    atom = step["atom"]
    params = dict(step.get("params", {}) or {})
    params.setdefault("_env_id", env_id)
    if asset_id not in knowledge_map:
        raise ValueError(f"Unknown asset id in atomic_plan: {asset_id}")
    a_type = id_to_type[asset_id]
    if atom not in ALLOWED_ATOMS.get(a_type, set()):
        raise ValueError(f"Atom '{atom}' is not allowed for asset type '{a_type}'")
    key = (a_type, atom)
    if key not in DISPATCH:
        raise ValueError(f"No dispatcher for {key}")
    obj = knowledge_map[asset_id]
    planning_scene = wrap_scene_with_import_poses(scene, import_pose_lookup, knowledge_map=knowledge_map)
    return DISPATCH[key](obj, planning_scene, params)


def build_task_context(
    task_json_path: str,
    sim,
    base_scene_cfg,
    workspace_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    post_spawn_fn=None,
) -> TaskContext:
    """
    读取任务 JSON -> 创建知识对象 -> 导入资产 -> 构建场景。
    不在这里展开 pose_pack：轨迹按原子任务、按当前世界状态即时编译。
    """
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)
    if "assets" not in task or "atomic_plan" not in task:
        raise ValueError("task json must contain 'assets' and 'atomic_plan'")
    # asset_id -> 知识对象
    knowledge_map: dict[str, Any] = {}
    # asset_id -> asset_type
    id_to_type: dict[str, str] = {}
    assets_cfg = []
    num_envs = base_scene_cfg.num_envs
    assets_cfg_2d = [[] for _ in range(num_envs)]
    source_urdf_by_base: dict[str, str | None] = {}
    concept_paired_bases = {
        (a["type"][:-8] if str(a.get("type", "")).endswith("_concept") else None)
        for a in task["assets"]
    }
    concept_paired_bases.discard(None)

    def _asset_base_type(type_name: str) -> str:
        return type_name[:-8] if type_name.endswith("_concept") else type_name

    # ---------- A) 按 JSON 动态创建知识对象并导入资产 ----------
    for asset in task["assets"]:
        if "id" not in asset or "type" not in asset:
            raise ValueError(f"asset item missing required fields: {asset}")
        asset_id = asset["id"]
        a_type = asset["type"]


        # #这里应该用一个函数随机得到一个类别的URDF文件
        # urdf_path = asset["urdf_path"]


        # Cuboid-like assets are generated procedurally and do not need URDF sampling.
        if a_type in CUBOID_TASK_TYPES:
            urdf_path = None
        else:
            base_type = _asset_base_type(a_type)
            if base_type not in source_urdf_by_base:
                pick_item = asset
                if a_type.endswith("_concept"):
                    pick_item = next(
                        (a for a in task["assets"] if a.get("type") == base_type),
                        {"type": base_type},
                    )
                source_urdf_by_base[base_type] = pick_random_urdf(
                    pick_item,
                    require_concept=base_type in concept_paired_bases,
                )
            urdf_path = source_urdf_by_base[base_type]
            if a_type.endswith("_concept"):
                print(f"[INFO] concept {a_type} shares PartNet URDF with {base_type}: {urdf_path}")



        if a_type not in ASSET_FACTORY:
            raise ValueError(f"Unknown asset type: {a_type}")
        if asset_id in knowledge_map:
            raise ValueError(f"Duplicate asset id: {asset_id}")
        obj = ASSET_FACTORY[a_type]()
        knowledge_map[asset_id] = obj
        id_to_type[asset_id] = a_type

        scene_name = f"my_{a_type}_{asset_id}"
        # Compatibility: some legacy asset classes still have import2sim(..., instance_name)
        # while newer ones support import2sim(..., instance_name, num_envs).
        try:
            cfgs_n = obj.import2sim(urdf_path, instance_name=scene_name, num_envs=num_envs)
        except TypeError as exc:
            msg = str(exc)
            if "num_envs" not in msg and "unexpected keyword" not in msg:
                raise
            cfgs_n = obj.import2sim(urdf_path, instance_name=scene_name)

        if not isinstance(cfgs_n, (list, tuple)):
            cfgs_n = [cfgs_n]

        if len(cfgs_n) != num_envs:
            raise ValueError(f"{scene_name}: expect {num_envs}, got {len(cfgs_n)}")
        for env_id, cfg in enumerate(cfgs_n):
            assets_cfg_2d[env_id].append(cfg)
        
        # if isinstance(imported, (list, tuple)):
        #     assets_cfg.extend(imported)
        # else:
        #     assets_cfg.append(imported)
    if len(assets_cfg_2d) == 0:
        raise RuntimeError("No AssetCfg imported from task assets.")

    # ---------- B) 构建场景 ----------
    builder = SceneBuilder(
        assets=assets_cfg_2d,
        base_scene_cfg=base_scene_cfg,
        workspace_offset=workspace_offset,
    )
    scene = builder.scene
    if post_spawn_fn is not None:
        post_spawn_fn()
    # 注意：先 reset/update，保证 scene 中 articulation data 可读
    sim.reset()
    builder.apply_visual_textures()
    # sim.reset() recooks PhysX from the USD; strip kitchen/warehouse colliders again.
    if post_spawn_fn is not None:
        post_spawn_fn()
    scene.update(sim.get_physics_dt())
    import_pose_lookup = attach_import_poses(
        knowledge_map, assets_cfg_2d, workspace_offset=workspace_offset
    )
    return TaskContext(
        scene=scene,
        knowledge_map=knowledge_map,
        id_to_type=id_to_type,
        atomic_plan=list(task["atomic_plan"]),
        task_name=task.get("task_name", "unknown_task"),
        task_json=task,
        import_pose_lookup=import_pose_lookup,
        apply_visual_textures=builder.apply_visual_textures,
    )


def compile_pose_pack(ctx: TaskContext, env_id: int = 0) -> list[dict]:
    """Eager flatten of the whole plan. Prefer TaskRuntime for closed-loop execution."""
    pose_pack = []
    for step in ctx.atomic_plan:
        seg = compile_atomic_step(
            step,
            ctx.knowledge_map,
            ctx.id_to_type,
            ctx.scene,
            env_id=env_id,
            import_pose_lookup=ctx.import_pose_lookup,
        )
        if seg is None:
            continue
        if isinstance(seg, list):
            pose_pack.extend(seg)
        else:
            pose_pack.append(seg)
    return pose_pack


def build_scene_and_pose_pack(task_json_path: str, sim, base_scene_cfg):
    """
    读取任务 JSON -> 动态创建知识对象 -> import2sim导入资产 -> 按atomic顺序拼接pose_pack
    返回: (scene, pose_pack, task_name, knowledge_map)
    """
    ctx = build_task_context(task_json_path=task_json_path, sim=sim, base_scene_cfg=base_scene_cfg)
    pose_pack = compile_pose_pack(ctx)
    return ctx.scene, pose_pack, ctx.task_name, ctx.knowledge_map