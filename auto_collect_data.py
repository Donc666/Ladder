import json
import copy
from dataclasses import dataclass, replace
from typing import Any, Callable

# 按你项目实际路径改 import
import kb_path  # noqa: F401
from my_utils import *
from microwave import mircrowave
from dispenser import dispenser
from table import table
from toolbox import toolbox
from screwdriver import screwdriver
from drill import drill
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
from pickable_assets import (
    PICKABLE_ATOMS,
    PICKABLE_DISPATCH,
    PICKABLE_FACTORY,
    assign_pick_candidates,
)

# TableTopSceneCfg / sim 这些用你当前 micwave_scene.py 里已有对象

# ---------- 1) 资产知识对象工厂 ----------
ASSET_FACTORY: dict[str, Callable[[], Any]] = {
    "microwave": lambda: mircrowave(),
    "dispenser": lambda: dispenser(),
    "table": lambda: table(),
    "toolbox": lambda: toolbox(),
    "screwdriver": lambda: screwdriver(),
    "drill": lambda: drill(),
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
ASSET_FACTORY.update(PICKABLE_FACTORY)
# ---------- 2) 每类资产允许的 atomic ----------
ALLOWED_ATOMS: dict[str, set[str]] = {
    "microwave": {"open", "close", "right", "on", "into", "open_release", "into_left"},
    "dispenser": {"put"},
    "table": {},
    "toolbox": set(),
    "screwdriver": set(),
    "drill": set(),
    "bottle":{"adjust","adjust_simultaneously_left","adjust_simultaneously_right","pick","place","pick_with_left_arm","shake","shake_horizontally","hand_over_left2right","hand_over_right2left","pick_simultaneously_left","pick_simultaneously_right"},
    "storagefurniture":{"open"},
    "laptop":{"open"},
    "window":{"open", "close"},
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
    "cabinet":{"open","into","close"},
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
    "bread":{"pick","place","left","right","pick_simultaneously_right"},
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
ALLOWED_ATOMS.update(PICKABLE_ATOMS)
# ---------- 3) atomic 调度表（按 type + atom） ----------
DISPATCH: dict[tuple[str, str], Callable[[Any, Any, dict], Any]] = {
    ("microwave", "open"): lambda obj, sc, p: obj.open(
        scene=sc, env_id=p.get("_env_id", 0),is_clockwise=p.get("is_clockwise", True)
    ),
    ("microwave", "close"): lambda obj, sc, p: obj.close(
        scene=sc, env_id=p.get("_env_id", 0), is_clockwise=p.get("is_clockwise", True)
    ),
    ("microwave", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("microwave", "on"): lambda obj, sc, p: obj.on(scene=sc,env_id=p.get("_env_id", 0)),
    ("microwave", "into"): lambda obj, sc, p: obj.into(scene=sc,env_id=p.get("_env_id", 0)),
    ("microwave", "open_release"): lambda obj, sc, p: obj.open_release(
        scene=sc, env_id=p.get("_env_id", 0), is_clockwise=p.get("is_clockwise", True)
    ),
    ("microwave", "into_left"): lambda obj, sc, p: obj.into_left(scene=sc, env_id=p.get("_env_id", 0)),
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
    ("window", "close"): lambda obj, sc, p: obj.close(scene=sc, env_id=p.get("_env_id", 0), fraction=p.get("fraction", 0.5)),
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
    ("cabinet", "close"): lambda obj, sc, p: obj.close(scene=sc,env_id=p.get("_env_id", 0)),
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
    ("bread", "place"): lambda obj, sc, p: obj.place(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "left"): lambda obj, sc, p: obj.left(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("bread", "pick_simultaneously_right"): lambda obj, sc, p: obj.pick_simultaneously_right(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("stapler", "press"): lambda obj, sc, p: obj.press(scene=sc,env_id=p.get("_env_id", 0)),
    ("toycar", "pick"): lambda obj, sc, p: obj.pick(scene=sc,env_id=p.get("_env_id", 0)),
    ("toycar", "right"): lambda obj, sc, p: obj.right(scene=sc,env_id=p.get("_env_id", 0)),
    ("breadbasket", "into"): lambda obj, sc, p: obj.into(scene=sc, env_id=p.get("_env_id", 0)),
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
DISPATCH.update(PICKABLE_DISPATCH)

USD_TASK_TYPES: set[str] = {
    "toolbox",
}

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


def _align_assets_to_env_origins(scene, assets_cfg_2d, workspace_offset) -> None:
    """Put each task asset on its own env, next to that env's robots.

    Robot bases follow the env xform. A one-env rigid body can keep the
    unshifted import position, which leaves its table between the two stations.
    """
    origins = getattr(scene, "env_origins", None)
    if origins is None or int(origins.shape[0]) < 2:
        return
    import torch

    ox, oy, oz = workspace_offset
    moved = 0
    for env_id, env_assets in enumerate(assets_cfg_2d):
        origin = origins[env_id].detach()
        for cfg in env_assets:
            key = f"{cfg.name}__env{env_id}"
            try:
                asset = scene[key]
            except KeyError:
                continue
            data = asset.data
            if not hasattr(data, "root_pos_w"):
                continue
            device = data.root_pos_w.device
            n = int(data.root_pos_w.shape[0])
            pose = torch.zeros((n, 7), device=device, dtype=data.root_pos_w.dtype)
            pose[:, 0] = float(cfg.pos[0]) + ox + float(origin[0])
            pose[:, 1] = float(cfg.pos[1]) + oy + float(origin[1])
            pose[:, 2] = float(cfg.pos[2]) + oz + float(origin[2])
            pose[:, 3:7] = torch.tensor(list(cfg.rot), device=device, dtype=pose.dtype)
            asset.write_root_pose_to_sim(pose)
            if hasattr(asset, "write_root_velocity_to_sim"):
                asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=device, dtype=pose.dtype))
            moved += 1
    if moved:
        scene.write_data_to_sim()
        print(f"[INFO] seated {moved} task assets onto their env origins")


def _add_wooden_box_lifts(task: dict, object_ids: list[str]) -> dict:
    """One flipped wooden box under each object, same rule as scan_object.

    The y scale is the height and is drawn from 0 to 0.1. The box is not fixed.
    The object's root is raised by that height once its spawn pose is known, so
    waypoints that read the root rise by the same amount.
    """
    present = {a.get("id") for a in task.get("assets", []) if isinstance(a, dict)}
    lifts = {}
    n = 1
    for aid in object_ids:
        if aid not in present:
            print(f"[WARN] {task.get('task_name')} has no asset {aid}; skip wooden box")
            continue
        while f"woodenbox_{n}" in present:
            n += 1
        box_id = f"woodenbox_{n}"
        n += 1
        y_scale = random.uniform(0.0, 0.1)
        box = {
            "id": box_id,
            "type": "woodenbox",
            "asset_kind": "auto",
            "flip_upside_down": True,
            "scale": (0.05, y_scale, 0.05),
        }
        task["assets"].append(box)
        present.add(box_id)
        lifts[aid] = {"box": box, "height": y_scale}
        print(f"[INFO] {aid} will sit on {box_id} scale_y={y_scale:.3f}")
    return lifts


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
    assign_pick_candidates(task)
    if task.get("task_name") == "pick_bottle_and_shake_horizontally":
        # This task only. bottle_1 is the one that is picked and shaken.
        _bands = {"bottle_1": (-0.65, -0.1), "bottle_2": (0.1, 0.65)}
        for asset in task["assets"]:
            band = _bands.get(asset.get("id"))
            if band is None:
                continue
            asset["pos"] = [
                random.uniform(0.7, 0.85),
                random.uniform(*band),
                0.32,
            ]
            print(f"[INFO] {asset['id']} pos: {asset['pos']}")
    if task.get("task_name") == "stack_bowls_and_stack_blocks":
        # This task only. Bowls stack on the right, blocks on the left.
        bowl_place = (0.5, -0.15)
        block_place = (0.5, 0.15)
        block_colors = (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
        )

        def _away(target, radius, xr, yr, taken, gap=0.15):
            tx, ty = target
            for _ in range(80):
                x = random.uniform(*xr)
                y = random.uniform(*yr)
                if (x - tx) ** 2 + (y - ty) ** 2 <= radius * radius:
                    continue
                if any((x - ox) ** 2 + (y - oy) ** 2 < gap * gap for ox, oy in taken):
                    continue
                return x, y
            raise RuntimeError(f"no spawn farther than {radius} m from {target}")

        taken_bowls = []
        taken_blocks = []
        for asset in task["assets"]:
            aid = asset.get("id")
            if aid in ("bowl_1", "bowl_2"):
                x, y = _away(bowl_place, 0.2, (0.40, 0.80), (-0.65, -0.02), taken_bowls)
                taken_bowls.append((x, y))
                asset["pos"] = [x, y, 0.35]
                asset["stack_place"] = [bowl_place[0], bowl_place[1]]
            elif aid in ("block_1", "block_2"):
                x, y = _away(block_place, 0.15, (0.40, 0.80), (0.02, 0.65), taken_blocks)
                taken_blocks.append((x, y))
                asset["pos"] = [x, y, 0.40]
                asset["stack_place"] = [block_place[0], block_place[1]]
                asset["cuboid_color"] = random.choice(block_colors)
            else:
                continue
            print(f"[INFO] {aid} pos: {asset['pos']} stack: {asset.get('stack_place')}")
        # Two of the four objects sit on a flipped wooden box, same rule as scan_object.
        # Grasp waypoints read the raised root z. The stack destination keeps the table height.
        lifted = random.sample(["bowl_1", "bowl_2", "block_1", "block_2"], 2)
        by_id = {a.get("id"): a for a in task["assets"]}
        for box_i, aid in enumerate(lifted, start=1):
            obj_asset = by_id[aid]
            x, y, z0 = (float(v) for v in obj_asset["pos"])
            box_y_scale = random.uniform(0.0, 0.1)
            height = box_y_scale * 1.0
            # Blocks are centered cuboids, 0.05 m on a side. Bowls rest on their root.
            half = 0.025 if str(aid).startswith("block") else 0.0
            obj_asset["pos"] = [x, y, z0 + height]
            obj_asset["stack_table_z"] = z0
            box_asset = by_id[f"woodenbox_{box_i}"]
            box_asset["pos"] = [x, y, z0 - half + height]
            box_asset["scale"] = (0.05, box_y_scale, 0.05)
            box_asset["flip_upside_down"] = True
            print(
                f"[INFO] {aid} on woodenbox_{box_i} "
                f"scale_y={box_y_scale:.3f} object_z={z0 + height:.3f}"
            )
    if task.get("task_name") == "place_phone_stand_and_place_object_scale":
        # The stapler slot is the random object. This task only.
        for asset in task["assets"]:
            if asset.get("id") != "stapler_1":
                continue
            asset["pos"] = [
                random.uniform(0.3, 0.7),
                random.uniform(0.15, 0.6),
                0.35,
            ]
            print(f"[INFO] {asset['id']} pos: {asset['pos']}")
    if task.get("task_name") == "scan_object":
        # This task only. The wooden box stands under teabox_1, flipped upside down.
        # Mesh up-length is 1 along local Y. Scale is applied before the x=90°
        # rotation, so the y component is the world height. An extra 180° about
        # local X turns that axis downward, and the root is raised by the height
        # so the flipped box still rests on the table. The box is not fixed.
        box_y_scale = random.uniform(0.0, 0.1)
        box_scale = (0.05, box_y_scale, 0.05)
        table_contact_z = 0.35
        box_height = box_scale[1] * 1.0
        object_z = table_contact_z + box_height
        x = random.uniform(0.40, 0.60)
        y = random.uniform(0.30, 0.60)
        for asset in task["assets"]:
            aid = asset.get("id")
            if aid == "woodenbox_1":
                asset["pos"] = [x, y, object_z]
                asset["scale"] = box_scale
                asset["flip_upside_down"] = True
            elif aid == "teabox_1":
                asset["pos"] = [x, y, object_z]
            else:
                continue
            print(f"[INFO] {aid} pos: {asset['pos']}")
        print(
            f"[INFO] scan_object wooden box scale={box_scale} "
            f"height={box_height:.3f} object_z={object_z:.3f}"
        )
    if task.get("task_name") == "open_cabinet_and_put_object_in":
        # This task only. The rubikscube_1 slot is the random object.
        for asset in task["assets"]:
            if asset.get("id") != "rubikscube_1":
                continue
            asset["pos"] = [
                random.uniform(0.4, 0.5),
                random.uniform(-0.65, -0.5),
                0.35,
            ]
            print(f"[INFO] {asset['id']} pos: {asset['pos']}")
    if task.get("task_name") == "place_shoe_in_box_and_place_cans_in_box":
        # Shoes trade places with cans. The two boxes stay swapped.
        def _xyz(xr, yr, dx=0.0):
            return [random.uniform(*xr) + dx, random.uniform(*yr), 0.35]

        swapped = {
            "shoe_1": _xyz((0.415, 0.41), (0.43, 0.435)),
            "shoe_2": _xyz((0.40, 0.42), (-0.16, -0.17)),
            "can_1": _xyz((0.40, 0.7), (0.32, 0.72)),
            "can_2": _xyz((0.40, 0.7), (-0.32, -0.72)),
            "shoebox_1": _xyz((0.45, 0.47), (0.1, 0.15), dx=-0.07),
            "plasticbox_1": _xyz((0.40, 0.55), (0.0, 0.1), dx=0.15),
        }
        for asset in task["assets"]:
            pos = swapped.get(asset.get("id"))
            if pos is None:
                continue
            asset["pos"] = pos
            asset.pop("pos_offset", None)
            print(f"[INFO] {asset['id']} swapped pos: {asset['pos']}")
    if task.get("task_name") == "click_bell_and_rotate_qrcode":
        # This task only. Both objects share one region and stay more than 0.15 m apart.
        def _separated(gap=0.15):
            xr, yr = (0.4, 0.7), (-0.6, 0.6)
            for _ in range(80):
                first = (random.uniform(*xr), random.uniform(*yr))
                second = (random.uniform(*xr), random.uniform(*yr))
                dx = first[0] - second[0]
                dy = first[1] - second[1]
                if dx * dx + dy * dy > gap * gap:
                    return first, second
            raise RuntimeError("no bell/qrcode pair farther than 0.15 m")

        bell_xy, qr_xy = _separated()
        for asset in task["assets"]:
            if asset.get("id") == "bell_1":
                asset["pos"] = [bell_xy[0], bell_xy[1], 0.35]
            elif asset.get("id") == "paymentsign_1":
                asset["pos"] = [qr_xy[0], qr_xy[1], 0.35]
            else:
                continue
            print(f"[INFO] {asset['id']} pos: {asset['pos']}")
        # y >= 0 is the left arm, matching bell.click and paymentsign.pick/rotate.
        bell_arm = "robot_left" if bell_xy[1] >= 0 else "robot_right"
        qr_arm = "robot_left" if qr_xy[1] >= 0 else "robot_right"
        same_arm = bell_arm == qr_arm
        for step in task.get("atomic_plan", []):
            if not isinstance(step, dict):
                continue
            if same_arm:
                step.pop("parallel", None)
            else:
                step["parallel"] = "arms"
        print(
            f"[INFO] click_bell_and_rotate_qrcode bell={bell_arm} qrcode={qr_arm} "
            f"{'sequential' if same_arm else 'simultaneous'}"
        )
    # Flipped wooden boxes, same placement as scan_object. Appended last so the
    # object's final spawn pose is known before the box is imported.
    wooden_lifts: dict = {}
    _task_name = task.get("task_name")
    if _task_name == "pick_bottle_and_shake_horizontally":
        wooden_lifts = _add_wooden_box_lifts(
            task, [random.choice(["bottle_1", "bottle_2"])]
        )
    elif _task_name == "open_cabinet_and_put_object_in":
        wooden_lifts = _add_wooden_box_lifts(task, ["rubikscube_1"])
    elif _task_name == "place_object_in_basket_and_pick":
        wooden_lifts = _add_wooden_box_lifts(
            task,
            random.sample(["toycar_1", "rubikscube_1", "cup_1", "mug_1"], 2),
        )
    elif _task_name == "place_bread_skillet":
        wooden_lifts = _add_wooden_box_lifts(task, ["bread_1"])
    elif _task_name == "place_shoe_in_box_and_place_cans_in_box":
        wooden_lifts = _add_wooden_box_lifts(
            task,
            random.sample(["shoe_1", "shoe_2", "can_1", "can_2"], 2),
        )
    elif _task_name == "place_a2b_right_and_grap_roller":
        wooden_lifts = _add_wooden_box_lifts(task, ["rubikscube_1", "roller_1"])
    elif _task_name == "click_bell_and_rotate_qrcode":
        wooden_lifts = _add_wooden_box_lifts(task, ["bell_1", "paymentsign_1"])
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

    # A close atom keeps the half-open spec for planning. If the plan also opens
    # that asset, the real door still spawns closed.
    preopen_bases: set[str] = set()
    opened_bases: set[str] = set()
    asset_type_by_id = {a.get("id"): a.get("type", "") for a in task["assets"] if isinstance(a, dict)}
    for step in task["atomic_plan"]:
        if not isinstance(step, dict):
            continue
        step_type = asset_type_by_id.get(step.get("asset"), "")
        if not step_type:
            continue
        atom = str(step.get("atom") or "")
        base = _asset_base_type(step_type)
        if atom == "close":
            preopen_bases.add(base)
        elif atom in ("open", "open_release"):
            opened_bases.add(base)

    # ---------- A) 按 JSON 动态创建知识对象并导入资产 ----------
    for asset in task["assets"]:
        if "id" not in asset or "type" not in asset:
            raise ValueError(f"asset item missing required fields: {asset}")
        asset_id = asset["id"]
        a_type = asset["type"]


        # #这里应该用一个函数随机得到一个类别的URDF文件
        # urdf_path = asset["urdf_path"]


        # Cuboid-like assets are generated procedurally and do not need URDF sampling.
        if a_type in CUBOID_TASK_TYPES or a_type in USD_TASK_TYPES:
            urdf_path = None
        elif isinstance(asset.get("urdf_path"), str) and asset["urdf_path"].strip():
            pinned = Path(asset["urdf_path"].strip())
            if not pinned.is_absolute():
                pinned = PROJECT_ROOT / pinned
            if not pinned.is_file():
                raise FileNotFoundError(f"{asset_id} URDF not found: {pinned}")
            urdf_path = str(pinned)
            print(f"[INFO] {asset_id} fixed URDF -> {urdf_path}")
        else:
            base_type = _asset_base_type(a_type)
            resample = bool(asset.get("resample_urdf"))
            # A later bottle, shoe, can, or bowl draws again instead of copying the first model.
            if base_type in ("bottle", "shoe", "can", "bowl") and base_type in source_urdf_by_base:
                resample = True
            if base_type not in source_urdf_by_base or resample:
                pick_item = asset
                if a_type.endswith("_concept"):
                    pick_item = next(
                        (a for a in task["assets"] if a.get("type") == base_type),
                        {"type": base_type},
                    )
                chosen = pick_random_urdf(
                    pick_item,
                    require_concept=base_type in concept_paired_bases,
                )
                if resample and base_type in source_urdf_by_base:
                    shared = source_urdf_by_base[base_type]
                    for _ in range(8):
                        if chosen != shared:
                            break
                        chosen = pick_random_urdf(
                            pick_item,
                            require_concept=base_type in concept_paired_bases,
                        )
                    print(f"[INFO] resample {asset_id} -> {chosen}")
                else:
                    source_urdf_by_base[base_type] = chosen
                urdf_path = chosen
            else:
                urdf_path = source_urdf_by_base[base_type]
            if a_type.endswith("_concept"):
                print(f"[INFO] concept {a_type} shares PartNet URDF with {base_type}: {urdf_path}")



        if a_type not in ASSET_FACTORY:
            raise ValueError(f"Unknown asset type: {a_type}")
        if asset_id in knowledge_map:
            raise ValueError(f"Duplicate asset id: {asset_id}")
        obj = ASSET_FACTORY[a_type]()
        if task.get("task_name") == "scan_object" and asset.get("id") == "teabox_1":
            obj.left_executor = "simultaneous_left"
            ALLOWED_ATOMS.setdefault(a_type, set()).add("scan")
            DISPATCH[(a_type, "scan")] = lambda obj, sc, p: obj.scan(
                scene=sc, env_id=p.get("_env_id", 0)
            )
        if (
            task.get("task_name") == "open_window_and_handover_microphone"
            and asset.get("id") == "microphone_1"
        ):
            obj.marker_per_waypoint = True
        if task.get("task_name") == "open_cabinet_and_put_object_in" and asset.get("id") in (
            "cabinet_1",
            "rubikscube_1",
        ):
            obj.marker_per_waypoint = True
        if a_type == "window" and "window" in preopen_bases and "window" not in opened_bases:
            frac = 0.5
            for step in task["atomic_plan"]:
                if (
                    isinstance(step, dict)
                    and step.get("asset") == asset_id
                    and step.get("atom") == "close"
                ):
                    frac = float((step.get("params") or {}).get("fraction", 0.5))
                    break
            obj.spawn_open_fraction = frac
        if _asset_base_type(a_type) in opened_bases and not a_type.endswith("_concept"):
            obj.spawn_closed = True
        if a_type == "table":
            if "pos" in asset:
                obj.table_root_pos = tuple(asset["pos"])
            if "scale" in asset:
                obj.scale_micwave = tuple(asset["scale"])
        if a_type in ("toolbox", "screwdriver", "drill"):
            if "pos" in asset:
                obj.spawn_pos = tuple(asset["pos"])
            if "scale" in asset:
                obj.scale = tuple(asset["scale"])
        spec = getattr(obj, "spec", None)
        if spec is not None and float(getattr(spec, "init_open_fraction", 0.0)) > 1e-6:
            if _asset_base_type(a_type) not in preopen_bases:
                obj.spec = replace(spec, init_open_fraction=0.0)
                print(f"[INFO] {asset_id} spawns closed; no close atom in this task")
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
        if (
            task.get("task_name") == "open_window_and_handover_microphone"
            and asset.get("id") == "microphone_1"
        ):
            for cfg in cfgs_n:
                cfg.disable_gravity = False
            print("[INFO] microphone_1 gravity enabled for open_window_and_handover_microphone")
        # One config is the env-local pose. Copy it into every env; the cloner offsets them.
        if len(cfgs_n) == 1 and num_envs > 1:
            cfgs_n = [copy.deepcopy(cfgs_n[0]) for _ in range(num_envs)]
        # Optional task-only spawn. Assets without pos keep their own sampler.
        if a_type != "table" and "pos" in asset:
            fixed_pos = tuple(float(v) for v in asset["pos"])
            for cfg in cfgs_n:
                cfg.pos = fixed_pos
        if asset.get("flip_upside_down"):
            for cfg in cfgs_n:
                cfg.scale = tuple(asset["scale"])
                upright = R.from_quat(wxyz_to_xyzw(cfg.rot))
                flipped = upright * R.from_euler("x", 180, degrees=True)
                cfg.rot = tuple(float(v) for v in xyzw_to_wxyz(flipped.as_quat()))
        color = asset.get("cuboid_color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            rgb = (float(color[0]), float(color[1]), float(color[2]))
            for cfg in cfgs_n:
                cfg.cuboid_color = rgb
        if asset.get("right_executor") in ("robot_left", "robot_right"):
            obj.right_executor = asset["right_executor"]
        offset = asset.get("pos_offset")
        if isinstance(offset, (list, tuple)) and len(offset) >= 3:
            dx, dy, dz = (float(offset[0]), float(offset[1]), float(offset[2]))
            for cfg in cfgs_n:
                px, py, pz = cfg.pos
                cfg.pos = (px + dx, py + dy, pz + dz)
        lift = wooden_lifts.get(asset_id)
        if lift is not None and cfgs_n:
            height = float(lift["height"])
            x, y, z = (float(v) for v in cfgs_n[0].pos)
            for cfg in cfgs_n:
                px, py, pz = (float(v) for v in cfg.pos)
                cfg.pos = (px, py, pz + height)
            # Flipped box root is the top face, so it meets the object root.
            lift["box"]["pos"] = [x, y, z + height]
            print(
                f"[INFO] {asset_id} on {lift['box']['id']} "
                f"scale_y={height:.3f} object_z={z + height:.3f}"
            )
        place = asset.get("stack_place")
        if isinstance(place, (list, tuple)) and len(place) >= 2:
            obj.stack_place = (float(place[0]), float(place[1]))
        if asset.get("stack_table_z") is not None:
            obj.stack_table_z = float(asset["stack_table_z"])
        if task.get("task_name") == "stack_bowls_and_stack_blocks" and a_type in ("bowl", "block"):
            obj.plan_position_tolerance = 0.01
            obj.reach_threshold = 0.03

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
    _align_assets_to_env_origins(scene, assets_cfg_2d, workspace_offset)
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