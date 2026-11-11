#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task_planner_agent.py

功能：
1) 将自然语言任务分解为结构化 JSON
2) 使用本地规则进行强校验
3) 失败自动重试修复
4) 保存为本地 json 文件

依赖：
    pip install openai pydantic

环境变量（建议）：
    export DEEPSEEK_API_KEY="xxx"
    export OPENAI_API_KEY="xxx"
    export KIMI_API_KEY="xxx"   # 可选
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import subprocess

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# ---------------------------
# 1) 你的领域定义（可按工程继续扩展）
# ---------------------------

AssetType = Literal["microwave", "dispenser", "table", "bottle", "storagefurniture","block","door"]
Executor = Literal["left", "right", "both", "auto"]

ALLOWED_ATOMS: dict[str, set[str]] = {
    "microwave": {"open", "close", "right", "on", "into", "open_release", "into_left"},
    "dispenser": {"put"},
    "table": {},
    "bottle": {
        "adjust_simultaneously_right",
        "pick",
        "pick_with_left_arm",
        "simultaneously_pick_with_left_arm",
        "simultaneously_pick_with_right_arm",
    },
    "storagefurniture":{"open"},
    "laptop":{"close"},
    "door":{"close"},
    "hammer":{"pick"},
    "block":{"beat","hand_over","stack_top","stack_middle","stack_bottom"},
    "bowl":{"pick"},
    "coaster":{"on"},
    "bell":{"click"},
    
}


class AssetItem(BaseModel):
    id: str = Field(..., description="唯一对象ID，例如 microwave_1")
    type: AssetType
    asset_kind: str = "auto" 



class AtomicStep(BaseModel):
    asset: str
    atom: str
    params: dict = Field(default_factory=dict)


class TaskPlan(BaseModel):
    task_name: str
    assets: List[AssetItem]
    atomic_plan: List[AtomicStep]


# ---------------------------
# 2) 模型配置
# ---------------------------

@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str

def _require_api_key(env_name: str) -> str:
    key = os.getenv(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"缺少环境变量 {env_name}。请先执行 export {env_name}=\"你的密钥\"，不要把密钥写进代码。"
        )
    return key


def _proxy_url_for_httpx() -> str | None:
    """Pick a proxy URL that httpx can use.

    Local tools often export ALL_PROXY=socks://127.0.0.1:7897. httpx rejects the
    socks:// scheme. A Clash mixed port on that address also accepts HTTP.
    """
    raw = ""
    for key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        raw = os.environ.get(key, "").strip()
        if raw:
            break
    if not raw:
        return None
    lowered = raw.lower()
    if lowered.startswith("socks://") or lowered.startswith("socks5://") or lowered.startswith("socks5h://"):
        hostport = raw.split("://", 1)[1].split("/", 1)[0]
        scheme = "socks5h" if lowered.startswith("socks5h://") else "socks5"
        try:
            import socksio  # noqa: F401
        except ImportError:
            return "http://" + hostport
        return f"{scheme}://{hostport}"
    return raw


def _openai_client(api_key: str, base_url: str) -> OpenAI:
    import httpx

    proxy = _proxy_url_for_httpx()
    if proxy:
        http_client = httpx.Client(proxy=proxy, trust_env=False, timeout=60.0)
    else:
        http_client = httpx.Client(trust_env=False, timeout=60.0)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def get_provider_config(provider: str) -> ProviderConfig:
    provider = provider.lower().strip()

    if provider == "deepseek":
        return ProviderConfig(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=_require_api_key("DEEPSEEK_API_KEY"),
        )

    if provider == "openai":
        return ProviderConfig(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=_require_api_key("OPENAI_API_KEY"),
        )

    if provider == "kimi":
        return ProviderConfig(
            model=os.getenv("KIMI_MODEL", "kimi-k3"),
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=_require_api_key("KIMI_API_KEY"),
        )

    raise ValueError(f"不支持的 provider: {provider}")


# ---------------------------
# 3) Prompt + JSON提取
# ---------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task_planner_agent.py

功能：
1) 将自然语言任务分解为结构化 JSON
2) 使用本地规则进行强校验
3) 失败自动重试修复
4) 保存为本地 json 文件

依赖：
    pip install openai pydantic

环境变量（建议）：
    export DEEPSEEK_API_KEY="xxx"
    export OPENAI_API_KEY="xxx"
    export KIMI_API_KEY="xxx"   # 可选
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import subprocess

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


# ---------------------------
# 1) 你的领域定义（可按工程继续扩展）
# ---------------------------

AssetType = str
Executor = Literal["left", "right", "both", "auto"]

ALLOWED_ATOMS: dict[str, set[str]] = {
    "microwave": {"open", "close", "right", "on", "into", "open_release", "into_left"},
    "dispenser": {"put"},
    "table": set(),
    "toolbox": set(),
    "screwdriver": set(),
    "drill": set(),
    "bottle": {"adjust", "adjust_simultaneously_left", "adjust_simultaneously_right", "pick", "place", "pick_with_left_arm", "shake", "shake_horizontally", "hand_over_left2right", "hand_over_right2left", "pick_simultaneously_left", "pick_simultaneously_right"},
    "storagefurniture": {"open"},
    "laptop": {"open"},
    "window": {"open", "close"},
    "window_concept": set(),
    "hammer": {"pick"},
    "block": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "pick", "on", "hand_over_left2right"},
    "bowl": {"pick", "stack_bottom", "stack_middle", "stack_top", "stack_bottom_simultaneously_right", "stack_middle_simultaneously_right"},
    "coaster": {"on"},
    "bell": {"click"},
    "alarmclock": {"click"},
    "block_green": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_green"},
    "block_red": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_red"},
    "block_blue": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_blue"},
    "block_small": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_small"},
    "block_middle": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_middle"},
    "block_big": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_big"},
    "shoe": {"place", "pick_simultaneously_left", "pick_simultaneously_right"},
    "fan": {"pick"},
    "seal": {"place", "pick"},
    "cabinet": {"open", "into", "close"},
    "door": {"close"},
    "remote": {"pick"},
    "displaystand": {"on"},
    "dustbin": {"into", "into_2"},
    "tabletrashbin": {"pick"},
    "roller": {"grap_simulataneously"},
    "microphone": {"hand_over"},
    "mug": {"pick", "place"},
    "rack": {"hang"},
    "can": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "kitchenpot": {"left", "pick_simultaneously_left", "pick_simultaneously_right"},
    "playingcards": {"place_away"},
    "mouse": {"pick", "left"},
    "bread": {"pick", "place", "left", "right", "pick_simultaneously_right"},
    "stapler": {"pick", "right", "press"},
    "toycar": {"pick", "right"},
    "breadbasket": {"into"},
    "basket": {"into_1", "pick", "into_2"},
    "plasticbox": {"into_1", "into_2"},
    "microwave_concept": set(),
    "plate": {"on"},
    "shoebox": {"into_1", "into_2"},
    "cup": {"pick"},
    "frenchfries": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "hamburg": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "tray": {"into_1", "into_2"},
    "skillet": {"pick_simultaneously_left", "into"},
    "electronicscale": {"on"},
    "rubikscube": {"pick"},
    "phonestand": {"on"},
    "phone": {"pick"},
    "paymentsign": {"pick", "rotate"},
    "scanner": {"pick_simultaneously_right", "scan"},
    "teabox": {"pick_simultaneously_left", "scan"},
    "switch": {"click"},
    "pillbottle": {"pick"},
    "laptop_concept": set(),
    "pot_concept": set(),
    "switch_concept": set(),
    "cabinet_concept": set(),
    "kettle": {"lift", "open"},
    "kettle_concept": set(),
    "toilet": {"flush", "open", "close"},
    "toilet_concept": set(),
    "suitcase": {"extend"},
    "suitcase_concept": set(),
    "printer": {"on"},
    "printer_concept": set(),
    "trashcan": {"open", "close"},
    "trashcan_concept": set(),
    "partnet_switch": {"on"},
    "partnet_switch_concept": set(),
    "storage_drawer": {"open"},
    "storage_drawer_concept": set(),
    "clock": {"advance"},
    "clock_concept": set(),
    "partnet_door": {"open", "close"},
    "partnet_door_concept": set(),
    "partnet_fan": {"rotate"},
    "partnet_fan_concept": set(),
    "coffeemachine": {"on"},
    "coffeemachine_concept": set(),
    "partnet_microwave": {"open"},
    "partnet_microwave_concept": set(),
    "display": {"extend", "tilt"},
    "display_concept": set(),
    "globe": {"rotate"},
    "globe_concept": set(),
    "lamp": {"aim", "on"},
    "lamp_concept": set(),
    "box": {"open", "pull"},
    "box_concept": set(),
    "partnet_pot": {"remove"},
    "partnet_pot_concept": set(),
    "stapler_lid": {"open"},
    "stapler_lid_concept": set(),
    "refrigerator": {"open", "open_other"},
    "refrigerator_concept": set(),
    "washingmachine": {"open"},
    "washingmachine_concept": set(),
    "partnet_window": {"close"},
    "partnet_window_concept": set(),
    "foldingchair": {"unfold"},
    "foldingchair_concept": set(),
    "oven": {"open", "close", "into"},
    "oven_concept": set(),
    "table_drawer": {"open", "close"},
    "table_drawer_concept": set(),
    "dishwasher": {"open", "press"},
    "dishwasher_concept": set(),
    "faucet": {"on", "off"},
    "faucet_concept": set(),
    "partnet_dispenser": {"open", "close"},
    "partnet_dispenser_concept": set(),
    "partnet_laptop": {"open"},
    "partnet_laptop_concept": set(),
}


class AssetItem(BaseModel):
    id: str = Field(..., description="唯一对象ID，例如 microwave_1")
    type: AssetType
    asset_kind: str = "auto" 



class AtomicStep(BaseModel):
    asset: str
    atom: str
    params: dict = Field(default_factory=dict)


class TaskPlan(BaseModel):
    task_name: str
    assets: List[AssetItem]
    atomic_plan: List[AtomicStep]


# ---------------------------
# 2) 模型配置
# ---------------------------

@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str

def get_provider_config(provider: str) -> ProviderConfig:
    provider = provider.lower().strip()

    if provider == "deepseek":
        return ProviderConfig(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=_require_api_key("DEEPSEEK_API_KEY"),
        )

    if provider == "openai":
        return ProviderConfig(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=_require_api_key("OPENAI_API_KEY"),
        )

    if provider == "kimi":
        return ProviderConfig(
            model=os.getenv("KIMI_MODEL", "kimi-k3"),
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=_require_api_key("KIMI_API_KEY"),
        )

    raise ValueError(f"不支持的 provider: {provider}")


# ---------------------------
# 3) Prompt + JSON提取
# ---------------------------

SYSTEM_PROMPT = """You are a robot task planning agent.
Your job is to convert a user's natural-language instruction into a strict JSON task plan.

You must follow these rules and can't ignore any of the rule below:
1) Output JSON only. Do not output markdown. Do not output explanations.
2) The top-level JSON fields must be:
- task_name: string
- assets: array
- atomic_plan: array
3) Every rigid-body task must include exactly one table asset:
- id must be "table_1", type must be "table".
- The table is the support surface already used by rigid tasks. Do not put table_1 in atomic_plan, because table has no atoms.
- This is required even when the instruction does not say "table".
- Articulated-body tasks do not need a table unless the instruction uses one. 

4) Semantic ordering rules:
- If microwave.into appears, the same microwave asset must have microwave.open earlier in the plan.
- If future storagefurniture "put-into" style atoms are introduced, open must come first.

5) Generate the most appropriate executable sequence.
6) Keep output keys and atom names exactly as specified above.
7) If user instruction explicitly requires simultaneous grasp/pick (e.g. contains "simultaneously", "at the same time", "同时"), you MUST use atom names that contain the substring "simultaneously" for those grasp steps. Do not use plain "pick" for those simultaneous steps.
8) The listing order of atoms inside each asset type is NOT a priority ranking.
   You must choose atoms only by semantic match with the instruction.
   Example: for bottle atoms {"adjust_simultaneously_right","pick",...,"pick_simultaneously_left","pick_simultaneously_right"},
   if instruction requires simultaneous execution, NEVER choose "pick"; choose simultaneous atoms.
8.1) Asset extraction is mandatory and must be complete:
- Parse the instruction first and extract all task-required physical assets.
- assets[] must include every object required to finish the task (acted object + target/container/support object).
- A rigid-body task must also include the table asset from rule 3.
- Do not invent irrelevant assets.
8.2) Asset type whitelist is mandatory:
- Every assets[i].type MUST come from the allowed asset type list below.
- If an object in instruction cannot be mapped to an allowed type, do not guess a new type; map it to the closest allowed type name.
8.3) Atom whitelist is mandatory:
- For each atomic_plan step, atom MUST be selected from that asset type's allowed atom set.
- Never use an atom that is not listed for that asset type.
8.4) Internal consistency is mandatory:
- Every atomic_plan.asset must appear in assets[].id.
- Keep atomic_plan minimal but complete (enough steps to accomplish the full instruction).
9) Allowed asset types are exactly the keys in the atom lists below. Do not invent type names.
10) Allowed atoms per asset type. An empty set means the asset may appear in assets[] but must not appear in atomic_plan.
Rigid-body atoms:
    "microwave": {"open", "close", "right", "on", "into", "open_release", "into_left"},
    "dispenser": {"put"},
    "table": {},
    "bottle": {"adjust", "adjust_simultaneously_left", "adjust_simultaneously_right", "pick", "place", "pick_with_left_arm", "shake", "shake_horizontally", "hand_over_left2right", "hand_over_right2left", "pick_simultaneously_left", "pick_simultaneously_right"},
    "storagefurniture": {"open"},
    "laptop": {"open"},
    "window": {"open", "close"},
    "window_concept": {},
    "hammer": {"pick"},
    "block": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "pick", "on", "hand_over_left2right"},
    "bowl": {"pick", "stack_bottom", "stack_middle", "stack_top", "stack_bottom_simultaneously_right", "stack_middle_simultaneously_right"},
    "coaster": {"on"},
    "bell": {"click"},
    "alarmclock": {"click"},
    "block_green": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_green"},
    "block_red": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_red"},
    "block_blue": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_blue"},
    "block_small": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_small"},
    "block_middle": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_middle"},
    "block_big": {"beat", "hand_over", "stack_top", "stack_middle", "stack_bottom", "place_big"},
    "shoe": {"place", "pick_simultaneously_left", "pick_simultaneously_right"},
    "fan": {"pick"},
    "seal": {"place", "pick"},
    "cabinet": {"open", "into", "close"},
    "door": {"close"},
    "remote": {"pick"},
    "displaystand": {"on"},
    "dustbin": {"into", "into_2"},
    "tabletrashbin": {"pick"},
    "roller": {"grap_simulataneously"},
    "microphone": {"hand_over"},
    "mug": {"pick", "place"},
    "rack": {"hang"},
    "can": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "kitchenpot": {"left", "pick_simultaneously_left", "pick_simultaneously_right"},
    "playingcards": {"place_away"},
    "mouse": {"pick", "left"},
    "bread": {"pick", "place", "left", "right", "pick_simultaneously_right"},
    "stapler": {"pick", "right", "press"},
    "toycar": {"pick", "right"},
    "breadbasket": {"into"},
    "basket": {"into_1", "pick", "into_2"},
    "plasticbox": {"into_1", "into_2"},
    "microwave_concept": {},
    "plate": {"on"},
    "shoebox": {"into_1", "into_2"},
    "cup": {"pick"},
    "frenchfries": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "hamburg": {"pick", "pick_simultaneously_left", "pick_simultaneously_right"},
    "tray": {"into_1", "into_2"},
    "skillet": {"pick_simultaneously_left", "into"},
    "electronicscale": {"on"},
    "rubikscube": {"pick"},
    "phonestand": {"on"},
    "phone": {"pick"},
    "paymentsign": {"pick", "rotate"},
    "scanner": {"pick_simultaneously_right", "scan"},
    "teabox": {"pick_simultaneously_left", "scan"},
    "switch": {"click"},
    "pillbottle": {"pick"},
    "laptop_concept": {},
    "pot_concept": {},
    "switch_concept": {},
    "cabinet_concept": {},
Articulated-body atoms:
    "kettle": {"lift", "open"},
    "toilet": {"flush", "open", "close"},
    "suitcase": {"extend"},
    "printer": {"on"},
    "trashcan": {"open", "close"},
    "partnet_switch": {"on"},
    "storage_drawer": {"open"},
    "clock": {"advance"},
    "partnet_door": {"open", "close"},
    "partnet_fan": {"rotate"},
    "coffeemachine": {"on"},
    "partnet_microwave": {"open"},
    "display": {"extend", "tilt"},
    "globe": {"rotate"},
    "lamp": {"aim", "on"},
    "box": {"open", "pull"},
    "partnet_pot": {"remove"},
    "stapler_lid": {"open"},
    "refrigerator": {"open", "open_other"},
    "washingmachine": {"open"},
    "partnet_window": {"close"},
    "foldingchair": {"unfold"},
    "oven": {"open", "close", "into"},
    "table_drawer": {"open", "close"},
    "dishwasher": {"open", "press"},
    "faucet": {"on", "off"},
    "partnet_dispenser": {"open", "close"},
    "partnet_laptop": {"open"},
For every articulated asset you use, also add a companion asset whose type is that name plus "_concept" (for example kettle and kettle_concept). Concept types have no atoms:
    "kettle_concept": {},
    "toilet_concept": {},
    "suitcase_concept": {},
    "printer_concept": {},
    "trashcan_concept": {},
    "partnet_switch_concept": {},
    "storage_drawer_concept": {},
    "clock_concept": {},
    "partnet_door_concept": {},
    "partnet_fan_concept": {},
    "coffeemachine_concept": {},
    "partnet_microwave_concept": {},
    "display_concept": {},
    "globe_concept": {},
    "lamp_concept": {},
    "box_concept": {},
    "partnet_pot_concept": {},
    "stapler_lid_concept": {},
    "refrigerator_concept": {},
    "washingmachine_concept": {},
    "partnet_window_concept": {},
    "foldingchair_concept": {},
    "oven_concept": {},
    "table_drawer_concept": {},
    "dishwasher_concept": {},
    "faucet_concept": {},
    "partnet_dispenser_concept": {},
    "partnet_laptop_concept": {},
11) For an articulated atom, set params to {"fraction": 0.5} unless the instruction gives a different amount. Rigid atoms use params {}.

assets item schema:
- id: string
- type: one of the allowed asset types listed above

atomic_plan item schema:
- asset: string (must reference assets[].id)
- atom: string
- params: object (must be present; use {} if empty)


"""

USER_PROMPT_TEMPLATE = """user instructions：
{instruction}

请直接输出 JSON。"""


def extract_json_from_text(text: str) -> str:
    """
    尽量从模型输出中提取第一个完整 JSON object。
    兼容：
    - 纯json
    - ```json ... ```
    - 前后有少量杂质文本
    """
    text = text.strip()

    # 先处理 fenced code block
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # 纯对象优先
    if text.startswith("{") and text.endswith("}"):
        return text

    # 通用：寻找第一个平衡大括号对象
    start = text.find("{")
    if start == -1:
        raise ValueError("模型输出中未找到 JSON 对象起始 '{'")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("模型输出中 JSON 大括号不平衡，提取失败")


# ---------------------------
# 4) 本地语义校验
# ---------------------------

def _instruction_requires_simultaneous(instruction: str | None) -> bool:
    if not instruction:
        return False
    text = instruction.lower()
    keywords = ("simultaneously", "at the same time", "同时", "一起", "双手同时")
    return any(k in text for k in keywords)


def _is_plain_pick_like(atom: str) -> bool:
    atom_l = atom.lower()
    # pick-like atoms without simultaneous marker are forbidden in simultaneous tasks.
    return ("pick" in atom_l) and ("simultaneous" not in atom_l)


_ARTICULATED_TYPES = {
    "kettle", "toilet", "suitcase", "printer", "trashcan", "partnet_switch",
    "storage_drawer", "clock", "partnet_door", "partnet_fan", "coffeemachine",
    "partnet_microwave", "display", "globe", "lamp", "box", "partnet_pot",
    "stapler_lid", "refrigerator", "washingmachine", "partnet_window",
    "foldingchair", "oven", "table_drawer", "dishwasher", "faucet",
    "partnet_dispenser", "partnet_laptop",
}


def _is_rigid_plan(plan: TaskPlan) -> bool:
    for asset in plan.assets:
        base = asset.type[:-8] if asset.type.endswith("_concept") else asset.type
        if base in _ARTICULATED_TYPES:
            return False
    return True


def _require_table_for_rigid(plan: TaskPlan) -> None:
    if not _is_rigid_plan(plan):
        return
    tables = [asset for asset in plan.assets if asset.type == "table"]
    if len(tables) != 1 or tables[0].id != "table_1":
        raise ValueError(
            "刚体任务必须包含且只包含一个桌子资产：id='table_1', type='table'。"
            "不要把桌子写进 atomic_plan。"
        )


def semantic_validate(plan: TaskPlan, instruction: str | None = None) -> None:
    if len(plan.assets) == 0:
        raise ValueError("assets 不能为空，必须从自然语言中提取任务所需资产")
    if len(plan.atomic_plan) == 0:
        raise ValueError("atomic_plan 不能为空，必须包含可执行步骤")

    asset_ids = [a.id for a in plan.assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("assets 中存在重复 id")
    _require_table_for_rigid(plan)

    for a in plan.assets:
        if a.type not in ALLOWED_ATOMS:
            raise ValueError(
                f"asset '{a.id}' 的 type='{a.type}' 不在允许类型中，允许={sorted(list(ALLOWED_ATOMS.keys()))}"
            )

    id_to_type = {a.id: a.type for a in plan.assets}
    used_assets: set[str] = set()
    open_seen: set[str] = set()
    require_simultaneous = _instruction_requires_simultaneous(instruction)
    for i, s in enumerate(plan.atomic_plan, start=1):
        if s.asset not in id_to_type:
            raise ValueError(f"step {i}: asset='{s.asset}' 不在 assets 中")
        used_assets.add(s.asset)
        obj_type = id_to_type[s.asset]
        allowed = ALLOWED_ATOMS.get(obj_type, set())
        if s.atom not in allowed:
            raise ValueError(
                f"step {i}: 动作不合法，{obj_type} 不支持 atom='{s.atom}'，允许={sorted(list(allowed))}"
            )
        if require_simultaneous and _is_plain_pick_like(s.atom):
            raise ValueError(
                f"step {i}: 指令要求同时执行，但 atom='{s.atom}' 不包含 simultaneously。"
                "请使用带 simultaneously 的原子任务（如 pick_simultaneously_left / pick_simultaneously_right）。"
            )
        if s.atom in ("open", "open_release"):
            open_seen.add(s.asset)
        if obj_type == "microwave" and s.atom in ("into", "into_left"):
            if s.asset not in open_seen:
                raise ValueError(f"step {i}: microwave.into 之前缺少对 {s.asset} 的 open")

    unused_assets = [
        aid for aid in asset_ids
        if aid not in used_assets and ALLOWED_ATOMS.get(id_to_type[aid], set())
    ]
    if unused_assets:
        raise ValueError(
            f"assets 中存在未被任何步骤使用的对象: {unused_assets}。请仅保留完成任务必需资产，或补充对应原子步骤。"
        )
    # E) table 不能被操作（如果有人写了 table 动作，上面已因 allowed=set 拦截）


# ---------------------------
# 5) Agent 主逻辑
# ---------------------------

class TaskPlannerAgent:
    def __init__(self, provider: str, temperature: float = 0.0, max_retry: int = 3):
        self.provider = provider
        self.temperature = temperature
        self.max_retry = max_retry

        cfg = get_provider_config(provider)
        self.model = cfg.model
        self.client = _openai_client(cfg.api_key, cfg.base_url)

    def _chat_once(self, messages: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    def plan(self, instruction: str) -> TaskPlan:
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction)},
        ]

        messages = list(base_messages)
        last_error = None

        for attempt in range(1, self.max_retry + 1):
            raw = self._chat_once(messages)

            try:
                json_text = extract_json_from_text(raw)
                data = json.loads(json_text)

                # 结构校验（Pydantic）
                plan = TaskPlan.model_validate(data)

                # 语义校验（本地规则）
                semantic_validate(plan, instruction=instruction)

                return plan

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = str(e)

                # 把错误反馈给模型，要求修复后重发纯JSON
                repair_msg = (
                    "上一次输出不合法，错误如下：\n"
                    f"{last_error}\n\n"
                    "请严格修复并仅返回 JSON。不要添加解释。"
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": repair_msg})

        raise RuntimeError(f"任务规划失败，重试 {self.max_retry} 次仍未成功。最后错误：{last_error}")


# ---------------------------
# 6) 对外函数：本地生成 + 保存
# ---------------------------

def generate_task_json(
    instruction: str,
    output_path: str = "task_plan.json",
    provider: str = "deepseek",
    temperature: float = 0.0,
    max_retry: int = 3,
) -> TaskPlan:
    agent = TaskPlannerAgent(
        provider=provider,
        temperature=temperature,
        max_retry=max_retry,
    )
    plan = agent.plan(instruction)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)

    return plan


# ---------------------------
# 7) CLI 入口
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Local task decomposition + JSON generation agent")
    parser.add_argument("--instruction", type=str, required=True, help="自然语言任务描述")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "openai", "kimi"])
    parser.add_argument("--output", type=str, default="./tasks/rigid/task_plan.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window.")
    parser.add_argument("--anti_interference", dest="anti_interference", action="store_true")
    parser.add_argument("--no_anti_interference", dest="anti_interference", action="store_false")
    parser.set_defaults(anti_interference=False)
    args = parser.parse_args()
    scene_py = "./run_task.py"
    try:
        plan = generate_task_json(
            instruction=args.instruction,
            output_path=args.output,
            provider=args.provider,
            temperature=args.temperature,
            max_retry=args.max_retry,
        )
        print("任务分解成功，已保存：", args.output)
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
        scene_cmd = [
            "python",
            scene_py,
            "--task",
            args.output,
            "--enable_cameras",
            "--num_envs",
            str(args.num_envs),
            "--anti_interference" if args.anti_interference else "--no_anti_interference",
        ]
        if args.headless:
            scene_cmd.append("--headless")
        subprocess.run(
            scene_cmd,
            check=True,
            cwd="./",
        )
    except Exception as e:
        print(" 任务分解失败：", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

USER_PROMPT_TEMPLATE = """user instructions：
{instruction}

请直接输出 JSON。"""


def extract_json_from_text(text: str) -> str:
    """
    尽量从模型输出中提取第一个完整 JSON object。
    兼容：
    - 纯json
    - ```json ... ```
    - 前后有少量杂质文本
    """
    text = text.strip()

    # 先处理 fenced code block
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # 纯对象优先
    if text.startswith("{") and text.endswith("}"):
        return text

    # 通用：寻找第一个平衡大括号对象
    start = text.find("{")
    if start == -1:
        raise ValueError("模型输出中未找到 JSON 对象起始 '{'")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("模型输出中 JSON 大括号不平衡，提取失败")


# ---------------------------
# 4) 本地语义校验
# ---------------------------

_ARTICULATED_TYPES = {
    "kettle", "toilet", "suitcase", "printer", "trashcan", "partnet_switch",
    "storage_drawer", "clock", "partnet_door", "partnet_fan", "coffeemachine",
    "partnet_microwave", "display", "globe", "lamp", "box", "partnet_pot",
    "stapler_lid", "refrigerator", "washingmachine", "partnet_window",
    "foldingchair", "oven", "table_drawer", "dishwasher", "faucet",
    "partnet_dispenser", "partnet_laptop",
}


def _is_rigid_plan(plan: TaskPlan) -> bool:
    for asset in plan.assets:
        base = asset.type[:-8] if asset.type.endswith("_concept") else asset.type
        if base in _ARTICULATED_TYPES:
            return False
    return True


def _require_table_for_rigid(plan: TaskPlan) -> None:
    if not _is_rigid_plan(plan):
        return
    tables = [asset for asset in plan.assets if asset.type == "table"]
    if len(tables) != 1 or tables[0].id != "table_1":
        raise ValueError(
            "刚体任务必须包含且只包含一个桌子资产：id='table_1', type='table'。"
            "不要把桌子写进 atomic_plan。"
        )


def semantic_validate(plan: TaskPlan, instruction: str | None = None) -> None:
    if len(plan.assets) == 0:
        raise ValueError("assets 不能为空，必须从自然语言中提取任务所需资产")
    if len(plan.atomic_plan) == 0:
        raise ValueError("atomic_plan 不能为空，必须包含可执行步骤")

    asset_ids = [a.id for a in plan.assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("assets 中存在重复 id")
    _require_table_for_rigid(plan)

    for a in plan.assets:
        if a.type not in ALLOWED_ATOMS:
            raise ValueError(
                f"asset '{a.id}' 的 type='{a.type}' 不在允许类型中，允许={sorted(list(ALLOWED_ATOMS.keys()))}"
            )

    id_to_type = {a.id: a.type for a in plan.assets}
    used_assets: set[str] = set()
    open_seen: set[str] = set()
    require_simultaneous = _instruction_requires_simultaneous(instruction)
    for i, s in enumerate(plan.atomic_plan, start=1):
        if s.asset not in id_to_type:
            raise ValueError(f"step {i}: asset='{s.asset}' 不在 assets 中")
        used_assets.add(s.asset)
        obj_type = id_to_type[s.asset]
        allowed = ALLOWED_ATOMS.get(obj_type, set())
        if s.atom not in allowed:
            raise ValueError(
                f"step {i}: 动作不合法，{obj_type} 不支持 atom='{s.atom}'，允许={sorted(list(allowed))}"
            )
        if require_simultaneous and _is_plain_pick_like(s.atom):
            raise ValueError(
                f"step {i}: 指令要求同时执行，但 atom='{s.atom}' 不包含 simultaneously。"
                "请使用带 simultaneously 的原子任务（如 pick_simultaneously_left / pick_simultaneously_right）。"
            )
        if s.atom in ("open", "open_release"):
            open_seen.add(s.asset)
        if obj_type == "microwave" and s.atom in ("into", "into_left"):
            if s.asset not in open_seen:
                raise ValueError(f"step {i}: microwave.into 之前缺少对 {s.asset} 的 open")
    # E) table 不能被操作（如果有人写了 table 动作，上面已因 allowed=set 拦截）
    unused_assets = [
        aid for aid in asset_ids
        if aid not in used_assets and ALLOWED_ATOMS.get(id_to_type[aid], set())
    ]
    if unused_assets:
        raise ValueError(
            f"assets 中存在未被任何步骤使用的对象: {unused_assets}。请仅保留完成任务必需资产，或补充对应原子步骤。"
        )


# ---------------------------
# 5) Agent 主逻辑
# ---------------------------

class TaskPlannerAgent:
    def __init__(self, provider: str, temperature: float = 0.0, max_retry: int = 3):
        self.provider = provider
        self.temperature = temperature
        self.max_retry = max_retry

        cfg = get_provider_config(provider)
        self.model = cfg.model
        self.client = _openai_client(cfg.api_key, cfg.base_url)

    def _chat_once(self, messages: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    def plan(self, instruction: str) -> TaskPlan:
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction)},
        ]

        messages = list(base_messages)
        last_error = None

        for attempt in range(1, self.max_retry + 1):
            raw = self._chat_once(messages)

            try:
                json_text = extract_json_from_text(raw)
                data = json.loads(json_text)

                # 结构校验（Pydantic）
                plan = TaskPlan.model_validate(data)

                # 语义校验（本地规则）
                semantic_validate(plan, instruction=instruction)

                return plan

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = str(e)

                # 把错误反馈给模型，要求修复后重发纯JSON
                repair_msg = (
                    "上一次输出不合法，错误如下：\n"
                    f"{last_error}\n\n"
                    "请严格修复并仅返回 JSON。不要添加解释。"
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": repair_msg})

        raise RuntimeError(f"任务规划失败，重试 {self.max_retry} 次仍未成功。最后错误：{last_error}")


# ---------------------------
# 6) 对外函数：本地生成 + 保存
# ---------------------------

def generate_task_json(
    instruction: str,
    output_path: str = "task_plan.json",
    provider: str = "deepseek",
    temperature: float = 0.0,
    max_retry: int = 3,
) -> TaskPlan:
    agent = TaskPlannerAgent(
        provider=provider,
        temperature=temperature,
        max_retry=max_retry,
    )
    plan = agent.plan(instruction)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)

    return plan


# ---------------------------
# 7) CLI 入口
# ---------------------------

def main():
    from pathlib import Path

    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Local task decomposition + JSON generation agent")
    parser.add_argument("--instruction", type=str, required=True, help="自然语言任务描述")
    parser.add_argument("--provider", type=str, default="openai", choices=["deepseek", "openai", "kimi"])
    parser.add_argument("--output", type=str, default=str(project_root / "tasks" / "task_plan.json"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retry", type=int, default=3)
    args = parser.parse_args()
    scene_py = str(project_root / "scripts" / "micwave_scene.py")
    try:
        plan = generate_task_json(
            instruction=args.instruction,
            output_path=args.output,
            provider=args.provider,
            temperature=args.temperature,
            max_retry=args.max_retry,
        )
        print("✅ 任务分解成功，已保存：", args.output)
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
        subprocess.run(
            ["conda", "run", "-n", "env_isaaclab", "python", scene_py, "--enable_cameras"],
            check=True,
            cwd=str(project_root),
        )
    except Exception as e:
        print("❌ 任务分解失败：", e, file=sys.stderr)
        sys.exit(1)