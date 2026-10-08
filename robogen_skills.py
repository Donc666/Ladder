"""Register one articulated category module per asset type.

Parameters and atomic methods live in ``manipulation knowledge base/<type>.py``.
This module only builds the factory, allowed-atom set, and dispatcher
that auto_collect_data.py already consumes.
"""

import kb_path  # noqa: F401

import box
import clock
import coffeemachine
import dishwasher
import display
import faucet
import foldingchair
import globe
import kettle
import lamp
import oven
import partnet_dispenser
import partnet_door
import partnet_fan
import partnet_laptop
import partnet_microwave
import partnet_pot
import partnet_switch
import partnet_window
import printer
import refrigerator
import stapler_lid
import storage_drawer
import suitcase
import table_drawer
import toilet
import trashcan
import washingmachine

MODULES = [
    suitcase,
    toilet,
    clock,
    oven,
    trashcan,
    display,
    faucet,
    foldingchair,
    stapler_lid,
    printer,
    kettle,
    table_drawer,
    partnet_door,
    box,
    coffeemachine,
    dishwasher,
    partnet_dispenser,
    partnet_fan,
    refrigerator,
    lamp,
    globe,
    washingmachine,
    partnet_pot,
    partnet_microwave,
    partnet_switch,
    partnet_window,
    partnet_laptop,
    storage_drawer,
]


def _call(method: str):
    return lambda obj, sc, p: getattr(obj, method)(
        scene=sc,
        env_id=p.get("_env_id", 0),
        fraction=p.get("fraction", 0.5),
    )


ROBOGEN_FACTORY = {}
ROBOGEN_ATOMS = {}
ROBOGEN_DISPATCH = {}

for _mod in MODULES:
    _name = _mod.SPEC.type_name
    _skill = getattr(_mod, _name)
    _concept = getattr(_mod, f"{_name}_concept")
    _atoms = set(_mod.ATOMS)
    _methods = {
        name
        for name, fn in _skill.__dict__.items()
        if callable(fn) and not name.startswith("_")
    }
    _missing = _atoms - _methods
    if _missing:
        raise RuntimeError(f"{_name} ATOMS has no method: {sorted(_missing)}")
    ROBOGEN_FACTORY[_name] = (lambda cls: lambda: cls())(_skill)
    ROBOGEN_FACTORY[f"{_name}_concept"] = (lambda cls: lambda: cls())(_concept)
    ROBOGEN_ATOMS[_name] = _atoms
    ROBOGEN_ATOMS[f"{_name}_concept"] = set()
    for _atom in _atoms:
        ROBOGEN_DISPATCH[(_name, _atom)] = _call(_atom)
