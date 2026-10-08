"""Add the manipulation knowledge base to sys.path.

Asset category modules live in a folder whose name has spaces, so they are
imported by module name after this path is inserted.
"""
import sys
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent / "manipulation knowledge base"
_kb = str(KB_ROOT)
if _kb not in sys.path:
    sys.path.insert(0, _kb)
