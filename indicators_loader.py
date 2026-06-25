"""
Discover and import every indicator class under indicators/<category>/ and
expose them as a flat list `all_classes` plus a name->class dict `by_name`.

Replaces the old notebook bootstrap cell. Usage:

    from indicators_loader import all_classes, by_name
    # all_classes -> list of 24 indicator classes, ready for strategy_funnel

It also puts each category folder on sys.path so the indicators' internal
`from SignalDecorator import signal` imports resolve.
"""
from __future__ import annotations

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_IND = os.path.join(_HERE, "indicators")
_CATEGORIES = ("divergence", "momentum", "trend_direction",
               "trend_strength", "volatility")

# --- make SignalDecorator + indicator modules importable by bare name ---
for _sub in _CATEGORIES + ("",):
    _p = os.path.join(_IND, _sub) if _sub else _IND
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _discover():
    classes, names = [], {}
    for cat in _CATEGORIES:
        folder = os.path.join(_IND, cat)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".py"):
                continue
            stem = fn[:-3]
            if stem in ("SignalDecorator", "__init__"):
                continue
            mod = importlib.import_module(stem)          # folder is on sys.path
            cls = getattr(mod, stem, None)
            if cls is None:                              # fallback: class defined here
                cls = next((v for v in vars(mod).values()
                            if isinstance(v, type) and v.__module__ == stem), None)
            if cls is None:
                raise ImportError(f"no class found in indicators/{cat}/{fn}")
            classes.append(cls)
            names[stem] = cls
    return classes, names


all_classes, by_name = _discover()


if __name__ == "__main__":
    print(f"loaded {len(all_classes)} indicator classes:")
    for c in all_classes:
        print(f"  {c.__name__:<18} category={getattr(c, 'category', '?')}")
