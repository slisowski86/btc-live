"""
Strategy pattern templates and combo-enumeration helpers.

This is the lean core extracted from the old combination_counter.py — only the
pieces strategy_funnel.py needs to build the strategy space:

  * the 8 PATTERNS_* templates
  * SAME_BASE_PAIRS (divergence/momentum exclusion)
  * indicator introspection (_get_category / _get_signal_capabilities / _get_signal_methods)
  * combo validation + enumeration (_is_valid_combination / _list_combos_for_patterns)
  * _expand_key (combo -> component dicts with concrete method names)

The bulk counting / enumeration / JSON-export API of combination_counter.py is
intentionally NOT included here.
"""
import ast
import inspect
import textwrap
from itertools import product as iproduct
from collections import defaultdict
from typing import Type, List, Tuple

# ---------------------------------------------------------------------------
# Pattern definitions
# Each element: (category, direction, signal_type)
# direction: 'long', 'short', 'both'      signal_type: 'continuous', 'discrete'
# ---------------------------------------------------------------------------

PATTERNS_2_ENTRY_LONG = [
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
]

PATTERNS_3_ENTRY_LONG = [
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
]

PATTERNS_2_EXIT_LONG = [
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
]

PATTERNS_3_EXIT_LONG = [
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
]

PATTERNS_2_ENTRY_SHORT = [
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
]

PATTERNS_3_ENTRY_SHORT = [
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('momentum', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('momentum', 'short', 'continuous'), ('divergence', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'short', 'discrete')),
    (('trend_direction', 'short', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'short', 'discrete')),
]

PATTERNS_2_EXIT_SHORT = [
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
]

PATTERNS_3_EXIT_SHORT = [
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('momentum', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('volatility', 'both', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('momentum', 'long', 'continuous'), ('divergence', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('trend_strength', 'both', 'continuous'), ('momentum', 'long', 'discrete')),
    (('trend_direction', 'long', 'continuous'), ('trend_strength', 'both', 'continuous'), ('divergence', 'long', 'discrete')),
]

# ---------------------------------------------------------------------------
# Same-base indicator pairs: a divergence indicator and its base momentum
# indicator cannot coexist in the same signal combination.
# ---------------------------------------------------------------------------
SAME_BASE_PAIRS: List[frozenset] = [
    frozenset({'RsiDiv',         'RSI'}),
    frozenset({'MACDDiv',        'MACD'}),
    frozenset({'EFTDiv',         'EFISH'}),
    frozenset({'EhlersReflexDiv','EHLERSCC'}),
    frozenset({'StochDiv',       'STOCH'}),
    frozenset({'CCIDiv',         'CCI'}),
]

# _IndicatorInfo = (class_name, category, capabilities_set)
_IndicatorInfo = Tuple[str, str, set]


# ---------------------------------------------------------------------------
# Indicator introspection
# ---------------------------------------------------------------------------

def _get_category(cls: Type) -> str:
    """Extract self.category from the class __init__ source via AST (None if absent)."""
    try:
        src = inspect.getsource(cls.__init__)
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == 'category'
                    and isinstance(node.value, ast.Constant)):
                return node.value.value
    except Exception:
        pass
    return None


def _get_signal_capabilities(cls: Type) -> set:
    """Set of (direction, signal_type) for all @signal methods on the class."""
    caps = set()
    for attr in vars(cls).values():
        if callable(attr) and hasattr(attr, '_signal_meta'):
            meta = attr._signal_meta
            caps.add((meta['direction'], meta['type']))
    return caps


def _get_signal_methods(cls: Type) -> dict:
    """Mapping {(direction, signal_type): [method_name, ...]} for @signal methods."""
    out: dict = defaultdict(list)
    for attr in vars(cls).values():
        if callable(attr) and hasattr(attr, '_signal_meta'):
            meta = attr._signal_meta
            out[(meta['direction'], meta['type'])].append(meta['name'])
    return out


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _share_base(name1: str, name2: str) -> bool:
    """True if name1 and name2 are a same-base divergence/momentum pair."""
    for pair in SAME_BASE_PAIRS:
        if name1 in pair and name2 in pair and name1 != name2:
            return True
    return False


def _is_valid_combination(assignment: Tuple[str, ...], pattern: tuple,
                          respect_same_base: bool = True) -> bool:
    """
    Constraint 1 — no continuous+discrete mix of the same indicator in a group.
    Constraint 2 (optional) — no same-base divergence+momentum pair.
    """
    n = len(assignment)
    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = assignment[i], assignment[j]
            if respect_same_base and _share_base(name_i, name_j):
                return False
            if name_i == name_j and pattern[i][2] != pattern[j][2]:
                return False
    return True


def _canonical_key(pattern: tuple, assignment: Tuple[str, ...]) -> tuple:
    """Order-independent key so permutations within identical slots dedupe."""
    slot_indices: dict = defaultdict(list)
    for i, slot in enumerate(pattern):
        slot_indices[slot].append(i)
    key = []
    for slot in sorted(slot_indices.keys()):
        names = tuple(sorted(assignment[i] for i in slot_indices[slot]))
        key.append((slot, names))
    return tuple(key)


# ---------------------------------------------------------------------------
# Combo enumeration
# ---------------------------------------------------------------------------

def _list_combos_for_patterns(infos: List[_IndicatorInfo], patterns: list,
                              respect_same_base: bool = True) -> List[Tuple[frozenset, tuple]]:
    """
    Enumerate all distinct valid combinations across the given patterns.
    Returns list of (indicator_set, canonical_key).
    """
    seen: set = set()
    results: List[Tuple[frozenset, tuple]] = []

    for pattern in patterns:
        slot_candidates: List[List[str]] = []
        for slot in pattern:
            slot_cat, slot_dir, slot_type = slot
            candidates = [
                name
                for name, cat, caps in infos
                if cat == slot_cat and (slot_dir, slot_type) in caps
            ]
            if not candidates:
                break
            slot_candidates.append(candidates)
        else:
            for assignment in iproduct(*slot_candidates):
                if _is_valid_combination(assignment, pattern, respect_same_base):
                    key = _canonical_key(pattern, assignment)
                    if key not in seen:
                        seen.add(key)
                        results.append((frozenset(assignment), key))

    return results


def _expand_key(key: tuple, name_to_methods: dict) -> List[dict]:
    """
    Expand a canonical combo key into component dicts with concrete method names.
    key : tuple of (slot, names) where slot = (category, direction, signal_type)
    """
    comps: List[dict] = []
    for slot, names in key:
        category, direction, signal_type = slot
        for nm in names:
            methods = name_to_methods.get(nm, {}).get((direction, signal_type), [])
            comps.append({
                'indicator':   nm,
                'category':    category,
                'direction':   direction,
                'signal_type': signal_type,
                'method':      methods[0] if len(methods) == 1 else methods,
            })
    return comps
