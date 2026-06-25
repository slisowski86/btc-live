"""
Multi-stage funnel for filtering trading-strategy combinations.

Pipeline
--------
    sample  ->  Stage 0 structural  ->  Stage 1 proxy Sharpe (no cost)
            ->  Stage 2 Sharpe with costs  ->  Stage 3 out-of-sample
            ->  Stage 4 walk-forward + Deflated Sharpe

Design
------
* A strategy = 4 signal groups (entry_long / exit_long / entry_short / exit_short),
  each group a list of component dicts {indicator, method, direction, signal_type}.
* Each signal group only has ~hundreds–thousands of valid combos; the trillions
  come from their product. So we enumerate the 4 small per-group lists and SAMPLE
  the product — full coverage without materialising 11.7T strategies.
* Every indicator signal is computed ONCE into an int8 array (signal cache); a
  backtest is then boolean algebra + a fast position loop. No TA recompute.

Requires: numpy, pandas. numba is optional (falls back to pure-Python loop).

Usage (notebook)
----------------
    import pandas as pd
    from strategy_funnel import StrategySpace, build_signal_cache, run_funnel

    data = pd.read_csv("df_test.csv", parse_dates=["Date"]).set_index("Date")
    survivors = run_funnel(all_classes, data, n_samples=200_000)
"""
from __future__ import annotations

import math
import random
from itertools import product as iproduct
from collections import OrderedDict

import numpy as np

import strategy_patterns as cc

# Optional numba acceleration -------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):                               # no-op decorator
        def wrap(f):
            return f
        return wrap(args[0]) if args and callable(args[0]) else wrap


GROUPS = ('entry_long', 'exit_long', 'entry_short', 'exit_short')
_BOTH_CATEGORIES = ('volatility', 'trend_strength')


# ---------------------------------------------------------------------------
# Strategy space: per-group detailed combos + product sampling
# ---------------------------------------------------------------------------

def _patterns_for(signal: str, max_pattern_size: int):
    p2 = {
        'entry_long':  cc.PATTERNS_2_ENTRY_LONG,
        'exit_long':   cc.PATTERNS_2_EXIT_LONG,
        'entry_short': cc.PATTERNS_2_ENTRY_SHORT,
        'exit_short':  cc.PATTERNS_2_EXIT_SHORT,
    }[signal]
    p3 = {
        'entry_long':  cc.PATTERNS_3_ENTRY_LONG,
        'exit_long':   cc.PATTERNS_3_EXIT_LONG,
        'entry_short': cc.PATTERNS_3_ENTRY_SHORT,
        'exit_short':  cc.PATTERNS_3_EXIT_SHORT,
    }[signal]
    return p2 + (p3 if max_pattern_size >= 3 else [])


def _split_group_variants(comps):
    """
    Expand one group's component list so each 'both'-direction component is
    tested as a long and a short variant. Mirrors the strategy-counter's
    _split_both_variants but for a single group. Yields component lists.
    """
    slots = [i for i, c in enumerate(comps) if c.get('direction') == 'both']
    if not slots:
        yield comps
        return
    for choice in iproduct(('long', 'short'), repeat=len(slots)):
        variant = [dict(c) for c in comps]
        for i, side in zip(slots, choice):
            variant[i]['base_direction'] = 'both'
            variant[i]['direction'] = side
            variant[i]['tested_as'] = side
        yield variant


def _signal_combos_detailed(indicator_classes, signal, max_pattern_size, split_both):
    """Return a list of component-lists (detailed combos) for one signal group."""
    infos = []
    name_to_methods = {}
    for clsx in indicator_classes:
        cat = cc._get_category(clsx)
        if cat is None:
            continue
        caps = cc._get_signal_capabilities(clsx)
        infos.append((clsx.__name__, cat, caps))
        name_to_methods[clsx.__name__] = cc._get_signal_methods(clsx)

    combos = cc._list_combos_for_patterns(infos, _patterns_for(signal, max_pattern_size))
    out = []
    for _set, key in combos:
        comps = cc._expand_key(key, name_to_methods)
        if split_both:
            out.extend(_split_group_variants(comps))
        else:
            out.append(comps)
    return out


class StrategySpace:
    """
    Holds the 4 per-group detailed-combo lists and samples the product.

    A sampled strategy is a dict {group: component_list}.
    """

    def __init__(self, indicator_classes, max_pattern_size=3, split_both=True,
                 divergence_in_exits_only=False):
        self.groups = {
            g: _signal_combos_detailed(indicator_classes, g, max_pattern_size, split_both)
            for g in GROUPS
        }
        if divergence_in_exits_only:
            # divergence is an early but noisy signal -> allow it only in exits
            # (false exit = opportunity cost; false entry = a real losing trade).
            def _no_div(combo):
                return not any(c['category'] == 'divergence' for c in combo)
            for g in ('entry_long', 'entry_short'):
                self.groups[g] = [c for c in self.groups[g] if _no_div(c)]
        self.sizes = {g: len(v) for g, v in self.groups.items()}

    def total(self) -> int:
        t = 1
        for g in GROUPS:
            t *= self.sizes[g]
        return t

    def random(self, rng: random.Random) -> dict:
        return {g: self.groups[g][rng.randrange(self.sizes[g])] for g in GROUPS}

    def iter_random(self, n: int, seed: int = 0, unique: bool = True):
        """Yield n sampled strategies (deduped by structural signature if unique)."""
        rng = random.Random(seed)
        seen = set()
        produced = 0
        attempts = 0
        max_attempts = n * 20
        while produced < n and attempts < max_attempts:
            attempts += 1
            strat = self.random(rng)
            if unique:
                sig = self._signature(strat)
                if sig in seen:
                    continue
                seen.add(sig)
            produced += 1
            yield strat

    @staticmethod
    def _signature(strat) -> tuple:
        return tuple(
            tuple(sorted((c['indicator'], c['method'] if isinstance(c['method'], str)
                          else tuple(c['method']), c.get('tested_as', c['direction']))
                         for c in strat[g]))
            for g in GROUPS
        )


# ---------------------------------------------------------------------------
# Signal cache: compute every indicator signal ONCE
# ---------------------------------------------------------------------------

def build_signal_cache(indicator_classes, data, show_progress: bool = True) -> dict:
    """
    Instantiate each indicator on `data` and evaluate every @signal method once.

    Returns {(indicator_name, method_name): np.int8 array aligned to data.index}.
    Indicators that fail to construct/evaluate are skipped (with a warning list
    attached as cache['_failed']).
    """
    cache = {}
    failed = []
    n = len(data)
    iterator = indicator_classes
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(indicator_classes, desc="signal cache", unit="indicator")
        except ImportError:
            pass
    for clsx in iterator:
        name = clsx.__name__
        try:
            inst = clsx(data)
        except Exception as exc:                              # pragma: no cover
            failed.append((name, repr(exc)))
            continue
        for attrname, attr in vars(clsx).items():
            if callable(attr) and hasattr(attr, '_signal_meta'):
                meta = attr._signal_meta
                try:
                    series = getattr(inst, attrname)()
                    # accept pandas Series (has .values) OR a raw numpy array/list
                    # np.array (not asarray) -> always a writable copy
                    raw = np.array(getattr(series, 'values', series), dtype=np.float64)
                    # NaN/inf (warmup bars etc.) -> 0 (no signal); avoids garbage int8 cast
                    raw[~np.isfinite(raw)] = 0.0
                    arr = raw.astype(np.int8)
                    if arr.shape[0] != n:
                        raise ValueError("length mismatch")
                    cache[(name, meta['name'])] = arr
                except Exception as exc:                      # pragma: no cover
                    failed.append((f"{name}.{meta['name']}", repr(exc)))
    cache['_failed'] = failed
    return cache


def _data_fingerprint(data) -> tuple:
    """Cheap identity of a dataframe so a persisted cache isn't reused for the
    wrong data (rows, span, columns)."""
    try:
        return (len(data), str(data.index[0]), str(data.index[-1]),
                tuple(map(str, data.columns)))
    except Exception:
        return (len(data),)


def get_signal_cache(indicator_classes, data, cache_path: str = None,
                     show_progress: bool = True) -> dict:
    """
    Build the signal cache once and persist it, so the slow indicators
    (e.g. divergence) are computed a single time and reloaded on later runs.

    If `cache_path` exists and matches the data fingerprint, it is loaded;
    otherwise the cache is built and (if `cache_path` is given) saved.
    Returns the same dict as build_signal_cache (including '_failed').
    """
    import os
    import pickle

    fp = _data_fingerprint(data)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                blob = pickle.load(fh)
            if blob.get('fingerprint') == fp:
                if show_progress:
                    print(f"signal cache: loaded {len(blob['cache']):,} arrays "
                          f"from {cache_path}")
                return blob['cache']
            elif show_progress:
                print(f"signal cache: {cache_path} fingerprint mismatch -> rebuilding")
        except Exception as exc:                              # pragma: no cover
            if show_progress:
                print(f"signal cache: failed to load {cache_path} ({exc}) -> rebuilding")

    cache = build_signal_cache(indicator_classes, data, show_progress=show_progress)
    if cache_path:
        tmp = cache_path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({'fingerprint': fp, 'cache': cache}, fh, protocol=4)
        os.replace(tmp, cache_path)
        if show_progress:
            print(f"signal cache: saved {len(cache):,} arrays to {cache_path}")
    return cache


def _component_active(comp, cache, n):
    """Boolean array: is this component 'active' on each bar?"""
    method = comp['method'] if isinstance(comp['method'], str) else comp['method'][0]
    key = (comp['indicator'], method)
    sig = cache.get(key)
    if sig is None:
        return np.zeros(n, dtype=bool)
    active = sig != 0
    # 'both'-regime tested on the short side => inverse polarity (regime OFF)
    if comp.get('base_direction') == 'both' and comp.get('tested_as') == 'short':
        active = sig == 0
    return active


def _group_trigger(comps, cache, n):
    """AND of all component-active arrays => this signal group fires."""
    trig = np.ones(n, dtype=bool)
    for c in comps:
        trig &= _component_active(c, cache, n)
    return trig


def strategy_triggers(strat, cache, n):
    """Return (EL, XL, ES, XS) boolean trigger arrays for a strategy."""
    return (
        _group_trigger(strat['entry_long'],  cache, n),
        _group_trigger(strat['exit_long'],   cache, n),
        _group_trigger(strat['entry_short'], cache, n),
        _group_trigger(strat['exit_short'],  cache, n),
    )


def _combo_key(comps) -> tuple:
    """
    Hashable key that uniquely identifies a signal group's TRIGGER behaviour:
    (indicator, resolved method, inverse-polarity flag) per component.
    The inverse flag captures a 'both' filter tested on the short side (regime
    OFF), which produces a different trigger than the long side (regime ON).
    """
    parts = []
    for c in comps:
        method = c['method'] if isinstance(c['method'], str) else c['method'][0]
        inverse = c.get('base_direction') == 'both' and c.get('tested_as') == 'short'
        parts.append((c['indicator'], method, inverse))
    return tuple(sorted(parts))


class _TriggerCache:
    """
    Lazy memoization of per-group trigger arrays. Each distinct combo's AND is
    computed once on first use and reused thereafter. Optional LRU cap bounds
    memory (least-recently-used combos are evicted and recomputed if seen again).
    """

    def __init__(self, signal_cache, n, max_cache=None):
        self.signal_cache = signal_cache
        self.n = n
        self.max = max_cache
        self.store = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, comps):
        key = _combo_key(comps)
        t = self.store.get(key)
        if t is not None:
            self.hits += 1
            if self.max:
                self.store.move_to_end(key)
            return t
        self.misses += 1
        t = _group_trigger(comps, self.signal_cache, self.n)
        self.store[key] = t
        if self.max and len(self.store) > self.max:
            self.store.popitem(last=False)     # evict least-recently-used
        return t

    def triggers(self, strat):
        """(EL, XL, ES, XS) for a strategy, via the cache."""
        return (self.get(strat['entry_long']),
                self.get(strat['exit_long']),
                self.get(strat['entry_short']),
                self.get(strat['exit_short']))

    @property
    def hit_rate(self):
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0


# ---------------------------------------------------------------------------
# Backtest core (position state machine)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _backtest_core(EL, XL, ES, XS, ret, cost):
    """
    Long/short state machine.
    pos held during bar i was set at close of i-1; earns ret[i].
    A position change (open/close/flip) costs `cost` (fraction) once.
    Returns (pnl_array, n_trades).
    """
    n = ret.shape[0]
    pnl = np.zeros(n)
    pos = 0
    trades = 0
    for i in range(n):
        pnl[i] = pos * ret[i]
        new_pos = pos
        if pos == 0:
            if EL[i]:
                new_pos = 1
            elif ES[i]:
                new_pos = -1
        elif pos == 1:
            if XL[i]:
                new_pos = 0
            if ES[i]:
                new_pos = -1
        else:  # pos == -1
            if XS[i]:
                new_pos = 0
            if EL[i]:
                new_pos = 1
        if new_pos != pos:
            pnl[i] -= cost
            trades += 1
            pos = new_pos
    return pnl, trades


@njit(cache=True)
def _backtest_trades(EL, XL, ES, XS, ret, cost):
    """
    Same state machine as _backtest_core but also records per-trade P&L
    (sum of bar returns over the holding period, minus the change cost).
    Returns (pnl_array, trade_pnls_array, n_trades).
    """
    n = ret.shape[0]
    pnl = np.zeros(n)
    trade_pnls = np.empty(n)
    pos = 0
    tcount = 0
    cur = 0.0
    for i in range(n):
        bar = pos * ret[i]
        pnl[i] = bar
        cur += bar
        new_pos = pos
        if pos == 0:
            if EL[i]:
                new_pos = 1
            elif ES[i]:
                new_pos = -1
        elif pos == 1:
            if XL[i]:
                new_pos = 0
            if ES[i]:
                new_pos = -1
        else:
            if XS[i]:
                new_pos = 0
            if EL[i]:
                new_pos = 1
        if new_pos != pos:
            pnl[i] -= cost
            cur -= cost
            if pos != 0:                 # a trade just closed (or flipped)
                trade_pnls[tcount] = cur
                tcount += 1
                cur = 0.0
            pos = new_pos
    if pos != 0:                          # close trailing open position
        trade_pnls[tcount] = cur
        tcount += 1
    return pnl, trade_pnls[:tcount], tcount


@njit(cache=True)
def _backtest_trades_stop(EL, XL, ES, XS, close, ret, atr, cost, trail):
    """
    Long/short state machine WITH an ATR trailing stop.

    A position closes when either the signal exit fires OR price retraces from
    its favourable extreme by `trail` * ATR:
        long  : exit if close <= peak  - trail*ATR
        short : exit if close >= trough + trail*ATR
    Returns (pnl_array, trade_pnls_array, n_trades).
    """
    n = ret.shape[0]
    pnl = np.zeros(n)
    trade_pnls = np.empty(n)
    pos = 0
    tcount = 0
    cur = 0.0
    peak = 0.0
    trough = 0.0
    for i in range(n):
        bar = pos * ret[i]
        pnl[i] = bar
        cur += bar

        stop_hit = False
        if pos == 1:
            if close[i] > peak:
                peak = close[i]
            if close[i] <= peak - trail * atr[i]:
                stop_hit = True
        elif pos == -1:
            if close[i] < trough:
                trough = close[i]
            if close[i] >= trough + trail * atr[i]:
                stop_hit = True

        new_pos = pos
        if pos == 0:
            if EL[i]:
                new_pos = 1
            elif ES[i]:
                new_pos = -1
        elif pos == 1:
            if XL[i] or stop_hit:
                new_pos = 0
            if ES[i]:
                new_pos = -1
        else:
            if XS[i] or stop_hit:
                new_pos = 0
            if EL[i]:
                new_pos = 1

        if new_pos != pos:
            pnl[i] -= cost
            cur -= cost
            if pos != 0:
                trade_pnls[tcount] = cur
                tcount += 1
                cur = 0.0
            pos = new_pos
            if pos == 1:
                peak = close[i]
            elif pos == -1:
                trough = close[i]
    if pos != 0:
        trade_pnls[tcount] = cur
        tcount += 1
    return pnl, trade_pnls[:tcount], tcount


def _returns_from_close(close: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(close, dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    return ret


def compute_atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder-style SMA of True Range)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    # rolling mean ATR
    atr = np.zeros(n)
    csum = np.cumsum(tr)
    atr[:period] = csum[:period] / np.arange(1, period + 1)
    atr[period:] = (csum[period:] - csum[:-period]) / period
    return atr


def compute_vol_target_size(close, target_ann_vol: float = 0.10, lookback: int = 96,
                            periods_per_year: float = 35040.0, max_leverage: float = 3.0):
    """
    Volatility-targeting position-size multiplier per bar.

    Sizes inversely to recent realized volatility so each trade contributes
    roughly constant risk: size = (target per-bar vol) / (realized vol).
    Capped at `max_leverage`. `target_ann_vol` is the annualized vol target
    (e.g. 0.10 = 10%). Returns a size array aligned to `close`.
    """
    close = np.asarray(close, dtype=np.float64)
    ret = _returns_from_close(close)
    n = ret.shape[0]
    # rolling std of returns (realized per-bar vol)
    vol = np.zeros(n)
    csum = np.cumsum(ret)
    csum2 = np.cumsum(ret * ret)
    for i in range(n):
        a = max(0, i - lookback + 1)
        m = i - a + 1
        s = csum[i] - (csum[a - 1] if a > 0 else 0.0)
        s2 = csum2[i] - (csum2[a - 1] if a > 0 else 0.0)
        var = s2 / m - (s / m) ** 2
        vol[i] = math.sqrt(var) if var > 0 else 0.0
    target_per_bar = target_ann_vol / math.sqrt(periods_per_year)
    size = np.zeros(n)
    nz = vol > 1e-12
    size[nz] = np.minimum(target_per_bar / vol[nz], max_leverage)
    return size


@njit(cache=True)
def _backtest_risk(EL, XL, ES, XS, close, ret, atr, size, cost, stop_atr, tp_atr):
    """
    Position-sized backtest with optional ATR stop-loss / take-profit.

    * size[i]  : position-size multiplier (locked at entry).
    * stop_atr : stop-loss distance in ATRs (0 = off).
    * tp_atr   : take-profit distance in ATRs (0 = off).
    Costs scale with position size. Returns (pnl, trade_pnls, n_trades).
    """
    n = ret.shape[0]
    pnl = np.zeros(n)
    trade_pnls = np.empty(n)
    pos = 0
    tcount = 0
    cur = 0.0
    entry_px = 0.0
    cur_size = 0.0
    for i in range(n):
        bar = pos * cur_size * ret[i]
        pnl[i] = bar
        cur += bar

        exit_now = False
        if pos == 1:
            if stop_atr > 0 and close[i] <= entry_px - stop_atr * atr[i]:
                exit_now = True
            if tp_atr > 0 and close[i] >= entry_px + tp_atr * atr[i]:
                exit_now = True
        elif pos == -1:
            if stop_atr > 0 and close[i] >= entry_px + stop_atr * atr[i]:
                exit_now = True
            if tp_atr > 0 and close[i] <= entry_px - tp_atr * atr[i]:
                exit_now = True

        new_pos = pos
        if pos == 0:
            if EL[i]:
                new_pos = 1
            elif ES[i]:
                new_pos = -1
        elif pos == 1:
            if XL[i] or exit_now:
                new_pos = 0
            if ES[i]:
                new_pos = -1
        else:
            if XS[i] or exit_now:
                new_pos = 0
            if EL[i]:
                new_pos = 1

        if new_pos != pos:
            pnl[i] -= cost * cur_size
            cur -= cost * cur_size
            if pos != 0:
                trade_pnls[tcount] = cur
                tcount += 1
                cur = 0.0
            pos = new_pos
            if pos != 0:
                entry_px = close[i]
                cur_size = size[i]
    if pos != 0:
        trade_pnls[tcount] = cur
        tcount += 1
    return pnl, trade_pnls[:tcount], tcount


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe(pnl: np.ndarray, periods_per_year: float) -> float:
    if pnl.size < 2:
        return 0.0
    sd = pnl.std()
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(pnl.mean() / sd * math.sqrt(periods_per_year))


def _max_drawdown(pnl: np.ndarray) -> float:
    """Max drawdown of the (arithmetic) equity curve, in return units (>=0)."""
    if pnl.size == 0:
        return 0.0
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    return float(dd.max()) if dd.size else 0.0


def compute_metrics(EL, XL, ES, XS, ret, cost, ppy,
                    close=None, atr=None, trail=None,
                    size=None, stop_atr=0.0, tp_atr=0.0) -> dict:
    """
    Full performance panel for one strategy on a (sliced) return series.

    If `trail`, `close` and `atr` are all provided, an ATR trailing stop is
    applied (exit on a `trail`*ATR retrace from the favourable extreme, in
    addition to the signal exits).

    Risk-management overlay (uses `_backtest_risk`): if `size` is given (a
    per-bar position-size multiplier, e.g. from compute_vol_target_size) and/or
    `stop_atr`/`tp_atr` > 0 (ATR stop-loss / take-profit, needs `close`+`atr`),
    the position-sized backtest is used. `size` defaults to all-ones (flat 1x)
    when only stops are requested. Takes precedence over `trail`.

    Returns a dict:
        total_return  – sum of bar P&L (arithmetic), in return units
        max_drawdown  – worst peak-to-trough of the equity curve
        sharpe        – annualized Sharpe
        sortino       – annualized Sortino (downside-only deviation)
        calmar        – annualized return / max drawdown
        win_loss      – avg win / avg loss (abs)
        profit_factor – gross profit / gross loss
        win_rate      – fraction of winning trades
        n_trades      – trade count
    """
    use_risk = (size is not None) or (stop_atr and stop_atr > 0) or (tp_atr and tp_atr > 0)
    if use_risk:
        n = ret.shape[0]
        sz = np.ones(n) if size is None else np.asarray(size, dtype=np.float64)
        cl = close if close is not None else np.zeros(n)
        at = atr if atr is not None else np.zeros(n)
        pnl, trade_pnls, ntr = _backtest_risk(EL, XL, ES, XS, cl, ret, at, sz,
                                              cost, float(stop_atr or 0.0),
                                              float(tp_atr or 0.0))
    elif trail is not None and close is not None and atr is not None:
        pnl, trade_pnls, ntr = _backtest_trades_stop(EL, XL, ES, XS, close, ret,
                                                     atr, cost, trail)
    else:
        pnl, trade_pnls, ntr = _backtest_trades(EL, XL, ES, XS, ret, cost)

    total_return = float(pnl.sum())
    mdd = _max_drawdown(pnl)
    shp = sharpe(pnl, ppy)

    # Sortino: mean / downside deviation
    downside = pnl[pnl < 0]
    dd_std = downside.std() if downside.size else 0.0
    sortino = float(pnl.mean() / dd_std * math.sqrt(ppy)) if dd_std > 0 else float('inf')

    # Calmar: annualized return / max drawdown
    ann_return = float(pnl.mean() * ppy)
    calmar = ann_return / mdd if mdd > 0 else float('inf')

    # Trade-level stats
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    win_loss = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    win_rate = float(wins.size / ntr) if ntr else 0.0

    return {
        'total_return':  total_return,
        'max_drawdown':  mdd,
        'sharpe':        shp,
        'sortino':       sortino,
        'calmar':        calmar,
        'win_loss':      win_loss,
        'profit_factor': profit_factor,
        'win_rate':      win_rate,
        'n_trades':      int(ntr),
    }


def _mean_fold_metrics(EL, XL, ES, XS, ret, folds, cost, ppy) -> dict:
    """
    Compute compute_metrics() on each walk-forward fold and average each metric
    across folds (infinite/NaN values are ignored in the mean).
    """
    per = [compute_metrics(EL[f], XL[f], ES[f], XS[f], ret[f], cost, ppy)
           for f in folds]
    out = {}
    for k in per[0]:
        vals = [d[k] for d in per]
        finite = [v for v in vals if v == v and v not in (float('inf'), float('-inf'))]
        out[k] = (sum(finite) / len(finite)) if finite else float('inf')
    return out


def _format_metrics(m: dict) -> str:
    def f(x):
        return "inf" if x == float('inf') else f"{x:.3f}"
    return ("    return={ret}  maxDD={dd}  Sharpe={sh}  Sortino={so}  "
            "Calmar={ca}  win/loss={wl}  PF={pf}  win%={wr}  trades={nt}").format(
        ret=f(m['total_return']), dd=f(m['max_drawdown']), sh=f(m['sharpe']),
        so=f(m['sortino']), ca=f(m['calmar']), wl=f(m['win_loss']),
        pf=f(m['profit_factor']), wr=f"{m['win_rate']*100:.1f}",
        nt=f"{m['n_trades']:.1f}")


def deflated_sharpe(observed_sr: float, n_trials: int, pnl: np.ndarray,
                    periods_per_year: float) -> float:
    """
    Simplified Deflated Sharpe Ratio (Bailey & López de Prado).
    Returns the probability (0..1) that the strategy's true Sharpe > 0 after
    accounting for `n_trials` independent tries. Higher = more trustworthy.
    """
    if pnl.size < 3:
        return 0.0
    sr = observed_sr / math.sqrt(periods_per_year)            # per-period SR
    # skew / kurtosis of per-bar pnl
    x = pnl - pnl.mean()
    sd = x.std()
    if sd == 0:
        return 0.0
    g1 = (x**3).mean() / sd**3
    g2 = (x**4).mean() / sd**4
    T = pnl.size
    # expected max SR over n_trials (Gumbel approximation), per period
    if n_trials < 2:
        sr0 = 0.0
    else:
        e = 0.5772156649
        z1 = _norm_ppf(1 - 1.0 / n_trials)
        z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = (z1 * (1 - e) + z2 * e)
        sr0 *= (1.0 / math.sqrt(T))                          # scale (rough)
    num = (sr - sr0) * math.sqrt(T - 1)
    den = math.sqrt(1 - g1 * sr + (g2 - 1) / 4.0 * sr**2)
    if den <= 0 or not np.isfinite(den):
        return 0.0
    return float(_norm_cdf(num / den))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam approximation)."""
    if p <= 0:
        return -1e9
    if p >= 1:
        return 1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Funnel orchestration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    'min_trades':      30,      # Stage 0
    'stage1_sharpe':   0.5,     # proxy Sharpe (no cost), in-sample
    'stage2_sharpe':   0.3,     # Sharpe with costs, in-sample
    'stage3_sharpe':   0.2,     # out-of-sample Sharpe with costs
    'wf_fold_sharpe':  0.0,     # Stage 4: min Sharpe EACH walk-forward fold must reach
    'wf_min_pos_frac': 0.6,     # Stage 4: fraction of folds that must clear wf_fold_sharpe
    'dsr_min':         0.90,    # Stage 4: min Deflated Sharpe probability
    'dsr_max_trials':  2000,    # Stage 4: cap on n_trials fed to DSR (effective independent trials)
}


def _evaluate_strategy(strat, trig_cache, n, ret, i_train, sl_train, sl_oos,
                       th, cost, ppy, wf_folds, n_trials, counts):
    """
    Push one strategy through all funnel stages.

    Returns (survivor_dict_or_None, fitness). `fitness` is the in-sample
    Sharpe-with-costs (the GA objective), defined for every strategy that clears
    Stage 0 (-inf only for structural failures).

    `trig_cache` is a _TriggerCache providing per-group trigger arrays (lazily
    memoized, so a combo's AND is computed at most once across the whole run).
    """
    EL, XL, ES, XS = trig_cache.triggers(strat)

    # ---- Stage 0: structural ----
    if int(EL[sl_train].sum() + ES[sl_train].sum()) < th['min_trades']:
        return None, -1e18
    pnl_is, trades_is = _backtest_core(EL[sl_train], XL[sl_train],
                                       ES[sl_train], XS[sl_train], ret[sl_train], 0.0)
    if trades_is < th['min_trades']:
        return None, -1e18
    counts['s0'] += 1
    sr_is = sharpe(pnl_is, ppy)

    # ---- Stage 2 metric (also the GA fitness): Sharpe with costs, in-sample ----
    pnl_c, _ = _backtest_core(EL[sl_train], XL[sl_train],
                              ES[sl_train], XS[sl_train], ret[sl_train], cost)
    sr_cost = sharpe(pnl_c, ppy)
    fitness = sr_cost

    # ---- Stage 1 gate ----
    if sr_is < th['stage1_sharpe']:
        return None, fitness
    counts['s1'] += 1
    # ---- Stage 2 gate ----
    if sr_cost < th['stage2_sharpe']:
        return None, fitness
    counts['s2'] += 1

    # ---- Stage 3: out-of-sample ----
    pnl_oos, trades_oos = _backtest_core(EL[sl_oos], XL[sl_oos],
                                         ES[sl_oos], XS[sl_oos], ret[sl_oos], cost)
    sr_oos = sharpe(pnl_oos, ppy)
    if trades_oos < max(5, th['min_trades'] // 3) or sr_oos < th['stage3_sharpe']:
        return None, fitness
    counts['s3'] += 1

    # ---- Stage 4: walk-forward + Deflated Sharpe ----
    folds = np.array_split(np.arange(i_train), wf_folds)
    fold_bar = th['wf_fold_sharpe']
    pos_folds = 0
    for f in folds:
        pf, _ = _backtest_core(EL[f], XL[f], ES[f], XS[f], ret[f], cost)
        if sharpe(pf, ppy) >= fold_bar:
            pos_folds += 1
    wf_frac = pos_folds / wf_folds
    if wf_frac < th['wf_min_pos_frac']:
        return None, fitness
    # Cap the trial count: the sampled strategies are highly correlated (built
    # from a small set of distinct signals), so the EFFECTIVE number of
    # independent trials is far below the raw count. Capping avoids DSR's
    # multiple-testing penalty over-rejecting everything.
    n_eff = n_trials if th['dsr_max_trials'] is None else min(n_trials, th['dsr_max_trials'])
    dsr = deflated_sharpe(sr_cost, max(2, n_eff), pnl_c, ppy)
    if dsr < th['dsr_min']:
        return None, fitness
    counts['s4'] += 1

    survivor = {
        'strategy':    strat,
        'n_trades':    int(trades_is),
        'sharpe_is':   round(sr_is, 3),
        'sharpe_cost': round(sr_cost, 3),
        'sharpe_oos':  round(sr_oos, 3),
        'wf_pos_frac': round(wf_frac, 2),
        'dsr':         round(dsr, 3),
    }
    return survivor, fitness


# ---- GA operators ----------------------------------------------------------

def _ga_tournament(scored, k, rng):
    best = None
    for _ in range(k):
        cand = scored[rng.randrange(len(scored))]
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1]


def _ga_crossover(pa, pb, rng):
    return {g: (pa[g] if rng.random() < 0.5 else pb[g]) for g in GROUPS}


def _ga_mutate(child, space, rng, rate):
    out = dict(child)
    for g in GROUPS:
        if rng.random() < rate:
            out[g] = space.groups[g][rng.randrange(space.sizes[g])]
    return out


def run_funnel(
    indicator_classes,
    data,
    method: str = 'random',          # 'random' or 'ga'
    # --- stage thresholds (explicit; override DEFAULT_THRESHOLDS) ---
    min_trades: int = None,          # Stage 0: min trades in-sample
    stage1_sharpe: float = None,     # Stage 1: proxy Sharpe (no cost), in-sample
    stage2_sharpe: float = None,     # Stage 2: Sharpe with costs, in-sample
    stage3_sharpe: float = None,     # Stage 3: out-of-sample Sharpe with costs
    wf_fold_sharpe: float = None,    # Stage 4: min Sharpe EACH walk-forward fold must reach
    wf_min_pos_frac: float = None,   # Stage 4: fraction of folds that must clear wf_fold_sharpe
    dsr_min: float = None,           # Stage 4: min Deflated Sharpe probability
    dsr_max_trials: int = None,      # Stage 4: cap n_trials for DSR (effective independent trials)
    thresholds: dict = None,         # optional dict; explicit args above take precedence
    # --- common ---
    cost: float = 0.00007,           # round-trip cost as a fraction (~0.7 pip EURUSD)
    train_frac: float = 0.6,
    oos_frac: float = 0.2,           # remaining (1-train-oos) reserved as final test
    train_end=None,                  # date split: end of train slice (overrides train_frac)
    oos_end=None,                    # date split: end of OOS slice (overrides oos_frac)
    periods_per_year: float = None,
    max_pattern_size: int = 3,
    split_both: bool = True,
    divergence_in_exits_only: bool = False,
    wf_folds: int = 5,
    seed: int = 0,
    show_progress: bool = True,
    keep_top: int = 200,
    max_trigger_cache: int = None,   # LRU cap on memoized combo triggers (None = unbounded)
    signal_cache_path: str = None,   # persist/reuse the computed indicator signals (one-time build)
    print_best: bool = False,        # print each new best (by fitness Sharpe) + metrics
    checkpoint: str = None,          # path to a checkpoint file: skip already-tested, resume
    checkpoint_every: int = None,    # save checkpoint every N new evaluations (GA: also per gen)
    checkpoint_mode: str = 'full',   # 'full' | 'compact' | 'hybrid' (see docstring)
    max_pattern_count: int = None,   # cap samples per pattern (skip once reached; needs collect_pattern_stats)
    pattern_stats_path: str = None,  # periodically export pattern stats CSV during the run
    progress_save_pct: float = 10.0, # export/checkpoint every this-% of search progress
    collect_pattern_stats: bool = False,  # accumulate per-pattern Sharpe for ALL Stage-0 passers
    # --- random sampling ---
    n_samples: int = 100_000,
    unique: bool = True,
    # --- systematic enumeration (method='enumerate') ---
    enum_start: int = 0,             # first strategy index (for parallel range-splitting)
    enum_stop: int = None,           # last+1 strategy index (None = end)
    # --- genetic algorithm ---
    ga_population: int = 500,
    ga_generations: int = 60,
    ga_mutation_rate: float = 0.15,
    ga_elitism: float = 0.05,
    ga_tournament_k: int = 4,
):
    """
    Run the multi-stage funnel.

    method='random' : sample `n_samples` strategies uniformly from the product
                      space and filter through the funnel.
    method='ga'     : evolve a population (`ga_*` params) using in-sample
                      Sharpe-with-costs as fitness; every individual ever
                      evaluated is also pushed through the full funnel, and all
                      strategies that pass become survivors.

    checkpoint : path to a pickle file. If set, prior runs' tested strategies
        are skipped and survivors accumulate across runs (resume).
    checkpoint_mode :
        'full'    – store every tried strategy's fitness. GA fully reuses prior
                    scores. Largest memory/file.
        'compact' – store only membership (no scores). Re-encountered strategies
                    are skipped/deprioritized; GA cannot reuse prior fitnesses.
                    Smallest memory. Best for random surveys.
        'hybrid'  – keep fitness for strategies that pass Stage 0 (the few that
                    matter for GA selection); membership-only for the many
                    structural failures. Good memory/feature balance.
    collect_pattern_stats : if True, accumulate per-pattern in-sample Sharpe for
        EVERY strategy that clears Stage 0 (survivor or not), so you can study
        which pattern combinations perform on average even when they fail the
        survivor gates. Exposed as counts['pattern_stats'] and persisted in the
        checkpoint; analyse with aggregate_pattern_stats().

    Returns (survivors, counts):
      survivors – list of dicts sorted by out-of-sample Sharpe, capped at
                  `keep_top`, each with strategy + metrics.
      counts    – dict of how many strategies passed each stage.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    # Explicit threshold args take precedence over the dict / defaults
    _explicit = {
        'min_trades':      min_trades,
        'stage1_sharpe':   stage1_sharpe,
        'stage2_sharpe':   stage2_sharpe,
        'stage3_sharpe':   stage3_sharpe,
        'wf_fold_sharpe':  wf_fold_sharpe,
        'wf_min_pos_frac': wf_min_pos_frac,
        'dsr_min':         dsr_min,
        'dsr_max_trials':  dsr_max_trials,
    }
    th.update({k: v for k, v in _explicit.items() if v is not None})

    close = np.asarray(data['Close'].values, dtype=np.float64)
    n = close.size
    ret = _returns_from_close(close)

    if periods_per_year is None:
        try:
            sec = np.median(np.diff(data.index.values).astype('timedelta64[s]').astype(float))
            periods_per_year = (365.0 * 24 * 3600) / sec
        except Exception:
            periods_per_year = 35040.0   # 15-min bars fallback
    ppy = periods_per_year

    # Split by date if train_end/oos_end given (converted to row indices),
    # otherwise by fraction. Date split only changes WHERE the boundary lands;
    # indicators are already computed over the full series and just sliced.
    if train_end is not None or oos_end is not None:
        import pandas as pd
        i_train = (int(data.index.searchsorted(pd.Timestamp(train_end)))
                   if train_end is not None else int(n * train_frac))
        i_oos = (int(data.index.searchsorted(pd.Timestamp(oos_end)))
                 if oos_end is not None else int(n * (train_frac + oos_frac)))
    else:
        i_train = int(n * train_frac)
        i_oos = int(n * (train_frac + oos_frac))
    sl_train = slice(0, i_train)
    sl_oos = slice(i_train, i_oos)

    cache = get_signal_cache(indicator_classes, data,
                             cache_path=signal_cache_path, show_progress=show_progress)
    failed = cache.pop('_failed', [])
    if failed and show_progress:
        print(f"[warn] {len(failed)} indicator/signal failures (e.g. {failed[:3]})")

    # Lazy per-combo trigger cache: each group-combo's AND is computed once and
    # reused across all strategies that share it (combos repeat heavily).
    trig_cache = _TriggerCache(cache, n, max_cache=max_trigger_cache)

    space = StrategySpace(indicator_classes, max_pattern_size, split_both,
                          divergence_in_exits_only=divergence_in_exits_only)
    if show_progress:
        print(f"space size = {space.total():,}; method = {method}; "
              f"periods/yr ~ {ppy:,.0f}")

    counts = {'evaluated': 0, 's0': 0, 's1': 0, 's2': 0, 's3': 0, 's4': 0}
    wf_fold_idx = np.array_split(np.arange(i_train), wf_folds)   # for fold metrics

    # Checkpoint: resume from prior runs. fit_cache holds every strategy ever
    # tried (signature -> fitness); survivors_by_sig accumulates all survivors.
    if checkpoint_mode not in ('full', 'compact', 'hybrid'):
        raise ValueError("checkpoint_mode must be 'full', 'compact' or 'hybrid'")
    ckpt = _load_checkpoint(checkpoint)
    fit_cache = ckpt['fit']               # signature -> fitness (kept scores)
    tried_set = ckpt['tried']             # signatures stored membership-only
    survivors_by_sig = ckpt['survivors']
    pattern_acc = ckpt['pattern_stats']   # pattern-sig -> [count, sum, sumsq] (in-sample)
    prior_tried = len(fit_cache) + len(tried_set)
    prior_survivors = len(survivors_by_sig)
    if checkpoint and show_progress and prior_tried:
        print(f"checkpoint[{checkpoint_mode}]: resumed {prior_tried:,} tested "
              f"strategies, {prior_survivors:,} survivors")

    best = {'sharpe_oos': max((s['sharpe_oos'] for s in survivors_by_sig.values()),
                              default=-1e18),
            'printed': 0}
    if print_best and prior_survivors and show_progress:
        print(f"print_best: prior best sharpe_oos = {best['sharpe_oos']:.3f} "
              f"(only survivors beating this will print)")

    def _maybe_save():
        if (checkpoint and checkpoint_every
                and counts['evaluated'] > 0
                and counts['evaluated'] % checkpoint_every == 0):
            _save_checkpoint(checkpoint, fit_cache, tried_set, survivors_by_sig,
                             pattern_acc)

    def _progress_save(frac):
        """Persist checkpoint + export pattern-stats CSV at a progress milestone."""
        if checkpoint:
            _save_checkpoint(checkpoint, fit_cache, tried_set, survivors_by_sig,
                             pattern_acc)
        if pattern_stats_path and pattern_acc:
            try:
                save_pattern_stats(pattern_acc, pattern_stats_path, min_count=1)
            except Exception as exc:                    # pragma: no cover
                print(f"  [progress] pattern-stats export failed: {exc}")
        if show_progress:
            print(f"  [progress {frac:.0%}] {len(pattern_acc):,} patterns, "
                  f"{counts['evaluated']:,} evaluated"
                  + (f" -> {pattern_stats_path}" if pattern_stats_path else ""))

    def get_fit(strat):
        sig = space._signature(strat)
        cached = fit_cache.get(sig)
        if cached is not None:           # already tested with a kept score
            return cached
        if sig in tried_set:             # tried before, score not kept (compact/failure)
            return -1e18                 # skip / deprioritize, don't re-evaluate
        # Per-pattern cap: once a pattern has enough samples, skip new strategies
        # of that pattern (no backtest) -> bounds stats, forces broader coverage.
        psig = None
        if collect_pattern_stats and max_pattern_count:
            psig = tuple(_pattern_combo(strat[g]) for g in GROUPS)
            pa = pattern_acc.get(psig)
            if pa is not None and pa[0] >= max_pattern_count:
                counts['pattern_skipped'] = counts.get('pattern_skipped', 0) + 1
                return -1e18             # pattern full -> skip / deprioritize
        counts['evaluated'] += 1
        n_trials = prior_tried + counts['evaluated']   # cumulative distinct trials
        surv, fit = _evaluate_strategy(strat, trig_cache, n, ret, i_train, sl_train,
                                       sl_oos, th, cost, ppy, wf_folds,
                                       n_trials, counts)
        # Store per checkpoint_mode:
        #   full    -> always keep the fitness (GA reuse for everything)
        #   compact -> membership only (smallest; GA can't reuse scores)
        #   hybrid  -> keep fitness for Stage-0 passers, membership for failures
        if checkpoint_mode == 'full':
            fit_cache[sig] = fit
        elif checkpoint_mode == 'compact':
            tried_set.add(sig)
        else:  # hybrid
            if fit > -1e17:
                fit_cache[sig] = fit
            else:
                tried_set.add(sig)
        # Accumulate per-pattern performance for EVERY Stage-0 passer (survivor
        # or not) — its in-sample Sharpe (the fitness).
        if collect_pattern_stats and fit > -1e17:
            if psig is None:
                psig = tuple(_pattern_combo(strat[g]) for g in GROUPS)
            pa = pattern_acc.get(psig)
            if pa is None:
                pattern_acc[psig] = [1, fit, fit * fit]
            else:
                pa[0] += 1; pa[1] += fit; pa[2] += fit * fit
        if surv is not None:
            survivors_by_sig[sig] = surv
            # New best among ALL survivors (this run + resumed), by OOS Sharpe
            if print_best and surv['sharpe_oos'] > best['sharpe_oos']:
                best['sharpe_oos'] = surv['sharpe_oos']
                best['printed'] += 1
                EL, XL, ES, XS = trig_cache.triggers(strat)
                m = _mean_fold_metrics(EL, XL, ES, XS, ret, wf_fold_idx, cost, ppy)
                print(f"\n*** NEW BEST  sharpe_oos={surv['sharpe_oos']:.3f}  "
                      f"sharpe_cost={surv['sharpe_cost']:.3f}  "
                      f"wf_pos_frac={surv['wf_pos_frac']}  dsr={surv['dsr']}  "
                      f"(new eval #{counts['evaluated']:,}) ***")
                print(describe_strategy(strat))
                print(f"  mean metrics across {len(wf_fold_idx)} folds:")
                print(_format_metrics(m))
        _maybe_save()
        return fit

    # ------------------------------------------------------------------ random
    if method == 'random':
        sampler = space.iter_random(n_samples, seed=seed, unique=unique)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                sampler = tqdm(sampler, total=n_samples, desc="random", unit="strat")
            except ImportError:
                pass
        step = max(1, int(n_samples * progress_save_pct / 100.0))
        processed = 0
        # Only do mid-run periodic saves when periodic pattern export is requested.
        do_progress = bool(pattern_stats_path)
        for strat in sampler:
            get_fit(strat)
            processed += 1
            if do_progress and processed % step == 0:
                _progress_save(processed / n_samples)

    # ---------------------------------------------------------------------- ga
    elif method == 'ga':
        rng = random.Random(seed)
        population = [space.random(rng) for _ in range(ga_population)]
        gens = range(ga_generations)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                gens = tqdm(gens, desc="ga", unit="gen")
            except ImportError:
                pass
        n_elite = max(1, int(ga_population * ga_elitism))
        gen_step = max(1, int(ga_generations * progress_save_pct / 100.0))
        for gi, _gen in enumerate(gens):
            scored = [(get_fit(ind), ind) for ind in population]
            scored.sort(key=lambda t: t[0], reverse=True)
            next_pop = [ind for _, ind in scored[:n_elite]]
            while len(next_pop) < ga_population:
                pa = _ga_tournament(scored, ga_tournament_k, rng)
                pb = _ga_tournament(scored, ga_tournament_k, rng)
                child = _ga_crossover(pa, pb, rng)
                child = _ga_mutate(child, space, rng, ga_mutation_rate)
                next_pop.append(child)
            population = next_pop
            if checkpoint:               # persist after each generation
                _save_checkpoint(checkpoint, fit_cache, tried_set, survivors_by_sig,
                                 pattern_acc)
            if pattern_stats_path and (gi + 1) % gen_step == 0:
                _progress_save((gi + 1) / ga_generations)

    # ----------------------------------------------------------- enumerate
    elif method == 'enumerate':
        # Test EVERY strategy in [enum_start, enum_stop) exactly once. Each index
        # maps bijectively to a strategy (mixed-radix over the 4 per-group pools),
        # so workers can process disjoint ranges in parallel.
        total = space.total()
        stop = total if enum_stop is None else min(enum_stop, total)
        sizes = [space.sizes[g] for g in GROUPS]

        def _strat_at(idx):
            s = {}
            for g, sz in zip(GROUPS, sizes):
                s[g] = space.groups[g][idx % sz]
                idx //= sz
            return s

        rng_iter = range(enum_start, stop)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                rng_iter = tqdm(rng_iter, total=stop - enum_start,
                                desc="enumerate", unit="strat")
            except ImportError:
                pass
        step = max(1, int((stop - enum_start) * progress_save_pct / 100.0))
        done = 0
        for idx in rng_iter:
            get_fit(_strat_at(idx))
            done += 1
            if (pattern_stats_path or checkpoint) and done % step == 0:
                _progress_save(done / max(1, stop - enum_start))
    else:
        raise ValueError("method must be 'random', 'ga' or 'enumerate'")

    if checkpoint:                       # final save
        _save_checkpoint(checkpoint, fit_cache, tried_set, survivors_by_sig,
                         pattern_acc)

    survivors = sorted(survivors_by_sig.values(),
                       key=lambda d: d['sharpe_oos'], reverse=True)[:keep_top]

    total_tried = len(fit_cache) + len(tried_set)
    counts['trigger_hit_rate'] = round(trig_cache.hit_rate, 4)
    counts['trigger_cached'] = len(trig_cache.store)
    counts['total_tried'] = total_tried
    counts['total_survivors'] = len(survivors_by_sig)
    if collect_pattern_stats:
        counts['pattern_stats'] = pattern_acc

    if show_progress:
        print("\nFunnel survival:")
        print(f"  evaluated (new)    : {counts['evaluated']:,}")
        print(f"  total tried (all)  : {total_tried:,}")
        print(f"  Stage0 structural  : {counts['s0']:,}")
        print(f"  Stage1 proxy       : {counts['s1']:,}")
        print(f"  Stage2 +costs      : {counts['s2']:,}")
        print(f"  Stage3 OOS         : {counts['s3']:,}")
        print(f"  Stage4 WF+DSR      : {counts['s4']:,}")
        print(f"  survivors (total)  : {len(survivors_by_sig):,} "
              f"(+{len(survivors_by_sig) - prior_survivors:,} this run)")
        print(f"  kept (top)         : {len(survivors):,}")
        print(f"  trigger cache      : {trig_cache.hits:,} hits / "
              f"{trig_cache.misses:,} miss ({trig_cache.hit_rate:.1%} hit, "
              f"{len(trig_cache.store):,} cached)")
        if print_best and best['printed'] == 0:
            if counts['s4'] == 0:
                print("  [print_best] nothing printed: 0 strategies passed all "
                      "stages — loosen the gates (wf_fold_sharpe / wf_min_pos_frac "
                      "/ dsr_min / stageN_sharpe).")
            else:
                print(f"  [print_best] nothing printed: survivors found but none "
                      f"beat the prior best sharpe_oos={best['sharpe_oos']:.3f} "
                      f"(use a fresh checkpoint or check survivors[0]).")

    return survivors, counts


# ---------------------------------------------------------------------------
# Checkpoint (resume across runs — skip already-evaluated strategies)
# ---------------------------------------------------------------------------

def _load_checkpoint(path):
    """
    Load a checkpoint pickle. Returns dict with:
        'fit'           : {strategy_signature: fitness}   (strategies with a kept score)
        'tried'         : set of signatures tried but stored membership-only
        'survivors'     : {strategy_signature: survivor_dict}
        'pattern_stats' : {pattern_signature: [count, sum, sumsq]}  (all Stage-0 passers)
    Missing/absent file -> empty state.
    """
    import os
    import pickle
    if path and os.path.exists(path):
        with open(path, 'rb') as fh:
            ckpt = pickle.load(fh)
        ckpt.setdefault('fit', {})
        ckpt.setdefault('tried', set())
        ckpt.setdefault('survivors', {})
        ckpt.setdefault('pattern_stats', {})
        return ckpt
    return {'fit': {}, 'tried': set(), 'survivors': {}, 'pattern_stats': {}}


def _save_checkpoint(path, fit_cache, tried_set, survivors_by_sig, pattern_stats=None):
    """Atomically persist fitness map + tried set + survivors + pattern stats."""
    import os
    import pickle
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump({'fit': fit_cache, 'tried': tried_set,
                     'survivors': survivors_by_sig,
                     'pattern_stats': pattern_stats or {}},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)   # atomic on the same filesystem


def split_enum_ranges(total, n_workers):
    """
    Split [0, total) into `n_workers` near-equal contiguous ranges.
    Returns a list of (start, stop) tuples — one per parallel worker.
    """
    base = total // n_workers
    rem = total % n_workers
    ranges = []
    start = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def merge_checkpoints(paths, out_path=None):
    """
    Merge several worker checkpoints into one (for parallel runs).

    Combines:
        fit          : dict union
        tried        : set union
        survivors    : dict union (keyed by signature)
        pattern_stats: per-key [count, sum, sumsq] summed (exact merge)

    If `out_path` is given, the merged checkpoint is written there.
    Returns the merged checkpoint dict.
    """
    merged = {'fit': {}, 'tried': set(), 'survivors': {}, 'pattern_stats': {}}
    for p in paths:
        ck = _load_checkpoint(p)
        merged['fit'].update(ck['fit'])
        merged['tried'] |= ck['tried']
        merged['survivors'].update(ck['survivors'])
        for sig, (n, s, ss) in ck.get('pattern_stats', {}).items():
            acc = merged['pattern_stats'].get(sig)
            if acc is None:
                merged['pattern_stats'][sig] = [n, s, ss]
            else:
                acc[0] += n; acc[1] += s; acc[2] += ss
    if out_path:
        _save_checkpoint(out_path, merged['fit'], merged['tried'],
                         merged['survivors'], merged['pattern_stats'])
    return merged


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_survivors(survivors, filepath, fmt: str = 'json', indent: int = 2) -> int:
    """
    Save funnel survivors to disk.

    fmt='json' : full structure (strategy components + all metrics).
    fmt='csv'  : one row per survivor, signal groups flattened to strings.

    Returns the number of survivors written.
    """
    if fmt == 'json':
        import json
        payload = {
            'count': len(survivors),
            'survivors': survivors,   # already JSON-friendly
        }
        with open(filepath, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=indent)

    elif fmt == 'csv':
        import csv

        def grp(strat, g):
            return ' AND '.join(
                f"{c['indicator']}.{c['method'] if isinstance(c['method'], str) else '/'.join(c['method'])}"
                f"[{c.get('tested_as', c['direction'])}]"
                for c in strat[g]
            )

        with open(filepath, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['rank', 'sharpe_oos', 'sharpe_cost', 'sharpe_is',
                        'n_trades', 'wf_pos_frac', 'dsr',
                        'entry_long', 'exit_long', 'entry_short', 'exit_short'])
            for i, s in enumerate(survivors, 1):
                st = s['strategy']
                w.writerow([i, s['sharpe_oos'], s['sharpe_cost'], s['sharpe_is'],
                            s['n_trades'], s['wf_pos_frac'], s['dsr'],
                            grp(st, 'entry_long'), grp(st, 'exit_long'),
                            grp(st, 'entry_short'), grp(st, 'exit_short')])
    else:
        raise ValueError("fmt must be 'json' or 'csv'")

    return len(survivors)


def load_survivors(filepath):
    """Load survivors saved as JSON (returns the list)."""
    import json
    with open(filepath, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data['survivors'] if isinstance(data, dict) and 'survivors' in data else data


# ---------------------------------------------------------------------------
# Pretty-print a strategy
# ---------------------------------------------------------------------------

def describe_strategy(strat) -> str:
    lines = []
    for g in GROUPS:
        parts = []
        for c in strat[g]:
            m = c['method'] if isinstance(c['method'], str) else '/'.join(c['method'])
            tag = c.get('tested_as', c['direction'])
            parts.append(f"{c['indicator']}.{m}[{tag}]")
        lines.append(f"  {g:12}: " + " AND ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: load OHLC and run end-to-end from a CSV
# ---------------------------------------------------------------------------

def load_ohlc(path, date_col: str = "Date"):
    """
    Load an OHLCV CSV into a datetime-indexed DataFrame.

    Expects columns: Date, Open, High, Low, Close, Volume (Volume optional).
    """
    import pandas as pd
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df


def run_on_csv(path, indicator_classes,
               save_json: str = None,
               save_csv: str = None,
               date_col: str = "Date",
               **run_kwargs):
    """
    One-call pipeline: load a CSV, run the funnel, optionally save survivors.

    Parameters
    ----------
    path              : CSV path, e.g. "all_data_EUR_USD.csv"
    indicator_classes : list of indicator classes
    save_json/save_csv: optional output paths for survivors
    **run_kwargs      : any run_funnel() argument (method, n_samples, ga_*,
                        thresholds, wf_fold_sharpe, keep_top, cost, ...)

    Returns (survivors, counts) — the same as run_funnel.
    """
    data = load_ohlc(path, date_col=date_col)
    if run_kwargs.get('show_progress', True):
        print(f"loaded {path}: {data.shape[0]:,} rows "
              f"({data.index.min()} -> {data.index.max()})")

    survivors, counts = run_funnel(indicator_classes, data, **run_kwargs)

    if save_json:
        save_survivors(survivors, save_json, fmt='json')
    if save_csv:
        save_survivors(survivors, save_csv, fmt='csv')
    return survivors, counts



# ---------------------------------------------------------------------------
# Pattern aggregation of SURVIVORS
#
# A "pattern" is the slot template a group was built from, e.g.
#   momentum:L:cont + momentum:L:disc
# It carries category + direction + signal_type per slot. 'both'-direction slots
# are reported as 'B' (their original template direction), not the long/short
# split. This aggregates the survivors produced by run_funnel/save_survivors to
# show which pattern combinations the surviving strategies share.
# ---------------------------------------------------------------------------

_DIR_ABBR = {'long': 'L', 'short': 'S', 'both': 'B'}
_TYPE_ABBR = {'continuous': 'cont', 'discrete': 'disc', 'regime': 'reg'}


def _slot_label(category, direction, signal_type) -> str:
    return f"{category}:{_DIR_ABBR.get(direction, direction)}:{_TYPE_ABBR.get(signal_type, signal_type)}"


def _pattern_combo(comps) -> str:
    """
    Pattern template of one signal group as a readable string, e.g.
    'momentum:L:cont + momentum:L:disc'. 'both' slots keep their original
    template direction (via base_direction).
    """
    slots = []
    for c in comps:
        direction = c.get('base_direction', c['direction'])   # restore 'both'
        slots.append(_slot_label(c['category'], direction, c['signal_type']))
    return ' + '.join(sorted(slots))


def aggregate_patterns(survivors, groups=GROUPS, metric: str = 'sharpe_oos',
                       min_count: int = 1, top: int = None):
    """
    Aggregate survivor strategies by their PATTERN combination.

    Parameters
    ----------
    survivors : a list of survivor dicts, OR a path to a survivors.json file
                (as written by save_survivors). Each survivor has a 'strategy'
                and metrics like 'sharpe_oos', 'sharpe_cost', 'sharpe_is',
                'n_trades', 'dsr'.
    groups    : which signal groups form the pattern combination key.
                Defaults to all four. Use e.g. ('entry_long','exit_long') to see
                only the long-side entry/exit pattern recipes.
    metric    : which survivor metric to average (default 'sharpe_oos').
    min_count : drop combinations seen fewer than this many times.
    top       : if set, return only the top-N rows.

    Returns
    -------
    pandas DataFrame: one row per pattern combination, with a column per group
    (the pattern string), plus [count, mean_<metric>, std_<metric>], sorted by
    mean_<metric> descending.
    """
    import pandas as pd

    if isinstance(survivors, str):           # a file path -> load it
        survivors = load_survivors(survivors)

    groups = tuple(groups)
    agg = {}   # combo_key -> [count, sum, sumsq]
    for s in survivors:
        strat = s['strategy']
        val = s.get(metric)
        if val is None:
            continue
        key = tuple(_pattern_combo(strat[g]) for g in groups)
        a = agg.setdefault(key, [0, 0.0, 0.0])
        a[0] += 1; a[1] += val; a[2] += val * val

    rows = []
    for key, (n, ssum, ssq) in agg.items():
        if n < min_count:
            continue
        mean = ssum / n
        std = max(0.0, ssq / n - mean * mean) ** 0.5
        rows.append((*key, n, mean, std))

    cols = list(groups) + ['count', f'mean_{metric}', f'std_{metric}']
    df = pd.DataFrame(rows, columns=cols).sort_values(
        f'mean_{metric}', ascending=False).reset_index(drop=True)
    return df.head(top) if top else df


def aggregate_pattern_stats(pattern_stats, groups=GROUPS, min_count: int = 1,
                            top: int = None):
    """
    Aggregate the per-pattern performance accumulator collected during a run with
    collect_pattern_stats=True (covers EVERY Stage-0 passer — survivor or not),
    so you can see how each pattern combination performs on average, not only the
    ones that cleared the survivor gates.

    Parameters
    ----------
    pattern_stats : the dict from counts['pattern_stats'] (or a loaded checkpoint's
                    'pattern_stats'). Maps a 4-tuple pattern signature (order =
                    GROUPS) -> [count, sum_sharpe, sumsq_sharpe] of IN-SAMPLE Sharpe.
    groups        : which signal groups form the aggregation key (default all four;
                    use e.g. ('entry_long','exit_long') for the long-side recipe).
    min_count     : drop combinations seen fewer than this many times.
    top           : if set, return only the top-N rows.

    Returns
    -------
    pandas DataFrame: a column per group + [count, mean_sharpe, std_sharpe],
    sorted by mean_sharpe descending.
    """
    import pandas as pd

    groups = tuple(groups)
    idxs = [GROUPS.index(g) for g in groups]
    agg = {}
    for sig, (n, ssum, ssq) in pattern_stats.items():
        key = tuple(sig[i] for i in idxs)
        a = agg.get(key)
        if a is None:
            agg[key] = [n, ssum, ssq]
        else:
            a[0] += n; a[1] += ssum; a[2] += ssq

    rows = []
    for key, (n, ssum, ssq) in agg.items():
        if n < min_count:
            continue
        mean = ssum / n
        std = max(0.0, ssq / n - mean * mean) ** 0.5
        rows.append((*key, n, mean, std))

    cols = list(groups) + ['count', 'mean_sharpe', 'std_sharpe']
    df = pd.DataFrame(rows, columns=cols).sort_values(
        'mean_sharpe', ascending=False).reset_index(drop=True)
    return df.head(top) if top else df


def save_pattern_stats(pattern_stats, filepath, groups=GROUPS, min_count: int = 1,
                       top: int = None, fmt: str = None):
    """
    Aggregate the per-pattern accumulator (see aggregate_pattern_stats) and save
    it as a table.

    filepath : output path. Format is inferred from the extension unless `fmt`
               is given: '.csv' -> CSV, '.parquet' -> Parquet, else CSV.
    groups   : aggregation key (default = whole-strategy 4-group combination).
    Returns the DataFrame that was written.
    """
    import os
    df = aggregate_pattern_stats(pattern_stats, groups=groups,
                                 min_count=min_count, top=top)
    ext = (fmt or os.path.splitext(filepath)[1].lstrip('.')).lower()
    if ext == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    return df


def export_pattern_stats_from_checkpoint(checkpoint_path, csv_path, groups=GROUPS,
                                         min_count: int = 1, top: int = None):
    """
    Read the (accumulated) pattern stats straight from a checkpoint pickle and
    write the aggregated table — without running the funnel. Useful to re-export
    the cumulative survey after several resumed runs.

    Returns the DataFrame written.
    """
    ckpt = _load_checkpoint(checkpoint_path)
    stats = ckpt.get('pattern_stats', {})
    if not stats:
        print(f"[warn] no pattern_stats in {checkpoint_path} "
              f"(was it run with collect_pattern_stats=True?)")
    return save_pattern_stats(stats, csv_path, groups=groups,
                              min_count=min_count, top=top)
