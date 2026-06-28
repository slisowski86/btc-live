"""
Protected cross-confirmed crypto strategy - the single source of truth.

This is the deployable strategy: the cross-confirmed basket (validated on BTC + ETH) with
vol-target sizing, a TREND-FILTER crash defense, and settable leverage. Both the live
paper trader and any backtest/replay should import `target_exposure` from here so there is
exactly one definition of the strategy.

Verified-best config (see protected_replay.ipynb):
  - vol-target sizing at TARGET_ANN_VOL
  - TREND FILTER: flatten NET-LONG exposure when price < MA_WIN-bar SMA (keeps shorts).
    Causal, robust across MA 150-700 and across BTC + ETH. Turned 10.2025 from a loss to a gain.
  - leverage with a hard cap

`target_exposure(df)` -> signed exposure (fraction of equity) for the LATEST closed bar of df.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd

import strategy_funnel as sf
from strategy_funnel import strategy_triggers
from indicators_loader import all_classes

# ----------------------------- config -----------------------------
BASKET_FILE    = "btc_basket_crossconfirmed.json"
TARGET_ANN_VOL = 0.12      # vol-target risk dial
LEVERAGE       = 1.0       # settable leverage (1 = unlevered; start here). 2-3x prudent ceiling.
MAX_LEVERAGE   = 2.0       # hard safety cap on exposure (after LEVERAGE)
USE_TREND      = True      # crash defense: flatten net-long when price < MA
MA_WIN         = 300       # trend MA window in bars (200-400 all robust; 300 = balanced)
REGIME_FLOOR   = 0.0       # exposure kept when risk-off (0 = flatten; 0.3 = reduce to 30%)
TREND_SYMMETRIC = True   # also flatten net-SHORT in an uptrend (defends short-squeeze drawdowns)
PPY            = 8760.0


def load_basket(path: str = None) -> list:
    return json.load(open(path or BASKET_FILE))


def _trend_gate(close, net, sl, symmetric):
    """
    Per-bar exposure gate from the trend filter (causal trailing MA).
    Always flattens net-LONG in a downtrend (price < MA). If `symmetric`, also flattens
    net-SHORT in an uptrend (price > MA) - defends against short squeezes in bull runs.
    Returns a multiplier array (REGIME_FLOOR where gated, else 1.0).
    """
    import numpy as np, pandas as pd
    sma = pd.Series(close).rolling(MA_WIN, min_periods=MA_WIN // 2).mean().values[sl]
    c = close[sl]
    gate = np.ones(net.shape[0])
    gate = np.where((c < sma) & (net > 0), REGIME_FLOOR, gate)          # downtrend -> cut longs
    if symmetric:
        gate = np.where((c > sma) & (net < 0), REGIME_FLOOR, gate)      # uptrend -> cut shorts
    return gate


def _position_now(EL, XL, ES, XS) -> int:
    """Run the long/short state machine; return the LAST bar's position (-1/0/+1)."""
    p = 0
    for i in range(EL.shape[0]):
        if p == 0:
            if EL[i]: p = 1
            elif ES[i]: p = -1
        elif p == 1:
            if XL[i]: p = 0
            if ES[i]: p = -1
        else:
            if XS[i]: p = 0
            if EL[i]: p = 1
    return p


def _position_series(EL, XL, ES, XS):
    """Full long/short position series (-1/0/+1) over the window."""
    n = EL.shape[0]
    pos = np.zeros(n, dtype=np.int8); p = 0
    for i in range(n):
        pos[i] = p
        if p == 0:
            if EL[i]: p = 1
            elif ES[i]: p = -1
        elif p == 1:
            if XL[i]: p = 0
            if ES[i]: p = -1
        else:
            if XS[i]: p = 0
            if EL[i]: p = 1
    return pos


def protected_pnl_series(df, start_date=None, end_date=None, cost: float = 0.0015,
                         basket: list = None, symmetric: bool = None):
    """
    Per-bar P&L of the PROTECTED basket (vol-target sizing + trend filter) at 1x leverage,
    over [start_date, end_date]. Indicators computed on the full df (warm-up preserved); only
    the evaluation window is sliced. Returns (pnl, net_position, dates).

    Historical counterpart of target_exposure() - same basket, sizing and trend filter - so
    simulators and the live trader share one definition of the strategy.
    """
    basket = basket or load_basket()
    n = len(df)
    close = df["Close"].values.astype(float)
    ret = sf._returns_from_close(close)
    atr = sf.compute_atr(df["High"].values, df["Low"].values, close, 14)
    size = sf.compute_vol_target_size(close, target_ann_vol=TARGET_ANN_VOL, periods_per_year=PPY)
    cache = sf.build_signal_cache(all_classes, df, show_progress=False); cache.pop("_failed", None)
    r0 = int(df.index.searchsorted(pd.Timestamp(start_date))) if start_date else 0
    r1 = int(df.index.searchsorted(pd.Timestamp(end_date))) if end_date else n
    sl = slice(r0, r1)
    pnls, poss = [], []
    for s in basket:
        EL, XL, ES, XS = strategy_triggers(s, cache, n)
        p, _, _ = sf._backtest_risk(EL[sl], XL[sl], ES[sl], XS[sl], close[sl], ret[sl],
                                    atr[sl], size[sl], cost, 0.0, 0.0)
        pnls.append(p); poss.append(_position_series(EL, XL, ES, XS)[sl])
    base = np.mean(pnls, axis=0)
    net = np.mean(np.array(poss), axis=0)
    if USE_TREND:
        sym = TREND_SYMMETRIC if symmetric is None else symmetric
        base = base * _trend_gate(close, net, sl, sym)
    return base, net, df.index[sl]


def target_exposure(df, basket: list = None):
    """
    Signed target exposure (fraction of equity) for the LATEST bar of `df`.

    `df` is an OHLCV DataFrame with enough lead-in for indicator + MA warm-up
    (>= MA_WIN + a few hundred bars recommended). Returns (exposure, detail).

    Pipeline: net basket position -> vol-target size -> trend filter (flatten net-long in a
    downtrend) -> * LEVERAGE -> clip to +/-MAX_LEVERAGE.
    """
    basket = basket or load_basket()
    n = len(df)
    close = df["Close"].values.astype(float)

    used = {c["indicator"] for s in basket for g in s for c in s[g]}
    classes = [c for c in all_classes if c.__name__ in used]
    cache = sf.build_signal_cache(classes, df, show_progress=False)
    cache.pop("_failed", None)

    positions = []
    for s in basket:
        EL, XL, ES, XS = strategy_triggers(s, cache, n)
        positions.append(_position_now(EL, XL, ES, XS))
    raw = float(np.mean(positions))                                   # net direction in [-1, 1]

    vol_size = float(sf.compute_vol_target_size(
        close, target_ann_vol=TARGET_ANN_VOL, periods_per_year=PPY)[-1])

    # trend filter: flatten net-long in a downtrend; if symmetric, also flatten net-short in
    # an uptrend (defends short squeezes). Causal trailing MA.
    gate, risk_off, sma = 1.0, False, float("nan")
    if USE_TREND:
        sma = float(pd.Series(close).rolling(MA_WIN, min_periods=MA_WIN // 2).mean().values[-1])
        down = bool(close[-1] < sma)
        risk_off = down
        if down and raw > 0:
            gate = REGIME_FLOOR                                  # downtrend -> cut long
        elif TREND_SYMMETRIC and (not down) and raw < 0:
            gate = REGIME_FLOOR                                  # uptrend -> cut short
            risk_off = True

    exposure = raw * vol_size * LEVERAGE * gate
    exposure = float(np.clip(exposure, -MAX_LEVERAGE, MAX_LEVERAGE))

    detail = {
        "positions": positions, "raw": round(raw, 3), "vol_size": round(vol_size, 3),
        "sma": round(sma, 2) if sma == sma else None, "risk_off": risk_off, "gate": gate,
        "n_long": positions.count(1), "n_short": positions.count(-1), "n_flat": positions.count(0),
    }
    return exposure, detail
