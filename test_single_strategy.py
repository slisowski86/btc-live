"""
Backtest a SINGLE strategy (e.g. one survivor) over a date range, with the full
metric panel for the strategy and for buy-and-hold, an indicators DataFrame, and
a cumulative-return plot.

Usage (notebook)
----------------
    from strategy_funnel import load_survivors
    from test_single_strategy import backtest_single_strategy

    survivors = load_survivors("survivors_EURUSD.json")
    result = backtest_single_strategy(
        survivors[0], all_classes, data="all_data_EUR_USD.csv",
        start_date="2021-01-01", end_date="2023-01-01",
        cost=0.00007, plot=True,
    )
    print(result['metrics'])        # strategy vs buy_hold, all metrics
    result['indicators'].head()     # df with every signal in the strategy
"""
from __future__ import annotations

import numpy as np

import strategy_funnel as sf
from strategy_funnel import (GROUPS, build_signal_cache, strategy_triggers,
                             compute_metrics, _returns_from_close, describe_strategy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_strategy(strategy):
    """Accept either a survivor dict (has 'strategy') or a raw strategy dict."""
    if isinstance(strategy, dict) and 'strategy' in strategy and 'entry_long' not in strategy:
        return strategy['strategy']
    return strategy


def _strategy_indicator_names(strat) -> list:
    """Distinct indicator names used anywhere in the strategy."""
    names = []
    for g in GROUPS:
        for c in strat[g]:
            if c['indicator'] not in names:
                names.append(c['indicator'])
    return names


def _positions(EL, XL, ES, XS) -> np.ndarray:
    """Reconstruct the position series (-1/0/+1) from the trigger arrays."""
    n = EL.shape[0]
    pos = np.zeros(n, dtype=np.int8)
    p = 0
    for i in range(n):
        pos[i] = p
        if p == 0:
            if EL[i]:
                p = 1
            elif ES[i]:
                p = -1
        elif p == 1:
            if XL[i]:
                p = 0
            if ES[i]:
                p = -1
        else:
            if XS[i]:
                p = 0
            if EL[i]:
                p = 1
    return pos


def _buy_hold_triggers(n):
    """Always-long: enter at bar 0, never exit."""
    EL = np.zeros(n, dtype=bool); EL[0] = True
    XL = np.zeros(n, dtype=bool)
    ES = np.zeros(n, dtype=bool)
    XS = np.zeros(n, dtype=bool)
    return EL, XL, ES, XS


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def backtest_single_strategy(strategy, indicator_classes, data,
                             start_date=None, end_date=None,
                             cost: float = 0.00007,
                             periods_per_year: float = None,
                             trail_atr: float = None,
                             atr_period: int = 14,
                             plot: bool = True,
                             title: str = None):
    """
    Backtest one strategy over [start_date, end_date] and compare to buy & hold.

    Parameters
    ----------
    strategy          : a survivor dict or a raw strategy dict (4 signal groups).
    indicator_classes : list of indicator classes (used to compute the signals).
    data              : a datetime-indexed OHLCV DataFrame, OR a CSV path.
    start_date/end_date : optional slice bounds (str or Timestamp). Inclusive.
    cost              : round-trip cost as a fraction (per position change).
    periods_per_year  : Sharpe annualization; inferred from bar spacing if None.
    trail_atr         : if set, apply an ATR trailing stop of this many ATRs
                        (e.g. 2.0). None = no stop (signal exits only).
    atr_period        : ATR lookback (default 14).
    plot              : draw the cumulative-return comparison.

    Returns
    -------
    dict with:
        'metrics'    : pandas DataFrame (rows = metric, cols = strategy/buy_hold)
        'indicators' : pandas DataFrame (price + every signal in the strategy
                       + triggers + position + cumulative returns)
        'strategy_pnl', 'bh_pnl' : per-bar P&L arrays
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    strat = _as_strategy(strategy)

    # ---- load / slice data ----
    if isinstance(data, str):
        df = sf.load_ohlc(data)
    else:
        df = data
    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]
    if len(df) < 5:
        raise ValueError("data slice too small to backtest")

    n = len(df)
    close = np.asarray(df['Close'].values, dtype=np.float64)
    ret = _returns_from_close(close)

    # ATR (only needed for the trailing stop)
    atr = None
    if trail_atr is not None:
        atr = sf.compute_atr(df['High'].values, df['Low'].values, close, atr_period)

    if periods_per_year is None:
        try:
            sec = np.median(np.diff(df.index.values).astype('timedelta64[s]').astype(float))
            periods_per_year = (365.0 * 24 * 3600) / sec
        except Exception:
            periods_per_year = 35040.0
    ppy = periods_per_year

    # ---- signal cache: only the indicators this strategy uses ----
    used = set(_strategy_indicator_names(strat))
    classes = [c for c in indicator_classes if c.__name__ in used]
    missing = used - {c.__name__ for c in classes}
    if missing:
        raise ValueError(f"indicator classes not provided for: {sorted(missing)}")
    cache = build_signal_cache(classes, df, show_progress=False)
    failed = cache.pop('_failed', [])
    if failed:
        print(f"[warn] signal failures: {failed}")

    # ---- triggers + backtests ----
    EL, XL, ES, XS = strategy_triggers(strat, cache, n)
    bEL, bXL, bES, bXS = _buy_hold_triggers(n)

    if trail_atr is not None:
        strat_pnl, _, _ = sf._backtest_trades_stop(EL, XL, ES, XS, close, ret, atr, cost, trail_atr)
    else:
        strat_pnl, _, _ = sf._backtest_trades(EL, XL, ES, XS, ret, cost)
    bh_pnl, _, _ = sf._backtest_trades(bEL, bXL, bES, bXS, ret, cost)

    m_strat = compute_metrics(EL, XL, ES, XS, ret, cost, ppy,
                              close=close, atr=atr, trail=trail_atr)
    m_bh = compute_metrics(bEL, bXL, bES, bXS, ret, cost, ppy)

    metrics = pd.DataFrame({'strategy': m_strat, 'buy_hold': m_bh})

    # ---- indicators dataframe ----
    out = pd.DataFrame(index=df.index)
    out['Close'] = close
    for g in GROUPS:
        for c in strat[g]:
            method = c['method'] if isinstance(c['method'], str) else c['method'][0]
            arr = cache.get((c['indicator'], method))
            col = f"{g}:{c['indicator']}.{method}"
            out[col] = arr if arr is not None else np.nan
    out['entry_long'] = EL
    out['exit_long'] = XL
    out['entry_short'] = ES
    out['exit_short'] = XS
    out['position'] = _positions(EL, XL, ES, XS)
    out['strat_cumret'] = np.cumsum(strat_pnl)
    out['bh_cumret'] = np.cumsum(bh_pnl)

    # ---- plot ----
    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, out['strat_cumret'], label='Strategy', linewidth=1.5)
        ax.plot(df.index, out['bh_cumret'], label='Buy & Hold',
                linewidth=1.2, alpha=0.8)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_title(title or "Cumulative return: strategy vs buy & hold "
                     f"({df.index[0].date()} → {df.index[-1].date()})")
        ax.set_ylabel("cumulative return (arithmetic)")
        ax.set_xlabel("date")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

    return {
        'metrics': metrics,
        'indicators': out,
        'strategy_pnl': strat_pnl,
        'bh_pnl': bh_pnl,
        'periods_per_year': ppy,
        'strategy': strat,
    }


def backtest_from_json(json_path, rank, indicator_classes, data, **kwargs):
    """
    Convenience: load survivors from JSON and backtest the `rank`-th one
    (0 = best by saved order). Extra kwargs go to backtest_single_strategy.
    """
    survivors = sf.load_survivors(json_path)
    if rank >= len(survivors):
        raise IndexError(f"only {len(survivors)} survivors; rank {rank} out of range")
    return backtest_single_strategy(survivors[rank], indicator_classes, data, **kwargs)


def validate_on_slice(survivors, indicator_classes, data,
                      start_date=None, end_date=None,
                      cost: float = 0.00007, periods_per_year: float = None,
                      trail_atr: float = None, atr_period: int = 14,
                      metric: str = 'sharpe', show_progress: bool = True):
    """
    Fast BATCH validation of many survivors on a data slice (e.g. fresh
    2023-2026 data). Builds the signal cache ONCE over the FULL series (so
    indicators get their warmup), then evaluates the metric only on the window
    [start_date, end_date] — far faster than per-survivor calls.

    IMPORTANT: indicators are computed on the full `data` (warmup preserved);
    start_date/end_date only restrict the EVALUATION window, not the data the
    indicators see. Pass a series with enough lead-in before start_date.

    Returns a list of (survivor, metric_value) in input order.
    """
    import pandas as pd
    if isinstance(data, str):
        df = sf.load_ohlc(data)
    else:
        df = data

    n = len(df)
    close = np.asarray(df['Close'].values, dtype=np.float64)
    ret = _returns_from_close(close)
    if periods_per_year is None:
        try:
            sec = np.median(np.diff(df.index.values).astype('timedelta64[s]').astype(float))
            periods_per_year = (365.0 * 24 * 3600) / sec
        except Exception:
            periods_per_year = 35040.0
    ppy = periods_per_year

    atr = None
    if trail_atr is not None:
        atr = sf.compute_atr(df['High'].values, df['Low'].values, close, atr_period)

    # build the cache on the FULL series (indicators warm up properly)
    cache = build_signal_cache(indicator_classes, df, show_progress=show_progress)
    cache.pop('_failed', None)

    # evaluation window (indices); indicators still computed on the full series
    lo = int(df.index.searchsorted(pd.Timestamp(start_date))) if start_date is not None else 0
    hi = int(df.index.searchsorted(pd.Timestamp(end_date))) if end_date is not None else n
    sl = slice(lo, hi)
    atr_w = atr[sl] if atr is not None else None

    iterator = survivors
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(survivors, total=len(survivors), desc="validate", unit="strat")
        except ImportError:
            pass

    out = []
    for s in iterator:
        strat = _as_strategy(s)
        EL, XL, ES, XS = strategy_triggers(strat, cache, n)
        m = compute_metrics(EL[sl], XL[sl], ES[sl], XS[sl], ret[sl], cost, ppy,
                             close=close[sl], atr=atr_w, trail=trail_atr)
        out.append((s, m[metric]))
    return out


def validate_gauntlet(survivors, indicator_classes, data,
                      reserved_start,
                      cost: float = 0.0015,
                      cost_stress=(0.0008, 0.0015, 0.0025),
                      periods_per_year: float = None,
                      target_vol: float = 0.20, vol_lookback: int = 96,
                      atr_period: int = 14,
                      min_sharpe: float = 0.3,
                      min_reserved_trades: int = 100,
                      require_beat_bh: bool = True,
                      print_best_every: int = None,
                      show_progress: bool = True):
    """
    Run every survivor through the full robustness gauntlet on data it was
    NEVER selected on (the reserved tail, [reserved_start : end]), plus
    consistency / cost / sizing stress. Returns a pandas DataFrame, one row per
    survivor, sorted by reserved Sharpe, with a boolean `pass` column.

    The six tests (all must hold to pass):
      1. reserved Sharpe >= `min_sharpe`           (works on untouched data)
      2. cost-stress Sharpe at the HIGHEST cost > 0 (edge survives real costs)
      3. vol-target-sized Sharpe > 0               (doesn't sign-flip under sizing)
      4. split-test: BOTH halves of the reserved tail > 0 (temporal consistency)
      5. n_trades on the reserved tail >= `min_reserved_trades`
         (enough trades that the Sharpe is statistically meaningful, not 3-4
         lucky trades). With ~25 trades a Sharpe of 1.3 is noise; default 100.
      6. reserved Sharpe > buy-and-hold Sharpe (`require_beat_bh`)
         (must beat just holding the asset over the same window — crucial for a
         long-biased asset like BTC where holding already scores a high Sharpe).

    Indicators are computed on the FULL series (warmup preserved); only the
    EVALUATION windows are restricted. `reserved_start` must be a date AFTER
    everything the search/funnel touched (train_end and oos_end).
    """
    import pandas as pd

    df = sf.load_ohlc(data) if isinstance(data, str) else data
    n = len(df)
    close = np.asarray(df['Close'].values, dtype=np.float64)
    ret = _returns_from_close(close)
    if periods_per_year is None:
        try:
            sec = np.median(np.diff(df.index.values).astype('timedelta64[s]').astype(float))
            periods_per_year = (365.0 * 24 * 3600) / sec
        except Exception:
            periods_per_year = 8760.0
    ppy = periods_per_year

    atr = sf.compute_atr(df['High'].values, df['Low'].values, close, atr_period)
    size = sf.compute_vol_target_size(close, target_ann_vol=target_vol,
                                      lookback=vol_lookback, periods_per_year=ppy)

    cache = build_signal_cache(indicator_classes, df, show_progress=show_progress)
    cache.pop('_failed', None)

    r0 = int(df.index.searchsorted(pd.Timestamp(reserved_start)))
    mid = (r0 + n) // 2                       # split the reserved tail in half
    res = slice(r0, n); h1 = slice(r0, mid); h2 = slice(mid, n)

    # buy-and-hold Sharpe on the reserved tail (always-long = the raw returns);
    # constant across strategies — it's the market the strategy must beat.
    bh_res = sf.sharpe(ret[res], ppy)
    bh_h1 = sf.sharpe(ret[h1], ppy); bh_h2 = sf.sharpe(ret[h2], ppy)

    iterator = survivors
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(survivors, total=len(survivors), desc="gauntlet", unit="strat")
        except ImportError:
            pass

    try:
        from tqdm.auto import tqdm as _tqdm
        _write = _tqdm.write
    except ImportError:
        _write = print

    rows = []
    best_pass = None
    n_pass_run = 0
    for idx, s in enumerate(iterator, 1):
        strat = _as_strategy(s)
        EL, XL, ES, XS = strategy_triggers(strat, cache, n)

        def shp(sl, c, sz=None):
            m = compute_metrics(EL[sl], XL[sl], ES[sl], XS[sl], ret[sl], c, ppy,
                                close=close[sl], atr=atr[sl],
                                size=(size[sl] if sz is not None else None))
            return m['sharpe'], m['n_trades'], m['max_drawdown']

        s_res, n_res, mdd = shp(res, cost)
        s_sized, _, _ = shp(res, cost, sz=True)
        s_lo, _, _ = shp(res, cost_stress[0])
        s_hi, _, _ = shp(res, cost_stress[-1])
        s_h1, _, _ = shp(h1, cost)
        s_h2, _, _ = shp(h2, cost)

        passed = (s_res >= min_sharpe and s_hi > 0 and s_sized > 0
                  and s_h1 > 0 and s_h2 > 0
                  and n_res >= min_reserved_trades
                  and (not require_beat_bh or s_res > bh_res))
        row = {
            'reserved_sharpe': round(s_res, 3),
            'bh_sharpe': round(bh_res, 3), 'vs_bh': round(s_res - bh_res, 3),
            'cost_lo': round(s_lo, 3), 'cost_hi': round(s_hi, 3),
            'vol_sized': round(s_sized, 3),
            'split_h1': round(s_h1, 3), 'split_h2': round(s_h2, 3),
            'n_trades': int(n_res), 'max_dd': round(mdd, 4),
            'pass': bool(passed), 'strategy': strat,
        }
        rows.append(row)

        # live partial results: track best passer + print every N
        if passed:
            n_pass_run += 1
            if best_pass is None or s_res > best_pass['reserved_sharpe']:
                best_pass = row
        if print_best_every and idx % print_best_every == 0:
            if best_pass is not None:
                _write(f"  [{idx:>6}] passed={n_pass_run:>4} | best reserved={best_pass['reserved_sharpe']:+.2f} "
                       f"vs_bh={best_pass['vs_bh']:+.2f} h1={best_pass['split_h1']:+.2f} "
                       f"h2={best_pass['split_h2']:+.2f} trades={best_pass['n_trades']}")
            else:
                _write(f"  [{idx:>6}] passed=0 | best so far: reserved={max(r['reserved_sharpe'] for r in rows):+.2f} "
                       f"(none passing yet)")

    out = pd.DataFrame(rows).sort_values('reserved_sharpe', ascending=False).reset_index(drop=True)
    if show_progress and len(out):
        n_pass = int(out['pass'].sum())
        thin = int((out['n_trades'] < min_reserved_trades).sum())
        beat = int((out['reserved_sharpe'] > bh_res).sum())
        print(f"\nGAUNTLET: {n_pass}/{len(out)} survivors passed all six tests "
              f"(reserved tail from {reserved_start}).")
        print(f"  buy & hold Sharpe over this window = {bh_res:+.2f}  "
              f"(h1 {bh_h1:+.2f}, h2 {bh_h2:+.2f})  <- the bar to beat")
        print(f"  {thin}/{len(out)} too few trades (<{min_reserved_trades}); "
              f"{beat}/{len(out)} beat buy & hold")
        cols = ['reserved_sharpe', 'bh_sharpe', 'vs_bh', 'cost_hi', 'vol_sized',
                'split_h1', 'split_h2', 'n_trades', 'pass']
        with pd.option_context('display.max_rows', 30):
            print(out[cols].to_string())
    return out


def inspect_passers(passers, indicator_classes, data, reserved_start,
                    cost: float = 0.0015, periods_per_year: float = None,
                    plot_n: int = 3, show_progress: bool = True):
    """
    For each (deduplicated) passing strategy, on the reserved tail, quantify
    whether its P&L is STEADY or driven by a few OUTLIER trades, and plot the
    `plot_n` best. Returns a DataFrame sorted by concentration (steadiest first).

    Key columns:
        net            – total P&L on the reserved tail (return units)
        sharpe         – reserved-tail Sharpe
        win_rate       – fraction of winning trades
        top1_share     – best single trade as a fraction of net P&L
        top3_share     – best 3 trades as a fraction of net P&L
        net_ex_top3    – net P&L with the 3 best trades removed
        steady         – True if top3_share < 0.5 AND net_ex_top3 > 0
                         (profit is broad-based, not a few lucky trades)

    `passers` may be the gauntlet DataFrame (uses its 'strategy' column) or a
    list of survivor/strategy dicts.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    items = passers['strategy'].tolist() if hasattr(passers, 'columns') else list(passers)

    df = sf.load_ohlc(data) if isinstance(data, str) else data
    n = len(df)
    close = np.asarray(df['Close'].values, dtype=np.float64)
    ret = _returns_from_close(close)
    if periods_per_year is None:
        try:
            sec = np.median(np.diff(df.index.values).astype('timedelta64[s]').astype(float))
            periods_per_year = (365.0 * 24 * 3600) / sec
        except Exception:
            periods_per_year = 8760.0
    ppy = periods_per_year

    cache = build_signal_cache(indicator_classes, df, show_progress=show_progress)
    cache.pop('_failed', None)
    r0 = int(df.index.searchsorted(pd.Timestamp(reserved_start)))
    res = slice(r0, n)
    dates = df.index[r0:n]

    # dedupe by strategy signature
    rows, seen = [], set()
    import json as _json
    for it in items:
        strat = _as_strategy(it)
        key = _json.dumps(strat, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        EL, XL, ES, XS = strategy_triggers(strat, cache, n)
        pnl, tp, ntr = sf._backtest_trades(EL[res], XL[res], ES[res], XS[res], ret[res], cost)
        net = float(tp.sum())
        order = np.sort(tp)[::-1] if ntr else np.array([0.0])
        top1 = float(order[0]); top3 = float(order[:3].sum())
        net_ex3 = net - top3
        denom = net if abs(net) > 1e-12 else float('nan')
        rows.append({
            'sharpe': round(sf.sharpe(pnl, ppy), 3),
            'net': round(net, 4), 'n_trades': int(ntr),
            'win_rate': round(float((tp > 0).sum()) / ntr, 3) if ntr else 0.0,
            'top1_share': round(top1 / denom, 3),
            'top3_share': round(top3 / denom, 3),
            'net_ex_top3': round(net_ex3, 4),
            'steady': bool((top3 / denom < 0.5) and net_ex3 > 0) if denom == denom else False,
            'strategy': strat,
        })

    out = pd.DataFrame(rows).sort_values('top3_share').reset_index(drop=True)

    if plot_n and len(out):
        for rank, (_, row) in enumerate(out.head(plot_n).iterrows()):
            strat = row['strategy']
            EL, XL, ES, XS = strategy_triggers(strat, cache, n)
            pnl, tp, _ = sf._backtest_trades(EL[res], XL[res], ES[res], XS[res], ret[res], cost)
            fig, ax = plt.subplots(1, 2, figsize=(13, 3.4),
                                   gridspec_kw={'width_ratios': [2, 1]})
            ax[0].plot(dates, np.cumsum(pnl), lw=1.4)
            ax[0].axhline(0, color='gray', lw=0.6, ls='--')
            ax[0].set_title(f"#{rank} equity (reserved tail)  Sharpe {row['sharpe']:+.2f}  "
                            f"net {row['net']:+.3f}  trades {row['n_trades']}")
            ax[0].grid(alpha=0.3)
            colors = ['#2a9d4a' if x > 0 else '#c0392b' for x in tp]
            ax[1].bar(range(len(tp)), tp, color=colors)
            ax[1].axhline(0, color='gray', lw=0.6)
            verdict = "STEADY" if row['steady'] else "LUMPY"
            ax[1].set_title(f"per-trade P&L [{verdict}]  top3={row['top3_share']:.0%} of net")
            ax[1].grid(alpha=0.3)
            fig.tight_layout()

    if show_progress and len(out):
        cols = ['sharpe', 'net', 'n_trades', 'win_rate', 'top1_share',
                'top3_share', 'net_ex_top3', 'steady']
        n_steady = int(out['steady'].sum())
        print(f"\nTRADE-CONCENTRATION: {n_steady}/{len(out)} distinct strategies are STEADY "
              f"(top-3 trades < 50% of net AND still profitable without them).")
        print("  lumpy = Sharpe driven by a few outlier trades -> don't trust it.\n")
        with pd.option_context('display.max_rows', 30):
            print(out[cols].to_string())
    return out


if __name__ == '__main__':
    print(__doc__)
