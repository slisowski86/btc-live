"""
Aggregate trading statistics from paper_log.csv.

The trader logs one row per closed bar (exposure + equity mark). This reads that log and
produces:

  * EQUITY metrics   - total return, CAGR, annualised vol, Sharpe, Sortino, Calmar, max
                       drawdown, % positive bars, time in market
  * TRADE metrics    - round-trips reconstructed from the exposure series: count, win rate,
                       avg win / avg loss, profit factor, expectancy, best / worst, avg bars
                       held, long-vs-short split
  * EXPOSURE metrics - % of bars long / short / flat, average & max |exposure|
  * FUNDING          - cumulative perp funding paid (if recorded)
  * PERIOD returns   - monthly equity returns table

Usage:
    py stats.py                 # print the report
    py stats.py --email         # print + email it (uses EMAIL_* in secrets.env)
    py stats.py --monthly       # also print the monthly-returns table
    py stats.py --trades out.csv# dump the reconstructed round-trip trades to CSV
    py stats.py --source testnet# equity source: auto (default) | paper | testnet
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

LOG_FILE = "paper_log.csv"
PPY      = 8760.0           # hourly bars per year (matches protected_strategy.PPY)


# ----------------------------- load -----------------------------
def load_log(path=LOG_FILE, source="auto") -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"no {path} found - run the trader first to generate a log")
    df = pd.read_csv(path)
    if "bar_time" in df:
        df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")
        df = df.dropna(subset=["bar_time"]).sort_values("bar_time").reset_index(drop=True)

    # choose the equity series: testnet (real exchange balance) when present, else paper
    tn = df["testnet_equity"] if "testnet_equity" in df else pd.Series(dtype=float)
    have_tn = tn.notna().sum() > 1
    if source == "testnet" or (source == "auto" and have_tn):
        df["equity"] = pd.to_numeric(df.get("testnet_equity"), errors="coerce").ffill()
        df["_src"] = "testnet"
    else:
        df["equity"] = pd.to_numeric(df["paper_equity"], errors="coerce")
        df["_src"] = "paper"
    df = df.dropna(subset=["equity"]).reset_index(drop=True)
    if len(df) < 2:
        raise SystemExit("log has fewer than 2 usable rows - not enough to aggregate yet")
    return df


# ----------------------------- equity metrics -----------------------------
def equity_metrics(df) -> dict:
    eq = df["equity"].values.astype(float)
    r = np.zeros(len(eq)); r[1:] = eq[1:] / eq[:-1] - 1.0
    n = len(eq); years = n / PPY
    total = eq[-1] / eq[0] - 1.0
    sd = r.std(ddof=1) if n > 2 else 0.0
    downside = r[r < 0]
    dsd = downside.std(ddof=1) if downside.size > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1.0 if years > 0 and eq[0] > 0 else float("nan")
    max_dd = float(dd.min())
    exposure = df["target_exposure"].astype(float).values if "target_exposure" in df else np.zeros(n)
    return {
        "rows": n, "span_days": round(years * 365.25, 1),
        "first_bar": str(df["bar_time"].iloc[0]) if "bar_time" in df else None,
        "last_bar": str(df["bar_time"].iloc[-1]) if "bar_time" in df else None,
        "start_equity": round(float(eq[0]), 2), "end_equity": round(float(eq[-1]), 2),
        "total_return": total, "cagr": cagr,
        "ann_vol": sd * np.sqrt(PPY),
        "sharpe": (r.mean() / sd * np.sqrt(PPY)) if sd > 0 else float("nan"),
        "sortino": (r.mean() / dsd * np.sqrt(PPY)) if dsd > 0 else float("nan"),
        "max_drawdown": max_dd,
        "calmar": (cagr / abs(max_dd)) if max_dd < 0 and cagr == cagr else float("nan"),
        "pct_positive_bars": float((r > 0).mean()),
        "time_in_market": float((exposure != 0).mean()),
        "source": df["_src"].iloc[0],
    }


# ----------------------------- trade reconstruction -----------------------------
def reconstruct_trades(df) -> pd.DataFrame:
    """Round-trips from the exposure series: a trade is a run of constant non-zero sign.
    Entry/exit equity are taken where the sign changes; P&L is net (costs+funding baked in)."""
    eq = df["equity"].values.astype(float)
    exp = df["target_exposure"].astype(float).values if "target_exposure" in df else np.zeros(len(df))
    sign = np.sign(exp).astype(int)
    close = df["close"].values.astype(float) if "close" in df else np.full(len(df), np.nan)
    times = df["bar_time"].values if "bar_time" in df else np.arange(len(df))

    trades, cur = [], None
    for i in range(len(df)):
        s = sign[i]
        if cur is not None and s != cur["dir"]:                 # close current position
            cur.update(exit_i=i, exit_eq=eq[i], exit_time=times[i], exit_close=close[i])
            trades.append(cur); cur = None
        if cur is None and s != 0:                              # open a new position
            cur = {"dir": s, "entry_i": i, "entry_eq": eq[i],
                   "entry_time": times[i], "entry_close": close[i]}
    if cur is not None:                                         # still open at the end
        cur.update(exit_i=len(df) - 1, exit_eq=eq[-1], exit_time=times[-1],
                   exit_close=close[-1], open=True)
        trades.append(cur)

    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades)
    t["open"] = t.get("open", False)
    t["direction"] = np.where(t["dir"] > 0, "long", "short")
    t["pnl_pct"] = t["exit_eq"] / t["entry_eq"] - 1.0
    t["pnl_usd"] = t["exit_eq"] - t["entry_eq"]
    t["bars_held"] = t["exit_i"] - t["entry_i"]
    return t[["direction", "entry_time", "exit_time", "entry_close", "exit_close",
              "bars_held", "pnl_pct", "pnl_usd", "open"]]


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0}
    closed = trades[~trades["open"]]
    p = closed["pnl_usd"].values
    wins, losses = p[p > 0], p[p < 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "n_trades": int(len(closed)), "n_open": int(trades["open"].sum()),
        "n_long": int((closed["direction"] == "long").sum()),
        "n_short": int((closed["direction"] == "short").sum()),
        "win_rate": float(len(wins) / len(p)) if len(p) else float("nan"),
        "avg_win_usd": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_usd": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(gp / gl) if gl > 0 else float("inf") if gp > 0 else float("nan"),
        "expectancy_usd": float(p.mean()) if len(p) else float("nan"),
        "best_usd": float(p.max()) if len(p) else float("nan"),
        "worst_usd": float(p.min()) if len(p) else float("nan"),
        "avg_bars_held": float(closed["bars_held"].mean()) if len(closed) else float("nan"),
        "long_win_rate": _wr(closed, "long"), "short_win_rate": _wr(closed, "short"),
    }


def _wr(closed, direction):
    d = closed[closed["direction"] == direction]
    return float((d["pnl_usd"] > 0).mean()) if len(d) else float("nan")


# ----------------------------- exposure / funding / periods -----------------------------
def exposure_metrics(df) -> dict:
    if "target_exposure" not in df:
        return {}
    e = df["target_exposure"].astype(float)
    return {"pct_long": float((e > 0).mean()), "pct_short": float((e < 0).mean()),
            "pct_flat": float((e == 0).mean()),
            "avg_abs_exposure": float(e.abs().mean()), "max_abs_exposure": float(e.abs().max())}


def funding_metrics(df) -> dict:
    out = {}
    if "funding_cum" in df:
        out["funding_cum_pct"] = float(pd.to_numeric(df["funding_cum"], errors="coerce").iloc[-1])
    if "funding_impact" in df:
        out["funding_sum_pct"] = float(pd.to_numeric(df["funding_impact"], errors="coerce").sum())
    return out


def monthly_returns(df) -> pd.Series:
    if "bar_time" not in df:
        return pd.Series(dtype=float)
    s = df.set_index("bar_time")["equity"].resample("ME").last().dropna()
    return s.pct_change().dropna()


def daily_table(df) -> pd.DataFrame:
    """Day-by-day stats from the bar log: end balance, daily P&L/return, cumulative return,
    drawdown, average exposure, time-in-market, and exposure changes (proxy for trades)."""
    if "bar_time" not in df:
        return pd.DataFrame()
    g = df.set_index("bar_time")
    eqd = g["equity"].resample("D").last().dropna()
    if eqd.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"balance": eqd.round(2)})
    out["day_pnl"] = eqd.diff().round(2)
    out["day_return"] = eqd.pct_change().round(6)
    out["cum_return"] = (eqd / eqd.iloc[0] - 1.0).round(6)
    out["drawdown"] = (eqd / eqd.cummax() - 1.0).round(6)
    if "target_exposure" in g:
        e = g["target_exposure"].astype(float)
        out["avg_exposure"] = e.resample("D").mean().round(3)
        out["time_in_market"] = (e != 0).resample("D").mean().round(3)
        sign = np.sign(e)
        out["exposure_changes"] = (sign != sign.shift()).astype(int).resample("D").sum()
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    return out.reset_index()


# ----------------------------- assemble + format -----------------------------
def summarize(df):
    trades = reconstruct_trades(df)
    return {"equity": equity_metrics(df), "trades": trade_metrics(trades),
            "exposure": exposure_metrics(df), "funding": funding_metrics(df)}, trades


def _pct(x):  return "n/a" if x != x else f"{x:+.2%}"
def _num(x):  return "n/a" if x != x else f"{x:.2f}"
def _usd(x):  return "n/a" if x != x else f"{x:+,.2f}"


def format_report(stats: dict, monthly: pd.Series | None = None) -> str:
    e, t, x, f = stats["equity"], stats["trades"], stats["exposure"], stats["funding"]
    L = []
    L.append("=" * 56)
    L.append(f"  TRADING STATISTICS  ({e['source']} equity)")
    L.append("=" * 56)
    L.append(f"  Period:        {e['first_bar']}  ->  {e['last_bar']}")
    L.append(f"  Bars:          {e['rows']}   ({e['span_days']} days)")
    L.append(f"  Equity:        {e['start_equity']:,.2f}  ->  {e['end_equity']:,.2f} USDT")
    L.append("")
    L.append("  -- returns --")
    L.append(f"  Total return:  {_pct(e['total_return'])}")
    L.append(f"  CAGR:          {_pct(e['cagr'])}")
    L.append(f"  Ann. vol:      {_pct(e['ann_vol'])}")
    L.append(f"  Sharpe:        {_num(e['sharpe'])}")
    L.append(f"  Sortino:       {_num(e['sortino'])}")
    L.append(f"  Calmar:        {_num(e['calmar'])}")
    L.append(f"  Max drawdown:  {_pct(e['max_drawdown'])}")
    L.append(f"  Positive bars: {e['pct_positive_bars']:.1%}")
    L.append(f"  Time in mkt:   {e['time_in_market']:.1%}")
    L.append("")
    L.append("  -- trades (round-trips) --")
    if t.get("n_trades", 0) == 0:
        L.append(f"  No closed trades yet (open positions: {t.get('n_open', 0)})")
    else:
        L.append(f"  Trades:        {t['n_trades']}   (long {t['n_long']} / short {t['n_short']}"
                 f"{', ' + str(t['n_open']) + ' open' if t['n_open'] else ''})")
        L.append(f"  Win rate:      {t['win_rate']:.1%}   (long {t['long_win_rate']:.0%} /"
                 f" short {t['short_win_rate']:.0%})")
        L.append(f"  Avg win:       {_usd(t['avg_win_usd'])} USDT")
        L.append(f"  Avg loss:      {_usd(t['avg_loss_usd'])} USDT")
        L.append(f"  Profit factor: {_num(t['profit_factor'])}")
        L.append(f"  Expectancy:    {_usd(t['expectancy_usd'])} USDT / trade")
        L.append(f"  Best / worst:  {_usd(t['best_usd'])} / {_usd(t['worst_usd'])} USDT")
        L.append(f"  Avg held:      {t['avg_bars_held']:.0f} bars")
    if x:
        L.append("")
        L.append("  -- exposure --")
        L.append(f"  Long / short / flat:  {x['pct_long']:.0%} / {x['pct_short']:.0%} / {x['pct_flat']:.0%}")
        L.append(f"  Avg |exposure|:       {x['avg_abs_exposure']:.2f}   (max {x['max_abs_exposure']:.2f})")
    if f:
        L.append("")
        L.append("  -- funding --")
        if "funding_cum_pct" in f:
            L.append(f"  Cumulative funding:   {f['funding_cum_pct']:+.3%} of equity")
    if monthly is not None and not monthly.empty:
        L.append("")
        L.append("  -- monthly returns --")
        for ts, v in monthly.items():
            L.append(f"  {ts:%Y-%m}:  {v:+.2%}")
    L.append("=" * 56)
    return "\n".join(L)


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", action="store_true", help="email the report (uses EMAIL_* config)")
    ap.add_argument("--monthly", action="store_true", help="include the monthly-returns table")
    ap.add_argument("--trades", metavar="CSV", help="dump reconstructed round-trip trades to CSV")
    ap.add_argument("--daily-csv", metavar="CSV", help="write a day-by-day stats CSV from the log")
    ap.add_argument("--source", default="auto", choices=["auto", "paper", "testnet"],
                    help="equity series to aggregate (default: auto)")
    ap.add_argument("--log", default=LOG_FILE, help=f"log file (default: {LOG_FILE})")
    a = ap.parse_args()

    df = load_log(a.log, a.source)
    stats, trades = summarize(df)
    monthly = monthly_returns(df) if (a.monthly or a.email) else None
    report = format_report(stats, monthly)
    print(report)

    if a.trades and not trades.empty:
        trades.to_csv(a.trades, index=False)
        print(f"\n[stats] {len(trades)} round-trips written to {a.trades}")

    if a.daily_csv:
        dt_tbl = daily_table(df)
        if dt_tbl.empty:
            print("[stats] no daily rows to write (need bar_time + >=1 day of data)")
        else:
            dt_tbl.to_csv(a.daily_csv, index=False)
            print(f"\n[stats] {len(dt_tbl)} daily rows written to {a.daily_csv}")

    if a.email:
        try:
            from paper_trader import load_secrets
            load_secrets()
        except Exception:
            pass
        import notifier
        e = stats["equity"]
        subj = (f"Trading stats  {_pct(e['total_return'])}  "
                f"DD {_pct(e['max_drawdown'])}  Sharpe {_num(e['sharpe'])}")
        ok = notifier.send_email(subj, report)
        print(f"[stats] email {'sent' if ok else 'NOT sent (check EMAIL_* config)'}")


if __name__ == "__main__":
    main()
