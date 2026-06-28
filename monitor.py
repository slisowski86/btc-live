"""
Live monitoring + reporting for the trader. Tracks balance, last-trade P&L, overall P&L and
max drawdown across restarts (monitor_state.json), writes a heartbeat the watchdog reads, and
sends email alerts via notifier.py:

  * a TRADE email whenever exposure changes / an order executes
  * an ERROR email when a cycle raises (throttled so a stuck error can't spam you)
  * a CRASH email when the process exits unexpectedly (also covered externally by watchdog.py)
  * a DAILY SUMMARY email (balance / last-trade / overall / max drawdown) at the first cycle of
    each new UTC day, including a Claude strategy review ONLY when Claude flags something

Every entry point is best-effort and swallows its own exceptions: monitoring must never be able
to take down the trader.
"""
from __future__ import annotations
import os, json, csv, time, datetime as dt, traceback

import notifier
try:
    import claude_review
except Exception:
    claude_review = None

STATE_FILE       = "monitor_state.json"
HEARTBEAT_FILE   = "heartbeat.txt"
DAILY_CSV        = "daily_stats.csv"     # one aggregated row appended per UTC day
ERROR_THROTTLE_S = 1800     # at most one identical error email per 30 min


# ----------------------------- state -----------------------------
def _load():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {}


def _save(st):
    try:
        json.dump(st, open(STATE_FILE, "w"), indent=2)
    except Exception as e:
        print(f"[monitor] state save failed: {e}")


def _init(st, equity):
    st.setdefault("start_equity", equity)
    st.setdefault("peak_equity", equity)
    st.setdefault("max_drawdown", 0.0)
    st.setdefault("equity_at_last_trade", equity)
    st.setdefault("equity_prev_summary", equity)     # balance at the last daily snapshot
    st.setdefault("last_trade_pnl", 0.0)
    st.setdefault("n_trades", 0)
    st.setdefault("trades_today", 0)
    st.setdefault("last_summary_date", None)
    return st


def _append_daily_csv(row: dict):
    """Append one day's aggregated stats to DAILY_CSV (header written once). Best-effort."""
    try:
        new = not os.path.exists(DAILY_CSV)
        with open(DAILY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[monitor] daily csv write failed: {e}")


def _utc_today():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def heartbeat(note: str = ""):
    """Touch the heartbeat file so the external watchdog knows the trader is alive."""
    try:
        with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {note}".strip())
    except Exception:
        pass


def _update_equity(st, equity):
    if equity is None or equity <= 0:
        return
    if equity > st["peak_equity"]:
        st["peak_equity"] = equity
    dd = equity / st["peak_equity"] - 1.0
    if dd < st["max_drawdown"]:
        st["max_drawdown"] = dd


# ----------------------------- per-bar hook -----------------------------
def on_bar(equity, *, bar_time, close, exposure, prev_exposure, traded,
           order_desc="", detail=None, cum_return=None, live=False):
    """Call once per processed bar. Updates stats, heartbeat, trade + daily-summary emails."""
    try:
        _on_bar(equity, bar_time=bar_time, close=close, exposure=exposure,
                prev_exposure=prev_exposure, traded=traded, order_desc=order_desc,
                detail=detail or {}, cum_return=cum_return, live=live)
    except Exception as e:
        print(f"[monitor] on_bar error (ignored): {e}")


def _on_bar(equity, *, bar_time, close, exposure, prev_exposure, traded,
            order_desc, detail, cum_return, live):
    st = _init(_load(), equity)
    _update_equity(st, equity)
    heartbeat(f"bar={bar_time} equity={equity:.2f} exp={exposure:+.2f}")

    # daily summary first (reports/resets the just-finished day before today's trade counts)
    _maybe_daily_summary(st, equity, bar_time, close, exposure, detail, live)

    if traded:
        last_pnl = equity - st["equity_at_last_trade"]
        st["last_trade_pnl"] = last_pnl
        st["equity_at_last_trade"] = equity
        st["n_trades"] += 1
        st["trades_today"] = st.get("trades_today", 0) + 1
        overall = equity - st["start_equity"]
        overall_pct = (equity / st["start_equity"] - 1.0) if st["start_equity"] else 0.0
        kind = "LIVE" if live else "PAPER"
        body = (
            f"{kind} trade executed.\n\n"
            f"  Time:            {bar_time}\n"
            f"  BTC price:       {close:,.2f}\n"
            f"  Exposure:        {prev_exposure:+.2f}  ->  {exposure:+.2f}\n"
            f"  Order:           {order_desc or '-'}\n\n"
            f"  Current balance: {equity:,.2f} USDT\n"
            f"  P&L last trade:  {last_pnl:+,.2f} USDT (since previous trade)\n"
            f"  Overall P&L:     {overall:+,.2f} USDT ({overall_pct:+.2%})\n"
            f"  Max drawdown:    {st['max_drawdown']:.2%}\n"
            f"  Trades total:    {st['n_trades']}\n"
        )
        notifier.send_email(f"Trade @ {close:,.0f}  exposure {exposure:+.2f}", body)

    _save(st)


def _maybe_daily_summary(st, equity, bar_time, close, exposure, detail, live):
    today = _utc_today()
    if st.get("last_summary_date") == today:
        return
    first_ever = st.get("last_summary_date") is None
    trades_prev_day = st.get("trades_today", 0)
    st["last_summary_date"] = today
    st["trades_today"] = 0
    if first_ever:
        st["equity_prev_summary"] = equity     # baseline; first full day is measured from here
        return                                  # no full prior day yet on the very first cycle

    overall = equity - st["start_equity"]
    overall_pct = (equity / st["start_equity"] - 1.0) if st["start_equity"] else 0.0
    prev_eq = st.get("equity_prev_summary", st["start_equity"])
    day_pnl = equity - prev_eq
    day_ret = (equity / prev_eq - 1.0) if prev_eq else 0.0

    # persist the day-by-day stats row, then advance the daily baseline
    _append_daily_csv({
        "date": today, "mode": "live" if live else "paper",
        "balance": round(equity, 2),
        "day_pnl": round(day_pnl, 2), "day_return": round(day_ret, 6),
        "overall_pnl": round(overall, 2), "overall_pct": round(overall_pct, 6),
        "max_drawdown": round(st["max_drawdown"], 6), "peak_equity": round(st["peak_equity"], 2),
        "exposure": round(float(exposure), 3),
        "trades_day": trades_prev_day, "trades_total": st["n_trades"],
        "btc_price": round(close, 2),
    })
    st["equity_prev_summary"] = equity
    stats = {
        "as_of": str(bar_time), "mode": "live" if live else "paper",
        "current_balance": round(equity, 2),
        "start_equity": round(st["start_equity"], 2),
        "day_pnl": round(day_pnl, 2),
        "day_return": round(day_ret, 4),
        "overall_pnl": round(overall, 2),
        "overall_pct": round(overall_pct, 4),
        "last_trade_pnl": round(st.get("last_trade_pnl", 0.0), 2),
        "max_drawdown": round(st["max_drawdown"], 4),
        "peak_equity": round(st["peak_equity"], 2),
        "current_exposure": round(float(exposure), 3),
        "n_trades_total": st["n_trades"],
        "trades_prev_day": trades_prev_day,
        "btc_price": round(close, 2),
        "detail": detail,
    }
    obs = claude_review.review_performance(stats) if claude_review else None

    body = (
        f"Daily summary - {today} (UTC)\n\n"
        f"  Current balance:   {equity:,.2f} USDT\n"
        f"  Day P&L:           {day_pnl:+,.2f} USDT ({day_ret:+.2%})\n"
        f"  Overall P&L:       {overall:+,.2f} USDT ({overall_pct:+.2%})\n"
        f"  Last trade P&L:    {st.get('last_trade_pnl', 0.0):+,.2f} USDT\n"
        f"  Max drawdown:      {st['max_drawdown']:.2%}\n"
        f"  Peak equity:       {st['peak_equity']:,.2f} USDT\n"
        f"  Exposure now:      {exposure:+.2f}\n"
        f"  Trades (prev day): {trades_prev_day}\n"
        f"  Trades (total):    {st['n_trades']}\n"
        f"  BTC price:         {close:,.2f}\n"
    )
    if obs:
        body += f"\n--- Claude strategy review ---\n{obs}\n"
    notifier.send_email(
        f"Daily summary {today}  balance {equity:,.0f} ({overall_pct:+.1%})", body)


# ----------------------------- lifecycle hooks -----------------------------
def on_start(mode: str):
    """Trader started - heartbeat + a startup email."""
    try:
        heartbeat("startup")
        notifier.send_email(
            "Trader started",
            f"The live trader has started.\n"
            f"  Mode: {mode}\n"
            f"  Time: {dt.datetime.now(dt.timezone.utc).isoformat()} (UTC)\n")
    except Exception as e:
        print(f"[monitor] on_start failed: {e}")


def on_error(exc: Exception, context: str = "run_once"):
    """A cycle raised but the trader keeps running. Throttle identical repeats."""
    try:
        sig = f"{context}:{type(exc).__name__}:{exc}"
        st = _load()
        now = time.time()
        if st.get("last_error_sig") == sig and (now - st.get("last_error_time", 0)) < ERROR_THROTTLE_S:
            return
        st["last_error_sig"] = sig
        st["last_error_time"] = now
        _save(st)
        body = (
            f"An error occurred in the live trader ({context}):\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}\n"
            f"The trader will retry on the next cycle.\n")
        notifier.send_email(f"ERROR in {context}: {type(exc).__name__}", body)
    except Exception as e:
        print(f"[monitor] on_error failed: {e}")


def on_crash(exc: Exception):
    """The process is exiting unexpectedly. Call from the outermost except."""
    try:
        body = (
            f"The live trader has CRASHED and is stopping:\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}\n"
            f"No more bars will be processed until it is restarted.\n")
        notifier.send_email(f"CRASH - trader stopped: {type(exc).__name__}", body)
    except Exception as e:
        print(f"[monitor] on_crash failed: {e}")
