"""
Offline simulator that exercises EVERY monitoring path without an exchange, live bars, or waiting.

It drives notifier / monitor / claude_review (and a tiny mock exchange) through each situation so
you can confirm the alert emails look right and actually arrive. No network, no strategy compute.

Scenarios (--scenario):
    startup    'Trader started' email
    profit     a winning TRADE email  (balance up, +last-trade P&L)
    loss       a losing TRADE email   (balance down, -last-trade P&L)
    order      a trade routed through a MOCK Kraken futures exchange (real order description)
    orderfail  a mock exchange order that FAILS (shows it does NOT send a false trade email)
    error      an ERROR email         (simulated run_once exception, e.g. a network drop)
    crash      a CRASH email          (simulated fatal exit)
    daily      a healthy DAILY SUMMARY (Claude usually says 'OK' -> no review appended)
    alert      a DAILY SUMMARY in deep drawdown (Claude should ADD a strategy review)
    watchdog   mark the heartbeat stale and run the watchdog check -> 'trader DOWN' email
    all        run the whole sequence in order

Examples:
    py mock_sim.py --scenario all                 # send every alert (uses EMAIL_* / ANTHROPIC_API_KEY)
    py mock_sim.py --scenario alert               # test the Claude-opinion path specifically
    py mock_sim.py --scenario all --dry           # mute email; just print what WOULD send
    py mock_sim.py --scenario all --no-claude     # skip the Claude API calls

State is written to mock_*.json / mock_heartbeat.txt - NOT your live monitor_state.json - and is
removed at the end unless --keep.
"""
from __future__ import annotations
import argparse, os, json, time, datetime as dt

import monitor
import notifier

# isolate all monitor state in mock files so the live trader's state is never touched
monitor.STATE_FILE     = "mock_monitor_state.json"
monitor.HEARTBEAT_FILE = "mock_heartbeat.txt"
monitor.DAILY_CSV      = "mock_daily_stats.csv"
_MOCK_FILES = ["mock_monitor_state.json", "mock_heartbeat.txt", "mock_watchdog_state.json",
               "mock_daily_stats.csv"]


# ----------------------------- helpers -----------------------------
def _load_secrets(path="secrets.env"):
    """Lightweight secrets loader (avoids importing the heavy paper_trader module)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if not (v.startswith('"') or v.startswith("'")):
            h = v.find("#")                       # strip trailing inline comment (# after space)
            if h > 0 and v[h - 1].isspace():
                v = v[:h].strip()
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _seed_state(**kw):
    base = {"start_equity": 10_000.0, "peak_equity": 10_000.0, "max_drawdown": 0.0,
            "equity_at_last_trade": 10_000.0, "last_trade_pnl": 0.0,
            "n_trades": 0, "trades_today": 0, "last_summary_date": None}
    base.update(kw)
    json.dump(base, open(monitor.STATE_FILE, "w"), indent=2)


def _today():     return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
def _yesterday(): return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).strftime("%Y-%m-%d")
def _now():       return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _hdr(name, desc):
    print("\n" + "-" * 60)
    print(f"  SCENARIO: {name}  -  {desc}")
    print("-" * 60)


class MockExchange:
    """Minimal ccxt-like swap account - just enough to look like a real fill/failure."""
    def __init__(self, equity=10_000.0, pos=0.0, fail=False):
        self.equity, self.pos, self.fail, self.id = equity, pos, fail, "mock-kraken"

    def amount_to_precision(self, symbol, amt):
        return f"{float(amt):.3f}"

    def create_order(self, symbol, type_, side, amt):
        if self.fail:
            raise Exception("insufficient margin (mock failure)")
        self.pos += float(amt) * (1 if side == "buy" else -1)
        return {"id": f"mock-{side}-{int(time.time())}"}


# ----------------------------- scenarios -----------------------------
def sc_startup(_):
    _hdr("startup", "trader-started email")
    monitor.on_start("MOCK LIVE (kraken futures / swap)")


def sc_profit(_):
    _hdr("profit", "winning trade email (+200 since last trade)")
    _seed_state(equity_at_last_trade=10_100.0, peak_equity=10_300.0,
                n_trades=5, last_summary_date=_today())     # today -> no daily summary now
    monitor.on_bar(10_300.0, bar_time=_now(), close=61_500, exposure=0.8, prev_exposure=0.0,
                   traded=True, order_desc="buy 0.050 BTC @market", detail={"raw": 1}, live=True)


def sc_loss(_):
    _hdr("loss", "losing trade email (-250 since last trade)")
    _seed_state(equity_at_last_trade=10_100.0, peak_equity=10_300.0,
                n_trades=6, last_summary_date=_today())
    monitor.on_bar(9_850.0, bar_time=_now(), close=58_900, exposure=0.0, prev_exposure=0.8,
                   traded=True, order_desc="sell 0.050 BTC @market", detail={"raw": 0}, live=True)


def sc_order(_):
    _hdr("order", "trade via a MOCK exchange (real order description + id)")
    ex = MockExchange(equity=10_250.0)
    amt = ex.amount_to_precision("BTC/USDT:USDT", 0.047)
    o = ex.create_order("BTC/USDT:USDT", "market", "buy", amt)
    desc = f"buy {amt} @market (id {o['id']})"
    _seed_state(equity_at_last_trade=10_080.0, peak_equity=10_250.0,
                n_trades=8, last_summary_date=_today())
    monitor.on_bar(ex.equity, bar_time=_now(), close=61_200, exposure=0.7, prev_exposure=0.0,
                   traded=True, order_desc=desc, detail={"raw": 1}, live=True)
    print(f"  mock fill -> position now {ex.pos:+.3f} BTC")


def sc_orderfail(_):
    _hdr("orderfail", "mock order FAILS -> no false trade email")
    ex = MockExchange(equity=10_000.0, fail=True)
    try:
        ex.create_order("BTC/USDT:USDT", "market", "buy", "0.050")
        desc = "buy 0.050 @market"
    except Exception as e:
        desc = f"FAILED: {e}"
    traded = bool(desc) and not desc.startswith("FAILED")
    _seed_state(n_trades=8, last_summary_date=_today())
    monitor.on_bar(ex.equity, bar_time=_now(), close=61_000, exposure=0.7, prev_exposure=0.0,
                   traded=traded, order_desc=desc, detail={"raw": 1}, live=True)
    print(f"  order_desc = {desc!r}")
    print(f"  traded     = {traded}  (failed order -> no trade email is correct)")


def sc_error(_):
    _hdr("error", "ERROR email (simulated run_once exception)")
    try:
        raise ConnectionError("Kraken kline fetch timed out (mock)")
    except Exception as e:
        monitor.on_error(e, "run_once (mock)")


def sc_crash(_):
    _hdr("crash", "CRASH email (simulated fatal exit)")
    try:
        raise RuntimeError("unhandled fatal error in --loop (mock)")
    except Exception as e:
        monitor.on_crash(e)


def sc_daily(_):
    _hdr("daily", "healthy daily summary (Claude likely 'OK' -> no review)")
    _seed_state(peak_equity=10_250.0, equity_at_last_trade=10_180.0, last_trade_pnl=70.0,
                n_trades=12, trades_today=2, max_drawdown=-0.03, last_summary_date=_yesterday())
    monitor.on_bar(10_180.0, bar_time=f"{_today()} 00:00:00", close=61_000, exposure=0.6,
                   prev_exposure=0.6, traded=False, detail={"raw": 1}, live=True)


def sc_alert(_):
    _hdr("alert", "daily summary in DEEP DRAWDOWN (Claude should add a review)")
    _seed_state(peak_equity=12_000.0, equity_at_last_trade=10_200.0, last_trade_pnl=-380.0,
                n_trades=22, trades_today=4, max_drawdown=-0.19, last_summary_date=_yesterday())
    monitor.on_bar(9_750.0, bar_time=f"{_today()} 00:00:00", close=57_000, exposure=-0.8,
                   prev_exposure=-0.8, traded=False,
                   detail={"raw": -1, "risk_off": False, "sma": 59_000}, live=True)


def sc_watchdog(_):
    _hdr("watchdog", "stale heartbeat -> 'trader DOWN' email")
    stale = 130 * 60                                    # watchdog default: ~2h since last beat
    monitor.heartbeat("stale-test")
    old = time.time() - 3 * 3600                        # pretend last beat was 3h ago
    os.utime(monitor.HEARTBEAT_FILE, (old, old))
    age = time.time() - os.path.getmtime(monitor.HEARTBEAT_FILE)
    if age > stale:
        notifier.send_email(
            "WATCHDOG: trader appears DOWN",
            f"The live trader has not updated its heartbeat.\n  last heartbeat {age/3600:.1f}h ago\n"
            f"  staleness threshold: {stale/3600:.1f}h\n\n"
            f"(This is a mock_sim test - your real trader is unaffected.)\n")
        print(f"  heartbeat {age/3600:.1f}h old > {stale/3600:.1f}h -> DOWN email sent")
    else:
        print("  heartbeat not stale enough (unexpected)")


SCENARIOS = {
    "startup": sc_startup, "profit": sc_profit, "loss": sc_loss, "order": sc_order,
    "orderfail": sc_orderfail, "error": sc_error, "crash": sc_crash,
    "daily": sc_daily, "alert": sc_alert, "watchdog": sc_watchdog,
}
ALL_ORDER = ["startup", "profit", "loss", "order", "orderfail", "error",
             "daily", "alert", "watchdog", "crash"]


def _cleanup():
    for f in _MOCK_FILES:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + list(SCENARIOS), help="which situation to simulate")
    ap.add_argument("--dry", action="store_true", help="mute email; just print what would send")
    ap.add_argument("--no-claude", action="store_true", help="skip the Claude API review calls")
    ap.add_argument("--keep", action="store_true", help="keep the mock_* state files")
    a = ap.parse_args()

    _load_secrets()
    if a.dry:
        os.environ["EMAIL_ENABLED"] = "0"

        def _preview(subject, body, prefix="[BTC-LIVE]"):   # show, don't send
            print("\n    ===== EMAIL (preview, not sent) =====")
            print(f"    Subject: {prefix} {subject}".rstrip())
            for ln in body.splitlines():
                print("    | " + ln)
            print("    =====================================")
            return True
        notifier.send_email = _preview                      # monitor.notifier is the same module
    if a.no_claude:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    print(f"email: {'MUTED (--dry)' if a.dry else ('ON -> ' + os.environ.get('EMAIL_TO', '?')) if notifier.email_configured() else 'OFF (set EMAIL_* in secrets.env)'}"
          f"  |  claude: {'off' if a.no_claude or not os.environ.get('ANTHROPIC_API_KEY') else 'on'}")

    _cleanup()
    names = ALL_ORDER if a.scenario == "all" else [a.scenario]
    for name in names:
        SCENARIOS[name](a)
    print("\nall scenarios done.")
    if not a.keep:
        _cleanup()
    else:
        print(f"kept: {', '.join(f for f in _MOCK_FILES if os.path.exists(f))}")


if __name__ == "__main__":
    main()
