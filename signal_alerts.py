"""
Manual-trading SIGNAL ALERTS.  No exchange API, no keys, no KYC.

It only READS public BTC price data, runs the protected strategy, and EMAILS you when the
recommended position changes - OPEN / CLOSE / FLIP. You place the trades yourself, on whatever
exchange you like. This is a signal service, not an order bot.

  Data     : Kraken Futures public 1h klines (no account needed; reuses paper_trader.fetch_klines)
  Strategy : protected_strategy.target_exposure (the same basket + vol-target + trend filter)
  Email    : notifier.py  (same EMAIL_* config in secrets.env)

It emails ONLY when the recommended DIRECTION changes (flat <-> long <-> short), so you get an
alert exactly when you need to act - not every hour.

    py signal_alerts.py --once    # one check (use with Windows Task Scheduler, hourly)
    py signal_alerts.py --loop    # run forever, waking ~15s after each hour close
    py signal_alerts.py --test    # send a test email and exit
"""
from __future__ import annotations
import argparse, json, os, time, datetime as dt

import protected_strategy as ps
from protected_strategy import target_exposure, load_basket
from paper_trader import fetch_klines, load_secrets   # reuse the Kraken public data feed
import notifier
import monitor                                          # only for the heartbeat (watchdog support)

STATE_FILE = "signal_state.json"
DIRN = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def _sign(x, eps=1e-9):
    return 1 if x > eps else -1 if x < -eps else 0


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {"direction": 0, "last_bar": None}


def save_state(st):
    json.dump(st, open(STATE_FILE, "w"), indent=2)


def _action(old, new):
    """Plain-language instruction for a direction change old->new (each -1/0/1), or None."""
    if old == new:
        return None
    if old == 0:
        return f"OPEN {DIRN[new]}"
    if new == 0:
        return f"CLOSE {DIRN[old]} (go flat)"
    return f"FLIP {DIRN[old]} to {DIRN[new]}"


def run_once(basket):
    df = fetch_klines()
    bar_time = str(df.index[-1])
    close = float(df["Close"].iloc[-1])
    st = load_state()
    monitor.heartbeat(f"signal bar={bar_time}")
    if st.get("last_bar") == bar_time:
        print(f"[{dt.datetime.now():%H:%M:%S}] no new closed bar ({bar_time}); skipping")
        return

    exposure, d = target_exposure(df, basket)
    new_dir = _sign(exposure)
    old_dir = int(st.get("direction", 0))
    action = _action(old_dir, new_dir)

    print(f"[{dt.datetime.now():%H:%M:%S}] {bar_time}  close {close:.0f}  "
          f"signal {DIRN[new_dir]} (exp {exposure:+.2f})"
          + (f"  ->  {action}" if action else "  (no change)"))

    if action:
        if new_dir != 0:
            suggested = (f"open {DIRN[new_dir].lower()} ~{abs(exposure) * 100:.0f}% of capital "
                         f"(vol-target, leverage {ps.LEVERAGE}x)")
        else:
            suggested = "close the position / stay flat"
        body = (
            f"SIGNAL: {action}\n\n"
            f"  Time:         {bar_time} (UTC)\n"
            f"  BTC price:    {close:,.2f}\n"
            f"  Direction:    {DIRN[old_dir]}  ->  {DIRN[new_dir]}\n"
            f"  Suggested:    {suggested}\n"
            f"  Net basket:   raw {d['raw']:+.2f}  (L{d['n_long']}/S{d['n_short']}/F{d['n_flat']})\n"
            f"  Trend filter: {'RISK-OFF (defending)' if d['risk_off'] else 'ok'} "
            f"(price vs MA{ps.MA_WIN})\n\n"
            f"Place this trade manually on your exchange. This is a signal alert, not an order.\n"
        )
        notifier.send_email(f"{action}  -  BTC {close:,.0f}", body)

    st.update(direction=new_dir, last_bar=bar_time, last_exposure=round(float(exposure), 3))
    save_state(st)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g.add_argument("--test", action="store_true", help="send a test email and exit")
    a = ap.parse_args()

    load_secrets()
    if a.test:
        ok = notifier.send_email("Signal alerts test",
                                 "If you can read this, signal-alert emails are configured.")
        print("test email sent" if ok else "NOT sent (check EMAIL_* in secrets.env)")
        return

    basket = load_basket()
    print(f"SIGNAL ALERTS (manual trading) | basket {len(basket)} strategies | "
          f"trend filter {ps.USE_TREND}(MA{ps.MA_WIN}) | "
          f"email {'ON -> ' + os.environ.get('EMAIL_TO', '') if notifier.email_configured() else 'OFF'}")

    if a.once:
        try:
            run_once(basket)
        except Exception as e:
            print(f"[{dt.datetime.now():%H:%M:%S}] error: {e}")
            monitor.on_error(e, "signal run_once --once")
        return

    try:
        while True:
            try:
                run_once(basket)
            except Exception as e:
                print(f"[{dt.datetime.now():%H:%M:%S}] error: {e}")
                monitor.on_error(e, "signal run_once")
            now = time.time()
            nxt = (int(now // 3600) + 1) * 3600 + 15
            time.sleep(max(5, nxt - now))
    except KeyboardInterrupt:
        print("stopped by user")
    except BaseException as e:
        monitor.on_crash(e)
        raise


if __name__ == "__main__":
    main()
