"""
External watchdog: checks that the trader's heartbeat file is fresh and emails if it is not.

A hard crash (power loss, OOM, the process killed, the VPS rebooting) leaves the trader unable
to email for itself - so run this INDEPENDENTLY of the trader, on a schedule:

    Windows Task Scheduler:  py watchdog.py            every 15-30 min
    Linux cron:              */20 * * * * cd /opt/BTC_Live && python watchdog.py
    systemd timer:           pair watchdog.service with a 20-min OnCalendar timer

It alerts ONCE when the heartbeat goes stale and ONCE when it recovers (state in
watchdog_state.json), so a down trader doesn't spam you every run.

The trader writes a heartbeat each hourly cycle, so "> ~2h since last heartbeat" = down.
"""
from __future__ import annotations
import os, json, time, datetime as dt

import notifier

WD_STATE       = "watchdog_state.json"
STALE_SECONDS  = 2 * 3600 + 600             # 2h10m: one missed hourly cycle + slack


def _age_seconds(path):
    if not os.path.exists(path):
        return None
    return time.time() - os.path.getmtime(path)


def main():
    # load secrets so EMAIL_* is available, and reuse the trader's heartbeat path
    try:
        from paper_trader import load_secrets
        load_secrets()
    except Exception as e:
        print(f"[watchdog] could not load secrets.env: {e}")
    try:
        from monitor import HEARTBEAT_FILE as HB
    except Exception:
        HB = "heartbeat.txt"

    st = {}
    if os.path.exists(WD_STATE):
        try:
            st = json.load(open(WD_STATE))
        except Exception:
            st = {}
    was_down = bool(st.get("down", False))

    age = _age_seconds(HB)
    if age is None:
        down, detail = True, "no heartbeat file found (trader may have never started)"
    else:
        down = age > STALE_SECONDS
        detail = f"last heartbeat {age / 3600:.1f}h ago"

    if down and not was_down:
        notifier.send_email(
            "WATCHDOG: trader appears DOWN",
            f"The live trader has not updated its heartbeat.\n  {detail}\n"
            f"  Staleness threshold: {STALE_SECONDS / 3600:.1f}h\n\n"
            f"Check the server / process and restart the trader if needed.\n")
        print(f"[watchdog] ALERT sent - {detail}")
    elif was_down and not down:
        notifier.send_email(
            "WATCHDOG: trader RECOVERED",
            f"The live trader heartbeat is fresh again ({detail}).\n")
        print(f"[watchdog] recovery email sent - {detail}")
    else:
        print(f"[watchdog] {'DOWN' if down else 'ok'} - {detail}")

    json.dump({"down": down, "checked": dt.datetime.now(dt.timezone.utc).isoformat(),
               "detail": detail}, open(WD_STATE, "w"), indent=2)


if __name__ == "__main__":
    main()
