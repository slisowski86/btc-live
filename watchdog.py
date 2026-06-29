"""
Healthcheck / watchdog for the trading script. Detects when the trader has stopped - crashed,
hung, or the whole machine rebooted - and alerts you. Optionally pings an external dead-man's-
switch and/or auto-restarts the trader.

HOW IT KNOWS THE TRADER IS ALIVE
  The trader writes a heartbeat file (heartbeat.txt) every cycle (monitor.heartbeat). If that file
  goes stale beyond the threshold, the trader is considered DOWN. This catches both a crash (file
  stops updating) and a hang (file stops updating) - more reliable than a bare "is the PID alive".

RUN IT (independently of the trader)
  Scheduled one-shot (recommended on a server) - every ~15-20 min:
      Windows Task Scheduler:  py watchdog.py
      Linux cron:              */20 * * * * cd /opt/BTC_Live && python watchdog.py
  Standalone loop (if you can't schedule) - checks itself every N minutes:
      py watchdog.py --loop --interval-min 15

OPTIONS
  --stale-min M    minutes with no heartbeat before "down" (default 130 ~= 2h, one missed cycle)
  --url URL        external dead-man's-switch. On every OK check it GETs URL; on a DOWN check it
                   GETs URL + "/fail". If the WHOLE SERVER dies, this watchdog can't email you -
                   but the external service (e.g. a free https://healthchecks.io check) notices the
                   pings stopped and alerts you. Or set HEALTHCHECK_URL in secrets.env.
  --restart "CMD"  shell command to relaunch the trader when it's down (auto-recovery), e.g.
                   --restart "py paper_trader.py --loop --live"

It emails ONCE on DOWN and ONCE on RECOVER (state in watchdog_state.json) - no spam.
"""
from __future__ import annotations
import argparse, os, json, time, datetime as dt, subprocess
import urllib.request

import notifier

WD_STATE       = "watchdog_state.json"
HEARTBEAT_FILE = "heartbeat.txt"


def load_secrets(path="secrets.env"):
    """Load KEY=VALUE lines from secrets.env into env (strips inline comments). Light - no heavy imports."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if not (v.startswith('"') or v.startswith("'")):
            h = v.find("#")
            if h > 0 and v[h - 1].isspace():
                v = v[:h].strip()
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _ping(url, fail=False):
    """Best-effort GET to an external dead-man's-switch. Never raises."""
    if not url:
        return
    target = url.rstrip("/") + "/fail" if fail else url
    try:
        urllib.request.urlopen(target, timeout=15).read()
    except Exception as e:
        print(f"[watchdog] dead-man's-switch ping failed: {e}")


def _restart(cmd):
    """Relaunch the trader, detached. Best-effort."""
    if not cmd:
        return False
    try:
        kwargs = {"shell": True}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(cmd, **kwargs)
        print(f"[watchdog] restart launched: {cmd}")
        return True
    except Exception as e:
        print(f"[watchdog] restart failed: {e}")
        return False


def check(stale_seconds, url=None, restart_cmd=None):
    """One health check: email + ping + optional restart on transition. Returns True if down."""
    st = {}
    if os.path.exists(WD_STATE):
        try:
            st = json.load(open(WD_STATE))
        except Exception:
            st = {}
    was_down = bool(st.get("down", False))

    if os.path.exists(HEARTBEAT_FILE):
        age = time.time() - os.path.getmtime(HEARTBEAT_FILE)
        down = age > stale_seconds
        detail = f"last heartbeat {age / 3600:.1f}h ago"
    else:
        down, detail = True, "no heartbeat file found (trader may have never started)"

    if down and not was_down:
        restarted = _restart(restart_cmd)
        notifier.send_email(
            "WATCHDOG: trader appears DOWN",
            f"The trading script has not updated its heartbeat.\n  {detail}\n"
            f"  threshold {stale_seconds / 3600:.1f}h\n\n"
            + (f"Auto-restart was launched: {restart_cmd}\n" if restarted else
               "Check the server / process and restart the trader if needed.\n"))
        print(f"[watchdog] ALERT sent - {detail}")
    elif was_down and not down:
        notifier.send_email("WATCHDOG: trader RECOVERED",
                            f"The trading script heartbeat is fresh again ({detail}).\n")
        print(f"[watchdog] recovery email sent - {detail}")
    elif down:
        _restart(restart_cmd)                    # keep trying to bring it back while down
        print(f"[watchdog] DOWN - {detail}")
    else:
        print(f"[watchdog] ok - {detail}")

    _ping(url, fail=down)                         # external dead-man's-switch
    json.dump({"down": down, "checked": dt.datetime.now(dt.timezone.utc).isoformat(),
               "detail": detail}, open(WD_STATE, "w"), indent=2)
    return down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="run forever instead of a single check")
    ap.add_argument("--interval-min", type=float, default=15, help="loop check interval (default 15 min)")
    ap.add_argument("--stale-min", type=float, default=130, help="no-heartbeat minutes -> down (default 130)")
    ap.add_argument("--url", default=None, help="external dead-man's-switch URL (or HEALTHCHECK_URL env)")
    ap.add_argument("--restart", default=None, help="shell command to relaunch the trader when down")
    a = ap.parse_args()

    load_secrets()
    url = a.url or os.environ.get("HEALTHCHECK_URL")
    stale = a.stale_min * 60
    print(f"watchdog | stale>{a.stale_min:.0f}min | "
          f"deadman={'on' if url else 'off'} | autorestart={'on' if a.restart else 'off'} | "
          f"{'loop ' + str(a.interval_min) + 'min' if a.loop else 'once'}")

    if not a.loop:
        check(stale, url, a.restart)
        return
    try:
        while True:
            check(stale, url, a.restart)
            time.sleep(max(30, a.interval_min * 60))
    except KeyboardInterrupt:
        print("watchdog stopped by user")


if __name__ == "__main__":
    main()
