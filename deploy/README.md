# Deploying BTC_Live to a DigitalOcean droplet (web console)

Runs the trader + watchdog 24/7 with auto-restart. Tested on Ubuntu 22.04/24.04. The bot is
tiny — the cheapest droplet (1 vCPU / 1 GB RAM) is plenty.

Everything below is typed into the droplet's **web console** (Droplet → Access → Launch Console).
The only thing the web console can't do is copy files off your PC — so the code comes in via `git`.

## 1. Get the code onto the droplet (via GitHub)

**On your PC (one time)** — push the project to a *private* GitHub repo. `secrets.env` is
git-ignored, so your keys never leave your machine:

```powershell
cd C:\Users\sliso\BTC_Live
git add -A
git commit -m "deploy"
git remote add origin https://github.com/<youruser>/BTC_Live.git   # create the repo first (Private)
git branch -M main
git push -u origin main
```

**On the droplet (web console):**

```bash
sudo apt-get update -y && sudo apt-get install -y git
git clone https://github.com/<youruser>/BTC_Live.git
cd BTC_Live
```

A private repo will ask for a username + password — use a **GitHub Personal Access Token**
(github.com → Settings → Developer settings → Fine-grained tokens, read-only on this repo) as the
password.

## 2. Install the environment

```bash
bash deploy/setup_vps.sh
```

Installs Miniconda + a `btc_live` conda env (numpy/pandas/numba/**TA-Lib**/ccxt). TA-Lib installs
cleanly here via conda-forge — no manual C build. Takes a few minutes.

## 3. Create secrets.env on the droplet

It is NOT in the repo (git-ignored), so create it by hand and paste your values:

```bash
nano secrets.env
```
Paste the Kraken keys, email settings, `LIVE_CONFIRM`, and (optional) `HEALTHCHECK_URL` /
`ANTHROPIC_API_KEY` — see `secrets.env.example` for the exact names. Save with Ctrl-O, Enter,
Ctrl-X.

Then verify the connection and one cycle:
```bash
PY=~/miniconda3/envs/btc_live/bin/python
$PY test_connection.py            # real Kraken (read-only)
$PY paper_trader.py --once        # one paper cycle: basket loaded, exposure, equity
```

## 4. Install the 24/7 services

```bash
bash deploy/install_service.sh                    # paper (no real orders)
# or, when you're ready for real trading (needs LIVE_CONFIRM=YES in secrets.env):
bash deploy/install_service.sh "--loop --live"    # REAL Kraken futures
```

This installs two systemd services that start on boot and auto-restart on crash:
- **btc-paper** — the trader (`paper_trader.py <args>`)
- **btc-watchdog** — `watchdog.py --loop` (emails you if the trader stops; pings your
  dead-man's-switch if `HEALTHCHECK_URL` is set)

## 5. Operate it

```bash
journalctl -u btc-paper -f         # live trader logs
journalctl -u btc-watchdog -f      # watchdog logs
sudo systemctl status btc-paper    # state
sudo systemctl restart btc-paper   # after editing config / pulling new code
sudo systemctl stop btc-paper btc-watchdog
```

Update code later: `cd ~/BTC_Live && git pull && sudo systemctl restart btc-paper btc-watchdog`.

## Notes

- **Data + execution are Kraken Futures** (BTC/USD perp). Data needs no keys; trading uses
  `KRAKEN_FUTURES_KEY/SECRET` (real) or `KRAKEN_FUTURES_DEMO_*` (demo).
- **Going live is two switches:** `LIVE_CONFIRM=YES` in `secrets.env` *and* the `--live` arg on the
  service. Start small (`LEVERAGE=1.0` in `protected_strategy.py`).
- State (`paper_state.json`), the log (`paper_log.csv`), and `daily_stats.csv` persist across
  restarts, so the bot resumes cleanly.
- Not latency-sensitive (acts on the hourly close) — any droplet region works; uptime matters more.
