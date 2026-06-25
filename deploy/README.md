# Deploying BTC_Live to a VPS

Runs the protected paper trader 24/7 with auto-restart. Tested for Ubuntu 22.04/24.04
(x86_64 or ARM). The bot is tiny — 1 vCPU / 1–2 GB RAM / ~15 GB disk is plenty.

## 1. Get the code onto the VPS

From your PC (PowerShell), copy the project (excluding caches):

```powershell
scp -r C:\Users\sliso\BTC_Live user@YOUR_VPS_IP:~/BTC_Live
```

…or `git clone` it if you've pushed it to a repo. Either way you need the folder at
`~/BTC_Live` on the server (with `protected_strategy.py`, `paper_trader.py`,
`btc_basket_crossconfirmed.json`, the `indicators/` folder, etc.).

## 2. Install the environment

```bash
cd ~/BTC_Live
bash deploy/setup_vps.sh
```

This installs Miniconda + a `btc_live` conda env (numpy/pandas/numba/**TA-Lib**/…),
then verifies the strategy imports and the basket loads.

Quick manual check:
```bash
~/miniconda3/envs/btc_live/bin/python paper_trader.py --once
```
You should see one line: basket loaded, current exposure, paper equity.

## 3. Install the 24/7 service

```bash
bash deploy/install_service.sh
```

This creates `/etc/systemd/system/btc-paper.service` (with your user + paths filled in),
enables it (starts on boot), and starts it now. It runs `paper_trader.py --loop` and
**auto-restarts** on crash or reboot.

## 4. Operate it

```bash
journalctl -u btc-paper -f        # live logs
tail -f ~/BTC_Live/paper_service.log
sudo systemctl status btc-paper   # state
sudo systemctl restart btc-paper  # after editing config
sudo systemctl stop btc-paper     # stop
sudo systemctl disable --now btc-paper   # stop + don't start on boot
```

## Config

All strategy/risk settings live in `protected_strategy.py` (leverage, MA window, target
vol) and `paper_trader.py` (cost, funding). Edit, then `sudo systemctl restart btc-paper`.

State persists in `paper_state.json`; the hourly log is `paper_log.csv`. Both survive
restarts, so the bot resumes cleanly.

## Notes

- **Paper only** — `paper_trader.py` places no real orders; it logs target orders + paper
  equity. To go live you swap the logged order for a real exchange call (e.g. via `ccxt`).
- **TA-Lib** installs cleanly here via `conda-forge` (no manual C-library build, unlike
  Windows).
- The bot is **not latency-sensitive** (acts on the hourly close), so any VPS region works;
  uptime matters more than location.
