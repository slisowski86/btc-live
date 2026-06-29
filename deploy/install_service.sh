#!/usr/bin/env bash
# Install + start systemd services that run the trader 24/7 (auto-restart on crash/reboot)
# plus the watchdog. Run after setup_vps.sh:
#     bash deploy/install_service.sh                       # paper (no orders)
#     bash deploy/install_service.sh "--loop --testnet"    # Kraken demo orders
#     bash deploy/install_service.sh "--loop --live"       # REAL Kraken orders (needs LIVE_CONFIRM=YES)
set -euo pipefail

RUN_ARGS="${1:---loop}"          # default: paper. Pass "--loop --live" for real trading.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="btc_live"
PYTHON="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
USER_NAME="$(whoami)"
UNIT="/etc/systemd/system/btc-paper.service"
WD_UNIT="/etc/systemd/system/btc-watchdog.service"

[ -x "$PYTHON" ] || { echo "ERROR: $PYTHON not found — run deploy/setup_vps.sh first"; exit 1; }

echo "installing $UNIT (user=$USER_NAME, dir=$PROJECT_DIR, args=$RUN_ARGS)"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=BTC_Live protected trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON paper_trader.py $RUN_ARGS
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/paper_service.log
StandardError=append:$PROJECT_DIR/paper_service.log

[Install]
WantedBy=multi-user.target
EOF

echo "installing $WD_UNIT (watchdog: heartbeat alerts + dead-man's-switch)"
sudo tee "$WD_UNIT" >/dev/null <<EOF
[Unit]
Description=BTC_Live watchdog (alerts when the trader stops)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON watchdog.py --loop --interval-min 15
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/watchdog.log
StandardError=append:$PROJECT_DIR/watchdog.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable btc-paper btc-watchdog
sudo systemctl restart btc-paper btc-watchdog
sleep 2
sudo systemctl --no-pager status btc-paper | head -n 12 || true
echo
echo "Services running. Useful commands:"
echo "  live logs : journalctl -u btc-paper -f      (watchdog: journalctl -u btc-watchdog -f)"
echo "  stop      : sudo systemctl stop btc-paper btc-watchdog"
echo "  restart   : sudo systemctl restart btc-paper"
echo "  disable   : sudo systemctl disable --now btc-paper btc-watchdog"
