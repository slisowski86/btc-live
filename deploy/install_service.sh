#!/usr/bin/env bash
# Install + start the systemd service that runs the protected paper trader 24/7
# (auto-restart on crash/reboot). Run after setup_vps.sh:
#     bash deploy/install_service.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="btc_live"
PYTHON="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
USER_NAME="$(whoami)"
UNIT="/etc/systemd/system/btc-paper.service"

[ -x "$PYTHON" ] || { echo "ERROR: $PYTHON not found — run deploy/setup_vps.sh first"; exit 1; }

echo "installing $UNIT (user=$USER_NAME, dir=$PROJECT_DIR)"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=BTC_Live protected paper trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON paper_trader.py --loop
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/paper_service.log
StandardError=append:$PROJECT_DIR/paper_service.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable btc-paper
sudo systemctl restart btc-paper
sleep 2
sudo systemctl --no-pager status btc-paper | head -n 12 || true
echo
echo "Service running. Useful commands:"
echo "  live logs : journalctl -u btc-paper -f     (or: tail -f $PROJECT_DIR/paper_service.log)"
echo "  stop      : sudo systemctl stop btc-paper"
echo "  restart   : sudo systemctl restart btc-paper"
echo "  disable   : sudo systemctl disable --now btc-paper"
