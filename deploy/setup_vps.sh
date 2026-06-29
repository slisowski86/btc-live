#!/usr/bin/env bash
# One-shot VPS setup for BTC_Live: installs Miniconda + the btc_live env (incl. TA-Lib).
# Works on x86_64 and ARM (aarch64) Ubuntu/Debian. Run as a normal (non-root) user:
#     bash deploy/setup_vps.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="btc_live"
MINICONDA_DIR="$HOME/miniconda3"

echo "== BTC_Live VPS setup =="
echo "project : $PROJECT_DIR"
echo "env     : $ENV_NAME"

# 1. Miniconda (handles TA-Lib's C library cleanly via conda-forge) -------------
if [ ! -d "$MINICONDA_DIR" ]; then
  echo "-- installing Miniconda ..."
  ARCH="$(uname -m)"
  if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
  else
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  fi
  command -v wget >/dev/null 2>&1 || { sudo apt-get update -y && sudo apt-get install -y wget; }
  wget -qO /tmp/miniconda.sh "$URL"
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm -f /tmp/miniconda.sh
else
  echo "-- Miniconda already present"
fi
# shellcheck disable=SC1091
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# accept Anaconda channel Terms of Service (required by recent conda) -----------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 2. environment ---------------------------------------------------------------
if conda env list | grep -qE "^$ENV_NAME[[:space:]]"; then
  echo "-- env $ENV_NAME already exists; ensuring deps ..."
else
  echo "-- creating env $ENV_NAME ..."
  conda create -n "$ENV_NAME" -c conda-forge python=3.12 -y
fi
conda install -n "$ENV_NAME" -c conda-forge \
  ta-lib numba numpy pandas plotly tqdm matplotlib ccxt -y
# optional: Claude daily review in the email summary
conda run -n "$ENV_NAME" pip install anthropic >/dev/null 2>&1 || \
  echo "-- note: 'anthropic' not installed (optional; only for the Claude review)"

# 3. verify --------------------------------------------------------------------
echo "-- verifying ..."
cd "$PROJECT_DIR"
conda run -n "$ENV_NAME" python -c "import numpy,pandas,numba,talib,matplotlib; print('deps OK')"
conda run -n "$ENV_NAME" python -c "import protected_strategy as ps; print('basket', len(ps.load_basket()), 'strategies | MA', ps.MA_WIN)"

PYBIN="$MINICONDA_DIR/envs/$ENV_NAME/bin/python"
echo
echo "DONE. Python: $PYBIN"
echo "Quick test:  cd $PROJECT_DIR && $PYBIN paper_trader.py --once"
echo "Then install the service:  bash deploy/install_service.sh"
