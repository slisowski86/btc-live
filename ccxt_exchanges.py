"""
Binance USDT-M FUTURES connector for the live trader.  *** REAL MONEY ***

There is no usable Binance futures TESTNET via ccxt (it is dead / retired), so this connects to
LIVE Binance futures: shorts and leverage work, but every order placed is real. Order placement
is gated twice in paper_trader.py - it requires BOTH the `--live` flag AND LIVE_CONFIRM=YES in
secrets.env - so connecting here never by itself trades.

Keys (real, with FUTURES trading permission enabled, from Binance -> API Management):
    BINANCE_KEY / BINANCE_SECRET    (in secrets.env; never commit)

Returns (exchange, symbol, market_type) = (ccxt.binanceusdm, "BTC/USDT:USDT", "swap").
"""
from __future__ import annotations
import os


def make_binance_futures(leverage):
    """Connect to LIVE Binance USDT-M futures. Real account, real balance, real orders."""
    import ccxt
    key = os.environ.get("BINANCE_KEY")
    sec = os.environ.get("BINANCE_SECRET")
    if not key or not sec:
        raise SystemExit("set BINANCE_KEY / BINANCE_SECRET (real futures API keys) in secrets.env")

    ex = ccxt.binanceusdm({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.load_markets()
    symbol = "BTC/USDT:USDT"

    lev = max(1, int(round(leverage)))
    try:
        ex.set_leverage(lev, symbol)
    except Exception as e:
        print(f"[binance-futures] set_leverage note: {e}")
    try:
        bal = float(ex.fetch_balance().get("USDT", {}).get("total") or 0.0)
    except Exception as e:
        bal = 0.0
        print(f"[binance-futures] balance note: {e}")

    print(f"[binance-futures] LIVE connected | balance {bal:.2f} USDT | "
          f"leverage {lev}x | symbol {symbol}")
    return ex, symbol, "swap"


def make_exchange(name=None, leverage=1):
    """Single supported exchange: live Binance USDT-M futures (name is ignored)."""
    return make_binance_futures(leverage)
