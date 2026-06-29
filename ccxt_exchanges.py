"""
Kraken Futures connector for the live trader. USD-settled BTC perpetual (BTC/USD:USD).

  demo=True  -> Kraken FUTURES demo (shorts + leverage, fake money). set_sandbox_mode(True).
  demo=False -> Kraken FUTURES real account (shorts + leverage). *** REAL MONEY ***

Order placement is gated in paper_trader.py by --live AND LIVE_CONFIRM=YES, so connecting here
never by itself trades.

Keys (secrets.env; never commit):
    KRAKEN_FUTURES_DEMO_KEY / KRAKEN_FUTURES_DEMO_SECRET   (demo-futures.kraken.com)
    KRAKEN_FUTURES_KEY / KRAKEN_FUTURES_SECRET             (futures.kraken.com)

Returns (exchange, symbol, market_type) = (ccxt.krakenfutures, "BTC/USD:USD", "swap").
"""
from __future__ import annotations
import os


def _balance(ex, ccys=("USD", "USDT", "USDC")):
    """Best-effort account equity: first non-zero of the given settle currencies."""
    try:
        b = ex.fetch_balance()
        for c in ccys:
            t = b.get(c, {}).get("total")
            if t:
                return float(t), c
        tot = b.get("total") or {}
        cands = [(float(tot[c]), c) for c in ccys if tot.get(c)]
        if cands:
            return max(cands)
    except Exception as e:
        print(f"[kraken] balance note: {e}")
    return 0.0, ccys[0]


def make_kraken(leverage=1, demo=False):
    """Kraken Futures - USD-settled BTC perp. demo=True uses the futures demo (fake money)."""
    import ccxt
    if demo:
        key = os.environ.get("KRAKEN_FUTURES_DEMO_KEY")
        sec = os.environ.get("KRAKEN_FUTURES_DEMO_SECRET")
        kname = "KRAKEN_FUTURES_DEMO_KEY"
    else:
        key = os.environ.get("KRAKEN_FUTURES_KEY")
        sec = os.environ.get("KRAKEN_FUTURES_SECRET")
        kname = "KRAKEN_FUTURES_KEY"
    if not key or not sec:
        raise SystemExit(f"set {kname} / {kname.replace('KEY', 'SECRET')} in secrets.env")

    ex = ccxt.krakenfutures({"apiKey": key, "secret": sec, "enableRateLimit": True})
    if demo:
        ex.set_sandbox_mode(True)
    try:
        ex.load_markets()
    except Exception as e:
        if demo:
            raise SystemExit(
                f"Kraken DEMO futures is unreachable (demo-futures.kraken.com): {e}\n"
                "Kraken's demo environment is frequently down for maintenance (HTTP 503). "
                "Try again later, or run paper mode (omit --testnet), or use --live with a "
                "small balance once you're ready.")
        raise SystemExit(f"Kraken futures unreachable (futures.kraken.com): {e}")

    symbol, lev = "BTC/USD:USD", max(1, int(round(leverage)))
    try:
        ex.set_leverage(lev, symbol)
    except Exception as e:
        print(f"[kraken] set_leverage note: {e}")
    bal, ccy = _balance(ex)
    tag = "DEMO" if demo else "LIVE"
    print(f"[kraken-{tag.lower()}] {tag} futures connected | balance {bal:.2f} {ccy} | "
          f"leverage {lev}x | symbol {symbol}")
    return ex, symbol, "swap"


def make_exchange(mode="testnet", leverage=1):
    """mode='live' -> real futures; anything else -> demo futures."""
    return make_kraken(leverage, demo=(mode != "live"))
