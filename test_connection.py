"""
Read-only connection test for Kraken Futures. Verifies your API keys work and the account is
reachable. Places NO orders and changes NO settings - it only READS (price, balance, position).

    py test_connection.py             # REAL Kraken futures (KRAKEN_FUTURES_KEY / _SECRET)
    py test_connection.py --testnet   # DEMO Kraken futures (KRAKEN_FUTURES_DEMO_KEY / _SECRET)

Run the demo check before demo trading, and the real check before ever using --live. Pinpoints:
keys missing/misnamed, wrong/disabled key, trading permission off, or the demo being down (503).
"""
from __future__ import annotations
import argparse, os, sys


def _load_secrets(path="secrets.env"):
    if not os.path.exists(path):
        sys.exit("FAIL: secrets.env not found in this folder")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testnet", action="store_true", help="use the Kraken futures DEMO keys")
    a = ap.parse_args()
    _load_secrets()

    try:
        import ccxt
    except ImportError:
        sys.exit("FAIL: ccxt not installed  ->  pip install ccxt")

    kname = "KRAKEN_FUTURES_DEMO_KEY" if a.testnet else "KRAKEN_FUTURES_KEY"
    key, sec = os.environ.get(kname), os.environ.get(kname.replace("KEY", "SECRET"))
    if not key or not sec:
        sys.exit(f"FAIL: {kname} / {kname.replace('KEY','SECRET')} not set in secrets.env")
    symbol = "BTC/USD:USD"
    print(f"keys loaded: {kname} ...{key[-4:]} (len {len(key)})  | "
          f"Kraken {'DEMO' if a.testnet else 'REAL'} futures")

    ex = ccxt.krakenfutures({"apiKey": key, "secret": sec, "enableRateLimit": True})
    if a.testnet:
        ex.set_sandbox_mode(True)

    # 1) public reachability + live price
    try:
        ex.load_markets()
        px = ex.fetch_ticker(symbol)["last"]
        print(f"OK  public : {symbol} last price {px:,.2f}")
    except Exception as e:
        host = "demo-futures.kraken.com" if a.testnet else "futures.kraken.com"
        sys.exit(f"FAIL: cannot reach Kraken ({host}): {e}\n"
                 + ("      Kraken's demo is often down for maintenance (HTTP 503) - try later."
                    if a.testnet else ""))

    # 2) private: balance - authenticates the key AND checks trading permission
    try:
        bal = ex.fetch_balance()
        print("OK  auth   : wallet reached")
        totals = {c: float(v) for c, v in (bal.get("total") or {}).items() if v and float(v) != 0}
        if totals:
            for c, v in totals.items():
                print(f"            {c} total {v:,.4f}")
        else:
            print("            (no non-zero balance - the futures wallet looks unfunded)")
    except ccxt.AuthenticationError as e:
        sys.exit(f"FAIL: authentication error - bad/disabled key or secret: {e}")
    except ccxt.PermissionDenied as e:
        sys.exit(f"FAIL: permission denied - enable trading on the API key: {e}")
    except Exception as e:
        sys.exit(f"FAIL: balance call failed: {e}\n"
                 f"      likely: clock skew or trading permission not enabled")

    # 3) open position (read-only)
    try:
        pos = [p for p in ex.fetch_positions([symbol]) if p.get("contracts")]
        if pos:
            for p in pos:
                print(f"            open position: {p.get('side')} {p.get('contracts')} "
                      f"@ {p.get('entryPrice')}")
        else:
            print("            open position: none (flat)")
    except Exception as e:
        print(f"            position read note: {e}")

    print("\nCONNECTION OK - read-only, no orders placed, no settings changed.")


if __name__ == "__main__":
    main()
