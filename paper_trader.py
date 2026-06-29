"""
Stage-0 paper-trading logger for the cross-confirmed BTC basket.

Each closed 1h bar it: fetches live BTC bars, computes each basket strategy's current
position, NETS them, applies volatility-target sizing (target_ann_vol) with a hard
leverage cap, and LOGS the target exposure / order size + a paper-equity mark.

Data + execution are Kraken Futures (BTC/USD perp). By default NO real orders are placed
(paper accounting only):
  --testnet : DEMO orders on Kraken futures (shorts+leverage, no real money)
  --live    : REAL orders on Kraken futures (real money; needs LIVE_CONFIRM=YES in secrets.env)

    py paper_trader.py --once             # one paper cycle (Task Scheduler, hourly)
    py paper_trader.py --loop             # paper, run forever
    py paper_trader.py --loop --testnet   # Kraken futures DEMO (full strategy, no real money)
    py paper_trader.py --loop --live      # REAL Kraken futures (needs LIVE_CONFIRM=YES)

Outputs:
    paper_log.csv     - one row per processed bar (signals, exposure, order, equity)
    paper_state.json  - persisted state so restarts resume correctly
"""
from __future__ import annotations
import argparse, json, os, time, datetime as dt
import numpy as np, pandas as pd

# the protected strategy (basket + vol-target + trend filter + leverage) is the SINGLE
# source of truth - all strategy/sizing config lives in protected_strategy.py
import protected_strategy as ps
from protected_strategy import target_exposure, load_basket
import monitor    # live monitoring + email alerts (best-effort; never raises)

# ---------------- execution config (data/accounting only) ----------------
DATA_SYMBOL     = "BTC/USD:USD"     # Kraken USD-perp; signal klines + funding come from here
INTERVAL        = "1h"
COST            = 0.0015    # round-trip cost fraction (for paper P&L)
WARMUP_BARS     = 1500      # bars fetched for indicator + MA warm-up (>= MA_WIN + buffer)
START_EQUITY    = 10_000.0
APPLY_FUNDING   = True      # deduct perp funding (the cost a real leveraged account pays)
LOG_FILE        = "paper_log.csv"
STATE_FILE      = "paper_state.json"


# ---------------- data (Kraken Futures public, no keys needed) ----------------
_PUBLIC = [None]   # cached keyless ccxt.krakenfutures client for market data


def _public_kraken():
    if _PUBLIC[0] is None:
        import ccxt
        ex = ccxt.krakenfutures({"enableRateLimit": True})
        ex.load_markets()
        _PUBLIC[0] = ex
    return _PUBLIC[0]


def fetch_klines(symbol=DATA_SYMBOL, interval=INTERVAL, n_bars=WARMUP_BARS):
    """Fetch the last n_bars CLOSED 1h klines from Kraken Futures (public). Returns OHLCV df.
    Kraken returns up to ~2000 bars in one call, so no pagination needed for the warm-up window."""
    ex = _public_kraken()
    tf_ms = 3_600_000
    now_ms = ex.milliseconds()
    batch = ex.fetch_ohlcv(symbol, timeframe=interval, limit=n_bars + 50)
    recs = []
    for r in batch:
        ts = int(r[0])
        if ts + tf_ms > now_ms:                       # drop the in-progress bar
            continue
        recs.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5] or 0.0)))
    recs = recs[-n_bars:]
    df = pd.DataFrame(recs, columns=["t", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("Date").drop(columns="t")


def fetch_funding(symbol=DATA_SYMBOL, limit=50):
    """Recent SETTLED Kraken perp funding rates: list of (fundingTime_ms, rate). Empty on failure."""
    try:
        rows = _public_kraken().fetch_funding_rate_history(symbol, limit=limit)
        return [(int(r["timestamp"]), float(r["fundingRate"])) for r in rows]
    except Exception as e:
        print(f"[funding] fetch failed ({e}); skipping funding this cycle")
        return []


def funding_in_interval(prev_ms, cur_ms):
    """Sum of funding rates that SETTLED in (prev_ms, cur_ms]. Long pays positive funding."""
    if not APPLY_FUNDING or prev_ms is None:
        return 0.0, None
    total = 0.0; last = None
    for ft, fr in fetch_funding():
        if prev_ms < ft <= cur_ms:
            total += fr; last = fr
    return total, last


# signals + sizing + trend filter all live in protected_strategy.target_exposure


# ---------------- state / paper accounting ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"exposure": 0.0, "equity": START_EQUITY, "prev_close": None, "last_bar": None}


def save_state(st):
    json.dump(st, open(STATE_FILE, "w"), indent=2)


def log_row(row):
    hdr = not os.path.exists(LOG_FILE)
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=hdr, index=False)


# ---------------- exchange execution (optional, --testnet / --live) ----------------
# exchange connection lives in ccxt_exchanges.py (Kraken Futures: demo, or real perp futures);
# the symbol + market_type are returned from there. Order placement below is exchange-agnostic ccxt.
MIN_ORDER_USD = 5.0                 # skip orders smaller than this notional
SECRETS_FILE  = "secrets.env"       # KEY=VALUE lines; git-ignored; never commit


def _clean_secret_value(v):
    """Strip surrounding quotes and any trailing inline `# comment` (a # after whitespace)."""
    v = v.strip()
    if not (v.startswith('"') or v.startswith("'")):
        h = v.find("#")
        if h > 0 and v[h - 1].isspace():
            v = v[:h].strip()
    return v.strip().strip('"').strip("'")


def load_secrets(path=SECRETS_FILE):
    """Load KEY=VALUE lines from secrets.env into the environment (if present)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), _clean_secret_value(v))


def _position_and_equity(ex, symbol, market_type):
    """Current BTC position (signed) + settle-currency equity, for spot or swap.
    Settle currency is read from the market (USD on Kraken futures)."""
    bal = ex.fetch_balance()
    if market_type == "spot":
        base = symbol.split("/")[0]                       # BTC
        btc = float(bal.get(base, {}).get("free") or 0.0)
        usdt = float(bal.get("USDT", {}).get("free") or 0.0)
        return btc, usdt + btc * _last_price_hint[0]      # equity = USDT + BTC value
    settle = "USDT"
    try:
        settle = ex.market(symbol).get("settle") or "USDT"
    except Exception:
        pass
    equity = float(bal.get(settle, {}).get("total")
                   or bal.get("USDT", {}).get("total") or 0.0)
    if equity == 0.0:                                     # fall back to any USD-like collateral
        tot = bal.get("total") or {}
        cands = [float(tot[c]) for c in ("USD", "USDT", "USDC") if tot.get(c)]
        equity = max(cands) if cands else 0.0
    pos = 0.0
    for p in ex.fetch_positions([symbol]):
        if p.get("symbol") == symbol and p.get("contracts"):
            sign = -1.0 if p.get("side") == "short" else 1.0
            pos = float(p["contracts"]) * sign
    return pos, equity


_last_price_hint = [0.0]   # set each cycle so spot equity can value the BTC balance


def exec_to_target(ex, symbol, market_type, target_exp, price):
    """Place a market order to move to target exposure. spot = long-only (exposure clamped 0..1)."""
    _last_price_hint[0] = price
    if market_type == "spot":
        target_exp = max(0.0, min(1.0, target_exp))       # no shorts, no leverage on spot
    pos, equity = _position_and_equity(ex, symbol, market_type)
    target_qty = (equity * target_exp) / price
    delta = target_qty - pos
    notional = abs(delta) * price
    out = {"equity": round(equity, 2), "pos_before": round(pos, 6),
           "target_qty": round(target_qty, 6), "delta_qty": round(delta, 6), "order": None}
    if notional < MIN_ORDER_USD:
        return out
    side = "buy" if delta > 0 else "sell"
    amt = ex.amount_to_precision(symbol, abs(delta))
    if float(amt) <= 0:
        return out
    tag = "LIVE" if market_type == "swap" else "demo"    # swap = real futures; spot = testnet
    try:
        o = ex.create_order(symbol, "market", side, amt)
        out["order"] = f"{side} {amt} @market (id {o.get('id')})"
        print(f"[{tag}] ORDER {side} {amt} BTC  (~${notional:.0f})")
    except Exception as e:
        out["order"] = f"FAILED: {e}"
        print(f"[{tag}] order FAILED: {e}")
    return out


def run_once(basket, exchange=None, symbol=None, market_type=None):
    df = fetch_klines()
    bar_time = str(df.index[-1])
    close = float(df["Close"].iloc[-1])
    st = load_state()
    if st["last_bar"] == bar_time:
        print(f"[{dt.datetime.now():%H:%M:%S}] no new closed bar ({bar_time}); skipping")
        return

    cur_ms = int(pd.Timestamp(bar_time).value // 10**6)
    prev_ms = int(pd.Timestamp(st["last_bar"]).value // 10**6) if st.get("last_bar") else None

    # mark the PRIOR exposure over the bar(s) since last processed
    if st["prev_close"] is not None:
        bar_ret = (close - st["prev_close"]) / st["prev_close"]
        st["equity"] *= (1.0 + st["exposure"] * bar_ret)

    # perp FUNDING: long pays positive funding -> impact = -exposure * sum(rates) settled this interval
    fund_rate, fund_impact = 0.0, 0.0
    fr_sum, fr_last = funding_in_interval(prev_ms, cur_ms)
    if fr_sum != 0.0:
        fund_impact = -st["exposure"] * fr_sum
        st["equity"] *= (1.0 + fund_impact)
        fund_rate = fr_last if fr_last is not None else 0.0
        st["funding_cum"] = st.get("funding_cum", 0.0) + fund_impact

    target, d = target_exposure(df, basket)        # netting + vol-size + trend filter + leverage

    # cost on exposure change
    delta = target - st["exposure"]
    if abs(delta) > 1e-9:
        st["equity"] *= (1.0 - abs(delta) * COST)

    order_usd = delta * st["equity"]
    order_qty = order_usd / close
    cum_ret = st["equity"] / START_EQUITY - 1.0

    # LIVE: place a REAL order on Kraken futures to move to the target exposure (only if --live)
    tn = exec_to_target(exchange, symbol, market_type, target, close) if exchange is not None else None

    row = {
        "bar_time": bar_time, "close": round(close, 2),
        "n_long": d["n_long"], "n_short": d["n_short"], "n_flat": d["n_flat"],
        "net_raw": d["raw"], "vol_size": d["vol_size"],
        "sma": d["sma"], "risk_off": d["risk_off"], "gate": d["gate"],
        "target_exposure": round(target, 3), "prev_exposure": round(st["exposure"], 3),
        "delta_exposure": round(delta, 3),
        "order_side": ("BUY" if delta > 0 else "SELL" if delta < 0 else "-"),
        "order_qty_btc": round(order_qty, 6), "order_usd": round(order_usd, 2),
        "funding_rate": round(fund_rate, 6), "funding_impact": round(fund_impact, 6),
        "funding_cum": round(st.get("funding_cum", 0.0), 5),
        "paper_equity": round(st["equity"], 2), "cum_return": round(cum_ret, 4),
        "testnet_equity": (tn["equity"] if tn else None),
        "testnet_order": (tn["order"] if tn else None),
    }
    log_row(row)
    fund_note = f"  funding {fund_impact*100:+.3f}%" if fund_impact != 0.0 else ""
    flag = "  [TREND-OFF: longs flattened]" if d["risk_off"] and d["raw"] > 0 else ""
    print(f"[{dt.datetime.now():%H:%M:%S}] {bar_time}  close {close:.0f}  "
          f"net {d['raw']:+.2f} (L{d['n_long']}/S{d['n_short']}/F{d['n_flat']})  "
          f"-> exposure {target:+.2f}  {row['order_side']} {abs(order_qty):.4f}BTC  "
          f"equity {st['equity']:.0f} ({cum_ret:+.1%}){fund_note}{flag}")

    # ---- live monitoring + email alerts (trade / daily summary / heartbeat) ----
    if tn is not None:                                   # on an exchange (testnet/real)
        live_equity = tn["equity"] if tn.get("equity") is not None else st["equity"]
        traded = bool(tn.get("order")) and not str(tn["order"]).startswith("FAILED")
        order_desc = tn.get("order") or "-"
    else:                                                # paper accounting
        live_equity = st["equity"]
        traded = abs(delta) > 1e-9
        order_desc = (f"{row['order_side']} {abs(order_qty):.6f} BTC (~${abs(order_usd):,.0f})"
                      if traded else "-")
    monitor.on_bar(live_equity, bar_time=bar_time, close=close, exposure=target,
                   prev_exposure=st["exposure"], traded=traded, order_desc=order_desc,
                   detail=d, cum_return=cum_ret, live=(exchange is not None))

    st.update(exposure=target, prev_close=close, last_bar=bar_time)
    save_state(st)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    g2 = ap.add_mutually_exclusive_group()
    g2.add_argument("--testnet", action="store_true",
                    help="Kraken futures DEMO orders (shorts+leverage, no real money)")
    g2.add_argument("--live", action="store_true",
                    help="place REAL orders on Kraken futures (REAL MONEY)")
    a = ap.parse_args()

    load_secrets()      # always: loads EMAIL_* / ANTHROPIC_API_KEY (+ Kraken keys) into env
    exchange = symbol = market_type = None
    if a.testnet or a.live:
        import ccxt_exchanges
        lev = max(1, int(round(ps.MAX_LEVERAGE)))
        if a.live:
            if os.environ.get("LIVE_CONFIRM", "").strip().upper() != "YES":
                raise SystemExit("--live refused: set LIVE_CONFIRM=YES in secrets.env to enable "
                                 "REAL Kraken-futures trading (real money).")
            print("=" * 60)
            print("  !!!  LIVE TRADING ON KRAKEN FUTURES - REAL MONEY  !!!")
            print("=" * 60)
        exchange, symbol, market_type = ccxt_exchanges.make_kraken(lev, demo=not a.live)

    basket = load_basket()
    mode = ("PAPER (no orders)" if exchange is None else
            f"Kraken {'LIVE (REAL orders)' if a.live else 'DEMO'} ({market_type})")
    print(f"loaded basket of {len(basket)} strategies | target_vol {ps.TARGET_ANN_VOL} "
          f"| leverage {ps.LEVERAGE}x | max_lev {ps.MAX_LEVERAGE} | "
          f"trend filter {ps.USE_TREND}(MA{ps.MA_WIN}) | funding {APPLY_FUNDING} | {mode}")
    if monitor.notifier.email_configured():
        print(f"[monitor] email alerts ENABLED -> {os.environ.get('EMAIL_TO')}")
    else:
        print("[monitor] email alerts OFF (set EMAIL_* in secrets.env to enable)")

    if a.once:
        try:
            run_once(basket, exchange, symbol, market_type)
        except Exception as e:
            print(f"[{dt.datetime.now():%H:%M:%S}] error: {e}")
            monitor.on_error(e, "run_once --once")
        return

    monitor.on_start(mode)
    try:
        while True:
            try:
                run_once(basket, exchange, symbol, market_type)
            except Exception as e:
                print(f"[{dt.datetime.now():%H:%M:%S}] error: {e}")
                monitor.on_error(e, "run_once")
            now = time.time()
            nxt = (int(now // 3600) + 1) * 3600 + 15      # 15s after the next hour close
            time.sleep(max(5, nxt - now))
    except KeyboardInterrupt:
        print("stopped by user")
    except BaseException as e:                            # unexpected fatal -> crash email
        monitor.on_crash(e)
        raise


if __name__ == "__main__":
    main()
