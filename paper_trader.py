"""
Stage-0 paper-trading logger for the cross-confirmed BTC basket.

Each closed 1h bar it: fetches live BTC bars, computes each basket strategy's current
position, NETS them, applies volatility-target sizing (target_ann_vol) with a hard
leverage cap, and LOGS the target exposure / order size + a paper-equity mark.
NO real orders are placed.

    py paper_trader.py --once      # one cycle (use with Windows Task Scheduler, hourly)
    py paper_trader.py --loop      # run forever, waking ~15s after each hour close

Outputs:
    paper_log.csv     - one row per processed bar (signals, exposure, order, paper equity)
    paper_state.json  - persisted state so restarts resume correctly
"""
from __future__ import annotations
import argparse, json, os, time, datetime as dt, urllib.request
import numpy as np, pandas as pd

# the protected strategy (basket + vol-target + trend filter + leverage) is the SINGLE
# source of truth - all strategy/sizing config lives in protected_strategy.py
import protected_strategy as ps
from protected_strategy import target_exposure, load_basket

# ---------------- execution config (data/accounting only) ----------------
SYMBOL          = "BTCUSDT"
INTERVAL        = "1h"
COST            = 0.0015    # round-trip cost fraction (for paper P&L)
WARMUP_BARS     = 1500      # bars fetched for indicator + MA warm-up (>= MA_WIN + buffer)
START_EQUITY    = 10_000.0
APPLY_FUNDING   = True      # deduct perp funding (the cost a real leveraged account pays)
LOG_FILE        = "paper_log.csv"
STATE_FILE      = "paper_state.json"
BINANCE         = "https://api.binance.com/api/v3/klines"
FUNDING_URL     = "https://fapi.binance.com/fapi/v1/fundingRate"   # perp funding history


# ---------------- data ----------------
def fetch_klines(symbol=SYMBOL, interval=INTERVAL, n_bars=WARMUP_BARS):
    """Fetch the last n_bars CLOSED klines from Binance (paginated). Returns OHLCV df."""
    out = {}
    end = None
    while len(out) < n_bars + 5:
        url = f"{BINANCE}?symbol={symbol}&interval={interval}&limit=1000"
        if end:
            url += f"&endTime={end}"
        with urllib.request.urlopen(url, timeout=30) as r:
            rows = json.loads(r.read())
        if not rows:
            break
        for k in rows:
            out[int(k[0])] = k
        end = int(rows[0][0]) - 1
        if len(rows) < 1000:
            break
    now_ms = int(time.time() * 1000)
    recs = []
    for k in sorted(out):
        close_time = int(k if False else out[k][6])  # ms
        if close_time >= now_ms:
            continue                                  # drop the in-progress bar
        recs.append((int(out[k][0]), float(out[k][1]), float(out[k][2]),
                     float(out[k][3]), float(out[k][4]), float(out[k][5])))
    recs = recs[-n_bars:]
    df = pd.DataFrame(recs, columns=["t", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("Date").drop(columns="t")


def fetch_funding(symbol=SYMBOL, limit=50):
    """Recent SETTLED perp funding rates: list of (fundingTime_ms, rate). Empty on failure."""
    try:
        with urllib.request.urlopen(f"{FUNDING_URL}?symbol={symbol}&limit={limit}", timeout=30) as r:
            rows = json.loads(r.read())
        return [(int(x["fundingTime"]), float(x["fundingRate"])) for x in rows]
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


# ---------------- testnet execution (optional, --testnet) ----------------
CCXT_SYMBOL   = "BTC/USDT:USDT"     # Binance USDT-M perpetual (ccxt notation)
MIN_ORDER_USD = 5.0                 # skip orders smaller than this notional
SECRETS_FILE  = "secrets.env"       # KEY=VALUE lines; git-ignored; never commit


def load_secrets(path=SECRETS_FILE):
    """Load KEY=VALUE lines from secrets.env into the environment (if present)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_exchange():
    """Connect to the Binance Futures TESTNET via ccxt using env-stored keys."""
    try:
        import ccxt
    except ImportError:
        raise SystemExit("ccxt not installed:  pip install ccxt")
    key = os.environ.get("BINANCE_TESTNET_KEY")
    sec = os.environ.get("BINANCE_TESTNET_SECRET")
    if not key or not sec:
        raise SystemExit("missing keys: set BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET "
                         f"(env vars or {SECRETS_FILE})")
    ex = ccxt.binanceusdm({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.set_sandbox_mode(True)                       # <-- Binance futures TESTNET (fake money)
    ex.load_markets()
    lev = max(2, int(round(ps.MAX_LEVERAGE)))
    try:
        ex.set_leverage(lev, CCXT_SYMBOL)
    except Exception as e:
        print(f"[testnet] set_leverage warning: {e}")
    bal = float(ex.fetch_balance()["USDT"]["total"])
    print(f"[testnet] connected | balance {bal:.2f} USDT | leverage {lev}x | symbol {CCXT_SYMBOL}")
    return ex


def _current_position(ex):
    """Signed current position size (contracts) on the testnet, + USDT equity."""
    equity = float(ex.fetch_balance()["USDT"]["total"])
    pos = 0.0
    for p in ex.fetch_positions([CCXT_SYMBOL]):
        if p.get("symbol") == CCXT_SYMBOL and p.get("contracts"):
            sign = -1.0 if p.get("side") == "short" else 1.0
            pos = float(p["contracts"]) * sign
    return pos, equity


def exec_to_target(ex, target_exp, price):
    """Place a market order on the testnet to move to target exposure. Returns a detail dict."""
    pos, equity = _current_position(ex)
    target_qty = (equity * target_exp) / price          # leveraged notional / price
    delta = target_qty - pos
    notional = abs(delta) * price
    out = {"equity": round(equity, 2), "pos_before": round(pos, 6),
           "target_qty": round(target_qty, 6), "delta_qty": round(delta, 6), "order": None}
    if notional < MIN_ORDER_USD:
        return out
    side = "buy" if delta > 0 else "sell"
    amt = ex.amount_to_precision(CCXT_SYMBOL, abs(delta))
    if float(amt) <= 0:
        return out
    try:
        o = ex.create_order(CCXT_SYMBOL, "market", side, amt)
        out["order"] = f"{side} {amt} @market (id {o.get('id')})"
        print(f"[testnet] ORDER {side} {amt} BTC  (~${notional:.0f})")
    except Exception as e:
        out["order"] = f"FAILED: {e}"
        print(f"[testnet] order FAILED: {e}")
    return out


def run_once(basket, exchange=None):
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

    # TESTNET: place a real (simulated) order to move to the target exposure
    tn = exec_to_target(exchange, target, close) if exchange is not None else None

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

    st.update(exposure=target, prev_close=close, last_bar=bar_time)
    save_state(st)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--loop", action="store_true")
    ap.add_argument("--testnet", action="store_true",
                    help="place real SIMULATED orders on the Binance Futures testnet")
    a = ap.parse_args()

    exchange = None
    if a.testnet:
        load_secrets()
        exchange = make_exchange()

    basket = load_basket()
    mode = "TESTNET orders" if exchange is not None else "PAPER (no orders)"
    print(f"loaded basket of {len(basket)} strategies | target_vol {ps.TARGET_ANN_VOL} "
          f"| leverage {ps.LEVERAGE}x | max_lev {ps.MAX_LEVERAGE} | "
          f"trend filter {ps.USE_TREND}(MA{ps.MA_WIN}) | funding {APPLY_FUNDING} | {mode}")

    if a.once:
        run_once(basket, exchange)
        return
    while True:
        try:
            run_once(basket, exchange)
        except Exception as e:
            print(f"[{dt.datetime.now():%H:%M:%S}] error: {e}")
        now = time.time()
        nxt = (int(now // 3600) + 1) * 3600 + 15      # 15s after the next hour close
        time.sleep(max(5, nxt - now))


if __name__ == "__main__":
    main()
