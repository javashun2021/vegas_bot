#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegas Channel (EMA144/169) Daily Bot with Volume/RSI filters for Binance via ccxt.

Features:
- Timeframe: 1d (daily close only)
- Indicators: EMA(144), EMA(169), RSI(14), SMA(volume, 20), ATR(14)
- Filters: Trend + Volume (vol>vol_sma) + RSI threshold (long>50, short<50)
- Risk mgmt: Risk-per-trade sizing from stop distance; ATR-based SL/TP (configurable multiple)
- Modes: Spot or USD-M Futures; Testnet toggle; DRY_RUN toggle
- Order handling: market entry + stop/take-profit for futures; spot uses OCO when supported (fallback prints warnings)
- New candle detection: persists last traded daily ts in a local state file (./_state.json)

! IMPORTANT !
- Install deps:  pip install ccxt pandas numpy
- Set API keys via environment variables or fill below:
    BINANCE_API_KEY, BINANCE_API_SECRET
- Start with DRY_RUN=True and/or USE_TESTNET=True
"""

import os, json, time, math, traceback, sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt

# ===================== User Config =====================
API_KEY = os.getenv("BINANCE_API_KEY", "your_key_here")
API_SECRET = os.getenv("BINANCE_API_SECRET", "your_secret_here")

SYMBOL          = "ETH/USDT"   # e.g., "ETH/USDT"
TIMEFRAME       = "1d"
EMA_FAST        = 144
EMA_SLOW        = 169
RSI_LEN         = 14
VOL_SMA_LEN     = 20
ATR_LEN         = 14

USE_FUTURES     = True         # True: USD-M Futures; False: Spot
LEVERAGE        = 2            # for futures
DRY_RUN         = True         # True: simulate only (prints), False: real orders
USE_TESTNET     = True         # recommend True at first

RISK_PER_TRADE  = 0.01         # risk 1% of quote balance per trade
ATR_SL_MULT     = 1.0          # SL distance in ATR
ATR_TP_MULT     = 2.0          # TP distance in ATR

STATE_FILE      = "eth_state.json" # persist last-candle-time processed to avoid duplicate trading
POLL_SECS       = 60            # loop check interval
# =======================================================

def log(*a):
    print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("WARN: failed to save state:", e)

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def fetch_klines(exchange, symbol: str, timeframe: str, limit: int = 600) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def get_exchange():
    if USE_FUTURES:
        ex = ccxt.binanceusdm({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
    else:
        ex = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    if USE_TESTNET:
        ex.set_sandbox_mode(True)
    return ex

def ensure_leverage(exchange, symbol, lev: int):
    if not USE_FUTURES:
        return
    try:
        market = exchange.market(symbol)
        exchange.set_leverage(lev, market["id"])
        log(f"Leverage set to {lev} for {symbol}")
    except Exception as e:
        log("WARN: set_leverage failed:", e)

def fetch_quote_balance(exchange) -> float:
    try:
        if USE_FUTURES:
            bal = exchange.fetch_balance(params={"type": "future"})
        else:
            bal = exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log("WARN: fetch_balance failed:", e)
        return 0.0

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["vol_sma"] = df["volume"].rolling(VOL_SMA_LEN).mean()
    df["atr"] = atr(df, ATR_LEN)
    return df.dropna()

def new_daily_closed_candle(df: pd.DataFrame, last_done_iso: str) -> bool:
    # Determine the *last closed* candle timestamp (2nd last row if exchange includes current forming bar)
    # ccxt fetch_ohlcv returns CLOSED candles for historical; last row is the last CLOSED candle.
    last_ts = df["ts"].iloc[-1]
    last_iso = last_ts.isoformat()
    return last_iso != last_done_iso, last_iso

def amount_to_precision(exchange, symbol, amount):
    return float(exchange.amount_to_precision(symbol, amount))

def price_to_precision(exchange, symbol, price):
    return float(exchange.price_to_precision(symbol, price))

def generate_signal(row_prev, row):
    """
    Use previous closed bar context (row_prev) and the latest closed bar (row).
    Entry rules (daily close confirmation):
      Long:
        - Trend: close>EMA144 and EMA144>EMA169
        - "Pullback then reclaim": previous day close within channel [min(EMA144,EMA169), max(...)]
          and today close > EMA144
        - Filters: RSI > 50 and volume > vol_sma
      Short (futures only):
        - Trend: close<EMA144 and EMA144<EMA169
        - Pullback then reject: prev in channel and today close < EMA144
        - Filters: RSI < 50 and volume > vol_sma
    """
    in_channel_prev = (min(row_prev["ema_fast"], row_prev["ema_slow"]) <= row_prev["close"] <= max(row_prev["ema_fast"], row_prev["ema_slow"]))

    # Long
    long_trend = (row["close"] > row["ema_fast"]) and (row["ema_fast"] > row["ema_slow"])
    long_reclaim = in_channel_prev and (row["close"] > row["ema_fast"])
    long_filters_ok = (row["rsi"] > 50.0) and (row["volume"] > row["vol_sma"])

    if long_trend and long_reclaim and long_filters_ok:
        # ATR-based SL/TP
        sl = min(row["ema_slow"], row["close"] - ATR_SL_MULT * row["atr"])
        tp = row["close"] + ATR_TP_MULT * row["atr"]
        return "long", float(sl), float(tp)

    # Short (futures only)
    if USE_FUTURES:
        short_trend = (row["close"] < row["ema_fast"]) and (row["ema_fast"] < row["ema_slow"])
        short_reject = in_channel_prev and (row["close"] < row["ema_fast"])
        short_filters_ok = (row["rsi"] < 50.0) and (row["volume"] > row["vol_sma"])

        if short_trend and short_reject and short_filters_ok:
            sl = max(row["ema_slow"], row["close"] + ATR_SL_MULT * row["atr"])
            tp = row["close"] - ATR_TP_MULT * row["atr"]
            return "short", float(sl), float(tp)

    return None, None, None

def calc_qty_from_risk(balance_quote, entry_price, stop_price):
    risk_cap = balance_quote * RISK_PER_TRADE
    sl_dist = abs(entry_price - stop_price)
    if sl_dist <= 0:
        return 0.0
    qty = risk_cap / sl_dist
    if USE_FUTURES:
        qty *= LEVERAGE  # notional sizing for futures
    return max(0.0, qty)

def place_entry_and_exits(exchange, symbol, side, qty, entry_price, stop_price, take_profit):
    side_ccxt = "buy" if side == "long" else "sell"
    if DRY_RUN:
        log(f"[DRY_RUN] ENTRY {side_ccxt} {qty} {symbol} @~{entry_price}; SL={stop_price}; TP={take_profit}")
        return

    # Market entry
    order = exchange.create_order(symbol, "market", side_ccxt, qty)
    log("Entry order placed:", order.get("id"))

    if USE_FUTURES:
        reduce_side = "sell" if side_ccxt == "buy" else "buy"
        # Stop (stop-market)
        try:
            exchange.create_order(symbol, type="stop_market", side=reduce_side, amount=qty,
                                  params={"stopPrice": price_to_precision(exchange, symbol, stop_price),
                                          "reduceOnly": True})
            log("Futures stop set.")
        except Exception as e:
            log("WARN: Futures stop failed:", e)

        # Take Profit (take-profit-market)
        try:
            exchange.create_order(symbol, type="take_profit_market", side=reduce_side, amount=qty,
                                  params={"stopPrice": price_to_precision(exchange, symbol, take_profit),
                                          "reduceOnly": True})
            log("Futures take-profit set.")
        except Exception as e:
            log("WARN: Futures take-profit failed:", e)
    else:
        # Spot: try OCO (sell-only); for long-only systems this is fine
        if side_ccxt != "buy":
            log("WARN: Spot short not supported; skip exit placement.")
            return
        try:
            # Some markets support OCO; required params can vary by market
            exchange.create_order(symbol, "oco", "sell", qty,
                                  price_to_precision(exchange, symbol, take_profit),
                                  params={"stopPrice": price_to_precision(exchange, symbol, stop_price)})
            log("Spot OCO (TP/SL) placed.")
        except Exception as e:
            log("WARN: Spot OCO unsupported here; you must manage exits via a watcher. Err:", e)

def main_loop():
    if not API_KEY or not API_SECRET or "your_key_here" in API_KEY:
        log("WARNING: API keys not set. DRY_RUN will be forced True.")
        global DRY_RUN
        DRY_RUN = True

    exchange = get_exchange()
    markets = exchange.load_markets()
    if SYMBOL not in markets:
        raise RuntimeError(f"Symbol {SYMBOL} not in markets. Check case / availability.")

    if USE_FUTURES:
        ensure_leverage(exchange, SYMBOL, LEVERAGE)

    state = load_state()
    last_done_iso = state.get("last_daily_iso", "")

    log(f"Bot started. SYMBOL={SYMBOL} TF={TIMEFRAME} Futures={USE_FUTURES} Testnet={USE_TESTNET} DRY_RUN={DRY_RUN}")
    while True:
        try:
            df_raw = fetch_klines(exchange, SYMBOL, TIMEFRAME, limit=600)
            df = compute_indicators(df_raw)
            ready, last_iso = new_daily_closed_candle(df, last_done_iso)

            last = df.iloc[-1]
            prev = df.iloc[-2]

            log(f"Last daily close={last['close']:.2f} ts={last['ts']} | ready_new_candle={ready}")

            if ready:
                # Generate signal on newly closed candle
                sig, sl, tp = generate_signal(prev, last)
                log(f"Signal={sig} SL={sl} TP={tp}")
                if sig:
                    entry = float(last["close"])
                    balance = fetch_quote_balance(exchange)
                    qty = calc_qty_from_risk(balance, entry, sl)
                    qty = amount_to_precision(exchange, SYMBOL, qty)
                    if qty > 0:
                        place_entry_and_exits(exchange, SYMBOL, sig, qty, entry, sl, tp)
                    else:
                        log("Qty too small; skip order.")
                else:
                    log("No signal.")

                # mark candle done
                last_done_iso = last_iso
                state["last_daily_iso"] = last_done_iso
                save_state(state)

        except Exception as e:
            log("ERROR:", e)
            traceback.print_exc()

        time.sleep(POLL_SECS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log("Bye.")
