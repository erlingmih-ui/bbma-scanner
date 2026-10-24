"""
Multi-Timeframe Market Analysis Script (Android-friendly)
-------------------------------------------------------------
Analyzes trend, support/resistance, and entry signals across
15-minute, 1-hour, 4-hour, and daily timeframes for:
  XAUUSD (Gold), BTCUSD, GBPUSD, EURUSD

Fetches data directly from Yahoo Finance's public chart API via
'requests' - no yfinance needed, since yfinance depends on packages
(curl_cffi, lxml) that don't have prebuilt wheels for Android and
require a C compiler Pydroid3/Termux may not have.

Install dependencies (Pydroid3: menu -> Pip -> search & install):
    pip install requests pandas numpy

Run:
    python market_analysis.py
"""

import time
import numpy as np
import pandas as pd

try:
    import requests
except ImportError as e:
    raise SystemExit(f"Missing dependency 'requests'. Run: pip install requests\n{e}")

SYMBOLS = {
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
}

# Yahoo chart API valid "range" values: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
TIMEFRAMES = {
    "15m": {"interval": "15m", "range": "1mo", "resample": None},
    "1h":  {"interval": "60m", "range": "3mo", "resample": None},
    "4h":  {"interval": "60m", "range": "3mo", "resample": "4h"},
    "1d":  {"interval": "1d",  "range": "6mo", "resample": None},
}

SR_WINDOW = 20
RSI_PERIOD = 14
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


def fetch_data(ticker, interval, rng, resample=None, retries=3):
    url = BASE_URL.format(ticker)
    params = {"interval": interval, "range": rng}

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            result = data["chart"]["result"][0]
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    else:
        raise ValueError(f"Failed to fetch {ticker} ({interval}): {last_err}")

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
    }, index=pd.to_datetime(timestamps, unit="s"))
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({interval})")

    if resample:
        df = df.resample(resample).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last",
        }).dropna()

    return df


def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    return df


def get_trend(df):
    if len(df) < 50 or df["SMA50"].isna().all():
        return "Not enough data for trend"
    last = df.iloc[-1]
    price, sma20, sma50 = last["Close"], last["SMA20"], last["SMA50"]
    if pd.isna(sma20) or pd.isna(sma50):
        return "Not enough data for trend"
    if price > sma20 > sma50:
        return "Uptrend"
    elif price < sma20 < sma50:
        return "Downtrend"
    return "Sideways / mixed"


def get_support_resistance(df, window=SR_WINDOW):
    recent = df.tail(window)
    return recent["Low"].min(), recent["High"].max()


def suggest_entry(df, trend, support, resistance):
    price = df.iloc[-1]["Close"]
    rsi = df.iloc[-1]["RSI"]
    dist_to_support = (price - support) / price * 100
    dist_to_resistance = (resistance - price) / price * 100

    notes = []
    if trend == "Uptrend":
        if dist_to_support < 1.5:
            notes.append(f"Near support ({support:.4f}) -> possible BUY zone")
        if not pd.isna(rsi) and rsi < 40:
            notes.append(f"RSI {rsi:.1f} = pullback, watch for bounce")
        if not notes:
            notes.append(f"Wait for pullback toward support ({support:.4f})")
    elif trend == "Downtrend":
        if dist_to_resistance < 1.5:
            notes.append(f"Near resistance ({resistance:.4f}) -> possible SELL zone")
        if not pd.isna(rsi) and rsi > 60:
            notes.append(f"RSI {rsi:.1f} = exhaustion, watch for rejection")
        if not notes:
            notes.append(f"Wait for rally toward resistance ({resistance:.4f})")
    else:
        notes.append("No clear trend - range trade S/R or wait")
    return notes


def analyze_timeframe(ticker, tf_name, tf_cfg):
    df = fetch_data(ticker, tf_cfg["interval"], tf_cfg["range"], tf_cfg["resample"])
    df = add_indicators(df)

    price = df.iloc[-1]["Close"]
    trend = get_trend(df)
    support, resistance = get_support_resistance(df)
    rsi = df.iloc[-1]["RSI"]
    notes = suggest_entry(df, trend, support, resistance)
    last_time = df.index[-1]

    print(f"  [{tf_name}] as of {last_time}")
    print(f"    Price: {price:.4f} | Trend: {trend}")
    line = f"    Support: {support:.4f}  Resistance: {resistance:.4f}"
    if not pd.isna(rsi):
        line += f"  RSI: {rsi:.1f}"
    print(line)
    for note in notes:
        print(f"    - {note}")

    return {
        "price": price, "trend": trend, "support": support,
        "resistance": resistance, "rsi": rsi,
    }


# Which timeframes feed each trading style.
# Scalping: fast timeframes, tight entries off the smaller structure.
# Swing: slower timeframes, wider entries off the bigger structure.
STYLE_TFS = {
    "Scalping": ["15m", "1h"],
    "Swing":    ["4h", "1d"],
}
STOP_BUFFER = {"Scalping": 0.003, "Swing": 0.01}  # 0.3% / 1% beyond the level


def build_style_suggestion(results, style_name):
    tfs = STYLE_TFS[style_name]
    valid = {tf: results[tf] for tf in tfs if tf in results}
    if len(valid) < len(tfs):
        return [f"{style_name}: missing data for {', '.join(tfs)}"]

    trends = {r["trend"] for r in valid.values()}
    buffer = STOP_BUFFER[style_name]
    fast_tf = tfs[0]           # faster timeframe sets the precise entry level
    fast = valid[fast_tf]
    slow_tf = tfs[1]
    slow = valid[slow_tf]
    current_price = fast["price"]

    if trends == {"Uptrend"}:
        entry = fast["support"]
        stop = entry * (1 - buffer)
        target = slow["resistance"]
        dist_pct = (current_price - entry) / current_price * 100
        dist_note = (
            f"  Price is already at/near entry ({dist_pct:.2f}% away)"
            if abs(dist_pct) < 0.3
            else f"  Price is {dist_pct:.2f}% above entry - not touched yet, set an alert"
        )
        return [
            f"{style_name} BUY bias ({fast_tf} & {slow_tf} both uptrend)",
            f"  Entry ~{entry:.4f}  Stop ~{stop:.4f}  Target ~{target:.4f}",
            dist_note,
        ]
    elif trends == {"Downtrend"}:
        entry = fast["resistance"]
        stop = entry * (1 + buffer)
        target = slow["support"]
        dist_pct = (entry - current_price) / current_price * 100
        dist_note = (
            f"  Price is already at/near entry ({dist_pct:.2f}% away)"
            if abs(dist_pct) < 0.3
            else f"  Price is {dist_pct:.2f}% below entry - not touched yet, set an alert"
        )
        return [
            f"{style_name} SELL bias ({fast_tf} & {slow_tf} both downtrend)",
            f"  Entry ~{entry:.4f}  Stop ~{stop:.4f}  Target ~{target:.4f}",
            dist_note,
        ]
    else:
        detail = ", ".join(f"{tf}={valid[tf]['trend']}" for tf in tfs)
        return [f"{style_name}: timeframes disagree ({detail}) - no clean setup, wait"]


def analyze_symbol(name, ticker):
    print(f"\n===== {name} =====")
    results = {}
    for tf_name, tf_cfg in TIMEFRAMES.items():
        try:
            results[tf_name] = analyze_timeframe(ticker, tf_name, tf_cfg)
        except Exception as e:
            print(f"  [{tf_name}] Error: {e}")

    print("  --- Trade style suggestions ---")
    for style_name in STYLE_TFS:
        for line in build_style_suggestion(results, style_name):
            print(f"  {line}")


def main():
    print("Multi-Timeframe Market Analysis Report")
    print("(Educational use only - not financial advice)")
    for name, ticker in SYMBOLS.items():
        analyze_symbol(name, ticker)


if __name__ == "__main__":
    main()
