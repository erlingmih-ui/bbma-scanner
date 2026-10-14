"""Données M5 / M15 / H1 depuis Dukascopy (gratuit, sans clé, sans retard de 15 min).

Module NON testé en réseau par son auteur : tout échec renvoie None et bbma_core retombe sur yfinance.
Pour désactiver : USE_DUKASCOPY = False (ici), ou variable d'environnement BBMA_DUKASCOPY=0.
D1 / W1 / MN restent sur yfinance.
"""
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

import pandas as pd

log = logging.getLogger("bbma.duka")

try:
    import dukascopy_python as dk
    from dukascopy_python import instruments as _ins
except Exception:                       # paquet absent : on reste sur yfinance
    dk = None
    _ins = None

USE_DUKASCOPY = os.environ.get("BBMA_DUKASCOPY", "1") != "0"
OFFER_SIDE = "BID"                      # "BID" ou "ASK" (BID = courbe habituelle des graphiques)
TIMEOUT = 40                            # secondes max par téléchargement
BREAKER_FAILS = 4                       # échecs consécutifs avant pause
BREAKER_PAUSE = 300                     # secondes de pause (yfinance prend le relais)

INTERVALS = {"5m": "INTERVAL_MIN_5", "15m": "INTERVAL_MIN_15", "1h": "INTERVAL_HOUR_1"}

_pool = ThreadPoolExecutor(4, thread_name_prefix="duka")
_lock = threading.Lock()
_fails = 0
_paused_until = 0.0
_inst_cache = {}


def active(interval):
    """Dukascopy gère-t-il cet intervalle (réglage actif et paquet présent) ?"""
    return bool(USE_DUKASCOPY and dk is not None and interval in INTERVALS)


def instrument(pair):
    """Code Dukascopy ('EUR/USD'...) d'une paire 'EURUSD', ou None si inconnue."""
    pair = pair.upper()
    if pair in _inst_cache:
        return _inst_cache[pair]
    code = None
    if _ins is not None and re.fullmatch(r"[A-Z]{6}", pair):
        suffix = f"_{pair[:3]}_{pair[3:]}"
        for name in dir(_ins):
            if name.startswith("INSTRUMENT_") and name.endswith(suffix):
                code = getattr(_ins, name)
                break
    _inst_cache[pair] = code
    return code


def _days(period):
    m = re.fullmatch(r"(\d+)([dwy])", str(period))
    if not m:
        return None
    return int(m.group(1)) * {"d": 1, "w": 7, "y": 365}[m.group(2)]


def _normalize(df):
    """DataFrame Open/High/Low/Close à index UTC croissant, ou None."""
    if df is None or not len(df):
        return None
    df = df.rename(columns=lambda c: str(c).lower())
    if not {"open", "high", "low", "close"} <= set(df.columns):
        return None
    df = df[["open", "high", "low", "close"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close"]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def fetch(pair, interval, period):
    """OHLC (colonnes Open/High/Low/Close, index UTC) ou None. Ne lève jamais d'exception."""
    global _fails, _paused_until
    if not active(interval):
        return None
    code = instrument(pair)
    days = _days(period)
    if code is None or days is None:
        return None
    if time.time() < _paused_until:
        return None
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()
    start = end - pd.Timedelta(days=days).to_pytimedelta()
    side = dk.OFFER_SIDE_ASK if OFFER_SIDE == "ASK" else dk.OFFER_SIDE_BID
    try:
        fut = _pool.submit(dk.fetch, code, getattr(dk, INTERVALS[interval]), side,
                           start, end, max_retries=2)
        df = _normalize(fut.result(timeout=TIMEOUT))
    except _Timeout:
        log.warning("Dukascopy %s %s : délai dépassé", pair, interval)
        df = None
    except Exception as e:
        log.warning("Dukascopy %s %s : %s", pair, interval, e)
        df = None
    with _lock:
        if df is None:
            _fails += 1
            if _fails >= BREAKER_FAILS:
                _paused_until = time.time() + BREAKER_PAUSE
                _fails = 0
                log.warning("Dukascopy en pause %d s, yfinance utilisé", BREAKER_PAUSE)
        else:
            _fails = 0
    return df
