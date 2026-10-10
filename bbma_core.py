"""Logique pure du scanner BBMA : données, détection, synthèse. Aucune dépendance à Streamlit."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("bbma")

# Symboles yfinance particuliers (le reste = PAIRE=X)
SYMBOLS = {"XAUUSD": "GC=F", "BTCUSD": "BTC-USD"}

def sym(pair):
    return SYMBOLS.get(pair, pair + "=X")

# TF -> (intervalle yfinance, période, rééchantillonnage)
TFS = {"MN": ("1mo", "10y", None), "W1": ("1wk", "5y", None), "D1": ("1d", "2y", None),
       "H4": ("1h", "180d", "4h"), "H1": ("1h", "60d", None), "M15": ("15m", "30d", None),
       "M5": ("5m", "30d", None)}
PRIORITY = ["MOM", "EXM", "EXT", "MHV", "CSAK", "RE"]   # priorité de détection (cycle BBMA)

TF_ORDER = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]   # du plus haut au plus bas
SIG_DIRECTEUR = {"MOM", "EXM", "EXT", "RE"}              # signaux de structure
SIG_ENTREE = {"RE", "MHV", "CSAK", "EXT"}                # signaux déclencheurs (EXT inclus pour M15/M5)

RETRY = 2          # essais par téléchargement
PAUSE = 1.5        # secondes entre essais (multipliées par le numéro d'essai)
MIN_BARS = 60      # historique minimum pour calculer un signal
DL_WORKERS = 6

# Correction 2 : le filtre D1 utilise la dernière bougie D1 CLÔTURÉE, comme le signal.
# Mettre False pour retrouver le comportement d'origine (bougie en formation incluse).
D1_FILTER_CLOSED_ONLY = True


def wma(s, n):
    """WMA vectorisée (poids 1..n, le plus récent pèse n). Résultat identique à rolling().apply()."""
    a = s.to_numpy(dtype=float)
    w = np.arange(1, n + 1, dtype=float)
    out = np.convolve(a, w[::-1], mode="full")[: len(a)] / w.sum()
    out[: n - 1] = np.nan
    return pd.Series(out, index=s.index)


def detect(df, lb=15):
    """Signal de la dernière bougie : (code, sens) avec sens 'B' (buy) ou 'S' (sell).
    Priorité : MOM > EXM (Extreme Magic, MA10) > EXT (MA5) > MHV > CSAK (Candle Arah) > RE.
    TP Wajib = flag déclenché par EXM/EXT uniquement (affiché à côté du code)."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    top, low = mid + 2 * sd, mid - 2 * sd
    h5, h10, l5, l10 = wma(h, 5), wma(h, 10), wma(l, 5), wma(l, 10)
    rec = lambda s, n: s.fillna(False).astype(bool).rolling(n, min_periods=1).max().astype(bool)

    mom_b, mom_s = c > top, c < low                                             # MOM (clôture hors BB)
    exm_s, exm_b = h10 > top, l10 < low                                         # Extreme Magic (MA10)
    ext_s, ext_b = h5 > top, l5 < low                                           # Extreme (MA5)
    any_s, any_b = exm_s | ext_s, exm_b | ext_b
    mhv_s = rec(any_s.shift(1), lb) & (h >= top) & (c < top)                    # MHV (fenêtre 15 bougies)
    mhv_b = rec(any_b.shift(1), lb) & (l <= low) & (c > low)
    arah_s = (c < mid) & (c < l5) & (c < l10)                                   # CSAK / Candle Arah
    arah_b = (c > mid) & (c > h5) & (c > h10)
    # tendance mémorisée (m_lastTrendDir) : dernier MOM / Arah
    trend = pd.Series(np.where(mom_b | arah_b, 1, np.where(mom_s | arah_s, -1, np.nan)), index=c.index).ffill()
    # Zones MA5/MA10
    zone_buy_hi, zone_buy_lo = np.maximum(l5, l10), np.minimum(l5, l10)
    zone_sell_hi, zone_sell_lo = np.maximum(h5, h10), np.minimum(h5, h10)

    # RE strict : bougie précédente HORS zone, bougie actuelle DANS la zone
    re_b = ((trend == 1)
            & (c.shift(1) < zone_buy_lo.shift(1))
            & (c >= zone_buy_lo) & (c <= zone_buy_hi)
            & (l <= zone_buy_hi))
    re_s = ((trend == -1)
            & (c.shift(1) > zone_sell_hi.shift(1))
            & (c >= zone_sell_lo) & (c <= zone_sell_hi)
            & (h >= zone_sell_lo))

    sig = {"MOM": (mom_b, mom_s), "EXM": (exm_b, exm_s), "EXT": (ext_b, ext_s),
           "MHV": (mhv_b, mhv_s), "CSAK": (arah_b, arah_s), "RE": (re_b, re_s)}
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[-1]: return code, "B"
        if s.iloc[-1]: return code, "S"
    return None, None


def synthese(cells, tf_dispo, use_filter, d1_trend):
    """Synthèse multi-TF BBMA Oma Ally.
    cells : {tf: (code, sens)} ; tf_dispo : TF scannés ; d1_trend : 'B', 'S' ou None.
    Retourne (tf_entree, code_entree, sens) ou (None, None, None)."""
    tfs_ord = [t for t in TF_ORDER if t in tf_dispo]

    # 1. TF directeur : le plus haut TF ayant un signal de structure
    idx = None
    for i, t in enumerate(tfs_ord):
        if cells.get(t, (None, None))[0] in SIG_DIRECTEUR:
            idx = i
            break
    if idx is None:
        return None, None, None
    tf_dir = tfs_ord[idx]
    code_dir, sens = cells[tf_dir]

    # 4. Un TF supérieur au directeur qui contredit le sens -> pas de trade
    for t in tfs_ord[:idx]:
        code, d = cells.get(t, (None, None))
        if code and d != sens:
            return None, None, None

    # 5. Filtre D1 (EMA50) : sens contre la tendance D1 -> pas de trade
    if use_filter and d1_trend and d1_trend != sens:
        return None, None, None

    # 2. TF d'entrée plus bas que le directeur, même sens (le plus proche du directeur)
    for t in tfs_ord[idx + 1:]:
        code, d = cells.get(t, (None, None))
        if code in SIG_ENTREE and d == sens:
            return t, code, sens

    # 3. Aucun TF d'entrée trouvé : le directeur sert d'entrée
    return tf_dir, code_dir, sens


def confiance_score(cells, tfs, sens_ref):
    """Part des TF scannés ayant un signal dans le même sens que l'entrée BBMA (0 à 1)."""
    if not tfs or sens_ref is None:
        return None
    n = sum(1 for t in tfs if cells.get(t, (None, None))[1] == sens_ref)
    return n / len(tfs)


def _history(pair, interval, period):
    """Télécharge les bougies avec réessais espacés. Retourne un DataFrame OHLC ou None."""
    for attempt in range(RETRY):
        try:
            df = yf.Ticker(sym(pair)).history(period=period, interval=interval)
            if df is not None and len(df):
                df = df[["Open", "High", "Low", "Close"]].dropna()
                if len(df):
                    return df
        except Exception as e:
            log.warning("yfinance %s %s %s (essai %d) : %s", pair, interval, period, attempt + 1, e)
        if attempt < RETRY - 1:
            time.sleep(PAUSE * (attempt + 1))
    return None


def _fetch_all(pairs, tfs, trend):
    """Un seul téléchargement par (paire, intervalle, période).
    Le D1 du filtre de tendance est le même téléchargement que le TF D1 : il n'est pas refait."""
    keys = {(p, TFS[t][0], TFS[t][1]) for p in pairs for t in tfs}
    if trend:
        keys |= {(p, "1d", "2y") for p in pairs}
    keys = list(keys)
    with ThreadPoolExecutor(DL_WORKERS) as ex:
        return dict(zip(keys, ex.map(lambda k: _history(*k), keys)))


def _scan_one(pair, tf, data):
    """((code, sens), ok) pour une paire sur un timeframe, à partir des données déjà téléchargées."""
    interval, period, rs = TFS[tf]
    df = data.get((pair, interval, period))
    if df is None:
        return (None, None), False
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    # Exclure la dernière bougie en cours de formation (critique sur M5/M15)
    df = df.iloc[:-1]
    if len(df) < MIN_BARS:
        return (None, None), False
    try:
        return detect(df), True
    except Exception:
        return (None, None), False


def _d1_trend(pair, data):
    """Filtre D1 : 'B' si clôture > EMA50, 'S' si <, None si données indisponibles."""
    df = data.get((pair, "1d", "2y"))
    if df is None:
        return None
    if D1_FILTER_CLOSED_ONLY:
        df = df.iloc[:-1]
    if len(df) < MIN_BARS:
        return None
    c = df["Close"]
    return "B" if c.iloc[-1] > c.ewm(span=50, adjust=False).mean().iloc[-1] else "S"


def scan(pairs, tfs, trend=True):
    """Scan des paires sur les TF demandés.
    Retourne (res, trends, failed, total, ts). trend=False : pas de téléchargement D1 pour le filtre."""
    pairs, tfs = list(pairs), list(tfs)
    data = _fetch_all(pairs, tfs, trend)
    jobs = [(p, t) for p in pairs for t in tfs]
    res, failed = {}, 0
    for p, t in jobs:
        r, ok = _scan_one(p, t, data)
        res[(p, t)] = r
        failed += not ok
    trends = {p: _d1_trend(p, data) for p in pairs} if trend else {}
    return res, trends, failed, len(jobs), datetime.now(timezone.utc)
