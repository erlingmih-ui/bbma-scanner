"""Logique pure du scanner BBMA : données, détection, synthèse. Aucune dépendance à Streamlit."""
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

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

TF_ORDER = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]   # du plus haut au plus bas

RETRY = 2          # essais par téléchargement
PAUSE = 1.5        # secondes entre essais (multipliées par le numéro d'essai)
MIN_BARS = 60      # historique minimum pour calculer un signal
DL_WORKERS = 6

# Correction 2 : le filtre D1 utilise la dernière bougie D1 CLÔTURÉE, comme le signal.
# Mettre False pour retrouver le comportement d'origine (bougie en formation incluse).
D1_FILTER_CLOSED_ONLY = True

# Cycle BBMA : un CSAK n'est retenu que s'il suit un MHV de même sens (lui-même précédé d'un Extreme)
# dans les CSAK_WINDOW bougies précédentes. Mettre False pour retrouver le CSAK brut d'origine.
# Seul le CSAK affiché est filtré : la tendance mémorisée du RE reste calculée sur le CSAK brut.
CSAK_CYCLE_ONLY = True
CSAK_WINDOW = 15

# Cache disque des bougies : un fichier Parquet par (paire, intervalle), dans ./cache à côté de ce module.
# Les bougies fermées sont conservées et s'accumulent d'un lancement à l'autre.
CACHE_DIR = Path(__file__).resolve().parent / "cache"
# Période du premier téléchargement complet (= période maximale utilisée par TFS pour cet intervalle)
CACHE_PERIOD = {"5m": "30d", "15m": "30d", "1h": "180d", "1d": "2y", "1wk": "5y", "1mo": "10y"}
# Fenêtre de rattrapage pour une mise à jour incrémentale (au-delà : téléchargement complet)
INCR_DAYS = {"5m": 5, "15m": 5, "1h": 10, "1d": 30}
BAR_DELTA = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1),
             "1d": pd.Timedelta(days=1), "1wk": pd.Timedelta(weeks=1)}
_file_locks = {}
_locks_guard = threading.Lock()


def wma(s, n):
    """WMA vectorisée (poids 1..n, le plus récent pèse n). Résultat identique à rolling().apply()."""
    a = s.to_numpy(dtype=float)
    w = np.arange(1, n + 1, dtype=float)
    out = np.convolve(a, w[::-1], mode="full")[: len(a)] / w.sum()
    out[: n - 1] = np.nan
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------------
# MOTEUR DE CYCLE BBMA (v2) — machine à états par sens.
# Cycle : EXTREME -> MHV -> CSAK -> MOM -> RE. Chaque étape est indispensable à la suivante.
# Hypothèses (à valider, toutes paramétrables) :
#   H1  MID = SMA20 de la clôture ; bandes = MID ± 2 écarts-types (population).
#   H2  Zones MA : WMA5 / WMA10 des plus bas (achat) ou des plus hauts (vente).
#   H3  Mèche de rejet : >= RE_WICK de l'amplitude, du côté de la zone.
#   H4  Clôture RE : 'above_low' = clôture au-dessus du bord bas de la zone RE ; 'inside' = dans la zone.
#   H5  Zone RE = [min(bord bas MA, MID) ; bord haut MA] : le RE peut se produire sur la MA ou sur la MID.
#       Bougie précédente : clôture hors zone, du côté de la tendance.
#   H6  Chaque étape doit survenir dans CSAK_WINDOW bougies après l'étape précédente.
#   H7  Réinitialisation : clôture du mauvais côté de la MID à partir de CSAK, ou délai dépassé.
#   H8  Le premier RE valide après MOM clôt le cycle ; un nouveau cycle peut démarrer.
#   H9  EXT / EXM : départ possible du cycle (EXT) et alertes d'affichage, jamais signal d'entrée.
# ---------------------------------------------------------------------------
MID_N = 20
RE_WICK = 0.30            # H3
RE_CLOSE = "above_low"    # H4 : "above_low" ou "inside"


def _frame(df, sens):
    """Indicateurs dans le sens du trade. Pour une vente, les prix sont inversés (x -> -x) :
    la logique de cycle est écrite une seule fois, pour l'achat, et s'applique aux deux sens."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    mid = c.rolling(MID_N).mean()
    sd = c.rolling(MID_N).std(ddof=0)
    top, low = mid + 2 * sd, mid - 2 * sd
    if sens == "B":
        O, H, L, C, MID, TOP, LOW = o, h, l, c, mid, top, low
    else:
        O, H, L, C, MID, TOP, LOW = -o, -l, -h, -c, -mid, -low, -top
    f = {"O": O, "H": H, "L": L, "C": C, "MID": MID, "TOP": TOP, "LOW": LOW,
         "H5": wma(H, 5), "H10": wma(H, 10), "L5": wma(L, 5), "L10": wma(L, 10)}
    return {k: v.to_numpy(dtype=float) for k, v in f.items()}


def _cycle(f, win):
    """Machine à états du cycle pour un sens. Retourne {code: tableau booléen} (EXT, EXM inclus)."""
    O, H, L, C, MID, TOP, LOW = (f[k] for k in ("O", "H", "L", "C", "MID", "TOP", "LOW"))
    H5, H10, L5, L10 = f["H5"], f["H10"], f["L5"], f["L10"]
    n = len(C)
    with np.errstate(invalid="ignore"):
        ext = (L5 < LOW) | (L10 < LOW)                       # EXT (MA5) — départ du cycle
        exm = L10 < LOW                                      # EXM (MA10) — alerte seulement
        mhv_c = (L <= LOW) & (C > LOW)                       # MHV : mèche sous la bande, clôture revenue
        csak_c = (C > MID) & (C > H5) & (C > H10)            # CSAK : clôture au-delà MID et MA5/10
        mom_c = C > TOP                                      # MOM : clôture hors bande
        zone_hi = np.maximum(L5, L10)
        zone_lo = np.minimum(L5, L10)
        re_hi = zone_hi                                      # H5
        re_lo = np.minimum(zone_lo, MID)                     # H5 : la zone descend jusqu'à la MID
        prev_out = np.r_[False, C[:-1] > re_hi[:-1]]         # H5 : bougie précédente hors zone, côté tendance
        wick = (np.minimum(O, C) - L) >= RE_WICK * (H - L)   # H3
        close_ok = (C >= re_lo) if RE_CLOSE == "above_low" else ((C >= re_lo) & (C <= re_hi))  # H4
        re_base = (C > O) & wick & close_ok & (L <= re_hi) & (C >= MID)

    out = {k: np.zeros(n, dtype=bool) for k in ("MHV", "CSAK", "MOM", "RE")}
    stage, t = 0, -10**9          # 0 aucun, 1 EXT, 2 MHV, 3 CSAK, 4 MOM
    for i in range(n):
        if stage and i - t > win:
            stage = 0                                        # H6 : délai dépassé
        if stage >= 3 and C[i] < MID[i]:
            stage = 0                                        # H7 : MID franchie dans le mauvais sens
        if stage == 4 and re_base[i] and prev_out[i]:
            out["RE"][i] = True; stage = 0; continue         # H8
        if stage == 3 and mom_c[i]:
            out["MOM"][i] = True; stage, t = 4, i; continue
        if stage == 2 and csak_c[i]:
            out["CSAK"][i] = True; stage, t = 3, i; continue # une seule CSAK : la machine ne revient pas à 2
        if stage == 1 and mhv_c[i]:
            out["MHV"][i] = True; stage, t = 2, i; continue
        if ext[i] and stage in (0, 1):
            stage, t = 1, i                                  # H9 : EXT arme le cycle (rafraîchi)
    out["EXT"], out["EXM"] = ext, exm
    return out


def _signals(df, lb=None, cycle=None):
    """Signaux de toutes les bougies : {code: (série buy, série sell)}. Calcul causal.
    `cycle` est conservé pour compatibilité : la machine à états est toujours active."""
    win = CSAK_WINDOW if lb is None else lb
    eb = _cycle(_frame(df, "B"), win)
    es = _cycle(_frame(df, "S"), win)
    ser = lambda a: pd.Series(a, index=df.index)
    return {k: (ser(eb[k]), ser(es[k])) for k in ("MOM", "EXM", "EXT", "MHV", "CSAK", "RE")}


def _at(sig, pos):
    """(code, sens) de la bougie à la position pos, par priorité. EXT/EXM ne sont jamais retournés."""
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[pos]: return code, "B"
        if s.iloc[pos]: return code, "S"
    return None, None


def detect(df, lb=None, cycle=None):
    """Signal d'entrée de la dernière bougie : (code, sens) ou (None, None).
    Priorité RE > MOM > CSAK > MHV (un RE n'est plus masqué par un MOM)."""
    return _at(_signals(df, lb), -1)


def detect_all(df, lb=None, cycle=None):
    """Tous les signaux de la dernière bougie : liste de (code, sens). Sert à ne rien masquer."""
    sig = _signals(df, lb)
    out = []
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[-1]: out.append((code, "B"))
        if s.iloc[-1]: out.append((code, "S"))
    return out


def alerts(df, lb=None):
    """Alertes d'affichage uniquement (EXT / EXM) sur la dernière bougie. Non utilisées par la décision."""
    sig = _signals(df, lb)
    return [(code, side) for code in ("EXT", "EXM") for side, s in (("B", sig[code][0]), ("S", sig[code][1]))
            if s.iloc[-1]]


RECENT_BARS = 5     # profondeur de recherche du dernier signal (affichage seulement, hors synthèse)


def recent_signal(df, lookback=RECENT_BARS, lb=None, cycle=None):
    """Dernier signal parmi les `lookback` bougies qui précèdent la dernière : (code, sens, age) ou None."""
    sig = _signals(df, lb)
    for age in range(1, min(lookback, len(df) - 1) + 1):
        code, sens = _at(sig, -1 - age)
        if code:
            return code, sens, age
    return None


# Codes de la décision. EXT et EXM n'y figurent plus (règle 4).
PRIORITY = ["RE", "MOM", "CSAK", "MHV"]
SIG_DIRECTEUR = {"MOM", "RE"}          # structure
SIG_ENTREE = {"RE", "MHV", "CSAK"}     # déclencheurs
# Filtre EMA50 D1 par type de signal (règle 6). True = la tendance D1 doit être dans le sens.
D1_FILTER = {"RE": True, "MOM": True, "CSAK": True, "MHV": False}


def synthese(cells, tf_dispo, use_filter, d1_trend):
    """Synthèse multi-TF BBMA.
    cells : {tf: (code, sens)} ; tf_dispo : TF scannés ; d1_trend : 'B', 'S' ou None.
    Retourne (tf_entree, code_entree, sens) ou (None, None, None)."""
    tfs_ord = [t for t in TF_ORDER if t in tf_dispo]

    idx = None                                                    # 1. TF directeur
    for i, t in enumerate(tfs_ord):
        if cells.get(t, (None, None))[0] in SIG_DIRECTEUR:
            idx = i
            break
    if idx is None:
        return None, None, None
    tf_dir = tfs_ord[idx]
    code_dir, sens = cells[tf_dir]

    for t in tfs_ord[:idx]:                                       # 4. TF supérieur contraire -> pas de trade
        code, d = cells.get(t, (None, None))
        if code and d != sens:
            return None, None, None

    entry = None                                                  # 2. TF d'entrée le plus proche
    for t in tfs_ord[idx + 1:]:
        code, d = cells.get(t, (None, None))
        if code in SIG_ENTREE and d == sens:
            entry = (t, code)
            break
    tf_e, code_e = entry if entry else (tf_dir, code_dir)         # 3. sinon le directeur

    if use_filter and d1_trend and d1_trend != sens and D1_FILTER.get(code_e, True):   # 5. filtre D1 par code
        return None, None, None
    return tf_e, code_e, sens


def confiance_score(cells, tfs, sens_ref):
    """Part des TF scannés ayant un signal dans le même sens que l'entrée BBMA (0 à 1)."""
    if not tfs or sens_ref is None:
        return None
    n = sum(1 for t in tfs if cells.get(t, (None, None))[1] == sens_ref)
    return n / len(tfs)


RSI_PERIOD = 14


def _wilder(x, n):
    """Lissage de Wilder : moyenne simple des n premières valeurs, puis récurrence avec alpha = 1/n."""
    x = x.copy()
    seed = x.iloc[1:n + 1].mean()
    x.iloc[: n] = np.nan                 # valeurs avant la graine : non calculées
    x.iloc[n] = seed
    return x.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=RSI_PERIOD):
    """RSI de Wilder (indépendant de la logique BBMA). Premier RSI disponible à la bougie n."""
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), n)
    avg_loss = _wilder(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + avg_gain / avg_loss)


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


def _lock_for(path):
    """Un verrou par fichier de cache : évite deux écritures simultanées (threads H1 et H4 par exemple)."""
    with _locks_guard:
        return _file_locks.setdefault(path, threading.Lock())


def _cache_path(pair, interval):
    safe = re.sub(r"[^A-Za-z0-9]", "_", pair)   # empêche les chemins hors du dossier cache
    return CACHE_DIR / f"{safe}_{interval}.parquet"


def _utc_index(df):
    """Index temporel en UTC (tz-aware), pour pouvoir comparer avec l'heure courante."""
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def _bar_end(ts, interval):
    """Instant de fermeture d'une bougie ouverte à ts."""
    if interval == "1mo":
        return ts + pd.DateOffset(months=1)
    return ts + BAR_DELTA[interval]


def _read_cache(path):
    if not path.exists():
        return None
    try:
        return _utc_index(pd.read_parquet(path))
    except Exception as e:
        log.warning("cache illisible, retéléchargement (%s) : %s", path.name, e)
        return None


def _write_cache(path, df):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)                       # écriture atomique : pas de fichier à moitié écrit
    except Exception as e:
        log.warning("écriture du cache impossible (%s) : %s", path.name, e)


def _history_cached(pair, interval, period):
    """Historique OHLC avec cache disque.

    - Cache à jour (la dernière bougie stockée est encore en formation) : aucun téléchargement.
    - Cache périmé : téléchargement incrémental, ou complet si le cache est trop ancien.
    - Premier lancement : téléchargement complet de CACHE_PERIOD.
    - Échec réseau : on renvoie le cache existant plutôt que rien.
    `period` est ignoré : la période réellement stockée est CACHE_PERIOD[interval] (maximum utilisé)."""
    path = _cache_path(pair, interval)
    with _lock_for(path):
        cached = _read_cache(path)
        now = pd.Timestamp.now(tz="UTC")

        if cached is not None:
            last = cached.index[-1]
            if now < _bar_end(last, interval):
                return cached                   # dernière bougie encore ouverte : rien de nouveau à fermer
            gap = now - last
            if interval in INCR_DAYS and gap < pd.Timedelta(days=INCR_DAYS[interval]):
                fetch_period = f"{INCR_DAYS[interval]}d"
            else:
                fetch_period = CACHE_PERIOD[interval]
        else:
            fetch_period = CACHE_PERIOD[interval]

        new = _history(pair, interval, fetch_period)
        if new is None:
            return cached
        new = _utc_index(new)
        merged = new if cached is None else pd.concat([cached, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _write_cache(path, merged)
        return merged


def _fetch_all(pairs, tfs, trend):
    """Un seul téléchargement par (paire, intervalle, période).
    Le D1 du filtre de tendance est le même téléchargement que le TF D1 : il n'est pas refait."""
    keys = {(p, TFS[t][0], TFS[t][1]) for p in pairs for t in tfs}
    if trend:
        keys |= {(p, "1d", "2y") for p in pairs}
    keys = list(keys)
    with ThreadPoolExecutor(DL_WORKERS) as ex:
        return dict(zip(keys, ex.map(lambda k: _history_cached(*k), keys)))


# Session H4 : source unique pour tout le scanner (scan, qualité, dashboard, backtest).
H4_SESSION = "UTC"                     # choix validé : bougies H4 à 00/04/08... UTC
SESSIONS = {"UTC": ("UTC", None), "NY17": ("America/New_York", "17h")}
AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


def resample_h4(df1h, session=None):
    """Bougies H4 à partir de bougies H1 (index UTC), ancrées selon la session."""
    session = session or H4_SESSION
    if session == "UTC":
        return df1h.resample("4h").agg(AGG).dropna()
    tz, offset = SESSIONS[session]
    local = df1h.tz_convert(tz).resample("4h", origin="start_day", offset=offset).agg(AGG).dropna()
    return local.tz_convert("UTC")


def _closed_bars(pair, tf, data):
    """Bougies fermées d'une paire sur un timeframe. H4 : resampling via resample_h4 (source unique).
    La bougie en cours de formation est exclue. None si historique insuffisant."""
    interval, period, rs = TFS[tf]
    df = data.get((pair, interval, period))
    if df is None:
        return None
    if tf == "H4":
        df = resample_h4(df)
    elif rs:
        df = df.resample(rs).agg(AGG).dropna()
    df = df.iloc[:-1]
    if len(df) < MIN_BARS:
        return None
    return df


def _scan_one(pair, tf, data):
    """((code, sens), ok) pour une paire sur un timeframe, à partir des données déjà téléchargées."""
    df = _closed_bars(pair, tf, data)
    if df is None:
        return (None, None), False
    try:
        return detect(df), True
    except Exception:
        return (None, None), False


def _rsi_one(pair, tf, data):
    """RSI 14 de la dernière bougie fermée, ou None si indisponible."""
    df = _closed_bars(pair, tf, data)
    if df is None:
        return None
    value = rsi(df["Close"]).iloc[-1]
    return None if pd.isna(value) else float(value)


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


def scan_with_rsi(pairs, tfs, trend=True):
    """Comme scan(), avec en plus le RSI de chaque couple (paire, TF) : un seul passage de téléchargement.
    Retourne (res, trends, failed, total, ts, rsi). rsi : {(paire, tf): float ou None}."""
    pairs, tfs = list(pairs), list(tfs)
    data = _fetch_all(pairs, tfs, trend)
    jobs = [(p, t) for p in pairs for t in tfs]
    res, failed = {}, 0
    for p, t in jobs:
        r, ok = _scan_one(p, t, data)
        res[(p, t)] = r
        failed += not ok
    trends = {p: _d1_trend(p, data) for p in pairs} if trend else {}
    rsi_values = {(p, t): _rsi_one(p, t, data) for p, t in jobs}
    return res, trends, failed, len(jobs), datetime.now(timezone.utc), rsi_values
