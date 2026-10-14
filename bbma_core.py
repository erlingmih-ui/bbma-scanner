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
PRIORITY = ["RE", "MOM", "EXM", "EXT", "MHV", "CSAK"]   # priorité de détection : RE en tête, une entrée valide n'est pas masquée

TF_ORDER = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]   # du plus haut au plus bas
SIG_DIRECTEUR = {"MOM", "EXM", "EXT", "CSAK", "RE"}      # signaux de structure / direction (CSAK = Candle Arah)
# Entrées BBMA Oma Ally : les setups RE et MHV. CSAK et MOM sont des signaux de direction, pas des entrées.
ENTRY_ALLOW_EXT = False       # True : EXT redevient un déclencheur d'entrée sur les TF inférieurs
SIG_ENTREE = {"RE", "MHV"} | ({"EXT"} if ENTRY_ALLOW_EXT else set())
# Repli « poursuite » : si aucun TF inférieur n'a RE/MHV, le TF directeur sert d'entrée.
# False (défaut) : seul un directeur déjà en RE/MHV peut servir d'entrée ; sinon pas d'entrée.
ALLOW_CHASE_FALLBACK = False

RETRY = 2          # essais par téléchargement
PAUSE = 1.5        # secondes entre essais (multipliées par le numéro d'essai)
MIN_BARS = 60      # historique minimum pour calculer un signal
DL_WORKERS = 6

# Correction 2 : le filtre D1 utilise la dernière bougie D1 CLÔTURÉE, comme le signal.
# Mettre False pour retrouver le comportement d'origine (bougie en formation incluse).
D1_FILTER_CLOSED_ONLY = True

# RE (Reentry) selon le guide BBMA OA :
#   - armé par un CSAK ou MOM dans le sens de la tendance, désarmé par un signal contraire ;
#   - retest = 1er chandelier qui revient dans la zone MA5/MA10 Low (achat) ou High (vente) depuis le bon côté ;
#   - le premier retest consomme l'armement, confirmé ou non ;
#   - confirmation = clôture du bon côté de la Mid BB (optionnelle, activable ici) ;
#   - entrée recherchée sur le chandelier SUIVANT le retest confirmé.
RE_USE_MID_BB = True
# MHV : « Must have reverse candle » (sources TradingView / guide). Interprétation : la bougie MHV
# doit être de sens contraire (achat : haussière ; vente : baissière). True = exigé.
MHV_REQUIRE_REVERSE_CANDLE = True
# HYPOTHÈSE non documentée (guide non trouvé) : après un premier retest non confirmé par la Mid BB,
# un second retest reste-t-il possible tant que la tendance est valide ? False = comportement actuel
# (le premier retest consomme l'armement).
RE_SECOND_RETEST_IF_UNCONFIRMED = False

# --- Cohérence backtest / live (point 1) ---
# Tous les signaux (détection, RE, MHV) sont calculés sur les SIGNAL_WINDOW dernières bougies fermées,
# en live comme en backtest. Un état (armement RE, extrême MHV) hérité d'un passé plus lointain
# n'influence donc jamais le résultat, et les deux modes produisent les mêmes signaux.
SIGNAL_WINDOW = 300

# --- Expiration de l'armement RE (point 2) ---
# Un armement (MOM ou CSAK) expire après RE_ARM_DURATION (point 1), ou dès que la clôture sort
# de l'autre côté de la bande de Bollinger (la tendance est invalidée). Durée à calibrer par TF.
RE_ARM_INVALIDATE = True
# Durée maximale d'armement, en temps réel, par timeframe (point 1). Convertie en bougies
# selon le TF détecté dans l'index. Valeurs de départ, à calibrer.
RE_ARM_DURATION = {
    "M5": pd.Timedelta(hours=3, minutes=20),   # ~40 bougies
    "M15": pd.Timedelta(hours=10),             # ~40 bougies
    "H1": pd.Timedelta(days=2),                # ~48 bougies
    "H4": pd.Timedelta(days=7),                # ~42 bougies
    "D1": pd.Timedelta(days=30),               # ~30 bougies
    "W1": pd.Timedelta(weeks=26),              # ~26 bougies
    "MN": pd.Timedelta(days=365),              # ~12 bougies
}
# Durée des bougies de chaque TF, pour retrouver le TF depuis l'écart médian de l'index
BAR_LEN = {"M5": pd.Timedelta(minutes=5), "M15": pd.Timedelta(minutes=15), "H1": pd.Timedelta(hours=1),
           "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(weeks=1),
           "MN": pd.Timedelta(days=30)}
RE_ARM_FALLBACK_BARS = 40   # utilisé si l'index est trop court pour deviner le TF

# --- Filtre de tendance sur les entrées (point 3) ---
# True : un MHV (et donc un RE) n'est retenu que s'il va dans le sens de la dernière impulsion
# directionnelle (dernier MOM ou CSAK). Le contre-sens n'est possible qu'après un nouveau MOM contraire.
TREND_GATE_ENTRIES = True

# --- MHV : un seul par extrême (point 4) ---
# Chaque extrême (EXM ou EXT) ne produit au plus qu'un MHV, dans la fenêtre de LB bougies.
MHV_ONCE_PER_EXTREME = True

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


def re_arm_bars(index):
    """Durée d'armement RE en bougies, pour le TF déduit de l'écart médian entre bougies de l'index."""
    if len(index) < 3:
        return RE_ARM_FALLBACK_BARS
    step = pd.Series(index).diff().median()
    tf = min(BAR_LEN, key=lambda k: abs(BAR_LEN[k] - step))
    return max(1, int(RE_ARM_DURATION[tf] / BAR_LEN[tf]))


def _reentry(c, h, l, mid, up, low, zl_b, zh_b, zl_s, zh_s, dir_b, dir_s, max_bars=RE_ARM_FALLBACK_BARS):
    """Reentry (RE) : (série buy, série sell) d'entrée, décalées d'une bougie.

    Armé par un signal directionnel, consommé au premier retest. Le retest est validé si
    la confirmation Mid BB (si RE_USE_MID_BB) est respectée ; l'entrée est sur la bougie suivante.
    Expiration (point 2) : l'armement tombe après max_bars bougies (voir re_arm_bars), ou si la clôture
    sort de la bande opposée (RE_ARM_INVALIDATE).
    """
    n = len(c)
    C, H, L, MID = c.to_numpy(float), h.to_numpy(float), l.to_numpy(float), mid.to_numpy(float)
    UP, LOW = up.to_numpy(float), low.to_numpy(float)
    ZLB, ZHB = zl_b.to_numpy(float), zh_b.to_numpy(float)
    ZLS, ZHS = zl_s.to_numpy(float), zh_s.to_numpy(float)
    DB, DS = dir_b.to_numpy(bool), dir_s.to_numpy(bool)
    confirm_b = np.zeros(n, bool)
    confirm_s = np.zeros(n, bool)
    armed_b = armed_s = False
    age_b = age_s = 0
    for i in range(1, n):
        # Expiration et invalidation de l'armement (point 2)
        if armed_b:
            age_b += 1
            if age_b > max_bars or (RE_ARM_INVALIDATE and C[i] < LOW[i]):
                armed_b = False
        if armed_s:
            age_s += 1
            if age_s > max_bars or (RE_ARM_INVALIDATE and C[i] > UP[i]):
                armed_s = False
        # Retest achat : 1er retour dans la zone MA Low depuis le haut
        if armed_b and C[i-1] > ZHB[i-1] and L[i] <= ZHB[i] and C[i] >= ZHB[i]:
            # Retest touche la zone, clôture ne passe pas les MA5/MA10 Low
            confirmed = (not RE_USE_MID_BB) or C[i] >= MID[i]
            if confirmed:
                confirm_b[i] = True
            if confirmed or not RE_SECOND_RETEST_IF_UNCONFIRMED:
                armed_b = False
        # Retest vente : 1er retour dans la zone MA High depuis le bas
        if armed_s and C[i-1] < ZLS[i-1] and H[i] >= ZLS[i] and C[i] <= ZLS[i]:
            # Retest touche la zone, clôture ne passe pas les MA5/MA10 High
            confirmed = (not RE_USE_MID_BB) or C[i] <= MID[i]
            if confirmed:
                confirm_s[i] = True
            if confirmed or not RE_SECOND_RETEST_IF_UNCONFIRMED:
                armed_s = False
        # Armement par signal directionnel ; le sens opposé désarme
        if DB[i]:
            armed_b, armed_s = True, False
            age_b = 0
        elif DS[i]:
            armed_s, armed_b = True, False
            age_s = 0
    re_b = pd.Series(np.roll(confirm_b, 1), index=c.index); re_b.iloc[0] = False
    re_s = pd.Series(np.roll(confirm_s, 1), index=c.index); re_s.iloc[0] = False
    return re_b, re_s


def _mhv_once(ext, trig, lb):
    """MHV avec consommation (point 4) : un extrême (ext) arme un MHV pendant lb bougies,
    et le premier MHV (trig) qui suit le consomme. Un nouvel extrême réarme.
    L'extrême doit être strictement antérieur à la bougie du MHV, comme dans l'ancienne règle."""
    e = ext.to_numpy(bool)
    t = trig.to_numpy(bool)
    out = np.zeros(len(e), bool)
    pending = None
    for i in range(len(e)):
        if pending is not None and i - pending > lb:
            pending = None
        if t[i] and pending is not None:
            out[i] = True
            pending = None
        if e[i]:
            pending = i
    return pd.Series(out, index=ext.index)


def _signals(df, lb=15):
    """Signaux de toutes les bougies : {code: (série buy, série sell)}. Calcul causal (aucune donnée future).
    Point 1 : seules les SIGNAL_WINDOW dernières bougies sont utilisées, en live comme en backtest."""
    df = df.iloc[-SIGNAL_WINDOW:]
    max_bars = re_arm_bars(df.index)
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    top, low = mid + 2 * sd, mid - 2 * sd
    h5, h10, l5, l10 = wma(h, 5), wma(h, 10), wma(l, 5), wma(l, 10)
    rec = lambda s, n: s.fillna(False).astype(bool).rolling(n, min_periods=1).max().astype(bool)

    mom_b, mom_s = c > top, c < low                                             # MOM (clôture hors BB)
    exm_s, exm_b = h10 > top, l10 < low                                         # Extreme Magic (MA10)
    ext_s, ext_b = h5 > top, l5 < low                                           # Extreme (MA5)
    any_s, any_b = exm_s | ext_s, exm_b | ext_b
    rev_s = (c < o) if MHV_REQUIRE_REVERSE_CANDLE else True                     # bougie de renversement
    rev_b = (c > o) if MHV_REQUIRE_REVERSE_CANDLE else True
    trig_s = (h >= top) & (c < top) & rev_s                                     # conditions MHV de la bougie
    trig_b = (l <= low) & (c > low) & rev_b
    if MHV_ONCE_PER_EXTREME:                                                    # point 4 : un MHV par extrême
        mhv_s = _mhv_once(any_s, trig_s, lb)
        mhv_b = _mhv_once(any_b, trig_b, lb)
    else:
        mhv_s = rec(any_s.shift(1), lb) & trig_s
        mhv_b = rec(any_b.shift(1), lb) & trig_b
    arah_s = (c < mid) & (c < l5) & (c < l10)                                   # CSAK / Candle Arah
    arah_b = (c > mid) & (c > h5) & (c > h10)
    # tendance mémorisée (m_lastTrendDir) : sens du dernier MOM / Arah (point 3)
    trend = pd.Series(np.where(mom_b | arah_b, 1, np.where(mom_s | arah_s, -1, np.nan)), index=c.index).ffill()
    if TREND_GATE_ENTRIES:
        mhv_b = mhv_b & (trend == 1)
        mhv_s = mhv_s & (trend == -1)
    # Zones MA5/MA10
    zone_buy_hi, zone_buy_lo = np.maximum(l5, l10), np.minimum(l5, l10)
    zone_sell_hi, zone_sell_lo = np.maximum(h5, h10), np.minimum(h5, h10)

    # RE : machine à états (voir _reentry et RE_USE_MID_BB). Armé seulement dans le sens de la tendance.
    re_b, re_s = _reentry(c, h, l, mid, top, low, zone_buy_lo, zone_buy_hi, zone_sell_lo, zone_sell_hi,
                          mom_b | arah_b, mom_s | arah_s, max_bars=max_bars)
    if TREND_GATE_ENTRIES:
        re_b = re_b & (trend == 1)
        re_s = re_s & (trend == -1)

    return {"MOM": (mom_b, mom_s), "EXM": (exm_b, exm_s), "EXT": (ext_b, ext_s),
            "MHV": (mhv_b, mhv_s), "CSAK": (arah_b, arah_s), "RE": (re_b, re_s)}


def _at(sig, pos):
    """(code, sens) de la bougie à la position pos (priorité BBMA), ou (None, None)."""
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[pos]: return code, "B"
        if s.iloc[pos]: return code, "S"
    return None, None


def detect(df, lb=15):
    """Signal de la dernière bougie : (code, sens) avec sens 'B' (buy) ou 'S' (sell).
    Priorité : MOM > EXM (Extreme Magic, MA10) > EXT (MA5) > MHV > CSAK (Candle Arah) > RE.
    TP Wajib = flag déclenché par EXM/EXT uniquement (affiché à côté du code)."""
    return _at(_signals(df, lb), -1)


RECENT_BARS = 5     # profondeur de recherche du dernier signal (affichage seulement, hors synthèse)


def recent_signal(df, lookback=RECENT_BARS, lb=15):
    """Dernier signal parmi les `lookback` bougies qui précèdent la dernière : (code, sens, age) ou None.
    age = nombre de bougies avant la dernière (1 = l'avant-dernière). Affichage uniquement :
    la synthèse, la confiance, la jauge et la qualité n'utilisent que detect()."""
    sig = _signals(df, lb)
    for age in range(1, min(lookback, len(df) - 1) + 1):
        code, sens = _at(sig, -1 - age)
        if code:
            return code, sens, age
    return None


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

    # 3. Aucun TF d'entrée trouvé : le directeur sert d'entrée seulement s'il est lui-même un setup
    #    d'entrée (RE/MHV), ou si le repli « poursuite » est activé.
    if ALLOW_CHASE_FALLBACK or code_dir in SIG_ENTREE:
        return tf_dir, code_dir, sens
    return None, None, None


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


def _closed_bars(pair, tf, data):
    """Bougies fermées d'une paire sur un timeframe (resampling, bougie en formation exclue), ou None."""
    interval, period, rs = TFS[tf]
    df = data.get((pair, interval, period))
    if df is None:
        return None
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    # Exclure la dernière bougie en cours de formation (critique sur M5/M15)
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
