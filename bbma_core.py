"""Logique pure du scanner BBMA : données, détection, synthèse. Aucune dépendance à Streamlit.

Version corrigée (audit Oma Ally, points 1 et 2) :
  1. Cycle Extreme -> MHV -> CSAK -> RE : le Reentry n'est armé que si un Extreme ou un MHV du même côté
     a précédé (RE_REQUIRE_CYCLE), l'armement expire (RE_ARM_MAX_BARS) et tombe sur un Extreme / MHV
     contraire ou sur une cassure de la Mid BB.
  2. CSAK = événement (première bougie) valable CSAK_VALID_BARS bougies, et non plus un état permanent ;
     la synthèse refuse l'entrée si un timeframe intermédiaire (entre directeur et entrée) contredit.
Tous les paramètres sont plus bas ; chacun peut être désactivé pour retrouver l'ancien comportement.
"""
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

# --- Correction 1 : machine d'états du cycle BBMA (Extreme -> MHV -> CSAK -> RE) ---
# CSAK n'est plus vrai sur chaque bougie de tendance : seule la 1re bougie (front montant) compte,
# et le signal reste affiché CSAK_VALID_BARS bougies (1 = la bougie de cassure seule), tant que la condition tient.
# False = ancien comportement (état permanent).
CSAK_EVENT_ONLY = True
CSAK_VALID_BARS = 3
# Le RE n'est armé que si un Extreme ou un MHV du même côté a eu lieu dans les CYCLE_WINDOW bougies
# précédentes (achat : Extreme sur la bande basse ; vente : sur la bande haute). False = ancien comportement.
RE_REQUIRE_CYCLE = True
CYCLE_WINDOW = 40
# Armement par MOM (clôture hors BB) en plus du CSAK. False : seul le CSAK arme le RE (cycle strict).
RE_ARM_BY_MOM = False
# Durée de vie de l'armement, en bougies. 0 = illimitée (ancien comportement).
RE_ARM_MAX_BARS = 30
# L'armement tombe sur un Extreme ou un MHV du côté contraire (objectif atteint : TP Wajib).
RE_DISARM_ON_OPPOSITE = True
# L'armement tombe si la clôture passe du mauvais côté de la Mid BB (tendance perdue).
RE_DISARM_ON_MID_BREAK = True

# --- Correction 2 : contrôle des timeframes intermédiaires dans la synthèse ---
# Un signal contraire sur un TF situé entre le directeur et le TF d'entrée annule l'entrée.
CHECK_INTERMEDIATE_TFS = True

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


def _reentry(c, h, l, mid, zl_b, zh_b, zl_s, zh_s, arm_b, arm_s, kill_b, kill_s):
    """Reentry (RE) : (série buy, série sell) d'entrée, décalées d'une bougie.

    Machine d'états par côté :
      - armée par arm_* (CSAK en événement, éventuellement MOM, après un Extreme / MHV du même côté) ;
      - désarmée par kill_* (Extreme / MHV contraire), par l'armement du côté opposé, par la cassure
        de la Mid BB (RE_DISARM_ON_MID_BREAK) ou à l'expiration (RE_ARM_MAX_BARS) ;
      - le premier retest de la zone MA5/10 consomme l'armement ; il est validé si la confirmation
        Mid BB (RE_USE_MID_BB) est respectée ; l'entrée est sur la bougie suivante.
    L'ordre à chaque bougie est : invalidations, retest, puis armement (calcul causal).
    """
    n = len(c)
    C, H, L, MID = c.to_numpy(float), h.to_numpy(float), l.to_numpy(float), mid.to_numpy(float)
    ZLB, ZHB = zl_b.to_numpy(float), zh_b.to_numpy(float)
    ZLS, ZHS = zl_s.to_numpy(float), zh_s.to_numpy(float)
    AB, AS = arm_b.to_numpy(bool), arm_s.to_numpy(bool)
    KB, KS = kill_b.to_numpy(bool), kill_s.to_numpy(bool)
    confirm_b = np.zeros(n, bool)
    confirm_s = np.zeros(n, bool)
    armed_b = armed_s = False
    since_b = since_s = 0
    for i in range(1, n):
        # 1. Invalidations de l'armement
        if armed_b:
            since_b += 1
            if (RE_DISARM_ON_OPPOSITE and KB[i]) or (RE_DISARM_ON_MID_BREAK and C[i] < MID[i]) \
                    or (RE_ARM_MAX_BARS and since_b > RE_ARM_MAX_BARS):
                armed_b = False
        if armed_s:
            since_s += 1
            if (RE_DISARM_ON_OPPOSITE and KS[i]) or (RE_DISARM_ON_MID_BREAK and C[i] > MID[i]) \
                    or (RE_ARM_MAX_BARS and since_s > RE_ARM_MAX_BARS):
                armed_s = False
        # 2. Retest achat : 1er retour dans la zone MA Low depuis le haut
        if armed_b and C[i-1] > ZHB[i-1] and L[i] <= ZHB[i] and C[i] >= ZHB[i]:
            confirmed = (not RE_USE_MID_BB) or C[i] >= MID[i]
            if confirmed:
                confirm_b[i] = True
            if confirmed or not RE_SECOND_RETEST_IF_UNCONFIRMED:
                armed_b = False
        # Retest vente : 1er retour dans la zone MA High depuis le bas
        if armed_s and C[i-1] < ZLS[i-1] and H[i] >= ZLS[i] and C[i] <= ZLS[i]:
            confirmed = (not RE_USE_MID_BB) or C[i] <= MID[i]
            if confirmed:
                confirm_s[i] = True
            if confirmed or not RE_SECOND_RETEST_IF_UNCONFIRMED:
                armed_s = False
        # 3. Armement ; le sens opposé désarme
        if AB[i]:
            armed_b, armed_s, since_b = True, False, 0
        elif AS[i]:
            armed_s, armed_b, since_s = True, False, 0
    re_b = pd.Series(np.roll(confirm_b, 1), index=c.index); re_b.iloc[0] = False
    re_s = pd.Series(np.roll(confirm_s, 1), index=c.index); re_s.iloc[0] = False
    return re_b, re_s


def _signals(df, lb=15):
    """Signaux de toutes les bougies : {code: (série buy, série sell)}. Calcul causal (aucune donnée future)."""
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
    mhv_s = rec(any_s.shift(1), lb) & (h >= top) & (c < top) & rev_s            # MHV (fenêtre 15 bougies)
    mhv_b = rec(any_b.shift(1), lb) & (l <= low) & (c > low) & rev_b
    arah_s_raw = (c < mid) & (c < l5) & (c < l10)                               # CSAK / Candle Arah (condition)
    arah_b_raw = (c > mid) & (c > h5) & (c > h10)
    # Correction 2 : CSAK = événement (1re bougie où la condition devient vraie), valable CSAK_VALID_BARS bougies.
    ev_s = arah_s_raw & ~arah_s_raw.shift(1, fill_value=False)
    ev_b = arah_b_raw & ~arah_b_raw.shift(1, fill_value=False)
    if CSAK_EVENT_ONLY:
        arah_s, arah_b = arah_s_raw & rec(ev_s, CSAK_VALID_BARS), arah_b_raw & rec(ev_b, CSAK_VALID_BARS)
    else:
        arah_s, arah_b = arah_s_raw, arah_b_raw
        ev_s, ev_b = arah_s_raw, arah_b_raw
    # Zones MA5/MA10
    zone_buy_hi, zone_buy_lo = np.maximum(l5, l10), np.minimum(l5, l10)
    zone_sell_hi, zone_sell_lo = np.maximum(h5, h10), np.minimum(h5, h10)

    # Correction 1 : cycle Extreme -> MHV -> CSAK -> RE. Le RE n'est armé qu'après un Extreme / MHV du même côté.
    no = pd.Series(False, index=c.index)
    cyc_b = rec(any_b | mhv_b, CYCLE_WINDOW) if RE_REQUIRE_CYCLE else ~no
    cyc_s = rec(any_s | mhv_s, CYCLE_WINDOW) if RE_REQUIRE_CYCLE else ~no
    arm_b = (ev_b | (mom_b if RE_ARM_BY_MOM else no)) & cyc_b
    arm_s = (ev_s | (mom_s if RE_ARM_BY_MOM else no)) & cyc_s
    kill_b = any_s | mhv_s          # Extreme / MHV de vente : l'achat a atteint son objectif
    kill_s = any_b | mhv_b
    re_b, re_s = _reentry(c, h, l, mid, zone_buy_lo, zone_buy_hi, zone_sell_lo, zone_sell_hi,
                          arm_b, arm_s, kill_b, kill_s)

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


def synthese_why(cells, tf_dispo, use_filter, d1_trend):
    """Synthèse multi-TF BBMA Oma Ally, avec le motif en cas de refus.
    cells : {tf: (code, sens)} ; tf_dispo : TF scannés ; d1_trend : 'B', 'S' ou None.
    Retourne ((tf_entree, code_entree, sens) ou (None, None, None), motif).
    Motifs : 'ok', 'aucun_directeur', 'tf_superieur_contraire', 'filtre_d1',
             'tf_intermediaire_contraire', 'aucune_entree'."""
    none = (None, None, None)
    tfs_ord = [t for t in TF_ORDER if t in tf_dispo]

    # 1. TF directeur : le plus haut TF ayant un signal de structure
    idx = None
    for i, t in enumerate(tfs_ord):
        if cells.get(t, (None, None))[0] in SIG_DIRECTEUR:
            idx = i
            break
    if idx is None:
        return none, "aucun_directeur"
    tf_dir = tfs_ord[idx]
    code_dir, sens = cells[tf_dir]

    # 2. Un TF supérieur au directeur qui contredit le sens -> pas de trade
    for t in tfs_ord[:idx]:
        code, d = cells.get(t, (None, None))
        if code and d != sens:
            return none, "tf_superieur_contraire"

    # 3. Filtre D1 (EMA50) : sens contre la tendance D1 -> pas de trade
    if use_filter and d1_trend and d1_trend != sens:
        return none, "filtre_d1"

    # 4. TF d'entrée plus bas que le directeur, même sens (le plus proche du directeur).
    #    Correction 2 : un signal contraire sur un TF situé entre le directeur et l'entrée annule le trade.
    #    Tout TF d'entrée plus bas traverserait ce même TF intermédiaire : inutile de chercher plus loin.
    for j in range(idx + 1, len(tfs_ord)):
        t = tfs_ord[j]
        code, d = cells.get(t, (None, None))
        if code in SIG_ENTREE and d == sens:
            return (t, code, sens), "ok"
        if CHECK_INTERMEDIATE_TFS and code and d != sens:
            return none, "tf_intermediaire_contraire"

    # 5. Aucun TF d'entrée trouvé : le directeur sert d'entrée seulement s'il est lui-même un setup
    #    d'entrée (RE/MHV), ou si le repli « poursuite » est activé.
    if ALLOW_CHASE_FALLBACK or code_dir in SIG_ENTREE:
        return (tf_dir, code_dir, sens), "ok"
    return none, "aucune_entree"


def synthese(cells, tf_dispo, use_filter, d1_trend):
    """Synthèse multi-TF BBMA Oma Ally.
    Retourne (tf_entree, code_entree, sens) ou (None, None, None). Voir synthese_why pour le motif."""
    return synthese_why(cells, tf_dispo, use_filter, d1_trend)[0]


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
