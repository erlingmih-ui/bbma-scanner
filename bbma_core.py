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
PRIORITY = ["RE", "REM", "MHV", "CSAK", "MOM", "EXM", "EXT"]   # priorité d'affichage ; detect_all renvoie tous les codes

TF_ORDER = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]   # du plus haut au plus bas
SIG_DIRECTEUR = {"MOM", "RE", "REM"}                     # signaux de structure (EXM/EXT : alerte de TP, plus de direction)
SIG_ENTREE = {"RE", "REM", "MHV", "CSAK"}                # signaux déclencheurs
SIG_REVERSAL = {"MHV"}                                   # setup de retournement : Extreme -> MHV
# Filtre D1 (EMA50) par type de setup : désactivé pour le retournement Extreme -> MHV (contre-tendance par nature)
SETUP_D1_FILTER = {"continuation": True, "retournement": False}

RETRY = 2          # essais par téléchargement
PAUSE = 1.5        # secondes entre essais (multipliées par le numéro d'essai)
MIN_BARS = 60      # historique minimum pour calculer un signal
DL_WORKERS = 6

# Correction 2 : le filtre D1 utilise la dernière bougie D1 CLÔTURÉE, comme le signal.
# Mettre False pour retrouver le comportement d'origine (bougie en formation incluse).
D1_FILTER_CLOSED_ONLY = True

# Cycle BBMA (machine à états par TF : EXTREME -> MHV -> CSAK -> MOM -> RE) : True = cycle strict,
# False = conditions brutes (sans chaînage d'étapes). Voir _run_cycle.
CSAK_CYCLE_ONLY = True
CSAK_WINDOW = 15      # bougies maximum entre le MHV et le CSAK
REJ_WICK = 0.25       # part minimale de la mèche de rejet dans l'amplitude de la bougie (RE / REM)

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


STAGE_NAMES = {1: "EXTREME", 2: "MHV", 3: "CSAK", 4: "MOM", 5: "RE"}


def _rec(arr, n):
    """True si arr était vrai sur l'une des n bougies précédentes (la bougie courante est exclue)."""
    s = np.concatenate(([False], np.asarray(arr, bool)[:-1])).astype(float)
    return pd.Series(s).rolling(n, min_periods=1).max().to_numpy() > 0


def _shift(a, fill=np.nan):
    """Décalage d'une bougie vers l'avant (la valeur précédente)."""
    out = np.empty_like(a)
    out[0] = fill
    out[1:] = a[:-1]
    return out


def _raw(df, lb=15):
    """Conditions brutes par bougie (numpy), sans mémoire de cycle. Causal : aucune donnée future.
    Les paires (achat, vente) sont des tuples de tableaux booléens."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    cs = df["Close"]
    mid_s = cs.rolling(20).mean()
    sd = cs.rolling(20).std(ddof=0)
    mid, top, low = mid_s.to_numpy(), (mid_s + 2 * sd).to_numpy(), (mid_s - 2 * sd).to_numpy()
    h5, h10 = wma(df["High"], 5).to_numpy(), wma(df["High"], 10).to_numpy()
    l5, l10 = wma(df["Low"], 5).to_numpy(), wma(df["Low"], 10).to_numpy()

    mom_b, mom_s = c > top, c < low                                   # MOM : clôture hors BB
    exm_s, exm_b = h10 > top, l10 < low                               # Extreme Magic (MA10)
    ext_s, ext_b = h5 > top, l5 < low                                 # Extreme (MA5)
    any_s, any_b = exm_s | ext_s, exm_b | ext_b
    mhv_cond_s, mhv_cond_b = (h >= top) & (c < top), (l <= low) & (c > low)   # retour dans la BB
    arah_s = (c < mid) & (c < l5) & (c < l10)                         # Candle Arah (CSAK potentiel)
    arah_b = (c > mid) & (c > h5) & (c > h10)

    # Zones MA5/MA10
    zb_hi, zb_lo = np.maximum(l5, l10), np.minimum(l5, l10)
    zs_hi, zs_lo = np.maximum(h5, h10), np.minimum(h5, h10)
    rng = h - l
    ok = rng > 0
    wick_b = ok & ((np.minimum(o, c) - l) >= REJ_WICK * rng)          # mèche basse : rejet côté achat
    wick_s = ok & ((h - np.maximum(o, c)) >= REJ_WICK * rng)          # mèche haute : rejet côté vente

    # RE sur zone MA5/10 : la bougie précédente est HORS zone, côté tendance (au-dessus pour un achat),
    # la bougie touche la zone, ne la franchit pas en clôture, et clôture dans le sens avec rejet.
    re_zone_b = (_shift(c) > _shift(zb_hi)) & (l <= zb_hi) & (c >= zb_lo) & (c > o) & wick_b
    re_zone_s = (_shift(c) < _shift(zs_lo)) & (h >= zs_lo) & (c <= zs_hi) & (c < o) & wick_s
    # RE sur MID BB : même logique avec la moyenne centrale
    re_mid_b = (_shift(c) > _shift(mid)) & (l <= mid) & (c > mid) & (c > o) & wick_b
    re_mid_s = (_shift(c) < _shift(mid)) & (h >= mid) & (c < mid) & (c < o) & wick_s

    return {"o": o, "h": h, "l": l, "c": c, "mid": mid,
            "mom": (mom_b, mom_s), "ext_any": (any_b, any_s), "exm": (exm_b, exm_s), "ext": (ext_b, ext_s),
            "mhv_cond": (mhv_cond_b, mhv_cond_s), "arah": (arah_b, arah_s),
            "re_zone": (re_zone_b, re_zone_s), "re_mid": (re_mid_b, re_mid_s)}


def _run_cycle(r, lb=15, cw=None):
    """Machine à états du cycle BBMA, par timeframe : EXTREME -> MHV -> CSAK -> MOM -> RE.

    Un code n'est produit que s'il découle de l'étape précédente du même sens :
      - MHV  : retour dans la BB, au plus lb bougies après le dernier Extreme ;
      - CSAK : première bougie qui clôture au-delà de la MID et des MA5/10 après le MHV (cw bougies max) ;
      - MOM  : clôture hors BB après le CSAK ; ensuite seulement une nouvelle cassure fraîche ;
      - RE / REM : rebond (rejet) sur la zone MA5/10 ou la MID BB, après un MOM.
    Annulation : clôture du mauvais côté de la MID BB à partir du CSAK ; délai dépassé avant CSAK ;
    Extreme de sens contraire = nouveau cycle (alerte de TP + armement d'un setup MHV).
    Retourne (signaux {code: (achat, vente)}, étapes, sens, ancre) ; l'ancre est le niveau extrême du cycle."""
    cw = CSAK_WINDOW if cw is None else cw
    c, h, l, mid = r["c"], r["h"], r["l"], r["mid"]
    n = len(c)
    out = {k: np.zeros((2, n), bool) for k in ("MHV", "CSAK", "MOM", "RE", "REM")}
    stage_a, dir_a, anchor_a = np.zeros(n, np.int8), np.zeros(n, np.int8), np.full(n, np.nan)
    ex_b, ex_s = r["ext_any"]
    stage, d, sbar, anc = 0, 0, 0, np.nan

    for i in range(n):
        if stage:
            k = 0 if d == 1 else 1
            if stage >= 3 and not np.isnan(mid[i]) and (c[i] - mid[i]) * d < 0:
                stage = 0                                              # MID BB franchie : cycle annulé
            elif stage == 1 and i - sbar > lb:
                stage = 0
            elif stage == 2 and i - sbar > cw:
                stage = 0
        if stage:
            k = 0 if d == 1 else 1
            if stage == 1 and r["mhv_cond"][k][i]:
                out["MHV"][k, i] = True
                stage, sbar = 2, i
            if stage == 2 and r["arah"][k][i]:
                out["CSAK"][k, i] = True
                stage, sbar = 3, i
            if stage >= 3:
                st0 = stage
                m = r["mom"][k]
                if m[i] and (stage == 3 or not (i > 0 and m[i - 1])):
                    out["MOM"][k, i] = True
                    stage, sbar = 4, i
                if st0 >= 4:
                    zb, mb = r["re_zone"][k][i], r["re_mid"][k][i]
                    if zb:
                        out["RE"][k, i] = True
                    if mb:
                        out["REM"][k, i] = True
                    if zb or mb:
                        stage, sbar = 5, i

        e = 1 if (ex_b[i] and not ex_s[i]) else -1 if (ex_s[i] and not ex_b[i]) else 0
        if e:
            prev_e = bool((ex_b if e == 1 else ex_s)[i - 1]) if i else False
            if stage == 0 or e != d:
                stage, d, sbar = 1, e, i
                anc = l[i] if e == 1 else h[i]
            elif stage == 1:
                sbar = i
            elif stage == 2 and not prev_e:                            # nouvel Extreme franc : on repart
                stage, sbar = 1, i
                anc = l[i] if e == 1 else h[i]
        if stage:
            anc = min(anc, l[i]) if d == 1 else max(anc, h[i])
        else:
            anc = np.nan
        stage_a[i], dir_a[i], anchor_a[i] = stage, (d if stage else 0), anc

    return {k: (v[0], v[1]) for k, v in out.items()}, stage_a, dir_a, anchor_a


def _analyze(df, lb=15, cycle=None):
    """(signaux, infos) de toutes les bougies. Causal. cycle : True = machine à états stricte,
    False = conditions brutes (CSAK = première Candle Arah, RE avec tendance mémorisée), None = CSAK_CYCLE_ONLY."""
    cycle = CSAK_CYCLE_ONLY if cycle is None else cycle
    r = _raw(df, lb)
    n = len(r["c"])
    if cycle:
        sig, stage, direc, anchor = _run_cycle(r, lb)
    else:
        arah_b, arah_s = r["arah"]
        first = lambda a: a & ~_shift(a.astype(float), 0.0).astype(bool)
        mhv_b = _rec(r["ext_any"][0], lb) & r["mhv_cond"][0]
        mhv_s = _rec(r["ext_any"][1], lb) & r["mhv_cond"][1]
        mom_b, mom_s = r["mom"]
        trend = pd.Series(np.where(mom_b | arah_b, 1, np.where(mom_s | arah_s, -1, np.nan))).ffill().to_numpy()
        sig = {"MOM": r["mom"], "MHV": (mhv_b, mhv_s), "CSAK": (first(arah_b), first(arah_s)),
               "RE": (r["re_zone"][0] & (trend == 1), r["re_zone"][1] & (trend == -1)),
               "REM": (r["re_mid"][0] & (trend == 1), r["re_mid"][1] & (trend == -1))}
        stage, direc, anchor = np.zeros(n, np.int8), np.zeros(n, np.int8), np.full(n, np.nan)
    sig["EXM"], sig["EXT"] = r["exm"], r["ext"]
    return sig, {"stage": stage, "dir": direc, "anchor": anchor}


def _signals(df, lb=15, cycle=None):
    """Signaux de toutes les bougies : {code: (tableau achat, tableau vente)}."""
    return _analyze(df, lb, cycle)[0]


def _at(sig, pos):
    """(code, sens) de la bougie à la position pos (priorité d'affichage), ou (None, None)."""
    for code in PRIORITY:
        b, s = sig[code]
        if b[pos]: return code, "B"
        if s[pos]: return code, "S"
    return None, None


def _all_at(sig, pos):
    """Tous les codes actifs à la position pos : [(code, sens), ...] dans l'ordre de priorité."""
    found = []
    for code in PRIORITY:
        b, s = sig[code]
        if b[pos]: found.append((code, "B"))
        if s[pos]: found.append((code, "S"))
    return found


def detect(df, lb=15, cycle=None):
    """Signal principal de la dernière bougie : (code, sens), sens 'B' ou 'S'.
    Priorité d'affichage : RE > REM > MHV > CSAK > MOM > EXM > EXT (RE et MHV ne sont plus masqués par MOM ;
    detect_all renvoie tous les codes). EXM/EXT = alerte de TP Wajib et armement d'un setup MHV, pas une direction."""
    return _at(_signals(df, lb, cycle), -1)


def detect_all(df, lb=15, cycle=None):
    """Tous les codes de la dernière bougie : [(code, sens), ...] (plusieurs possibles par bougie)."""
    return _all_at(_signals(df, lb, cycle), -1)


def cycle_state(df, lb=15, cycle=None):
    """État du cycle à la dernière bougie : (sens, étape) ou None. Étape 'EXTREME' = setup MHV armé."""
    info = _analyze(df, lb, cycle)[1]
    st = int(info["stage"][-1])
    if not st:
        return None
    return ("B" if info["dir"][-1] == 1 else "S"), STAGE_NAMES[st]


def cycle_anchor(df, lb=15, cycle=None):
    """Niveau extrême du cycle en cours à la dernière bougie (plus bas pour un achat, plus haut pour une vente),
    ou None si aucun cycle n'est actif. Sert de stop derrière l'Extreme / MHV."""
    a = _analyze(df, lb, cycle)[1]["anchor"][-1]
    return None if np.isnan(a) else float(a)


RECENT_BARS = 5     # profondeur de recherche du dernier signal (affichage seulement, hors synthèse)


def recent_signal(df, lookback=RECENT_BARS, lb=15, cycle=None):
    """Dernier signal parmi les `lookback` bougies qui précèdent la dernière : (code, sens, age) ou None.
    age = nombre de bougies avant la dernière (1 = l'avant-dernière). Affichage uniquement :
    la synthèse, la confiance, la jauge et la qualité n'utilisent que detect()."""
    sig = _signals(df, lb, cycle)
    for age in range(1, min(lookback, len(df) - 1) + 1):
        code, sens = _at(sig, -1 - age)
        if code:
            return code, sens, age
    return None


def synthese_ex(cells, tf_dispo, use_filter, d1_trend):
    """Synthèse multi-TF BBMA Oma Ally.
    cells : {tf: (code, sens)} ; tf_dispo : TF scannés ; d1_trend : 'B', 'S' ou None.
    Retourne (tf_entree, code_entree, sens, setup) ; setup = 'continuation' ou 'retournement'
    (MHV après Extreme sans signal de structure). (None, None, None, None) s'il n'y a pas d'entrée.
    Le filtre D1 s'applique par type de setup (SETUP_D1_FILTER) : désactivé pour le retournement."""
    tfs_ord = [t for t in TF_ORDER if t in tf_dispo]
    setup = "continuation"

    # 1. TF directeur : le plus haut TF ayant un signal de structure (MOM / RE / REM)
    idx = next((i for i, t in enumerate(tfs_ord) if cells.get(t, (None, None))[0] in SIG_DIRECTEUR), None)
    if idx is None:                       # sinon : setup de retournement Extreme -> MHV
        idx = next((i for i, t in enumerate(tfs_ord) if cells.get(t, (None, None))[0] in SIG_REVERSAL), None)
        setup = "retournement"
    if idx is None:
        return None, None, None, None
    tf_dir = tfs_ord[idx]
    code_dir, sens = cells[tf_dir]

    # 4. Un TF supérieur au directeur qui contredit le sens -> pas de trade
    #    (retournement : seuls les signaux de structure contraires annulent)
    for t in tfs_ord[:idx]:
        code, d = cells.get(t, (None, None))
        if code and d != sens and (setup == "continuation" or code in SIG_DIRECTEUR):
            return None, None, None, None

    # 5. Filtre D1 (EMA50), selon le type de setup
    if use_filter and SETUP_D1_FILTER.get(setup, True) and d1_trend and d1_trend != sens:
        return None, None, None, None

    # 2. TF d'entrée plus bas que le directeur, même sens (le plus proche du directeur)
    for t in tfs_ord[idx + 1:]:
        code, d = cells.get(t, (None, None))
        if code in SIG_ENTREE and d == sens:
            return t, code, sens, setup

    # 3. Aucun TF d'entrée trouvé : le directeur sert d'entrée
    return tf_dir, code_dir, sens, setup


def synthese(cells, tf_dispo, use_filter, d1_trend):
    """Comme synthese_ex, sans le type de setup : (tf_entree, code_entree, sens) ou (None, None, None)."""
    return synthese_ex(cells, tf_dispo, use_filter, d1_trend)[:3]




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


def resample_tf(df, tf):
    """Bougies d'un timeframe à partir de l'historique brut. Chemin unique pour le scan, la qualité et le backtest :
    le H4 passe toujours par bbma_session.resample_h4 (ancre H4_SESSION)."""
    if tf == "H4":
        import bbma_session                      # import tardif : bbma_session importe bbma_core
        return bbma_session.resample_h4(df)
    rs = TFS[tf][2]
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return df


def _closed_bars(pair, tf, data):
    """Bougies fermées d'une paire sur un timeframe (resampling, bougie en formation exclue), ou None."""
    interval, period, _ = TFS[tf]
    df = data.get((pair, interval, period))
    if df is None:
        return None
    df = resample_tf(df, tf)
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
