"""Force Buy / Sell « générale » d'une paire, issue de TradingView (en ligne).

Source : analyse technique TradingView (oscillateurs + moyennes mobiles) via le paquet
non officiel ``tradingview_ta``. Elle ne dépend pas de la méthode BBMA : elle remplace le calcul
de ``bbma_gauge.compute_gauge`` dans la colonne « Jauge ».

Pour chaque paire, on interroge plusieurs timeframes (M15 à W1). Chacun donne un nombre d'indicateurs
en BUY / SELL / NEUTRAL ; on en tire un score de -100 à +100, puis une moyenne pondérée par le rang du
timeframe (poids de ``bbma_gauge.TF_WEIGHT``).

Robustesse : aucune fonction publique ne lève d'exception. Si le paquet est absent, si TradingView ne
répond pas ou si une paire est inconnue, la paire vaut None et la cellule affiche « – ».
Les symboles et places de marché sont dans PAIR_MAP : à ajuster si une paire reste vide.
"""
import logging

import bbma_gauge as bg

try:
    from tradingview_ta import Interval, get_multiple_analysis
except Exception:                      # paquet absent ou incompatible : jauge « – », le scanner continue
    Interval = None
    get_multiple_analysis = None

log = logging.getLogger("bbma.strength")

# Timeframes interrogés : (nom pour le poids, nom de la constante Interval)
INTERVALS = (
    ("M15", "INTERVAL_15_MINUTES"),
    ("H1", "INTERVAL_1_HOUR"),
    ("H4", "INTERVAL_4_HOURS"),
    ("D1", "INTERVAL_1_DAY"),
    ("W1", "INTERVAL_1_WEEK"),
)
MIN_INTERVALS = 3          # au-dessous, la « force générale » n'est pas fiable : pas de jauge
TIMEOUT = 8                # secondes par requête

# Par défaut : forex sur FX_IDC puis OANDA.
DEFAULT_ROUTE = ("forex", ("FX_IDC", "OANDA"))
# Exceptions : paire -> (screener, places de marché essayées dans l'ordre)
PAIR_MAP = {
    "XAUUSD": ("forex", ("OANDA", "FX_IDC")),
    "XAGUSD": ("forex", ("OANDA", "FX_IDC")),
    "BTCUSD": ("crypto", ("BITSTAMP", "COINBASE")),
}


def route(pair):
    """(screener, places de marché) d'une paire."""
    return PAIR_MAP.get(pair, DEFAULT_ROUTE)


def interval_score(summary):
    """Score -100..+100 et votes (buy, sell) depuis un résumé TradingView ; None si inutilisable."""
    try:
        buy = int(summary.get("BUY", 0))
        sell = int(summary.get("SELL", 0))
        neutral = int(summary.get("NEUTRAL", 0))
    except Exception:
        return None
    n = buy + sell + neutral
    if n <= 0:
        return None
    return 100.0 * (buy - sell) / n, buy, sell


def aggregate(per_tf):
    """Fusionne {nom_tf: (score, buy, sell)} en {score, label, buy_pct, n_tf} ; None si trop peu de TF."""
    items = [(bg.TF_WEIGHT.get(tf, bg.DEFAULT_TF_WEIGHT), v) for tf, v in per_tf.items() if v is not None]
    if len(items) < MIN_INTERVALS:
        return None
    wsum = sum(w for w, _ in items)
    score = sum(w * v[0] for w, v in items) / wsum
    score = max(-100.0, min(100.0, score))
    votes = sum(w * (v[1] + v[2]) for w, v in items)
    buy_pct = 100.0 * sum(w * v[1] for w, v in items) / votes if votes else 50.0
    label = next((txt for seuil, txt in bg.LABELS if score >= seuil), bg.LABEL_FLOOR)
    return {"score": round(score), "label": label, "buy_pct": round(buy_pct), "n_tf": len(items)}


def _fetch_interval(pairs, interval):
    """{paire: (score, buy, sell)} pour un timeframe. Essaie les places de marché de rang 0, 1, ..."""
    out = {}
    for rank in range(2):
        groups = {}                                   # screener -> {"EXCH:PAIR": paire}
        for p in pairs:
            if p in out:
                continue
            screener, exchanges = route(p)
            if rank < len(exchanges):
                groups.setdefault(screener, {})[f"{exchanges[rank]}:{p}"] = p
        if not groups:
            break
        for screener, mapping in groups.items():
            try:
                res = get_multiple_analysis(screener=screener, interval=interval,
                                            symbols=list(mapping), timeout=TIMEOUT)
            except Exception as e:                    # réseau, limite, symbole refusé : on passe
                log.warning("TradingView %s %s : %s", screener, interval, e)
                continue
            for key, analysis in (res or {}).items():
                p = mapping.get(key)
                if p is None or analysis is None:
                    continue
                sc = interval_score(getattr(analysis, "summary", None) or {})
                if sc is not None:
                    out[p] = sc
    return out


def fetch_strength(pairs):
    """{paire: {score, label, buy_pct, n_tf} ou None}. Ne lève jamais d'exception."""
    pairs = list(pairs)
    result = {p: None for p in pairs}
    if get_multiple_analysis is None or Interval is None or not pairs:
        return result
    try:
        per_pair = {p: {} for p in pairs}
        for tf_name, const in INTERVALS:
            interval = getattr(Interval, const, None)
            if interval is None:
                continue
            for p, sc in _fetch_interval(pairs, interval).items():
                per_pair[p][tf_name] = sc
        for p in pairs:
            result[p] = aggregate(per_pair[p])
    except Exception as e:
        log.warning("force TradingView indisponible : %s", e)
    return result
