"""Jauge Buy / Sell locale et gratuite, sans TradingView.

Remplace bbma_strength.fetch_strength avec la même interface : {paire: {score, label, buy_pct, n_tf} ou None}.

Principe (comme un « résumé technique ») : pour chaque timeframe, on compte les indicateurs
qui votent BUY, SELL ou NEUTRE, à partir des bougies fermées déjà téléchargées par le scanner
(cache disque, aucun appel supplémentaire à TradingView).
  - Oscillateurs : RSI 14, Stochastique %K 14/3, CCI 20, Williams %R 14, MACD 12/26/9, Momentum 10.
  - Moyennes mobiles : EMA et SMA de 10, 20, 50, 100 et 200 périodes (vote : clôture au-dessus = BUY).
Le score par timeframe vaut 100 x (achats - ventes) / nombre de votes valides.
Les timeframes sont ensuite fusionnés avec les poids de bbma_gauge.TF_WEIGHT, comme la jauge d'origine.

Robustesse : ne lève jamais d'exception. Pas assez de données, ou téléchargement impossible : None (« – »).
Module ajouté : bbma_core.py, bbma_session.py et bbma_gauge.py ne sont pas modifiés.
"""
import logging

import numpy as np
import pandas as pd

import bbma_core as core
import bbma_gauge as bg
import bbma_session as bs

log = logging.getLogger("bbma.techgauge")

TFS = ("M15", "H1", "H4", "D1", "W1")
MIN_TF = 3                       # en dessous, la jauge n'est pas fiable : None
MA_PERIODS = (10, 20, 50, 100, 200)


def _cmp(a, b):
    """'B' si a > b, 'S' si a < b, 'N' si égal, None si une valeur manque."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return "B" if a > b else ("S" if a < b else "N")


def _votes(df):
    """Liste de votes ('B', 'S', 'N' ou None) pour une série de bougies fermées."""
    c, h, l = df["Close"], df["High"], df["Low"]
    last = c.iloc[-1]
    v = []

    # RSI 14 : survente / surachat
    r = core.rsi(c).iloc[-1]
    v.append(None if pd.isna(r) else ("B" if r < 30 else "S" if r > 70 else "N"))

    # Stochastique %K (14, lissage 3) : 20 / 80
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    raw_k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(3).mean().iloc[-1]
    v.append(None if pd.isna(k) else ("B" if k < 20 else "S" if k > 80 else "N"))

    # CCI 20 : -100 / +100 (calcul sur les 20 dernières bougies)
    tp = ((h + l + c) / 3).iloc[-20:]
    if len(tp) == 20:
        dev = np.mean(np.abs(tp - tp.mean()))
        cci = (tp.iloc[-1] - tp.mean()) / (0.015 * dev) if dev > 0 else np.nan
        v.append(None if pd.isna(cci) else ("B" if cci < -100 else "S" if cci > 100 else "N"))
    else:
        v.append(None)

    # Williams %R 14 : -80 / -20
    wr = -100 * (hh.iloc[-1] - last) / (hh.iloc[-1] - ll.iloc[-1]) if hh.iloc[-1] != ll.iloc[-1] else np.nan
    v.append(None if pd.isna(wr) else ("B" if wr < -80 else "S" if wr > -20 else "N"))

    # MACD 12/26/9 : ligne MACD au-dessus du signal
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    v.append(_cmp(macd.iloc[-1], sig.iloc[-1]))

    # Momentum 10 : clôture actuelle vs clôture de 10 bougies avant
    v.append(_cmp(last, c.iloc[-11]) if len(c) > 10 else None)

    # Moyennes mobiles : clôture au-dessus = achat
    for n in MA_PERIODS:
        v.append(_cmp(last, c.ewm(span=n, adjust=False).mean().iloc[-1]))
        v.append(_cmp(last, c.rolling(n).mean().iloc[-1]))
    return v


def tf_score(df):
    """(score -100..+100, achats, ventes) pour un timeframe, ou None si aucun vote valide."""
    votes = [x for x in _votes(df) if x is not None]
    if not votes:
        return None
    buy = votes.count("B")
    sell = votes.count("S")
    return 100.0 * (buy - sell) / len(votes), buy, sell


def aggregate(per_tf):
    """Fusionne {tf: (score, achats, ventes)} en {score, label, buy_pct, n_tf}, ou None si trop peu de TF."""
    items = [(bg.TF_WEIGHT.get(tf, bg.DEFAULT_TF_WEIGHT), v) for tf, v in per_tf.items() if v is not None]
    if len(items) < MIN_TF:
        return None
    wsum = sum(w for w, _ in items)
    score = max(-100.0, min(100.0, sum(w * v[0] for w, v in items) / wsum))
    votes = sum(w * (v[1] + v[2]) for w, v in items)
    buy_pct = 100.0 * sum(w * v[1] for w, v in items) / votes if votes else 50.0
    label = next((txt for seuil, txt in bg.LABELS if score >= seuil), bg.LABEL_FLOOR)
    return {"score": round(score), "label": label, "buy_pct": round(buy_pct), "n_tf": len(items)}


def fetch_strength(pairs):
    """{paire: {score, label, buy_pct, n_tf} ou None}. Ne lève jamais d'exception."""
    pairs = list(pairs)
    result = {p: None for p in pairs}
    if not pairs:
        return result
    try:
        data = core._fetch_all(pairs, list(TFS), trend=False)   # même téléchargement / cache que le scanner
        for p in pairs:
            per = {}
            for tf in TFS:
                df = bs.closed_bars(p, tf, data)                 # H4 respecte H4_SESSION
                if df is not None:
                    per[tf] = tf_score(df)
            result[p] = aggregate(per)
    except Exception as e:
        log.warning("jauge locale indisponible : %s", e)
    return result
