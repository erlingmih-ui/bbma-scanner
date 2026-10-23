"""Colonne « Qualité » de l'entrée BBMA, et Confiance pondérée.

Module ajouté au moteur existant : bbma_core.py n'est pas modifié. Les fonctions de
téléchargement, de détection et de synthèse sont réutilisées ; les poids, seuils et
libellés viennent de bbma_quality_codes.py.

Corrections par rapport à la version précédente :
  - anti-poursuite : un prix trop éloigné du mauvais côté de la MID ne rapporte plus de points ;
  - confiance pondérée : chaque timeframe compte selon son poids, les signaux contraires
    réduisent la confiance, et un timeframe sans données est exclu au lieu de pénaliser.
"""
import html
import logging
from datetime import datetime, timezone

import pandas as pd

import bbma_core as core
import bbma_fresh as bf
import bbma_gauge as bg
import bbma_levels as bl

import bbma_patterns as bpat
import bbma_session as bs

log = logging.getLogger("bbma")
from bbma_quality_codes import (
    ATR_PERIOD, CHASE_BELOW, CHASE_BELOW_MAX_ATR, CHASE_MISSING, CHASE_STEPS, CODE_WEIGHT,
    D1_AGREE, D1_DISAGREE, D1_UNKNOWN, GRADE_MIN, MAX_CONFLUENCE, MID_PERIOD,
    QUAL_LABELS, RSI_MISSING, RSI_POINTS, RSI_ZONE_OVERBOUGHT, RSI_ZONE_PULLBACK,
)


def atr(df, n=ATR_PERIOD):
    """Average True Range (moyenne simple sur n bougies)."""
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def features_one(pair, tf, data):
    """RSI et distance signée close -> MID BB en ATR, sur la dernière bougie fermée. None si indisponible."""
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return None
    c = df["Close"]
    mid = c.rolling(MID_PERIOD).mean().iloc[-1]
    a = atr(df).iloc[-1]
    r = core.rsi(c).iloc[-1]
    dist = None if pd.isna(a) or a <= 0 or pd.isna(mid) else float((c.iloc[-1] - mid) / a)
    last_close = df.index[-1] + bf.bar_length(tf)
    try:
        recent = core.recent_signal(df)      # dernier signal des bougies précédentes (affichage grisé)
    except Exception:
        recent = None
    try:
        recent_entry = recent_entry_signal(df)   # dernier RE / MHV récent (colonne Entrée BBMA, grisé)
    except Exception:
        recent_entry = None
    try:
        alive = bpat.alive_signal(df)            # signal de pattern encore en vie (colonne Pattern MTF)
    except Exception:
        alive = None
    try:
        levels = bl.levels_one(pair, tf, data)   # niveaux S/R (colonne Niveaux, informative)
    except Exception:
        log.exception("niveaux S/R indisponibles pour %s %s", pair, tf)   # visible dans le terminal
        levels = None
    return {"rsi": None if pd.isna(r) else float(r), "dist_atr": dist, "last_close": last_close,
            "recent": recent, "recent_entry": recent_entry, "alive": alive, "levels": levels}


def recent_entry_signal(df, lookback=core.RECENT_BARS):
    """Dernier signal d'ENTRÉE (RE ou MHV) parmi les `lookback` bougies qui précèdent la dernière :
    (code, sens, age) ou None. Contrairement à core.recent_signal, un CSAK plus récent ne le masque pas.
    Affichage uniquement."""
    sig = core._signals(df)
    for age in range(1, min(lookback, len(df) - 1) + 1):
        for code in ("RE", "MHV"):
            b, s = sig[code]
            if b.iloc[-1 - age]:
                return code, "B", age
            if s.iloc[-1 - age]:
                return code, "S", age
    return None


def _scan_one(pair, tf, data):
    """((code, sens), ok) avec les bougies fermées de bbma_session (ancre H4 prise en compte)."""
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return (None, None), False
    try:
        return core.detect(df), True
    except Exception:
        return (None, None), False


def scan_quality(pairs, tfs, trend=True):
    """Scan identique à core.scan_with_rsi, plus les caractéristiques par (paire, TF).
    Retourne (res, trends, failed, total, ts, feats)."""
    pairs, tfs = list(pairs), list(tfs)
    data = core._fetch_all(pairs, tfs, trend)
    jobs = [(p, t) for p in pairs for t in tfs]
    res, failed = {}, 0
    for p, t in jobs:
        r, ok = _scan_one(p, t, data)
        res[(p, t)] = r
        failed += not ok
    trends = {p: core._d1_trend(p, data) for p in pairs} if trend else {}
    feats = {(p, t): features_one(p, t, data) for p, t in jobs}
    return res, trends, failed, len(jobs), datetime.now(timezone.utc), feats


def confiance_ponderee(cells, tfs, sens_ref, avail):
    """Confiance pondérée par timeframe, entre 0 et 1.

    - Numérateur : poids des TF dont le signal va dans le sens de l'entrée, moins le poids des TF
      dont le signal est contraire (un signal contraire réduit la confiance).
    - Dénominateur : poids des TF disponibles. Un TF sans données n'est pas compté.
    - Poids par TF : bbma_gauge.TF_WEIGHT (MN 5 … M5 0.5).
    Retourne None s'il n'y a ni sens ni TF disponible."""
    if sens_ref is None:
        return None
    avail = set(avail)
    num = den = 0.0
    for t in tfs:
        if t not in avail:
            continue
        w = bg.TF_WEIGHT.get(t, bg.DEFAULT_TF_WEIGHT)
        den += w
        _, d = cells.get(t, (None, None))
        if d == sens_ref:
            num += w
        elif d is not None:
            num -= w
    if den == 0:
        return None
    return max(0.0, num) / den


def qualite_entree(code, sens, conf, d1, feat):
    """Score de qualité (0-100), grade A/B/C et détail par critère.
    Retourne (None, None, None) s'il n'y a pas d'entrée."""
    if not code or not sens:
        return None, None, None
    sign = 1 if sens == "B" else -1
    feat = feat or {}
    detail = {
        "confluence": round(MAX_CONFLUENCE * (conf or 0)),
        "signal": CODE_WEIGHT.get(code, 0),
    }

    # RSI : r_dir = RSI du point de vue du trade (élevé = favorable au sens)
    r = feat.get("rsi")
    if r is None:
        detail["rsi"] = RSI_MISSING
    else:
        r_dir = r if sign == 1 else 100 - r
        if r_dir > RSI_ZONE_OVERBOUGHT:
            detail["rsi"] = RSI_POINTS["overbought"]
        elif r_dir < RSI_ZONE_PULLBACK:
            detail["rsi"] = RSI_POINTS["pullback"]
        else:
            detail["rsi"] = RSI_POINTS["neutral"]

    # Anti-poursuite : ds > 0 = prix étendu dans le sens du trade ; ds < 0 = prix de l'autre côté de la MID
    d = feat.get("dist_atr")
    if d is None:
        detail["chase"] = CHASE_MISSING
    else:
        ds = d * sign
        if ds < 0:
            # Proche de la MID : points réduits. Loin de la MID du mauvais côté : 0 point.
            detail["chase"] = CHASE_BELOW if -ds <= CHASE_BELOW_MAX_ATR else 0
        else:
            detail["chase"] = next((pts for seuil, pts in CHASE_STEPS if ds <= seuil), 0)

    # Filtre D1
    if d1 == sens:
        detail["d1"] = D1_AGREE
    elif d1 is None:
        detail["d1"] = D1_UNKNOWN
    else:
        detail["d1"] = D1_DISAGREE

    score = sum(detail.values())
    grade = "A" if score >= GRADE_MIN["A"] else "B" if score >= GRADE_MIN["B"] else "C"
    return score, grade, detail


def qual_cell(score, grade, detail) -> str:
    """Cellule « Qualité » : grade et score /100, détail au survol."""
    if score is None:
        return '<td class="qual s3 none">–</td>'
    parts = " · ".join(f"{QUAL_LABELS[k]} {v}" for k, v in detail.items())
    tip = html.escape(f"Qualité {score}/100 · {parts}")
    return f'<td class="qual s3 {grade}" title="{tip}"><b>{grade}</b> <span class="qs">{score}</span></td>'
