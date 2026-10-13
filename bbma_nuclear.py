"""Scanner BBMA Nuclear, indépendant des signaux BBMA (MOM, EXM, EXT, MHV, CSAK, RE).

Règles, par timeframe, sur les bougies fermées (bbma_session.closed_bars) :
  - Bandes de Bollinger : période 20, déviation 2, sur la clôture.
  - EMA50 de la clôture.
  - NUCLEAR SELL : EMA50 > bande haute, pendant les 3 dernières bougies fermées.
  - NUCLEAR BUY  : EMA50 < bande basse, pendant les 3 dernières bougies fermées.
  - Si la condition échoue sur une seule de ces bougies : pas de signal.

Score = nombre de timeframes en Nuclear dans le sens majoritaire (ex. 4/7).
Force = somme des poids des timeframes concernés (MN 7, W1 6, D1 5, H4 4, H1 3, M15 2, M5 1).
En cas d'égalité entre achats et ventes, aucune direction n'est retenue.

Données : le scan télécharge une seule fois (core._fetch_all) et partage les bougies avec
les signaux BBMA. Aucun téléchargement supplémentaire n'est fait pour le Nuclear.
Module ajouté : bbma_core.py, bbma_quality.py et bbma_session.py ne sont pas modifiés.
"""
import html
from datetime import datetime, timezone

import bbma_core as core
import bbma_quality as bq
import bbma_session as bs

NUCLEAR_TFS = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]
TF_FORCE = {"MN": 7, "W1": 6, "D1": 5, "H4": 4, "H1": 3, "M15": 2, "M5": 1}
BB_PERIOD = 20
BB_DEV = 2
EMA_PERIOD = 50
CONFIRM_BARS = 3

SIDE_LABEL = {"S": "SELL", "B": "BUY"}
SIDE_RGB = {"S": "239,83,80", "B": "38,166,154"}     # rouge (vente), vert (achat)
QUAL_RANK = {"A": 3, "B": 2, "C": 1}


def nuclear_tf(pair, tf, data):
    """(disponible, direction) d'un couple (paire, TF). direction : 'S', 'B' ou None (aucun signal)."""
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return False, None
    c = df["Close"]
    mid = c.rolling(BB_PERIOD).mean()
    sd = c.rolling(BB_PERIOD).std(ddof=0)
    upper = mid + BB_DEV * sd
    lower = mid - BB_DEV * sd
    ema = c.ewm(span=EMA_PERIOD, adjust=False).mean()
    last = slice(-CONFIRM_BARS, None)
    if (ema > upper).iloc[last].all():
        return True, "S"
    if (ema < lower).iloc[last].all():
        return True, "B"
    return True, None


def summarize(pair, data):
    """Résumé Nuclear d'une paire sur les 7 timeframes."""
    per_tf = {t: nuclear_tf(pair, t, data) for t in NUCLEAR_TFS}
    avail = [t for t in NUCLEAR_TFS if per_tf[t][0]]
    sells = [t for t in avail if per_tf[t][1] == "S"]
    buys = [t for t in avail if per_tf[t][1] == "B"]
    if len(sells) > len(buys):
        side, on = "S", sells
        other = buys
    elif len(buys) > len(sells):
        side, on = "B", buys
        other = sells
    else:
        side, on, other = None, [], []
    return {
        "side": side,
        "score": len(on),
        "avail": len(avail),
        "force": sum(TF_FORCE[t] for t in on),
        "tfs": on,
        "other": other,
        "tied": side is None and bool(sells),
        "missing": [t for t in NUCLEAR_TFS if t not in avail],
    }


def scan_full(pairs, tfs, trend=True):
    """Scan BBMA + Nuclear en un seul téléchargement.

    Télécharge l'union des timeframes affichés et des 7 timeframes Nuclear. Le tableau n'affiche
    que les timeframes choisis ; le Nuclear utilise toujours les 7.
    Retourne (res, trends, failed, total, ts, feats, nuc)."""
    pairs, tfs = list(pairs), list(tfs)
    all_tfs = list(dict.fromkeys(tfs + NUCLEAR_TFS))
    data = core._fetch_all(pairs, all_tfs, trend)
    jobs = [(p, t) for p in pairs for t in all_tfs]
    res, failed = {}, 0
    for p, t in jobs:
        r, ok = bq._scan_one(p, t, data)
        res[(p, t)] = r
        failed += not ok
    trends = {p: core._d1_trend(p, data) for p in pairs} if trend else {}
    feats = {(p, t): bq.features_one(p, t, data) for p, t in jobs}
    nuc = {p: summarize(p, data) for p in pairs}
    return res, trends, failed, len(jobs), datetime.now(timezone.utc), feats, nuc


def sort_key(n):
    """Force, puis score Nuclear (priorités 1 et 2). Les qualité et confiance sont ajoutées dans app.py."""
    if not n or n["side"] is None:
        return (0, 0)
    return (n["force"], n["score"])


def tooltip(n):
    """Infobulle détaillée de la cellule Nuclear, lignes séparées par un retour à la ligne HTML."""
    if n is None:
        return "NUCLEAR : données absentes"
    if n["avail"] == 0:
        lines = ["NUCLEAR : données absentes"]
    elif n["side"] is None:
        head = "NUCLEAR : signaux opposés à égalité" if n["tied"] else "NUCLEAR : aucun signal"
        lines = [head]
    else:
        cond = "EMA50 > High BB" if n["side"] == "S" else "EMA50 < Low BB"
        lines = [f"NUCLEAR {SIDE_LABEL[n['side']]}", ""]
        lines += [f"{t} : {cond}" for t in n["tfs"]]
        if n["other"]:
            lines.append("Sens opposé : " + ", ".join(n["other"]))
        lines.append("")
        lines.append(f"Score : {n['score']}/{n['avail']} · Force : {n['force']}")
    if n["avail"] and n["missing"]:
        lines.append("Données absentes : " + ", ".join(n["missing"]))
    return "&#10;".join(html.escape(x) for x in lines)


def nuc_cell(n) -> str:
    """Cellule « Nuclear » : direction, score, timeframes concernés. Intensité selon le score."""
    tip = tooltip(n)
    if n is None or n["side"] is None:
        return f'<td class="nuc none" title="{tip}">-</td>'
    ratio = n["score"] / n["avail"] if n["avail"] else 0
    alpha = 0.18 + 0.67 * ratio                      # 1/7 : pâle, 7/7 : foncé
    text = "#fff" if alpha >= 0.5 else ("#ef5350" if n["side"] == "S" else "#26a69a")
    style = f"background:rgba({SIDE_RGB[n['side']]},{alpha:.2f});color:{text}"
    label = f"NUC {n['side']} {n['score']}/{n['avail']}"
    return (f'<td class="nuc" style="{style}" title="{tip}">'
            f'<div><b>{label}</b></div><div class="ntf">{" ".join(n["tfs"])}</div></td>')
