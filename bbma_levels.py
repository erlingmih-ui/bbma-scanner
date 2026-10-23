"""Niveaux de support et de résistance : colonne informative « Niveaux ».

Intégration minimale : bbma_quality.features_one ajoute la clé « levels », et app.py affiche la
colonne « Niveaux » (cellule de bbma_levels.levels_cell). Aucun grade, aucune synthèse et aucun
score de qualité n'en dépend : la colonne sert à observer avant toute intégration au score.

Sources de niveaux (calculées sur les bougies fermées uniquement) :
  - pivots (swings) : un haut ou un bas local sur une fenêtre de k bougies de chaque côté.
    Un pivot en i n'est retenu qu'à partir de la bougie i+k (pas de look-ahead) ;
  - PDH / PDL : plus haut et plus bas de la dernière journée D1 fermée (timeframes intraday).

Les pivots proches (à moins de TOL_ATR × ATR de la moyenne du groupe) sont regroupés en zones.
Le nombre de touches d'une zone sert de force. Un ancien haut peut redevenir support, donc
les hauts et les bas sont regroupés ensemble.

Toutes les distances sont exprimées en ATR, pour être comparables entre XAUUSD, BTCUSD et les paires FX.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

PIVOT_K = 3              # bougies de chaque côté pour valider un pivot
TOL_ATR = 0.5            # distance maximale (en ATR) pour regrouper deux pivots dans une même zone
ATR_PERIOD = 14
NEAR_ATR = 0.3           # clôture à moins de NEAR_ATR × ATR d'un niveau : « sur le niveau »
PDH_TFS = ("M5", "M15", "H1", "H4")   # PDH / PDL utiles seulement sur les timeframes intraday


def _atr(df, n=ATR_PERIOD):
    """Average True Range (moyenne simple sur n bougies), même formule que bbma_quality.atr."""
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def confirmed_pivots(df, k=PIVOT_K):
    """Prix des pivots (hauts et bas) confirmés dans df, dans l'ordre chronologique.

    Un pivot au centre c est retenu seulement si c + k <= len(df) - 1 : il faut k bougies
    après lui pour le confirmer. Ainsi, une bougie future ne peut jamais créer un pivot passé."""
    n = len(df)
    if n < 2 * k + 1:
        return []
    prices = []
    for col, sign in (("High", 1), ("Low", -1)):
        x = sign * df[col].to_numpy(float)               # pour les bas : on cherche un maximum de -Low
        win = sliding_window_view(x, 2 * k + 1)          # ligne j : fenêtre centrée sur j + k
        centre = win[:, k]
        # Pivot strict : la bougie centrale dépasse STRICTEMENT toutes ses voisines.
        # Sans cette règle, un plateau (égalités) ferait de chaque bougie un pivot.
        others_max = np.maximum(win[:, :k].max(axis=1), win[:, k + 1:].max(axis=1))
        is_pivot = centre > others_max
        prices.extend((sign * centre[is_pivot]).tolist())
    return sorted(prices)


def _cluster(prices, tol):
    """Regroupe des prix triés : un prix rejoint le groupe si il est à moins de tol de la moyenne du groupe.
    Retourne une liste de {price, touches}."""
    zones = []
    group = []
    for p in sorted(prices):
        if group and p - np.mean(group) > tol:
            zones.append(group)
            group = []
        group.append(p)
    if group:
        zones.append(group)
    return [{"price": float(np.mean(g)), "touches": len(g)} for g in zones]


def sr_levels(df, extra=(), k=PIVOT_K, tol_atr=TOL_ATR, near_atr=NEAR_ATR):
    """Niveaux de support et de résistance proches de la dernière bougie fermée de df.

    extra : niveaux supplémentaires fournis par l'appelant, sous la forme (prix, source) ; ex. (PDH, "PDH").
    Retourne None si l'ATR n'est pas calculable, sinon un dictionnaire :
      res       : résistance la plus proche au-dessus du prix, {atr, touches, src} ou None
      sup       : support le plus proche sous le prix, {atr, touches, src} ou None
      on_level  : True si la clôture est à moins de near_atr × ATR d'une zone
      atr       : valeur de l'ATR utilisée
    Les distances « atr » sont des multiples de l'ATR, toujours positives."""
    if len(df) <= ATR_PERIOD:
        return None
    a = _atr(df).iloc[-1]
    if pd.isna(a) or a <= 0:
        return None
    close = float(df["Close"].iloc[-1])

    zones = [dict(z, src="pivot") for z in _cluster(confirmed_pivots(df, k), tol_atr * a)]
    zones += [{"price": float(p), "touches": 1, "src": s} for p, s in extra]

    near = near_atr * a
    above = [z for z in zones if z["price"] > close + near]
    below = [z for z in zones if z["price"] < close - near]
    on = any(abs(z["price"] - close) <= near for z in zones)

    def pick(cands, nearest):
        if not cands:
            return None
        z = nearest(cands, key=lambda x: abs(x["price"] - close))
        return {"atr": abs(z["price"] - close) / a, "touches": z["touches"], "src": z["src"]}

    return {"res": pick(above, min), "sup": pick(below, min), "on_level": on, "atr": a}


def room_atr(levels, sens):
    """Espace avant le prix dans le sens du trade, en ATR.

    Achat : distance à la résistance la plus proche (obstacle). Vente : distance au support le plus proche.
    None si aucun niveau n'est connu dans ce sens, c'est-à-dire aucun obstacle identifié."""
    if not levels or sens not in ("B", "S"):
        return None
    lvl = levels["res"] if sens == "B" else levels["sup"]
    return lvl["atr"] if lvl else None


def appui_atr(levels, sens):
    """Niveau derrière le prix, dans le sens inverse du trade, en ATR (appui pour l'entrée).

    Achat : support le plus proche sous le prix. Vente : résistance la plus proche au-dessus du prix."""
    if not levels or sens not in ("B", "S"):
        return None
    lvl = levels["sup"] if sens == "B" else levels["res"]
    return lvl["atr"] if lvl else None


def _fr(x):
    """Nombre à une décimale, séparateur français."""
    return f"{x:.1f}".replace(".", ",")


def label(levels, sens):
    """Texte court pour la cellule « Niveaux » et son infobulle. « – » si pas de niveaux ou pas de sens."""
    if not levels or sens not in ("B", "S"):
        return "–"
    parts = []
    ob = room_atr(levels, sens)
    parts.append("aucun obstacle connu" if ob is None else f"obstacle à {_fr(ob)} ATR")
    ap = appui_atr(levels, sens)
    if ap is not None:
        parts.append(f"appui à {_fr(ap)} ATR")
    if levels["on_level"]:
        parts.append("sur un niveau")
    return " · ".join(parts)


def levels_one(pair, tf, data):
    """Niveaux d'un couple (paire, TF) à partir des données téléchargées (bbma_core._fetch_all).

    Importe bbma_session à l'intérieur de la fonction : le reste du module reste utilisable
    sans yfinance (tests, outils hors application). Retourne None si indisponible."""
    import bbma_session as bs
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return None
    extra = ()
    if tf in PDH_TFS:
        d1 = bs.closed_bars(pair, "D1", data)
        if d1 is not None and len(d1):
            day = d1.iloc[-1]                     # dernière journée D1 fermée = jour précédent
            extra = ((float(day["High"]), "PDH"), (float(day["Low"]), "PDL"))
    return sr_levels(df, extra=extra)


def levels_cell(levels, sens) -> str:
    """Cellule « Niveaux » : obstacle devant le prix dans le sens du trade, détail au survol."""
    import html
    if not levels or sens not in ("B", "S"):
        return '<td class="lvl none">–</td>'
    ob = room_atr(levels, sens)
    short = "libre" if ob is None else f"{_fr(ob)} ATR"
    tip = html.escape(label(levels, sens))
    return f'<td class="lvl" title="{tip}">{short}</td>'
