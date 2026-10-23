"""Niveaux de support et de résistance : colonne « Niveaux » du tableau.

Module totalement isolé du moteur BBMA :
  - il ne modifie aucun fichier existant et n'est appelé par aucun calcul de signal, de grade ou de score ;
  - il fait son propre scan (scan_levels) et renvoie ses propres erreurs ;
  - la colonne affiche, pour chaque paire, la résistance la plus proche au-dessus du prix et le support
    le plus proche en dessous, sur un timeframe de référence fixe (LEVELS_TF), indépendamment des signaux.

Sources de niveaux (bougies fermées uniquement) :
  - pivots (swings) : un haut ou un bas local dépassant strictement k bougies de chaque côté.
    Un pivot n'est retenu qu'après k bougies suivantes (pas de look-ahead) ;
  - PDH / PDL : plus haut et plus bas de la dernière journée D1 fermée (timeframes intraday).

Les pivots proches (à moins de TOL_ATR × ATR de la moyenne de leur groupe) forment une zone ;
le nombre de touches donne la force. Les hauts et les bas sont regroupés ensemble, car un ancien
sommet peut redevenir support. Toutes les distances sont en multiples d'ATR.
"""
import html
import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

log = logging.getLogger("bbma")

LEVELS_TF = "H1"         # timeframe affiché dans la colonne (indépendant des TF choisis pour les signaux)
PIVOT_K = 3              # bougies de chaque côté pour valider un pivot
TOL_ATR = 0.5            # distance maximale (en ATR) pour regrouper deux pivots dans une zone
ATR_PERIOD = 14
NEAR_ATR = 0.3           # clôture à moins de NEAR_ATR × ATR d'une zone : « sur un niveau »
PDH_TFS = ("M5", "M15", "H1", "H4")   # PDH / PDL seulement sur les timeframes intraday


def _atr(df, n=ATR_PERIOD):
    """Average True Range (moyenne simple sur n bougies)."""
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def confirmed_pivots(df, k=PIVOT_K):
    """Prix des pivots (hauts et bas) confirmés dans df, triés.

    Un pivot au centre c n'est retenu que si c + k <= len(df) - 1 : il faut k bougies après lui."""
    if len(df) < 2 * k + 1:
        return []
    prices = []
    for col, sign in (("High", 1), ("Low", -1)):
        x = sign * df[col].to_numpy(float)                 # pour les bas : on cherche un maximum de -Low
        win = sliding_window_view(x, 2 * k + 1)            # ligne j : fenêtre centrée sur j + k
        centre = win[:, k]
        # Pivot strict : la bougie centrale dépasse STRICTEMENT ses voisines (évite les plateaux)
        others_max = np.maximum(win[:, :k].max(axis=1), win[:, k + 1:].max(axis=1))
        prices.extend((sign * centre[centre > others_max]).tolist())
    return sorted(prices)


def _cluster(prices, tol):
    """Regroupe des prix triés : un prix rejoint le groupe s'il est à moins de tol de la moyenne du groupe.
    Retourne une liste de {price, touches}."""
    groups, group = [], []
    for p in sorted(prices):
        if group and p - np.mean(group) > tol:
            groups.append(group)
            group = []
        group.append(p)
    if group:
        groups.append(group)
    return [{"price": float(np.mean(g)), "touches": len(g)} for g in groups]


def sr_levels(df, extra=(), k=PIVOT_K, tol_atr=TOL_ATR, near_atr=NEAR_ATR):
    """Résistance et support les plus proches de la dernière bougie fermée de df.

    extra : niveaux fournis par l'appelant, sous la forme (prix, source), par exemple (PDH, "PDH").
    Retourne None si l'ATR n'est pas calculable, sinon :
      res      : {atr, touches, src} de la résistance la plus proche au-dessus du prix, ou None
      sup      : {atr, touches, src} du support le plus proche sous le prix, ou None
      on_level : True si la clôture est à moins de near_atr × ATR d'une zone
      atr      : valeur de l'ATR utilisée"""
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

    def nearest(cands):
        if not cands:
            return None
        z = min(cands, key=lambda x: abs(x["price"] - close))
        return {"atr": abs(z["price"] - close) / a, "touches": z["touches"], "src": z["src"]}

    return {"res": nearest(above), "sup": nearest(below), "on_level": on, "atr": a}


def levels_one(pair, tf, data):
    """Niveaux d'un couple (paire, TF) à partir des bougies téléchargées.

    bbma_session est importé ici pour que le reste du module reste utilisable sans yfinance.
    Retourne None si les données sont indisponibles."""
    import bbma_session as bs
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return None
    extra = ()
    if tf in PDH_TFS:
        d1 = bs.closed_bars(pair, "D1", data)
        if d1 is not None and len(d1):
            day = d1.iloc[-1]                  # dernière journée D1 fermée = jour précédent
            extra = ((float(day["High"]), "PDH"), (float(day["Low"]), "PDL"))
    return sr_levels(df, extra=extra)


def scan_levels(pairs, tfs=(LEVELS_TF,)):
    """Niveaux S/R pour les paires et timeframes demandés, avec son propre téléchargement.

    Indépendant de run_scan : une erreur ici n'affecte aucune autre colonne.
    Retourne {(paire, tf): (niveaux ou None, message d'erreur ou None)}."""
    import bbma_core as core
    pairs, tfs = list(pairs), list(tfs)
    fetch_tfs = list(dict.fromkeys(tfs + ["D1"]))          # D1 nécessaire pour PDH / PDL
    data = core._fetch_all(pairs, fetch_tfs, trend=False)
    out = {}
    for p in pairs:
        for t in tfs:
            try:
                lv = levels_one(p, t, data)
                out[(p, t)] = (lv, None if lv else "données indisponibles")
            except Exception as e:
                log.exception("niveaux S/R %s %s", p, t)
                out[(p, t)] = (None, f"{type(e).__name__} : {e}")
    return out


def _fr(x):
    """Nombre à une décimale, séparateur français."""
    return f"{x:.1f}".replace(".", ",")


def describe(levels):
    """Texte détaillé (infobulle) décrivant les niveaux."""
    if not levels:
        return "Niveaux non calculables"
    parts = []
    for name, lvl, word in (("Résistance", levels["res"], "au-dessus"), ("Support", levels["sup"], "en dessous")):
        if lvl:
            parts.append(f"{name} {word} à {_fr(lvl['atr'])} ATR ({lvl['touches']} touche(s), {lvl['src']})")
        else:
            parts.append(f"{name} : aucun niveau connu")
    if levels["on_level"]:
        parts.append("clôture sur un niveau")
    return " · ".join(parts)


RED, GREEN, GREY, AMBER = "#ef5350", "#26a69a", "#787b86", "#f0b90b"
NEAR_WARN_ATR = 0.5      # en dessous : niveau très proche, chiffre affiché en gras


def short(levels):
    """Texte d'une ligne pour l'infobulle et les tests : « Rés 2,0 · Sup 1,2 »."""
    up = "–" if not levels or not levels["res"] else _fr(levels["res"]["atr"])
    dn = "–" if not levels or not levels["sup"] else _fr(levels["sup"]["atr"])
    return f"Rés {up} · Sup {dn}"


def _side_html(name, lvl, color):
    """Une ligne « Rés 1,1 » ou « Sup 0,8 » ; en gras si le niveau est très proche."""
    if not lvl:
        return f'<div style="color:{GREY}">{name} –</div>'
    weight = "700" if lvl["atr"] < NEAR_WARN_ATR else "400"
    return f'<div style="color:{color};font-weight:{weight}">{name} {_fr(lvl["atr"])}</div>'


def _bar_html(levels):
    """Barre de place disponible : vert = sous le prix, rouge = au-dessus. Une partie fine = niveau proche."""
    res, sup = levels["res"], levels["sup"]
    if res and sup:
        tot = res["atr"] + sup["atr"]
        left = 50.0 if tot <= 0 else 100.0 * sup["atr"] / tot
    elif sup:
        left = 100.0
    elif res:
        left = 0.0
    else:
        left = 50.0
    return ('<div style="display:flex;height:4px;border-radius:2px;overflow:hidden;margin:3px 0">'
            f'<div style="width:{left:.0f}%;background:{GREEN}"></div>'
            f'<div style="flex:1;background:{RED}"></div></div>')


def levels_cell(levels, error=None) -> str:
    """Cellule « Niveaux » pour un couple (paire, LEVELS_TF). L'erreur est donnée en infobulle."""
    if not levels:
        tip = html.escape(error or "Niveaux non calculables (voir le terminal)")
        return f'<td class="lvl none" title="{tip}">n/d</td>'
    tip = html.escape(describe(levels))
    ring = f";box-shadow:0 0 0 1px {AMBER}" if levels["on_level"] else ""
    body = _side_html("Rés", levels["res"], RED) + _side_html("Sup", levels["sup"], GREEN)
    return (f'<td class="lvl" title="{tip}" style="font-size:0.8em;line-height:1.25{ring}">'
            f"{body}{_bar_html(levels)}</td>")
