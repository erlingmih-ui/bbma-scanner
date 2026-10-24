"""Niveaux de support et de résistance : colonne « Niveaux » du tableau.

Module isolé du moteur BBMA : il n'est appelé par aucun calcul de signal, de grade ou de score.
Il fait son propre scan (scan_levels) et renvoie ses propres erreurs.

Règles (bougies fermées uniquement) :
  - pivots (swings) : un haut ou un bas dépassant strictement k bougies de chaque côté. Un pivot n'est
    retenu qu'après k bougies suivantes (pas de look-ahead) ;
  - PDH / PDL : plus haut et plus bas de la dernière journée D1 fermée (timeframes intraday) ;
  - force d'une zone : somme des contacts, chacun pondéré par son âge (demi-vie HALF_LIFE_BARS bougies) ;
  - niveau franchi : si une clôture a dépassé le niveau de BREAK_ATR × ATR depuis son dernier contact, du côté
    opposé à celui où se trouve le prix, le niveau est ignoré. Un support traversé vers le bas n'est donc
    plus un support. La réaffirmation d'un niveau franchi (retest) n'est pas modélisée.

Toutes les distances sont des multiples d'ATR, pour être comparables entre instruments.
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
NEAR_ATR = 0.15          # clôture à moins de NEAR_ATR × ATR d'un niveau valide : « sur un niveau »
NEAR_WARN_ATR = 0.5      # niveau à moins de NEAR_WARN_ATR × ATR : chiffre en gras
BREAK_ATR = 0.3          # niveau franchi depuis son dernier contact : clôture au-delà de BREAK_ATR × ATR
HALF_LIFE_BARS = 240     # un contact perd la moitié de son poids toutes les 240 bougies (~10 jours en H1)
PDH_TFS = ("M5", "M15", "H1", "H4")   # PDH / PDL seulement sur les timeframes intraday

RED, GREEN, GREY, AMBER = "#ef5350", "#26a69a", "#787b86", "#f0b90b"


def _atr(df, n=ATR_PERIOD):
    """Average True Range (moyenne simple sur n bougies)."""
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def confirmed_pivots_idx(df, k=PIVOT_K):
    """Pivots (hauts et bas) confirmés : liste de (prix, indice de la bougie), triée par indice.

    Un pivot au centre c n'est retenu que si c + k <= len(df) - 1 : il faut k bougies après lui."""
    if len(df) < 2 * k + 1:
        return []
    out = []
    for col, sign in (("High", 1), ("Low", -1)):
        x = sign * df[col].to_numpy(float)                 # pour les bas : on cherche un maximum de -Low
        win = sliding_window_view(x, 2 * k + 1)            # ligne j : fenêtre centrée sur j + k
        centre = win[:, k]
        # Pivot strict : la bougie centrale dépasse STRICTEMENT ses voisines (évite les plateaux)
        others_max = np.maximum(win[:, :k].max(axis=1), win[:, k + 1:].max(axis=1))
        for j in np.nonzero(centre > others_max)[0]:
            out.append((float(sign * centre[j]), int(j + k)))
    return sorted(out, key=lambda t: t[1])


def confirmed_pivots(df, k=PIVOT_K):
    """Prix des pivots confirmés, triés."""
    return sorted(p for p, _ in confirmed_pivots_idx(df, k))


def _cluster(points, tol, n):
    """Regroupe des pivots (prix, indice) : un point rejoint le groupe s'il est à moins de tol de sa moyenne.

    n : nombre de bougies disponibles, utilisé pour pondérer les contacts par leur âge.
    Retourne une liste de zones {price, touches, strength, last, src}."""
    groups, group = [], []
    for p, i in sorted(points):
        if group and p - np.mean([q for q, _ in group]) > tol:
            groups.append(group)
            group = []
        group.append((p, i))
    if group:
        groups.append(group)
    zones = []
    for g in groups:
        idx = [i for _, i in g]
        zones.append({
            "price": float(np.mean([p for p, _ in g])),
            "touches": len(g),
            "strength": float(sum(0.5 ** ((n - 1 - i) / HALF_LIFE_BARS) for i in idx)),
            "last": max(idx),
            "src": "pivot",
        })
    return zones


def sr_levels(df, extra=(), k=PIVOT_K, tol_atr=TOL_ATR, near_atr=NEAR_ATR, break_atr=BREAK_ATR):
    """Résistance et support valides les plus proches de la dernière bougie fermée de df.

    extra : niveaux fournis par l'appelant, sous la forme (prix, source, indice du dernier contact).
    Retourne None si l'ATR n'est pas calculable, sinon :
      res      : {atr, touches, src, strength} de la résistance la plus proche au-dessus du prix, ou None
      sup      : {atr, touches, src, strength} du support le plus proche sous le prix, ou None
      on_level : True si la clôture est à moins de near_atr × ATR d'un niveau valide
      atr      : valeur de l'ATR utilisée
      close    : dernière clôture fermée (prix de référence)
      (chaque niveau contient aussi "price", le prix absolu du niveau)
      casses   : nombre de niveaux ignorés car déjà franchis depuis leur dernier contact"""
    if len(df) <= ATR_PERIOD:
        return None
    a = _atr(df).iloc[-1]
    if pd.isna(a) or a <= 0:
        return None
    closes = df["Close"].to_numpy(float)
    n, close = len(df), closes[-1]

    zones = _cluster(confirmed_pivots_idx(df, k), tol_atr * a, n)
    zones += [{"price": float(p), "touches": 1, "strength": 1.0, "last": int(i), "src": s}
              for p, s, i in extra]

    valid, broken_above, broken_below = [], [], []
    for z in zones:
        seg = closes[max(z["last"] + 1, 0):]                 # clôtures depuis le dernier contact
        if z["price"] > close:                               # résistance candidate : franchie vers le haut ?
            gone = seg.size > 0 and bool((seg > z["price"] + break_atr * a).any())
        else:                                                # support candidat : franchi vers le bas ?
            gone = seg.size > 0 and bool((seg < z["price"] - break_atr * a).any())
        if not gone:
            valid.append(z)
        elif z["price"] > close:
            broken_above.append(z)
        else:
            broken_below.append(z)

    near = near_atr * a
    above = [z for z in valid if z["price"] > close + near]
    below = [z for z in valid if z["price"] < close - near]
    on = any(abs(z["price"] - close) <= near for z in valid)

    def nearest(cands):
        if not cands:
            return None
        z = min(cands, key=lambda x: abs(x["price"] - close))
        return {"atr": abs(z["price"] - close) / a, "touches": z["touches"], "src": z["src"],
                "strength": round(z["strength"], 2), "price": z["price"]}

    return {
        "res": nearest(above),
        "sup": nearest(below),
        "res_franchi": nearest([z for z in broken_above if z["price"] > close + near]),   # affiché barré
        "sup_franchi": nearest([z for z in broken_below if z["price"] < close - near]),
        "on_level": on,
        "atr": a,
        "close": close,
        "casses": len(broken_above) + len(broken_below),
    }


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
            day = d1.iloc[-1]                                  # dernière journée D1 fermée = jour précédent
            start = int(df.index.searchsorted(d1.index[-1] + pd.Timedelta(days=1)))   # premier bougie du jour courant
            i = start - 1                                      # contrôle de franchissement à partir du jour courant
            extra = ((float(day["High"]), "PDH", i), (float(day["Low"]), "PDL", i))
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
    """Texte détaillé (infobulle)."""
    if not levels:
        return "Niveaux non calculables"
    parts = []
    for name, lvl, word in (("Résistance", levels["res"], "au-dessus"), ("Support", levels["sup"], "en dessous")):
        if lvl:
            parts.append(f"{name} {word} à {_fr(lvl['atr'])} ATR "
                         f"({lvl['touches']} touche(s), force {_fr(lvl['strength'])}, {lvl['src']})")
        else:
            parts.append(f"{name} : aucun niveau valide")
    for name, fr, word in (("Résistance", levels.get("res_franchi"), "au-dessus"),
                           ("Support", levels.get("sup_franchi"), "en dessous")):
        if fr:
            parts.append(f"{name} {word} franchi à {_fr(fr['atr'])} ATR (affiché barré, ignoré comme obstacle)")
    if levels.get("casses"):
        parts.append(f"{levels['casses']} niveau(x) déjà franchi(s) au total")
    if levels["on_level"]:
        parts.append("clôture sur un niveau")
    return " · ".join(parts)


def short(levels):
    """Texte d'une ligne : « Rés 2,0 · Sup 1,2 »."""
    up = "–" if not levels or not levels["res"] else _fr(levels["res"]["atr"])
    dn = "–" if not levels or not levels["sup"] else _fr(levels["sup"]["atr"])
    return f"Rés {up} · Sup {dn}"


def _side_html(name, lvl, color, franchi=None):
    """Lignes d'un côté : niveau valide (en gras si très proche) ; niveau franchi barré en gris s'il est plus proche."""
    lines = []
    if franchi and (lvl is None or franchi["atr"] < lvl["atr"]):
        lines.append(f'<div style="color:{GREY};text-decoration:line-through">{name} {_fr(franchi["atr"])}</div>')
    if not lvl:
        lines.append(f'<div style="color:{GREY}">{name} –</div>')
    else:
        weight = "700" if lvl["atr"] < NEAR_WARN_ATR else "400"
        lines.append(f'<div style="color:{color};font-weight:{weight}">{name} {_fr(lvl["atr"])}</div>')
    return "".join(lines)


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
    body = (_side_html("Rés", levels["res"], RED, levels.get("res_franchi"))
            + _side_html("Sup", levels["sup"], GREEN, levels.get("sup_franchi")))
    return (f'<td class="lvl" title="{tip}" style="font-size:0.8em;line-height:1.25{ring}">'
            f"{body}{_bar_html(levels)}</td>")
