"""Ancre de session du H4 et classification de l'entrée (pullback ou poursuite).

- H4_SESSION choisit où commencent les bougies H4 :
    "UTC"  : 00, 04, 08... UTC (comportement actuel de bbma_core)
    "NY17" : 17 h, 21 h, 01 h... heure de New York (clôture forex usuelle)
  Mettre la valeur qui correspond au graphique de votre broker.
- entry_kind distingue un vrai pullback (signal d'entrée sur un TF inférieur) d'une poursuite
  (le TF directeur sert d'entrée, faute de TF d'entrée). Une poursuite ne peut pas être grade A.
Module ajouté : bbma_core.py n'est pas modifié.

Correction : cap_grade ne modifie plus le score. Une poursuite au score élevé reçoit le grade B,
et le score réel reste affiché.
"""
import bbma_core as core
from bbma_quality_codes import GRADE_MIN

H4_SESSION = "UTC"
SESSIONS = {"UTC": ("UTC", None), "NY17": ("America/New_York", "17h")}
AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


def resample_h4(df1h, session=H4_SESSION):
    """Bougies H4 à partir de bougies H1 (UTC), ancrées selon la session choisie."""
    if session == "UTC":
        return df1h.resample("4h").agg(AGG).dropna()
    tz, offset = SESSIONS[session]
    local = df1h.tz_convert(tz).resample("4h", origin="start_day", offset=offset).agg(AGG).dropna()
    return local.tz_convert("UTC")


def closed_bars(pair, tf, data):
    """Comme core._closed_bars, mais le H4 respecte H4_SESSION. None si indisponible."""
    if tf != "H4":
        return core._closed_bars(pair, tf, data)
    raw = data.get((pair, core.TFS["H4"][0], core.TFS["H4"][1]))
    if raw is None:
        return None
    df = resample_h4(raw).iloc[:-1]          # la bougie H4 en cours n'est pas fermée
    return df if len(df) >= core.MIN_BARS else None


def entry_kind(cells, tf_dispo, tf_e, code_e):
    """'pullback', 'poursuite', ou None s'il n'y a pas d'entrée."""
    if not tf_e or not code_e:
        return None
    tfs_ord = [t for t in core.TF_ORDER if t in tf_dispo]
    idx = next((i for i, t in enumerate(tfs_ord)
                if cells.get(t, (None, None))[0] in core.SIG_DIRECTEUR), None)
    if idx is None:
        return None
    return "poursuite" if tf_e == tfs_ord[idx] else "pullback"


def cap_grade(score, grade, kind):
    """Une poursuite ne peut pas dépasser le grade B. Le score réel est conservé."""
    if kind == "poursuite" and grade == "A":
        return score, "B"
    return score, grade
