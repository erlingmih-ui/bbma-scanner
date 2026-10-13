"""Ancre de session du H4 et classification de l'entrée (pullback ou poursuite).

- H4_SESSION choisit où commencent les bougies H4 :
    "UTC"  : 00, 04, 08... UTC (comportement actuel de bbma_core)
    "NY17" : 17 h, 21 h, 01 h... heure de New York (clôture forex usuelle)
  Mettre la valeur qui correspond au graphique de votre broker.
- Le H4 n'a qu'un seul chemin : core.resample_tf -> resample_h4 (scan, qualité et backtest).
- entry_kind distingue un vrai pullback (signal d'entrée sur un TF inférieur) d'une poursuite
  (le TF directeur sert d'entrée, faute de TF d'entrée). Une poursuite ne peut pas être grade A.
Module ajouté : bbma_core.py n'est pas modifié.
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
    """Bougies fermées d'une paire sur un TF. Chemin unique : core._closed_bars passe par core.resample_tf,
    qui appelle resample_h4 (ancre H4_SESSION) pour le H4 : scan, qualité et backtest voient les mêmes bougies."""
    return core._closed_bars(pair, tf, data)


def entry_kind(cells, tf_dispo, tf_e, code_e):
    """'pullback', 'poursuite', ou None s'il n'y a pas d'entrée."""
    if not tf_e or not code_e:
        return None
    tfs_ord = [t for t in core.TF_ORDER if t in tf_dispo]
    idx = next((i for i, t in enumerate(tfs_ord)
                if cells.get(t, (None, None))[0] in core.SIG_DIRECTEUR), None)
    if idx is None:       # pas de structure : setup de retournement Extreme -> MHV
        idx = next((i for i, t in enumerate(tfs_ord)
                    if cells.get(t, (None, None))[0] in core.SIG_REVERSAL), None)
        return None if idx is None else "retournement"
    return "poursuite" if tf_e == tfs_ord[idx] else "pullback"


def cap_grade(score, grade, kind):
    """Une poursuite ne peut pas dépasser le grade B."""
    if kind == "poursuite" and grade == "A":
        return GRADE_MIN["A"] - 1, "B"
    return score, grade
