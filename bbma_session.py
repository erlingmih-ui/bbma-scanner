"""Classification de l'entrée (pullback ou poursuite) et grade plafonné.

La session H4 et le resampling H4 sont désormais définis UNIQUEMENT dans bbma_core
(core.H4_SESSION, core.resample_h4, core._closed_bars). Ce module les réexporte pour compatibilité.
"""
import bbma_core as core
from bbma_quality_codes import GRADE_MIN

H4_SESSION = core.H4_SESSION
resample_h4 = core.resample_h4


def closed_bars(pair, tf, data):
    """Bougies fermées, identiques à celles de core (H4 inclus)."""
    return core._closed_bars(pair, tf, data)


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
    """Une poursuite ne peut pas dépasser le grade B."""
    if kind == "poursuite" and grade == "A":
        return GRADE_MIN["A"] - 1, "B"
    return score, grade
