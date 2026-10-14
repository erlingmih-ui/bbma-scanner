"""Machine à états du cycle BBMA Oma Ally : Extreme -> MHV -> CSAK -> RE.

Pour chaque timeframe, deux cycles sont suivis en parallèle : achat (B) et vente (S).
Un cycle avance UNIQUEMENT dans l'ordre :

    0 aucun  ->  1 EXTREME  ->  2 MHV  ->  3 CSAK  ->  4 RE

  - Extreme (EXM ou EXT du même côté) : ouvre le cycle. Achat : MA sous la bande basse ; vente : MA
    au-dessus de la bande haute. Un nouvel Extreme du même côté ramène le cycle à l'étape 1
    (le prix a continué : le MHV précédent n'était pas valable).
  - MHV du même côté : seulement si le cycle est à l'étape 1.
  - CSAK du même côté : seulement si le cycle est à l'étape 2.
  - RE du même côté : seulement si le cycle est à l'étape 3.
Un cycle retombe à 0 quand :
  - un Extreme du côté opposé apparaît. Si le cycle était à l'étape 3 ou plus, c'est le TP Wajib
    (tp_for = côté de la position à clôturer). Le cycle opposé démarre alors à l'étape 1 ;
  - il dépasse l'âge maximal de son étape (MAX_AGE, en bougies) ;
  - la clôture passe du mauvais côté de la Mid BB à l'étape 3 ou plus (tendance perdue).

Les signaux de base viennent de bbma_core._signals : bbma_core.py n'est pas modifié, et ce module n'agit
pas sur la synthèse ni sur la colonne Qualité. Il ajoute une information : la phase du cycle.

Intégration minimale dans app.py (facultative) :
    import bbma_cycle as bcy
    states = bcy.scan_cycles(pairs, tfs)                       # {(paire, tf): état ou None}
    ... dans la boucle des cellules : bcy.cycle_cell(states.get((p, t)))
    ... et une fois dans le CSS de la page : bcy.CYCLE_CSS
"""
import html

import numpy as np
import pandas as pd

import bbma_core as core
import bbma_session as bs

# --- Paramètres -------------------------------------------------------------------------------
# Durée de vie maximale de chaque étape, en bougies depuis la dernière transition.
MAX_AGE = {1: 15, 2: 25, 3: 30, 4: 10}
# True : un signal d'une étape plus avancée peut ouvrir ou faire avancer le cycle sans passer par les
# étapes précédentes (ex. un CSAK sans Extreme ni MHV). False = ordre strict Oma Ally.
ALLOW_SKIP = False
# La clôture du mauvais côté de la Mid BB arrête un cycle déjà à l'étape CSAK ou RE.
MID_BREAK_RESETS = True
MID_PERIOD = 20
# Étapes qui constituent un point d'entrée dans la méthode (MHV et RE).
ENTRY_PHASES = (2, 4)

NAMES = {1: "EXTREME", 2: "MHV", 3: "CSAK", 4: "RE"}
NEXT = {1: "MHV", 2: "CSAK", 3: "RE", 4: "Extreme contraire (TP)"}
LETTERS = {1: "E", 2: "M", 3: "C", 4: "R"}
SIDE_RGB = {"B": "38,166,154", "S": "239,83,80"}
SIDE_TEXT = {"B": "achat", "S": "vente"}

CYCLE_CSS = """<style>
table.bb td.cyc{padding:6px 8px}
table.bb td.cyc .st{display:inline-flex;gap:3px;vertical-align:middle}
table.bb td.cyc .st i{font-style:normal;font-size:10px;font-weight:700;width:15px;height:15px;line-height:15px;
border-radius:4px;background:#2a2e39;color:#787b86}
table.bb td.cyc.B .st i.on{background:#26a69a;color:#fff}
table.bb td.cyc.S .st i.on{background:#ef5350;color:#fff}
table.bb td.cyc .nm{display:block;margin-top:3px;font-size:11px;color:#b2b5be}
table.bb td.cyc.none{color:#787b86}
</style>"""


# --- Cœur : la machine à états (fonction pure sur des tableaux) -------------------------------

def run_machine(ext, mhv, csak, re_, close, mid, allow_skip=None, max_age=None, mid_break=None):
    """Exécute le cycle sur toutes les bougies.

    ext, mhv, csak, re_ : chacun un couple (série achat, série vente) de booléens.
    close, mid : clôtures et Mid BB. Retourne un dict de tableaux numpy :
      phase_b, phase_s (0 à 4), age_b, age_s (bougies depuis la dernière transition),
      tp_b, tp_s (True si le cycle de ce côté, à l'étape 3 ou plus, vient d'être clos par un Extreme opposé)."""
    allow_skip = ALLOW_SKIP if allow_skip is None else allow_skip
    max_age = MAX_AGE if max_age is None else max_age
    mid_break = MID_BREAK_RESETS if mid_break is None else mid_break

    sig = {k: [np.asarray(x, dtype=bool) for x in v]
           for k, v in (("ext", ext), ("mhv", mhv), ("csak", csak), ("re", re_))}
    C = np.asarray(close, dtype=float)
    M = np.asarray(mid, dtype=float)
    n = len(C)
    phase = [np.zeros(n, int), np.zeros(n, int)]      # [achat, vente]
    age = [np.zeros(n, int), np.zeros(n, int)]
    tp = [np.zeros(n, bool), np.zeros(n, bool)]
    cur = [0, 0]
    ag = [0, 0]

    for i in range(n):
        for k in (0, 1):                               # 0 = achat, 1 = vente
            o = 1 - k
            was = cur[k]
            if was:
                ag[k] += 1
                wrong_side = (C[i] < M[i]) if k == 0 else (C[i] > M[i])      # NaN : False
                if sig["ext"][o][i]:
                    if was >= 3:
                        tp[k][i] = True                # TP Wajib pour la position de ce côté
                    cur[k], ag[k] = 0, 0
                elif ag[k] > max_age[was]:
                    cur[k], ag[k] = 0, 0
                elif mid_break and was >= 3 and wrong_side:
                    cur[k], ag[k] = 0, 0

            p = cur[k]
            if sig["ext"][k][i]:                       # un Extreme ouvre ou relance le cycle (étape 1)
                cur[k], ag[k] = 1, 0
            elif p == 1 and sig["mhv"][k][i]:
                cur[k], ag[k] = 2, 0
            elif p == 2 and sig["csak"][k][i]:
                cur[k], ag[k] = 3, 0
            elif p == 3 and sig["re"][k][i]:
                cur[k], ag[k] = 4, 0
            elif allow_skip:
                for target, key in ((4, "re"), (3, "csak"), (2, "mhv")):
                    if sig[key][k][i] and p < target:
                        cur[k], ag[k] = target, 0
                        break
            phase[k][i], age[k][i] = cur[k], ag[k]

    return {"phase_b": phase[0], "phase_s": phase[1], "age_b": age[0], "age_s": age[1],
            "tp_b": tp[0], "tp_s": tp[1]}


# --- Application à un DataFrame de bougies ----------------------------------------------------

def _inputs(df):
    sig = core._signals(df)
    ext = tuple(sig["EXM"][k] | sig["EXT"][k] for k in (0, 1))
    mid = df["Close"].rolling(MID_PERIOD).mean()
    return ext, sig["MHV"], sig["CSAK"], sig["RE"], df["Close"], mid


def cycle_series(df, **kw):
    """Phase du cycle à chaque bougie. DataFrame (index de df) : sens ('B', 'S' ou None), phase (0-4),
    age, tp_for ('B' / 'S' si un TP Wajib vient de se produire pour cette position, sinon None).
    Si les deux côtés ont une phase, le plus avancé est retenu (à égalité, le plus récent)."""
    ext, mhv, csak, re_, close, mid = _inputs(df)
    r = run_machine(ext, mhv, csak, re_, close, mid, **kw)
    pb, ps, ab, as_ = r["phase_b"], r["phase_s"], r["age_b"], r["age_s"]
    pick_b = (pb > ps) | ((pb == ps) & (pb > 0) & (ab <= as_))
    phase = np.where(pick_b, pb, ps)
    age = np.where(pick_b, ab, as_)
    sens = np.where(phase == 0, None, np.where(pick_b, "B", "S"))
    tp_for = np.where(r["tp_b"], "B", np.where(r["tp_s"], "S", None))
    return pd.DataFrame({"sens": sens, "phase": phase, "age": age, "tp_for": tp_for}, index=df.index)


def cycle_state(df, **kw):
    """État du cycle à la dernière bougie fermée : dict {sens, phase, name, next, age, entry, tp_for,
    steps}, ou None si aucun cycle n'est actif. steps = nombre d'étapes franchies (0 à 4)."""
    last = cycle_series(df, **kw).iloc[-1]
    if last["phase"] == 0:
        return None
    ph = int(last["phase"])
    tp = last["tp_for"]
    tp = tp if tp in ("B", "S") else None              # pandas peut convertir None en NaN
    return {"sens": last["sens"], "phase": ph, "name": NAMES[ph], "next": NEXT[ph],
            "age": int(last["age"]), "entry": ph in ENTRY_PHASES, "tp_for": tp, "steps": ph}


def cycle_one(pair, tf, data):
    """État du cycle d'un couple (paire, TF) à partir des données déjà téléchargées, ou None."""
    df = bs.closed_bars(pair, tf, data)
    if df is None:
        return None
    try:
        return cycle_state(df)
    except Exception:
        return None


def scan_cycles(pairs, tfs):
    """{(paire, tf): état ou None}. Les bougies viennent du cache disque de bbma_core
    (pas de téléchargement supplémentaire si le scan vient d'avoir lieu)."""
    pairs, tfs = list(pairs), list(tfs)
    data = core._fetch_all(pairs, tfs, False)
    return {(p, t): cycle_one(p, t, data) for p in pairs for t in tfs}


# --- Affichage --------------------------------------------------------------------------------

def tooltip(st):
    if st is None:
        return "Cycle BBMA : aucun cycle actif"
    chain = " > ".join(NAMES[i] + (" ✓" if i <= st["steps"] else "") for i in (1, 2, 3, 4))
    lines = [f"Cycle {SIDE_TEXT[st['sens']]} : étape {st['phase']}/4 ({st['name']})",
             chain,
             f"Prochaine étape : {st['next']}",
             f"Âge de l'étape : {st['age']} bougie(s)"]
    if st["entry"]:
        lines.append("Étape d'entrée (MHV / RE)")
    if st["tp_for"]:
        lines.append(f"TP Wajib : clôturer les positions {SIDE_TEXT[st['tp_for']]}")
    return "&#10;".join(html.escape(x) for x in lines)


def cycle_cell(st) -> str:
    """Cellule HTML : quatre pastilles E M C R (franchies = colorées), nom de l'étape en dessous."""
    tip = tooltip(st)
    if st is None:
        return f'<td class="cyc none" title="{tip}">-</td>'
    dots = "".join(f'<i class="{"on" if i <= st["steps"] else ""}">{LETTERS[i]}</i>' for i in (1, 2, 3, 4))
    tp = " · TP" if st["tp_for"] else ""
    return (f'<td class="cyc {st["sens"]}" title="{tip}"><span class="st">{dots}</span>'
            f'<span class="nm">{st["name"]}{tp}</span></td>')
