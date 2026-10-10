"""Moteur BBMA unique : progression R (D1 RE) -> E (H4) -> M (H1) par paire.

Source de vérité pour la colonne Progression ET la colonne Entrée.
Aucun téléchargement en plus : lit le cache disque de bbma_core, qui s'accumule sans limite.

Règles (à confirmer avec votre documentation BBMA) :
- Un setup démarre sur un D1 RE.
- Les étapes suivantes doivent être H4 puis H1, dans le même sens, et former un motif complet (REM, RRE, REE).
- Tout signal de structure de sens contraire (CONTRA_CODES) sur D1, H4 ou H1 annule le setup.
- Le setup expire MAX_AGE après le D1 RE, qu'il soit complet ou non.
- Entrée = motif complet et non expiré. Rien d'autre n'est une entrée.
"""
import numpy as np
import pandas as pd

import bbma_core as core

# Valeurs à confirmer avec votre documentation. MAX_AGE est un paramètre de travail, pas une règle validée.
MAX_AGE = pd.Timedelta(days=5)
MAX_NS = int(MAX_AGE.value)
CONTRA_CODES = {"MOM", "EXT", "MHV", "CSAK", "RE"}   # EXM est normalisé en EXT

PATTERNS = {                     # motifs complets (D1, H4, H1)
    "REM": ["RE", "EXT", "MHV"],
    "RRE": ["RE", "RE", "EXT"],
    "REE": ["RE", "EXT", "EXT"],
}
TF_STEPS = ["D1", "H4", "H1"]
RANK = {"D1": 0, "H4": 1, "H1": 2}          # à instant égal : D1 puis H4 puis H1
LETTER = {"RE": "R", "EXT": "E", "MHV": "M"}


def _norm(code):
    return "EXT" if code == "EXM" else code


def _events(pair, now):
    """Signaux sur bougies fermées D1, H4, H1, triés par instant : (instant_ns, rang, tf, code, sens)."""
    d1 = core._history_cached(pair, "1d", "max")
    h1 = core._history_cached(pair, "1h", "730d")
    if d1 is None or h1 is None:
        return None
    h4 = h1.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    frames = {"D1": (d1, pd.Timedelta(days=1)),
              "H4": (h4, pd.Timedelta(hours=4)),
              "H1": (h1, pd.Timedelta(hours=1))}
    ev = []
    for tf, (df, delta) in frames.items():
        ends = df.index + delta
        mask = np.asarray(ends <= now)            # bougies fermées seulement
        df, ends = df[mask], ends[mask]
        if len(df) < core.MIN_BARS:
            continue
        code, sens = core.codes_series(df)
        for t, c, s in zip(ends.asi8, code, sens):
            if c:
                ev.append((int(t), RANK[tf], tf, _norm(c), s))
    ev.sort(key=lambda x: (x[0], x[1]))
    return ev


def _replay(ev):
    """Rejoue les signaux dans l'ordre. Retourne (étapes [(tf, code, instant)], sens, instant du D1 RE)."""
    path, sens, start = [], None, None
    for t, _, tf, code, s in ev:
        if path and t - start > MAX_NS:           # fenêtre dépassée avant cet événement
            path, sens, start = [], None, None
        if tf == "D1" and code == "RE":           # nouveau setup
            path, sens, start = [(tf, code, t)], s, t
            continue
        if not path:
            continue
        if code in CONTRA_CODES and s != sens:    # signal contraire : on repart de zéro
            path, sens, start = [], None, None
            continue
        k = len(path)
        if k >= len(TF_STEPS) or tf != TF_STEPS[k] or s != sens:
            continue
        prefix = [c for _, c, _ in path]
        if any(seq[:k] == prefix and seq[k] == code for seq in PATTERNS.values()):
            path.append((tf, code, t))
    return path, sens, start


def progress(pair, now=None):
    """État d'une paire, ou None si aucun setup. Clés : sens, letters, steps, complete, expired, start, since.
    complete = nom du motif ('REM', 'RRE', 'REE') ou None. expired = fenêtre MAX_AGE dépassée à 'now'."""
    now = now or pd.Timestamp.now(tz="UTC")
    ev = _events(pair, now)
    if not ev:
        return None
    path, sens, start = _replay(ev)
    if not path:
        return None
    codes = [c for _, c, _ in path]
    complete = next((name for name, seq in PATTERNS.items() if seq == codes), None)
    return {
        "sens": sens,
        "letters": [LETTER[c] for c in codes],
        "steps": len(codes),
        "complete": complete,
        "expired": int(now.value) - start > MAX_NS,
        "start": pd.Timestamp(start, unit="ns", tz="UTC"),
        "since": pd.Timestamp(path[-1][2], unit="ns", tz="UTC"),
    }


def entree(pr):
    """Entrée finale : motif complet et non expiré, sinon None."""
    if pr and pr["complete"] and not pr["expired"]:
        return pr["complete"], pr["sens"]
    return None, None
