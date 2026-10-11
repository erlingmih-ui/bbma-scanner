"""Badge de fraîcheur : âge de la dernière bougie fermée, par paire et par timeframe.

Règles (âge = maintenant - fermeture de la dernière bougie fermée) :
  - ok    : âge < 2 périodes du timeframe ;
  - warn  : âge < max(24 h, 2 périodes) (week-end, fermeture de marché) ;
  - stale : au-delà, données à ne pas utiliser sans vérification.
Le statut d'une paire est le plus mauvais de ses timeframes.
Module ajouté au moteur existant : bbma_core.py n'est pas modifié.
"""
import html

import pandas as pd

BAR_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN": 43200}
FLOOR = pd.Timedelta(hours=24)
RANK = {"ok": 0, "warn": 1, "stale": 2}
LABEL = {"ok": "à jour", "warn": "en retard", "stale": "périmé", "none": "absent"}
COLOR = {"ok": "#639922", "warn": "#BA7517", "stale": "#E24B4A", "none": "#787b86"}


def bar_length(tf):
    return pd.Timedelta(minutes=BAR_MINUTES.get(tf, 60))


def status_tf(last_close, tf, now):
    """('ok' | 'warn' | 'stale' | 'none', âge ou None)."""
    if last_close is None:
        return "none", None
    age = now - last_close
    length = bar_length(tf)
    if age < 2 * length:
        return "ok", age
    if age < max(FLOOR, 2 * length):
        return "warn", age
    return "stale", age


def fmt_age(age):
    minutes = int(age.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {mins:02d}"
    return f"{hours // 24} j"


def pair_status(pair, feats, tfs, now=None):
    """(statut global, infobulle) pour une paire, à partir des caractéristiques de chaque TF."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    parts, known, missing = [], [], []
    for tf in tfs:
        st, age = status_tf((feats.get((pair, tf)) or {}).get("last_close"), tf, now)
        if st == "none":
            missing.append(tf)
            continue
        known.append(st)
        parts.append(f"{tf} : {LABEL[st]} ({fmt_age(age)})")
    if missing:
        parts.append("absent : " + ", ".join(missing))
    if not known:
        return "none", "Fraîcheur : aucune donnée"
    worst = max(known, key=RANK.get)
    return worst, "Fraîcheur · " + " · ".join(parts)


def dot_html(status, tip):
    """Pastille colorée à côté du nom de la paire."""
    return (f'<span class="fdot" style="background:{COLOR[status]}" '
            f'title="{html.escape(tip)}" aria-label="{html.escape(tip)}"></span>')
