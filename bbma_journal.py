"""Journal de test en avant : une ligne par entrée affichée, à revérifier plus tard sur le graphique.

Module isolé : il lit le résultat du scan et écrit un CSV. Il ne modifie ni les signaux, ni les grades.
Une entrée est identifiée par (paire, TF d'entrée, code, sens, bougie) : elle n'est écrite qu'une fois,
même si le scan tourne plusieurs fois sur la même bougie.

La colonne « resultat » est à remplir à la main, après vérification sur le graphique
(par exemple : atteint, stop, pas atteint). Le fichier est journal/entrees.csv, à côté de l'application,
au format UTF-8 avec séparateur « ; » (ouvrable directement dans Excel).
"""
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import bbma_core as core
import bbma_levels as bl
import bbma_quality as bq
import bbma_session as bs

log = logging.getLogger("bbma")
JOURNAL_PATH = Path(__file__).resolve().parent / "journal" / "entrees.csv"
SEP = ";"
COLUMNS = ["cle", "horodatage_scan", "bougie", "paire", "tf_entree", "code", "sens", "grade", "score",
           "rsi_tf", "obstacle_H1_ATR", "appui_H1_ATR", "sur_niveau_H1", "resultat", "note"]


def entry_rows(pairs, tfs, res, trends, feats, lvl, use_filter, scanned_at=None):
    """Une ligne par paire ayant une entrée, avec les mêmes règles que le tableau (synthèse, grade, niveaux)."""
    now = (scanned_at or datetime.now(timezone.utc)).isoformat(timespec="minutes")
    pairs, tfs = list(pairs), list(tfs)
    rows = []
    for p in pairs:
        cells = {t: res[(p, t)] for t in tfs if (p, t) in res}
        d1 = trends.get(p)
        tf_e, code_e, sens_e = core.synthese(cells, tfs, use_filter, d1)
        if not tf_e:
            continue
        avail = [t for t in tfs if feats.get((p, t)) is not None]
        score = bq.confiance_ponderee(cells, tfs, sens_e, avail)
        q_score, q_grade, _ = bq.qualite_entree(code_e, sens_e, score, d1, feats.get((p, tf_e)))
        q_score, q_grade = bs.cap_grade(q_score, q_grade, bs.entry_kind(cells, tfs, tf_e, code_e))
        f = feats.get((p, tf_e)) or {}
        levels, _ = (lvl or {}).get((p, bl.LEVELS_TF), (None, None))
        # Obstacle = côté du trade ; appui = côté opposé (même logique que le guide)
        obst = (levels["res"] if sens_e == "B" else levels["sup"]) if levels else None
        appui = (levels["sup"] if sens_e == "B" else levels["res"]) if levels else None
        bar = str(f.get("last_close", ""))
        rows.append({
            "cle": f"{p}|{tf_e}|{code_e}|{sens_e}|{bar}",
            "horodatage_scan": now,
            "bougie": bar,
            "paire": p,
            "tf_entree": tf_e,
            "code": code_e,
            "sens": sens_e,
            "grade": q_grade or "",
            "score": "" if q_score is None else q_score,
            "rsi_tf": "" if f.get("rsi") is None else round(f["rsi"], 1),
            "obstacle_H1_ATR": "" if not obst else round(obst["atr"], 2),
            "appui_H1_ATR": "" if not appui else round(appui["atr"], 2),
            "sur_niveau_H1": "" if not levels else ("oui" if levels["on_level"] else "non"),
            "resultat": "",
            "note": "",
        })
    return rows


def record(rows, path=JOURNAL_PATH):
    """Ajoute au journal les lignes nouvelles seulement. Retourne le nombre de lignes écrites."""
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8", newline="") as fh:
            known = {r["cle"] for r in csv.DictReader(fh, delimiter=SEP)}
    new, seen = [], set()
    for r in rows:
        if r["cle"] in known or r["cle"] in seen:
            continue
        seen.add(r["cle"])
        new.append(r)
    if not new:
        return 0
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=SEP, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(new)
    return len(new)
