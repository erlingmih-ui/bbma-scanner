"""Jauge Buy / Sell : mesure l'équilibre des signaux entre achat et vente pour une paire.

Score de -100 (vente forte) à +100 (achat fort). Chaque timeframe pèse selon son rang
(un signal H4 compte plus qu'un signal M5) et selon la force de son signal.
La tendance D1 (EMA50) ajoute un poids supplémentaire.
Module ajouté au moteur existant : bbma_core.py n'est pas modifié.
"""
import html

# --- Poids par timeframe (un TF supérieur compte davantage) ---
TF_WEIGHT = {"MN": 5.0, "W1": 4.0, "D1": 3.0, "H4": 2.0, "H1": 1.5, "M15": 1.0, "M5": 0.5}
DEFAULT_TF_WEIGHT = 1.0

# --- Force de chaque signal (1.0 = signal fort) ---
CODE_STRENGTH = {"MOM": 1.0, "EXM": 1.0, "EXT": 1.0, "RE": 0.8, "MHV": 0.8, "CSAK": 0.6}
DEFAULT_CODE_STRENGTH = 0.6

# --- Poids de la tendance D1 (EMA50) ---
TREND_WEIGHT = 3.0

# --- Seuils des libellés (score >= valeur) ---
LABELS = ((60, "Achat fort"), (20, "Achat"), (-20, "Neutre"), (-60, "Vente"))
LABEL_FLOOR = "Vente forte"


def compute_gauge(cells, tfs, d1):
    """Score Buy/Sell d'une paire.

    cells : {tf: (code, sens)} ; tfs : timeframes scannés ; d1 : 'B', 'S' ou None.
    Retourne None si aucun poids disponible, sinon dict {score, label, buy_pct}."""
    total = 0.0      # poids maximum possible (chaque TF scanné + tendance D1)
    signed = 0.0     # somme signée des forces
    buy = sell = 0.0

    for t in tfs:
        w = TF_WEIGHT.get(t, DEFAULT_TF_WEIGHT)
        total += w
        code, sens = cells.get(t, (None, None))
        if not code:
            continue
        s = w * CODE_STRENGTH.get(code, DEFAULT_CODE_STRENGTH)
        if sens == "B":
            signed += s
            buy += s
        else:
            signed -= s
            sell += s

    if d1 in ("B", "S"):
        total += TREND_WEIGHT
        signed += TREND_WEIGHT if d1 == "B" else -TREND_WEIGHT

    if total == 0 or (buy + sell == 0 and d1 not in ("B", "S")):
        return None                                # aucun signal ni tendance : pas de jauge

    score = max(-100.0, min(100.0, 100.0 * signed / total))
    label = next((txt for seuil, txt in LABELS if score >= seuil), LABEL_FLOOR)
    active = buy + sell
    buy_pct = 100.0 * buy / active if active else 50.0
    return {"score": round(score), "label": label, "buy_pct": round(buy_pct)}


def gauge_cell(g) -> str:
    """Cellule « Jauge » : barre à deux moitiés séparées (Sell à gauche en rouge, Buy à droite en vert),
    remplie depuis le centre selon le score, libellé du score, et part des votes Sell / Buy dessous."""
    if g is None:
        return '<td class="gauge none">–</td>'
    score = g["score"]
    buy_w = max(0, min(100, score))               # largeur du remplissage Buy (moitié droite)
    sell_w = max(0, min(100, -score))             # largeur du remplissage Sell (moitié gauche)
    cls = "buy" if score > 0 else "sell" if score < 0 else "neutral"
    buy_pct = int(g.get("buy_pct", 50))
    sell_pct = 100 - buy_pct
    tip = html.escape(f"{g['label']} · score {score:+d} · Buy {buy_pct}% / Sell {sell_pct}%")
    return (f'<td class="gauge" title="{tip}">'
            f'<div class="gbar"><div class="gh gs"><i style="width:{sell_w}%"></i></div>'
            f'<div class="gh gb"><i style="width:{buy_w}%"></i></div></div>'
            f'<div class="glab {cls}">{html.escape(g["label"])} {score:+d}</div>'
            f'<div class="gsplit"><span class="gs">S {sell_pct}%</span><span class="gb">B {buy_pct}%</span></div></td>')
