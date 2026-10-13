"""Interface Streamlit du scanner BBMA OA MTF.

La logique métier (téléchargement, détection, synthèse, confiance) est dans ``bbma_core``.
La colonne Qualité est calculée par ``bbma_quality`` (poids dans ``bbma_quality_codes``).
La jauge Buy/Sell est calculée par ``bbma_gauge``.
Ce module ne gère que l'affichage et les paramètres de la barre latérale.

Lancement : streamlit run app.py
"""
import html
import logging

import streamlit as st

import bbma_core as core
import bbma_fresh as bf
import bbma_session as bs
import bbma_gauge as bg
import bbma_pairs as bp
import bbma_quality as bq
from bbma_quality_codes import GRADE_MIN, MAX_CHASE, MAX_CONFLUENCE, MAX_D1, MAX_RSI, MAX_SIGNAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

APP_TITLE = "BBMA OA MTF Scanner"
DEFAULT_PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD"
MAX_PAIRS = 12
CACHE_TTL = 240          # secondes ; inférieur à AUTO_REFRESH pour qu'un vrai scan ait lieu à chaque cycle
SCAN_VERSION = 4         # à incrémenter quand la structure de feats (bbma_quality) change : invalide le cache
AUTO_REFRESH = 300       # secondes entre deux scans automatiques
RSI_LOW, RSI_HIGH = 30, 70   # repères visuels uniquement (couleur), aucun effet sur les signaux

STRONG_CODES = {"MOM", "EXM", "EXT"}   # signaux affichés en gras (trait latéral)
TPW_CODES = {"EXM", "EXT"}             # signaux pour lesquels le flag TP Wajib est affiché
ARROW = {"B": "▲", "S": "▼"}
NAMES = {
    "MOM": "Momentum",
    "EXM": "Extreme Magic (MA10)",
    "EXT": "Extreme (MA5)",
    "MHV": "Market Hilang Volume",
    "CSAK": "Candle Arah Kukuh",
    "RE": "Reentry",
}
LEGEND_CSS = """
.bb-legend{margin-top:14px;background:#1e222d;border:1px solid #2a2e39;border-radius:14px;padding:14px 16px;font:13px/1.5 -apple-system,"Trebuchet MS",Roboto,sans-serif;color:#b2b5be}
.bb-legend h4{margin:14px 0 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#787b86}
.bb-legend h4:first-child{margin-top:0}
.bb-legend b{color:#d1d4dc}
.bb-legend .chips{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0}
.bb-legend .chip{padding:3px 9px;border-radius:6px;font-weight:700;font-size:12px}
.bb-legend .chip.buy{background:rgba(38,166,154,.18);color:#26a69a}
.bb-legend .chip.sell{background:rgba(239,83,80,.18);color:#ef5350}
.bb-legend .chip.mid{background:#2962ff;color:#fff}
.bb-legend .chip.low{background:#2a2e39;color:#787b86}
.bb-legend .chip.qa{background:#26a69a;color:#fff}
.bb-legend .chip.qb{background:#2962ff;color:#fff}
.bb-legend .chip.tp{background:#2962ff;color:#fff}
.bb-legend .sigs{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:3px 14px}
.bb-legend .grad{height:8px;border-radius:4px;margin:6px 0 4px;background:linear-gradient(90deg,#ef5350,#2a2e39 50%,#26a69a)}
.bb-legend .scale{display:flex;justify-content:space-between;font-size:11px;color:#787b86}
.bb-legend .note{color:#787b86;font-size:12px;margin-top:4px}
"""


def legend_html(show_recent=False) -> str:
    """Légende organisée par sections, avec les mêmes couleurs que le tableau."""
    recent_note = ('<div class="note">Texte grisé « CODE ▲ −n » : la dernière bougie n’a pas de signal ; c’est le dernier signal, '
                   'n bougies plus tôt (information seulement, non utilisé par la synthèse).</div>') if show_recent else ""
    sigs = [("MOM", "Momentum"), ("EXM", "Extreme Magic (MA10)"), ("EXT", "Extreme (MA5)"),
            ("MHV", "Market Hilang Volume"), ("CSAK", "Candle Arah Kukuh"), ("RE", "Reentry (strict)")]
    sigs_html = "".join(f"<div><b>{code}</b> · {name}</div>" for code, name in sigs)
    return (
        '<div class="bb-legend">'
        '<h4>Choix des paires</h4>'
        '<div><b>Mode saisie</b> (case « Choisir dans la liste » décochée) : les <b>12 paires par défaut</b> sont '
        'toujours scannées. Pour en <b>ajouter</b>, écrivez-les dans la zone <b>« Paires à ajouter »</b>, séparées '
        'par un espace, par exemple <b>AUDJPY CHFJPY</b>. Validez avec Ctrl+Entrée ou en cliquant ailleurs.</div>'
        '<div><b>Mode liste</b> (case cochée) : la zone « Paires à ajouter » est ignorée. Choisissez les paires '
        'dans « Paires disponibles », à partir des 12 paires par défaut.</div>'
        '<div><b>1.</b> Pour <b>ajouter</b> une paire en mode saisie : écrivez son nom dans « Paires à ajouter ».</div>'
        '<div><b>2.</b> Pour <b>choisir</b> librement : cochez <b>Choisir dans la liste</b>, puis cliquez dans '
        '« Paires disponibles » et sélectionnez les paires voulues.</div>'
        '<div><b>3.</b> Pour <b>retirer</b> une paire ajoutée : effacez-la de la zone « Paires à ajouter ».</div>'
        '<div><b>4.</b> Pour revenir au réglage d’origine : cliquez sur <b>Réinitialiser (paires par défaut)</b> '
        '(mode liste) ou videz « Paires à ajouter » (mode saisie).</div>'
        f'<div class="note">Limite : {MAX_PAIRS} paires au total. Les paires ajoutées passent avant les paires par '
        f'défaut : si vous en ajoutez, les dernières paires par défaut sont retirées pour rester à {MAX_PAIRS}.</div>'
        '<div class="note">Votre choix en mode liste ne concerne que vous. Il est conservé d’une visite à l’autre si '
        'votre compte est identifié ; sinon il est perdu au rechargement de la page.</div>'
        '<h4>Direction</h4>'
        '<div class="chips"><span class="chip buy">▲ Buy</span><span class="chip sell">▼ Sell</span>'
        '<span class="chip tp">TPW = TP Wajib (avec EXM / EXT)</span></div>'
        '<h4>Signaux</h4><div class="sigs">' + sigs_html + '</div>'
        + recent_note +
        '<h4>Fraîcheur des données</h4>'
        '<div class="chips"><span class="chip buy">● à jour</span><span class="chip mid">● en retard</span>'
        '<span class="chip sell">● périmé</span></div>'
        '<div class="note">Pastille à côté de chaque paire : âge de la dernière bougie fermée, le pire timeframe compte. Détail au survol.</div>'
        '<h4>Entrée BBMA</h4>'
        '<div class="chips"><span class="chip buy">pullback</span><span class="chip mid">poursuite</span></div>'
        '<div class="note">Pullback : retour vers la zone MA, avec un signal d’entrée sur un TF inférieur. Poursuite : le TF directeur sert d’entrée, prix déjà étendu, grade plafonné à B. H4 ancré sur : ' + bs.H4_SESSION + '.</div>'
        '<div>TF directeur (MOM / EXM / EXT / RE) + TF d’entrée plus bas (RE / MHV / CSAK / EXT) dans le même sens. '
        'Annulé si un TF supérieur contredit, ou si le sens va contre le filtre D1 (clôture vs EMA50).</div>'
        '<h4>Confiance</h4>'
        '<div class="chips"><span class="chip buy">≥ 70 %</span><span class="chip mid">40 – 69 %</span>'
        '<span class="chip low">&lt; 40 %</span></div>'
        '<div class="note">Part des TF scannés dans le même sens que l’entrée.</div>'
        '<h4>RSI 14 sous chaque cellule</h4>'
        '<div class="chips"><span class="chip buy">&lt; 30</span><span class="chip sell">&gt; 70</span>'
        '<span class="chip low">entre les deux</span></div>'
        '<div class="note">Informatif : n’entre ni dans l’entrée BBMA ni dans la confiance.</div>'
        '<h4>Jauge Buy / Sell</h4>'
        '<div class="grad"></div>'
        '<div class="scale"><span>−100 vente forte</span><span>0</span><span>+100 achat fort</span></div>'
        '<div class="note">Pondérée par timeframe, force du signal et tendance D1. Détail au survol.</div>'
        '<h4>Qualité de l’entrée</h4>'
        f'<div class="chips"><span class="chip qa">A ≥ {GRADE_MIN["A"]}</span>'
        f'<span class="chip qb">B {GRADE_MIN["B"]} – {GRADE_MIN["A"] - 1}</span>'
        f'<span class="chip low">C &lt; {GRADE_MIN["B"]}</span></div>'
        f'<div>Score /100 : confluence ({MAX_CONFLUENCE}) · code d’entrée ({MAX_SIGNAL}) · '
        f'RSI à contre-sens ({MAX_RSI}) · anti-poursuite, distance à la MID BB en ATR ({MAX_CHASE}) · '
        f'accord D1 ({MAX_D1}). Détail au survol.</div>'
        '<div class="note">Les bougies non clôturées sont exclues du scan.</div>'
        '</div>'
    )


def guide_html() -> str:
    """Guide de lecture pour la décision d'achat ou de vente. Informatif : ce n'est pas un conseil."""
    return (
        '<div class="bb-legend">'
        '<h4>Guide de décision (informatif, pas un conseil)</h4>'
        '<div>Ce guide résume comment lire le tableau avant de décider un achat ou une vente. '
        'Le scanner ne fixe ni point d’entrée, ni stop, ni objectif : la décision et le risque restent les vôtres.</div>'
        '<h4>1. Y a-t-il une entrée ?</h4>'
        '<div>La colonne <b>Entrée BBMA</b> est remplie : une entrée est détectée. Un « – » signifie qu’il n’y a '
        'aucune décision à prendre sur cette paire.</div>'
        '<h4>2. Quel sens ?</h4>'
        '<div><span class="chip buy">▲ Achat</span> si le signal est haussier, <span class="chip sell">▼ Vente</span> '
        's’il est baissier. Le sens doit rester cohérent avec la <b>Jauge</b> et le filtre <b>D1</b>.</div>'
        '<h4>3. Quelle nature d’entrée ?</h4>'
        '<div><span class="chip buy">pullback</span> : retour vers la zone des moyennes, avec un signal sur un '
        'timeframe inférieur. <span class="chip mid">poursuite</span> : le prix est déjà étendu. Le grade est alors '
        'plafonné à B : prudence.</div>'
        '<h4>4. Quelle qualité ?</h4>'
        f'<div class="chips"><span class="chip qa">A ≥ {GRADE_MIN["A"]}</span>'
        f'<span class="chip qb">B {GRADE_MIN["B"]} – {GRADE_MIN["A"] - 1}</span>'
        f'<span class="chip low">C &lt; {GRADE_MIN["B"]}</span></div>'
        '<div>Grade A : les critères du scanner sont réunis, sans garantie de résultat. Grade B : critères partiels. '
        'Grade C : ce signal ne suffit pas à lui seul à décider.</div>'
        '<h4>5. Les autres colonnes confirment-elles ?</h4>'
        '<div><b>Confiance</b> ≥ 70 % : la plupart des timeframes vont dans le même sens. <b>Jauge</b> : sa couleur '
        'doit rester du côté de l’entrée. <b>RSI</b> sous chaque cellule : il pèse dans le score Qualité ; un RSI '
        'extrême dans le sens du trade le réduit.</div>'
        '<h4>6. Quand renoncer ?</h4>'
        '<div>Pastille de fraîcheur <b>en retard</b> ou <b>périmée</b> ; filtre D1 contraire ; grade C ; signal '
        'contraire sur un timeframe supérieur ; marché fermé ou données absentes.</div>'
        '<h4>7. Avant d’agir</h4>'
        '<div>Vérifiez le signal sur votre propre graphique, puis fixez votre risque et votre taille de position. '
        'Testez la méthode en compte démo. Le module <b>bbma_backtest.py</b> permet de mesurer les grades sur '
        'l’historique.</div>'
        '</div>'
    )


CSS = """<style>
.stApp{background:#131722}
.bb-wrap{overflow-x:auto;background:#1e222d;border:1px solid #2a2e39;border-radius:14px;box-shadow:0 8px 28px rgba(0,0,0,.5);padding:6px}
table.bb{border-collapse:separate;border-spacing:4px;width:100%;min-width:1040px;font:13px/1.2 -apple-system,"Trebuchet MS",Roboto,sans-serif}
table.bb th{color:#787b86;font-weight:600;padding:8px 10px;text-align:center;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
table.bb th.pair{color:#d1d4dc;text-align:left;font-size:13px;letter-spacing:0}
table.bb th.hmtf{color:#2962ff}
table.bb td{padding:8px 10px;text-align:center;border-radius:8px;white-space:nowrap;transition:transform .12s,box-shadow .12s,filter .12s}
table.bb td.none{color:#434651}
table.bb td.none .ghost{font-size:11px;font-weight:600;color:#5c6070;opacity:.8}
table.bb td.buy{background:rgba(38,166,154,.16);color:#26a69a}
table.bb td.sell{background:rgba(239,83,80,.16);color:#ef5350}
table.bb td.strong{box-shadow:inset 3px 0 0 currentColor}
table.bb td:not(.none):hover{transform:translateY(-1px);filter:brightness(1.3);box-shadow:0 4px 14px rgba(0,0,0,.55)}
table.bb tr:hover th.pair{color:#fff}
table.bb .tpw{margin-left:6px;padding:1px 5px;border-radius:4px;font-size:10px;background:#2962ff;color:#fff}
/* RSI sous chaque cellule */
table.bb .rsi{margin-top:3px;font-size:10px;font-weight:600;color:#787b86}
table.bb .rsi.lo{color:#26a69a}
table.bb .rsi.hi{color:#ef5350}
/* Entrée BBMA (synthèse multi-TF) */
table.bb td.synth{font-weight:700;font-size:13px;min-width:110px}
table.bb td.synth.none{background:#2a2e39;color:#787b86}
table.bb td.synth.buy{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.synth.sell{background:#ef5350;color:#fff;box-shadow:0 0 14px rgba(239,83,80,.55)}
table.bb td.synth.chase{background:#BA7517;color:#fff;box-shadow:none}
table.bb td.synth .kind{font-size:10px;font-weight:500;margin-top:2px;opacity:.9}
/* Confiance (part des TF en confluence) */
table.bb td.conf{font-weight:700;font-size:14px;min-width:70px}
table.bb td.conf.high{background:#26a69a;color:#fff}
table.bb td.conf.mid{background:#2962ff;color:#fff}
table.bb td.conf.low,table.bb td.conf.none{background:#2a2e39;color:#787b86}
/* Jauge Buy / Sell */
table.bb td.gauge{min-width:130px;padding:8px 12px}
table.bb td.gauge.none{background:#2a2e39;color:#787b86}
table.bb .gbar{position:relative;height:8px;border-radius:4px;background:linear-gradient(90deg,#ef5350,#2a2e39 50%,#26a69a)}
table.bb .gpt{position:absolute;top:-3px;width:4px;height:14px;border-radius:2px;background:#fff;transform:translateX(-50%)}
table.bb .glab{margin-top:5px;font-size:11px;font-weight:700}
table.bb .glab.buy{color:#26a69a}
table.bb .glab.sell{color:#ef5350}
table.bb .glab.neutral{color:#787b86}
/* Qualité de l'entrée (dernière colonne) */
table.bb td.qual{font-weight:700;font-size:14px;min-width:80px}
table.bb td.qual.A{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.qual.B{background:#2962ff;color:#fff}
table.bb td.qual.C,table.bb td.qual.none{background:#2a2e39;color:#787b86}
table.bb td.qual .qs{font-weight:500;font-size:11px;opacity:.9}
.fdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-left:6px;vertical-align:middle}
/* Colonne Pair figée à gauche sur téléphone */
table.bb th.pair{position:sticky;left:0;z-index:2;width:96px;min-width:96px;white-space:nowrap;background:#1e222d}
/* En-tête figé : reste collé en haut pendant le scroll de la page.
   overflow:visible est nécessaire : un cadre avec overflow-x:auto empêche le sticky de suivre la page.
   Sur écran étroit (< 1150 px), on garde le défilement horizontal (l'en-tête n'est alors plus figé). */
.bb-wrap{overflow:visible}
@media (max-width:1150px){.bb-wrap{overflow-x:auto}}
table.bb tr:first-child th{position:sticky;top:3.75rem;z-index:3;background:#1e222d;box-shadow:0 0 0 4px #1e222d}
table.bb tr:first-child th.pair{left:0;z-index:5}
""" + LEGEND_CSS + """</style>"""


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_scan(pairs: tuple, tfs: tuple, version: int, nonce: int = 0):
    """Scan complet (signaux, tendances D1, RSI, ATR), mis en cache.
    Retourne (res, trends, failed, total, ts, feats) ; feats contient le RSI et la distance ATR par (paire, TF)."""
    return bq.scan_quality(pairs, tfs, trend=True)   # version : fait partie de la clé de cache


def rsi_html(value) -> str:
    """Valeur RSI sous la cellule : couleur uniquement pour les extrêmes."""
    if value is None:
        return '<div class="rsi">–</div>'
    cls = "lo" if value < RSI_LOW else "hi" if value > RSI_HIGH else ""
    return f'<div class="rsi {cls}">{round(value)}</div>'


def cell_html(code, direction, rsi_value, recent=None) -> str:
    """Cellule d'un signal pour un couple (paire, timeframe), avec le RSI du même TF en dessous.
    recent : (code, sens, age) du dernier signal des bougies précédentes, affiché en grisé si la dernière n'en a pas."""
    sub = rsi_html(rsi_value)
    rsi_tip = "" if rsi_value is None else f" · RSI {round(rsi_value)}"
    if not code:
        if recent:
            rc, rd, age = recent
            ghost = f'<div class="ghost">{rc} {ARROW[rd]} −{age}</div>'
            side = "Buy" if rd == "B" else "Sell"
            tip = f"Aucun signal sur la dernière bougie · dernier : {NAMES.get(rc, rc)} · {side}, il y a {age} bougie(s){rsi_tip}"
            return f'<td class="none" title="{tip}">{ghost}{sub}</td>'
        return f'<td class="none" title="Aucun signal{rsi_tip}"><div>–</div>{sub}</td>'
    cls = "buy" if direction == "B" else "sell"
    strong = " strong" if code in STRONG_CODES else ""
    tpw = '<span class="tpw">TPW</span>' if code in TPW_CODES else ""
    side = "Buy" if direction == "B" else "Sell"
    tip = f"{NAMES.get(code, code)} · {side}{rsi_tip}"
    return (f'<td class="{cls}{strong}" title="{tip}">'
            f'<div><b>{code}</b> {ARROW[direction]}{tpw}</div>{sub}</td>')


def synth_cell(tf, code, direction, kind=None) -> str:
    """Cellule « Entrée BBMA » : timeframe, signal, sens et nature de l'entrée (pullback ou poursuite)."""
    if not tf:
        return '<td class="synth s1 none">–</td>'
    cls = "buy" if direction == "B" else "sell"
    side = "Buy" if direction == "B" else "Sell"
    label = {"pullback": "pullback", "poursuite": "poursuite"}.get(kind, "")
    if kind == "poursuite":
        cls += " chase"
    tip = f"Entrée {tf} · {NAMES.get(code, code)} · {side}"
    if label:
        tip += " · " + ("retour vers la zone MA" if kind == "pullback" else "prix déjà étendu, grade plafonné à B")
    tag = f'<div class="kind">{label}</div>' if label else ""
    return f'<td class="synth s1 {cls}" title="{tip}"><b>{tf}</b> {code} {ARROW[direction]}{tag}</td>'


def conf_cell(score) -> str:
    """Cellule « Confiance » : pourcentage coloré (vert ≥ 70 %, bleu 40–69 %, gris < 40 %)."""
    if score is None:
        return '<td class="conf s2 none">–</td>'
    pct = round(score * 100)
    cls = "high" if pct >= 70 else "mid" if pct >= 40 else "low"
    return f'<td class="conf s2 {cls}">{pct}%</td>'


def build_table(pairs, tfs, res, trends, feats, use_filter, show_recent=False) -> str:
    """Tableau unique : Pair, timeframes, Entrée BBMA, Confiance, Jauge, Qualité (dernière colonne)."""
    rows = []
    for p in pairs:
        cells, tf_tds = {}, []
        for t in tfs:
            code, direction = res[(p, t)]
            cells[t] = (code, direction)
            rsi_v = (feats.get((p, t)) or {}).get("rsi")
            recent = (feats.get((p, t)) or {}).get("recent") if show_recent else None
            tf_tds.append(cell_html(code, direction, rsi_v, recent))
        d1 = trends.get(p)
        tf_e, code_e, sens_e = core.synthese(cells, tfs, use_filter, d1)
        score = core.confiance_score(cells, tfs, sens_e) if sens_e else None
        gauge = bg.compute_gauge(cells, tfs, d1)
        q_score, q_grade, q_detail = bq.qualite_entree(
            code_e, sens_e, score, d1, feats.get((p, tf_e)))
        kind = bs.entry_kind(cells, tfs, tf_e, code_e)
        q_score, q_grade = bs.cap_grade(q_score, q_grade, kind)
        tds = tf_tds + [synth_cell(tf_e, code_e, sens_e, kind), conf_cell(score),
                        bg.gauge_cell(gauge), bq.qual_cell(q_score, q_grade, q_detail)]
        fr_status, fr_tip = bf.pair_status(p, feats, tfs)
        rows.append(f'<tr><th class="pair">{html.escape(p)}{bf.dot_html(fr_status, fr_tip)}</th>{"".join(tds)}</tr>')

    head = ('<tr><th class="pair">Pair</th>'
            + "".join(f"<th>{t}</th>" for t in tfs)
            + '<th class="s1 hmtf">Entrée BBMA</th><th class="s2 hmtf">Confiance</th>'
            + '<th class="s4 hmtf">Jauge</th><th class="s3 hmtf">Qualité</th></tr>')
    return f'<div class="bb-wrap"><table class="bb">{head}{"".join(rows)}</table></div>'


DISCLAIMER_CSS = """<style>
.bb-disclaimer{margin-top:14px;padding:12px 16px;background:#1e222d;border:1px solid #2a2e39;border-left:3px solid #BA7517;border-radius:10px;font:12px/1.55 -apple-system,"Trebuchet MS",Roboto,sans-serif;color:#787b86}
.bb-disclaimer b{color:#b2b5be}
</style>"""

DISCLAIMER_HTML = (
    '<div class="bb-disclaimer"><b>Avertissement.</b> Ce scanner est un outil d’analyse technique fourni à titre '
    'purement informatif et éducatif. Il ne constitue ni un conseil en investissement, ni une recommandation '
    'd’achat ou de vente, ni une incitation à trader. Le trading du forex, des métaux et des cryptomonnaies '
    'comporte un risque élevé de perte, pouvant dépasser le capital engagé. Les signaux peuvent être erronés '
    'ou retardés (données différées) et les performances passées ne préjugent pas des résultats futurs. '
    '<b>Vous restez seul responsable de vos décisions et de vos pertes éventuelles</b> ; '
    'consultez un conseiller financier agréé avant d’investir.</div>'
)


def _toggle_legend():
    """Bascule l'affichage de la légende (masquée par défaut)."""
    st.session_state["show_legend"] = not st.session_state.get("show_legend", False)


def _toggle_guide():
    """Bascule l'affichage du guide de décision (masqué par défaut)."""
    st.session_state["show_guide"] = not st.session_state.get("show_guide", False)


def sidebar_settings():
    """Affiche la barre latérale et retourne les paramètres choisis."""
    base = DEFAULT_PAIRS.split()                       # les 12 paires par défaut, toujours scannées
    raw = st.sidebar.text_area(
        f"Paires à ajouter (max {MAX_PAIRS})", "",
        placeholder="Exemple : AUDJPY CHFJPY",
        help="Ces paires s’ajoutent aux 12 paires par défaut. Écrivez-les en les séparant par un espace, "
             "par exemple AUDJPY CHFJPY. Validez avec Ctrl+Entrée ou en cliquant ailleurs. "
             "Cette zone est ignorée quand « Choisir dans la liste » est coché.")
    extra = [p for p in dict.fromkeys(raw.upper().split()) if p not in base]   # sans doublons ni défauts
    if len(extra) > MAX_PAIRS:
        st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires à ajouter.")
        extra = extra[:MAX_PAIRS]
    # Total limité à MAX_PAIRS : les paires ajoutées sont prioritaires, les dernières paires par défaut cèdent leur place.
    kept = base[:MAX_PAIRS - len(extra)]
    if len(kept) < len(base):
        st.sidebar.info(f"Limite de {MAX_PAIRS} paires : {len(base) - len(kept)} paire(s) par défaut retirée(s) "
                        f"({', '.join(base[len(kept):])}).")
    pairs = bp.selector(kept + extra)   # case « Choisir dans la liste » (propre à chaque utilisateur)

    tfs = st.sidebar.multiselect("Timeframes", list(core.TFS), list(core.TFS))
    use_filter = st.sidebar.checkbox("Filtre D1 (EMA50)", True)
    show_recent = st.sidebar.checkbox("Afficher les signaux récents (grisés)", False,
                                      help="Dernier signal des bougies précédentes quand la dernière n'en a pas. Information seulement.")
    auto = st.sidebar.checkbox(f"Scan automatique toutes les {AUTO_REFRESH // 60} min", True)
    if st.sidebar.button("Rafraîchir"):
        st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1
    return pairs, tfs, use_filter, auto, show_recent


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    pairs, tfs, use_filter, auto, show_recent = sidebar_settings()
    if not pairs or not tfs:
        st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
        st.stop()

    @st.fragment(run_every=AUTO_REFRESH if auto else None)   # ne relance que cette partie
    def live():
        with st.spinner("Scan en cours..."):
            res, trends, failed, total, ts, feats = run_scan(
                tuple(pairs), tuple(tfs), SCAN_VERSION, st.session_state.get("refresh_nonce", 0))

        if failed:
            st.warning(
                f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                "Clique sur Rafraîchir dans quelques instants."
            )

        st.markdown(CSS + build_table(pairs, tfs, res, trends, feats, use_filter, show_recent), unsafe_allow_html=True)
        legend_open = st.session_state.get("show_legend", False)
        st.button("Masquer la légende" if legend_open else "Afficher la légende",
                  key="btn_legend", on_click=_toggle_legend)
        if legend_open:
            st.markdown(legend_html(show_recent), unsafe_allow_html=True)
        guide_open = st.session_state.get("show_guide", False)
        st.button("Masquer le guide de décision" if guide_open else "Afficher le guide de décision",
                  key="btn_guide", on_click=_toggle_guide)
        if guide_open:
            st.markdown(guide_html(), unsafe_allow_html=True)
        suffix = " · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"
        st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC{suffix}")
        st.markdown(DISCLAIMER_CSS + DISCLAIMER_HTML, unsafe_allow_html=True)

    live()


if __name__ == "__main__":
    main()
