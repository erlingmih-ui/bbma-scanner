"""Interface Streamlit du scanner BBMA OA MTF.

La détection est dans ``bbma_core`` ; les motifs REM / RRE / REE et la progression sont dans ``bbma_progress``.
Ce module ne gère que l'affichage et les paramètres de la barre latérale.

Lancement : streamlit run app.py
"""
import html
import logging
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

import bbma_core as core
import bbma_progress as prog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

APP_TITLE = "BBMA OA MTF Scanner"
DEFAULT_PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD"
MAX_PAIRS = 12
CACHE_TTL = 240          # secondes ; inférieur à AUTO_REFRESH pour qu'un vrai scan ait lieu à chaque cycle
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
LEGEND = (
    "▲ Buy · ▼ Sell · MOM=Momentum · EXM=Extreme Magic (MA10) · EXT=Extreme (MA5) · "
    "TPW=TP Wajib (avec EXM/EXT) · MHV · CSAK=Candle Arah · "
    "RE=Reentry (strict : retour depuis l'extérieur de la zone MA5/MA10). "
    "Survole une cellule pour voir le nom du signal. "
    "Entrée : seul un motif complet et non expiré (REM, RRE ou REE) est une entrée. "
    "Progression : R = RE sur D1, E = EXT sur H4, M = MHV sur H1, et R sur H4 pour RRE. "
    f"Pointillés = en cours, gris = expiré. Fenêtre de validité : {prog.MAX_AGE.days} jours depuis le D1 RE. "
    "Un signal de structure contraire remet la progression à zéro. "
    "Confiance : part des TF scannés dans le même sens que l'entrée (vert ≥ 70% · bleu 40–69% · gris < 40%). "
    "RSI 14 (Wilder) sous chaque cellule : vert < 30, rouge > 70, gris entre les deux ; "
    "il est purement informatif et n'entre ni dans l'entrée ni dans la confiance. "
    "Les bougies non clôturées sont exclues du scan."
)

CSS = """<style>
.stApp{background:#131722}
.bb-wrap{overflow-x:auto;background:#1e222d;border:1px solid #2a2e39;border-radius:14px;box-shadow:0 8px 28px rgba(0,0,0,.5);padding:6px}
table.bb{border-collapse:separate;border-spacing:4px;width:100%;min-width:900px;font:13px/1.2 -apple-system,"Trebuchet MS",Roboto,sans-serif}
table.bb th{color:#787b86;font-weight:600;padding:8px 10px;text-align:center;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
table.bb th.pair{color:#d1d4dc;text-align:left;font-size:13px;letter-spacing:0}
table.bb th.hmtf{color:#2962ff}
table.bb td{padding:8px 10px;text-align:center;border-radius:8px;white-space:nowrap;transition:transform .12s,box-shadow .12s,filter .12s}
table.bb td.none{color:#434651}
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
/* Entrée (motif complet non expiré) */
table.bb td.synth{font-weight:700;font-size:13px;min-width:110px}
table.bb td.synth small{font-weight:400;font-size:11px;color:#787b86}
table.bb td.synth.none{background:#2a2e39;color:#787b86}
table.bb td.synth.buy{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.synth.sell{background:#ef5350;color:#fff;box-shadow:0 0 14px rgba(239,83,80,.55)}
/* Progression en cours : pointillés. Expiré : gris */
table.bb td.synth.partial{background:transparent;outline:1px dashed currentColor;outline-offset:-2px;box-shadow:none}
table.bb td.synth.partial.buy{color:#26a69a}
table.bb td.synth.partial.sell{color:#ef5350}
/* Confiance (part des TF en confluence) */
table.bb td.conf{font-weight:700;font-size:14px;min-width:70px}
table.bb td.conf.high{background:#26a69a;color:#fff}
table.bb td.conf.mid{background:#2962ff;color:#fff}
table.bb td.conf.low,table.bb td.conf.none{background:#2a2e39;color:#787b86}
/* Colonne Pair figée à gauche sur téléphone */
table.bb th.pair{position:sticky;left:0;z-index:2;width:84px;min-width:84px;background:#1e222d}
</style>"""


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_scan(pairs: tuple, tfs: tuple):
    """Signaux par TF et RSI, mis en cache. Retourne (res, trends, failed, total, ts, rsi)."""
    return core.scan_with_rsi(pairs, tfs, trend=False)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_progress(pairs: tuple):
    """Progression R / E / M par paire, depuis le cache disque. Indépendant des TF sélectionnés."""
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(pairs, ex.map(prog.progress, pairs)))


def rsi_html(value) -> str:
    """Valeur RSI sous la cellule : couleur uniquement pour les extrêmes."""
    if value is None:
        return '<div class="rsi">–</div>'
    cls = "lo" if value < RSI_LOW else "hi" if value > RSI_HIGH else ""
    return f'<div class="rsi {cls}">{round(value)}</div>'


def cell_html(code, direction, rsi_value) -> str:
    """Cellule d'un signal pour un couple (paire, timeframe), avec le RSI du même TF en dessous."""
    sub = rsi_html(rsi_value)
    rsi_tip = "" if rsi_value is None else f" · RSI {round(rsi_value)}"
    if not code:
        return f'<td class="none" title="Aucun signal{rsi_tip}"><div>–</div>{sub}</td>'
    cls = "buy" if direction == "B" else "sell"
    strong = " strong" if code in STRONG_CODES else ""
    tpw = '<span class="tpw">TPW</span>' if code in TPW_CODES else ""
    side = "Buy" if direction == "B" else "Sell"
    tip = f"{NAMES.get(code, code)} · {side}{rsi_tip}"
    return (f'<td class="{cls}{strong}" title="{tip}">'
            f'<div><b>{code}</b> {ARROW[direction]}{tpw}</div>{sub}</td>')


def entry_cell(code, direction) -> str:
    """Cellule « Entrée » : motif complet et non expiré uniquement (REM, RRE, REE)."""
    if not code:
        return '<td class="synth none" title="Aucun motif complet non expiré">–</td>'
    cls = "buy" if direction == "B" else "sell"
    side = "Buy" if direction == "B" else "Sell"
    tip = f"Motif complet {code} · {side} · fenêtre {prog.MAX_AGE.days} j depuis le D1 RE"
    return f'<td class="synth {cls}" title="{tip}"><b>{code}</b> {ARROW[direction]}</td>'


def prog_cell(pr) -> str:
    """Cellule « Progression » : lettres validées (R E M), pointillés si en cours, gris si expiré."""
    if not pr:
        return '<td class="none">–</td>'
    d = pr["sens"]
    cls = "buy" if d == "B" else "sell"
    letters = " ".join(pr["letters"])
    if pr["expired"]:
        tip = f"Setup expiré · démarré {pr['start']:%d/%m %H:%M} UTC"
        return f'<td class="synth none" title="{tip}"><b>{letters}</b> {ARROW[d]}<br><small>expiré</small></td>'
    done = pr["complete"]
    style = "synth" if done else "synth partial"
    label = done or f"{pr['steps']}/3 en cours"
    tip = f"{label} · démarré {pr['start']:%d/%m %H:%M} UTC · dernière étape {pr['since']:%d/%m %H:%M} UTC"
    return (f'<td class="{style} {cls}" title="{tip}"><b>{letters}</b> {ARROW[d]}'
            f'<br><small>{label}</small></td>')


def conf_cell(score) -> str:
    """Cellule « Confiance » : pourcentage coloré (vert ≥ 70 %, bleu 40–69 %, gris < 40 %)."""
    if score is None:
        return '<td class="conf s2 none">–</td>'
    pct = round(score * 100)
    cls = "high" if pct >= 70 else "mid" if pct >= 40 else "low"
    return f'<td class="conf s2 {cls}">{pct}%</td>'


def build_table(pairs, tfs, res, rsi_vals, progs) -> str:
    """Tableau unique : Pair, timeframes, puis Entrée, Confiance et Progression."""
    rows = []
    for p in pairs:
        cells, tf_tds = {}, []
        for t in tfs:
            code, direction = res[(p, t)]
            cells[t] = (code, direction)
            tf_tds.append(cell_html(code, direction, rsi_vals.get((p, t))))
        pr = progs.get(p)
        code_e, sens_e = prog.entree(pr)                 # source unique : le moteur de progression
        score = core.confiance_score(cells, tfs, sens_e) if sens_e else None
        tds = tf_tds + [entry_cell(code_e, sens_e), conf_cell(score), prog_cell(pr)]
        rows.append(f'<tr><th class="pair">{html.escape(p)}</th>{"".join(tds)}</tr>')

    head = ('<tr><th class="pair">Pair</th>'
            + "".join(f"<th>{t}</th>" for t in tfs)
            + '<th class="s1 hmtf">Entrée</th><th class="s2 hmtf">Confiance</th>'
            + '<th class="s3 hmtf">Progression</th></tr>')
    return f'<div class="bb-wrap"><table class="bb">{head}{"".join(rows)}</table></div>'


def sidebar_settings():
    """Affiche la barre latérale et retourne les paramètres choisis."""
    raw = st.sidebar.text_area(f"Paires (max {MAX_PAIRS})", DEFAULT_PAIRS)
    pairs = list(dict.fromkeys(raw.upper().split()))   # dédoublonnage en conservant l'ordre
    if len(pairs) > MAX_PAIRS:
        st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires.")
        pairs = pairs[:MAX_PAIRS]

    tfs = st.sidebar.multiselect("Timeframes", list(core.TFS), list(core.TFS))
    auto = st.sidebar.checkbox(f"Scan automatique toutes les {AUTO_REFRESH // 60} min", True)
    if st.sidebar.button("Rafraîchir"):
        st.cache_data.clear()
    return pairs, tfs, auto


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    pairs, tfs, auto = sidebar_settings()
    if not pairs or not tfs:
        st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
        st.stop()

    @st.fragment(run_every=AUTO_REFRESH if auto else None)   # ne relance que cette partie
    def live():
        with st.spinner("Scan en cours..."):
            res, _trends, failed, total, ts, rsi_vals = run_scan(tuple(pairs), tuple(tfs))
            progs = run_progress(tuple(pairs))

        if failed:
            st.warning(
                f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                "Clique sur Rafraîchir dans quelques instants."
            )

        st.markdown(CSS + build_table(pairs, tfs, res, rsi_vals, progs), unsafe_allow_html=True)
        st.caption(LEGEND)
        suffix = " · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"
        st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC{suffix}")

    live()


if __name__ == "__main__":
    main()
