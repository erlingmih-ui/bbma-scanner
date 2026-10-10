"""Interface Streamlit du scanner BBMA OA MTF.

La logique métier (téléchargement, détection, synthèse, confiance) est dans ``bbma_core``.
Ce module ne gère que l'affichage et les paramètres de la barre latérale.

Lancement : streamlit run app.py
"""
import html
import logging

import streamlit as st

import bbma_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

APP_TITLE = "BBMA OA MTF Scanner"
DEFAULT_PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD"
MAX_PAIRS = 12
CACHE_TTL = 240          # secondes ; inférieur à AUTO_REFRESH pour qu'un vrai scan ait lieu à chaque cycle
AUTO_REFRESH = 300       # secondes entre deux scans automatiques

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
    "Entrée BBMA : TF directeur (MOM/EXM/EXT/RE) + TF d'entrée plus bas (RE/MHV/CSAK/EXT) dans le même sens, "
    "annulé si un TF supérieur contredit ou si le sens va contre le filtre D1 (clôture vs EMA50). "
    "Confiance : part des TF scannés dans le même sens que l'entrée (vert ≥ 70% · bleu 40–69% · gris < 40%). "
    "Les bougies non clôturées sont exclues du scan."
)

CSS = """<style>
.stApp{background:#131722}
.bb-wrap{overflow-x:auto;background:#1e222d;border:1px solid #2a2e39;border-radius:14px;box-shadow:0 8px 28px rgba(0,0,0,.5);padding:6px}
table.bb{border-collapse:separate;border-spacing:4px;width:100%;min-width:780px;font:13px/1.2 -apple-system,"Trebuchet MS",Roboto,sans-serif}
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
/* Entrée BBMA (synthèse multi-TF) */
table.bb td.synth{font-weight:700;font-size:13px;min-width:110px}
table.bb td.synth.none{background:#2a2e39;color:#787b86}
table.bb td.synth.buy{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.synth.sell{background:#ef5350;color:#fff;box-shadow:0 0 14px rgba(239,83,80,.55)}
/* Confiance (part des TF en confluence) */
table.bb td.conf{font-weight:700;font-size:14px;min-width:70px}
table.bb td.conf.high{background:#26a69a;color:#fff}
table.bb td.conf.mid{background:#2962ff;color:#fff}
table.bb td.conf.low,table.bb td.conf.none{background:#2a2e39;color:#787b86}
</style>"""


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_scan(pairs: tuple, tfs: tuple):
    """Scan complet, mis en cache. Retourne (res, trends, failed, total, timestamp UTC)."""
    return core.scan(pairs, tfs, trend=True)


def cell_html(code, direction) -> str:
    """Cellule d'un signal pour un couple (paire, timeframe)."""
    if not code:
        return '<td class="none">–</td>'
    cls = "buy" if direction == "B" else "sell"
    strong = " strong" if code in STRONG_CODES else ""
    tpw = '<span class="tpw">TPW</span>' if code in TPW_CODES else ""
    side = "Buy" if direction == "B" else "Sell"
    tip = f"{NAMES.get(code, code)} · {side}"
    return f'<td class="{cls}{strong}" title="{tip}"><b>{code}</b> {ARROW[direction]}{tpw}</td>'


def synth_cell(tf, code, direction) -> str:
    """Cellule « Entrée BBMA » : timeframe d'entrée, signal et sens."""
    if not tf:
        return '<td class="synth none">–</td>'
    cls = "buy" if direction == "B" else "sell"
    side = "Buy" if direction == "B" else "Sell"
    tip = f"Entrée {tf} · {NAMES.get(code, code)} · {side}"
    return f'<td class="synth {cls}" title="{tip}"><b>{tf}</b> {code} {ARROW[direction]}</td>'


def conf_cell(score) -> str:
    """Cellule « Confiance » : pourcentage coloré (vert ≥ 70 %, bleu 40–69 %, gris < 40 %)."""
    if score is None:
        return '<td class="conf none">–</td>'
    pct = round(score * 100)
    cls = "high" if pct >= 70 else "mid" if pct >= 40 else "low"
    return f'<td class="conf {cls}">{pct}%</td>'


def build_table(pairs, tfs, res, trends, use_filter) -> str:
    """Construit le HTML complet du tableau de bord."""
    rows = []
    for p in pairs:
        cells, tds = {}, []
        for t in tfs:
            code, direction = res[(p, t)]
            cells[t] = (code, direction)
            tds.append(cell_html(code, direction))
        tf_e, code_e, sens_e = core.synthese(cells, tfs, use_filter, trends.get(p))
        score = core.confiance_score(cells, tfs, sens_e) if sens_e else None
        tds.append(synth_cell(tf_e, code_e, sens_e))
        tds.append(conf_cell(score))
        rows.append(f'<tr><th class="pair">{html.escape(p)}</th>{"".join(tds)}</tr>')

    head = (
        '<tr><th class="pair">Pair</th>'
        + "".join(f"<th>{t}</th>" for t in tfs)
        + '<th class="hmtf">Entrée BBMA</th><th class="hmtf">Confiance</th></tr>'
    )
    return f'<div class="bb-wrap"><table class="bb">{head}{"".join(rows)}</table></div>'


def sidebar_settings():
    """Affiche la barre latérale et retourne les paramètres choisis."""
    raw = st.sidebar.text_area(f"Paires (max {MAX_PAIRS})", DEFAULT_PAIRS)
    pairs = list(dict.fromkeys(raw.upper().split()))   # dédoublonnage en conservant l'ordre
    if len(pairs) > MAX_PAIRS:
        st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires.")
        pairs = pairs[:MAX_PAIRS]

    tfs = st.sidebar.multiselect("Timeframes", list(core.TFS), list(core.TFS))
    use_filter = st.sidebar.checkbox("Filtre D1 (EMA50)", True)
    auto = st.sidebar.checkbox(f"Scan automatique toutes les {AUTO_REFRESH // 60} min", True)
    if st.sidebar.button("Rafraîchir"):
        st.cache_data.clear()
    return pairs, tfs, use_filter, auto


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    pairs, tfs, use_filter, auto = sidebar_settings()
    if not pairs or not tfs:
        st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
        st.stop()

    @st.fragment(run_every=AUTO_REFRESH if auto else None)   # ne relance que cette partie
    def live():
        with st.spinner("Scan en cours..."):
            res, trends, failed, total, ts = run_scan(tuple(pairs), tuple(tfs))

        if failed:
            st.warning(
                f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                "Clique sur Rafraîchir dans quelques instants."
            )

        st.markdown(CSS + build_table(pairs, tfs, res, trends, use_filter), unsafe_allow_html=True)
        st.caption(LEGEND)
        suffix = " · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"
        st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC{suffix}")

    live()


if __name__ == "__main__":
    main()
