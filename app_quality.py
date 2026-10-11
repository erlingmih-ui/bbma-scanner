"""Interface Streamlit avec la colonne « Qualité » en dernière position.

Réutilise app.py (CSS, cellules, barre latérale, légende) sans le modifier.
Lancement : streamlit run app_quality.py
"""
import html

import streamlit as st

import app as ui                 # CSS, cellules, barre latérale et légende existants
import bbma_core as core         # synthèse et confiance, inchangées
import bbma_quality as bq
from bbma_quality_codes import GRADE_MIN, MAX_CHASE, MAX_CONFLUENCE, MAX_D1, MAX_RSI, MAX_SIGNAL

QUAL_CSS = """<style>
table.bb{min-width:900px}
table.bb td.qual{font-weight:700;font-size:14px;min-width:80px}
table.bb td.qual.A{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.qual.B{background:#2962ff;color:#fff}
table.bb td.qual.C,table.bb td.qual.none{background:#2a2e39;color:#787b86}
table.bb td.qual .qs{font-weight:500;font-size:11px;opacity:.9}
</style>"""

QUAL_LEGEND = (
    f" Qualité (A ≥ {GRADE_MIN['A']} · B {GRADE_MIN['B']}–{GRADE_MIN['A'] - 1} · C < {GRADE_MIN['B']}) : "
    f"score /100 = confluence ({MAX_CONFLUENCE}) + code d'entrée ({MAX_SIGNAL}) + RSI à contre-sens ({MAX_RSI}) "
    f"+ anti-poursuite, distance à la MID BB en ATR ({MAX_CHASE}) + accord D1 ({MAX_D1}). "
    "Survole la cellule pour le détail."
)


@st.cache_data(ttl=ui.CACHE_TTL, show_spinner=False)
def run_scan_quality(pairs: tuple, tfs: tuple):
    """Scan avec caractéristiques (RSI, distance ATR), mis en cache."""
    return bq.scan_quality(pairs, tfs, trend=True)


def build_table_quality(pairs, tfs, res, trends, feats, use_filter) -> str:
    """Tableau : Pair, timeframes, Entrée BBMA, Confiance, Qualité (dernière colonne)."""
    rows = []
    for p in pairs:
        cells, tf_tds = {}, []
        for t in tfs:
            code, direction = res[(p, t)]
            cells[t] = (code, direction)
            rsi_v = (feats.get((p, t)) or {}).get("rsi")
            tf_tds.append(ui.cell_html(code, direction, rsi_v))
        tf_e, code_e, sens_e = core.synthese(cells, tfs, use_filter, trends.get(p))
        score = core.confiance_score(cells, tfs, sens_e) if sens_e else None
        q_score, q_grade, q_detail = bq.qualite_entree(
            code_e, sens_e, score, trends.get(p), feats.get((p, tf_e)))
        tds = tf_tds + [ui.synth_cell(tf_e, code_e, sens_e), ui.conf_cell(score),
                        bq.qual_cell(q_score, q_grade, q_detail)]
        rows.append(f'<tr><th class="pair">{html.escape(p)}</th>{"".join(tds)}</tr>')

    head = ('<tr><th class="pair">Pair</th>'
            + "".join(f"<th>{t}</th>" for t in tfs)
            + '<th class="s1 hmtf">Entrée BBMA</th><th class="s2 hmtf">Confiance</th>'
            + '<th class="s3 hmtf">Qualité</th></tr>')
    return f'<div class="bb-wrap"><table class="bb">{head}{"".join(rows)}</table></div>'


def main():
    st.set_page_config(page_title=ui.APP_TITLE + " · Qualité", layout="wide")
    st.title(ui.APP_TITLE + " · Qualité")

    pairs, tfs, use_filter, auto = ui.sidebar_settings()
    if not pairs or not tfs:
        st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
        st.stop()

    @st.fragment(run_every=ui.AUTO_REFRESH if auto else None)
    def live():
        with st.spinner("Scan en cours..."):
            res, trends, failed, total, ts, feats = run_scan_quality(tuple(pairs), tuple(tfs))

        if failed:
            st.warning(
                f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                "Clique sur Rafraîchir dans quelques instants."
            )

        st.markdown(ui.CSS + QUAL_CSS + build_table_quality(pairs, tfs, res, trends, feats, use_filter),
                    unsafe_allow_html=True)
        st.caption(ui.LEGEND + QUAL_LEGEND)
        suffix = " · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"
        st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC{suffix}")

    live()


if __name__ == "__main__":
    main()
