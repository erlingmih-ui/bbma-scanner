import html
import logging

import streamlit as st

from bbma_core import TFS, confiance_score, scan, synthese

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

st.set_page_config(page_title="BBMA OA MTF Scanner", layout="wide")
st.title("BBMA OA MTF Scanner")

DEFAULT_PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD"
MAX_PAIRS = 12
REFRESH_S = 300                      # scan automatique toutes les 5 min
SLOW_TFS = {"MN", "W1", "D1"}        # évoluent au plus une fois par jour : cache 1 h
SLOW_TTL, FAST_TTL = 3600, 240       # FAST < 5 min : chaque cycle automatique refait un vrai scan


class _Incomplete(Exception):
    """Scan partiel (téléchargements en échec). Levée pour que Streamlit ne mette pas en cache
    des données manquantes : le cycle suivant réessaiera."""
    def __init__(self, payload):
        super().__init__("scan partiel")
        self.payload = payload


def _checked(pairs, tfs, trend):
    out = scan(pairs, tfs, trend=trend)
    if out[2]:                       # failed > 0
        raise _Incomplete(out)
    return out


@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def scan_slow(pairs, tfs):
    """TF lents + filtre D1 (le filtre n'est calculé que ici)."""
    return _checked(pairs, tfs, trend=True)


@st.cache_data(ttl=FAST_TTL, show_spinner=False)
def scan_fast(pairs, tfs):
    """TF rapides uniquement : pas de téléchargement D1 supplémentaire."""
    return _checked(pairs, tfs, trend=False)


def _unwrap(fn, pairs, tfs):
    try:
        return fn(pairs, tfs)
    except _Incomplete as e:
        return e.payload


def run_all(pairs, tfs):
    """Combine les deux caches. Retourne (res, trends, failed, total, ts) comme avant."""
    slow_tfs = tuple(t for t in tfs if t in SLOW_TFS)
    fast_tfs = tuple(t for t in tfs if t not in SLOW_TFS)
    s_res, trends, s_failed, s_total, _ = _unwrap(scan_slow, pairs, slow_tfs)
    f_res, _, f_failed, f_total, ts = _unwrap(scan_fast, pairs, fast_tfs)
    return {**s_res, **f_res}, trends, s_failed + f_failed, s_total + f_total, ts


pairs = list(dict.fromkeys(st.sidebar.text_area(f"Paires (max {MAX_PAIRS})", DEFAULT_PAIRS).upper().split()))
if len(pairs) > MAX_PAIRS:
    st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires.")
    pairs = pairs[:MAX_PAIRS]
tfs = st.sidebar.multiselect("Timeframes", list(TFS), list(TFS))
use_filter = st.sidebar.checkbox("Filtre D1 (EMA50)", True)
auto = st.sidebar.checkbox("Scan automatique toutes les 5 min", True)
if st.sidebar.button("Rafraîchir"):
    st.cache_data.clear()
if not pairs or not tfs:
    st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
    st.stop()

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

ARROW = {"B": "▲", "S": "▼"}

NAMES = {"MOM": "Momentum", "EXM": "Extreme Magic (MA10)", "EXT": "Extreme (MA5)", "MHV": "Market Hilang Volume",
         "CSAK": "Candle Arah Kukuh", "RE": "Reentry"}


def cell_html(code, d):
    if not code: return '<td class="none">–</td>'
    strong = " strong" if code in ("MOM", "EXM", "EXT") else ""
    tpw = '<span class="tpw">TPW</span>' if code in ("EXM", "EXT") else ""
    tip = NAMES.get(code, code) + (" · Buy" if d == "B" else " · Sell")
    return (f'<td class="{"buy" if d == "B" else "sell"}{strong}" title="{tip}">'
            f'<b>{code}</b> {ARROW[d]}{tpw}</td>')


def synth_cell(tf, code, d):
    """Cellule 'Entrée BBMA' : TF d'entrée, code du signal et sens."""
    if not tf: return '<td class="synth none">–</td>'
    cls = "buy" if d == "B" else "sell"
    tip = f"Entrée {tf} · {NAMES.get(code, code)}" + (" · Buy" if d == "B" else " · Sell")
    return f'<td class="synth {cls}" title="{tip}"><b>{tf}</b> {code} {ARROW[d]}</td>'


def conf_cell(score):
    """Cellule 'Confiance' : pourcentage avec code couleur (vert >= 70%, bleu 40-69%, gris < 40%)."""
    if score is None: return '<td class="conf none">–</td>'
    pct = round(score * 100)
    cls = "high" if pct >= 70 else "mid" if pct >= 40 else "low"
    return f'<td class="conf {cls}">{pct}%</td>'


@st.fragment(run_every=REFRESH_S if auto else None)   # relance seulement cette partie
def live():
    with st.spinner("Scan en cours..."):
        res, trends, failed, total, ts = run_all(tuple(pairs), tuple(tfs))
    if failed:
        st.warning(f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                   "Clique sur Rafraîchir dans quelques instants.")

    body = []
    for p in pairs:
        cells, tds = {}, []
        for t in tfs:
            code, d = res[(p, t)]
            cells[t] = (code, d)
            tds.append(cell_html(code, d))
        # Entrée BBMA (synthèse multi-TF, filtre D1 inclus) puis Confiance
        tf_e, code_e, sens_e = synthese(cells, tfs, use_filter, trends.get(p))
        score = confiance_score(cells, tfs, sens_e) if sens_e else None
        tds.append(synth_cell(tf_e, code_e, sens_e))
        tds.append(conf_cell(score))
        body.append(f'<tr><th class="pair">{html.escape(p)}</th>{"".join(tds)}</tr>')
    head = ('<tr><th class="pair">Pair</th>' + "".join(f"<th>{t}</th>" for t in tfs)
            + '<th class="hmtf">Entrée BBMA</th><th class="hmtf">Confiance</th></tr>')
    st.markdown(CSS + f'<div class="bb-wrap"><table class="bb">{head}{"".join(body)}</table></div>',
                unsafe_allow_html=True)
    st.caption("▲ Buy · ▼ Sell · MOM=Momentum · EXM=Extreme Magic (MA10) · EXT=Extreme (MA5) · TPW=TP Wajib (avec EXM/EXT) · MHV · CSAK=Candle Arah · RE=Reentry (strict : retour depuis l'extérieur de la zone MA5/MA10). "
               "Survole une cellule pour voir le nom du signal. "
               "Entrée BBMA : TF directeur (MOM/EXM/EXT/RE) + TF d'entrée plus bas (RE/MHV/CSAK/EXT) dans le même sens, "
               "annulé si un TF supérieur contredit ou si le sens va contre le filtre D1 (clôture D1 validée vs EMA50). "
               "Confiance : part des TF scannés dans le même sens que l'entrée (vert ≥ 70% · bleu 40–69% · gris < 40%). "
               "Les bougies non clôturées sont exclues du scan.")
    st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC" + (" · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"))


live()
