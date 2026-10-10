import numpy as np, pandas as pd, streamlit as st, yfinance as yf
import html
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import bbma_progress as prog

st.set_page_config(page_title="BBMA OA MTF Scanner", layout="wide")
st.title("BBMA OA MTF Scanner")

DEFAULT_PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD"
MAX_PAIRS = 12
# Symboles yfinance particuliers (le reste = PAIRE=X)
SYMBOLS = {"XAUUSD": "GC=F", "BTCUSD": "BTC-USD"}
def sym(pair): return SYMBOLS.get(pair, pair + "=X")
# TF -> (yfinance interval, period, resample)
TFS = {"MN": ("1mo", "10y", None), "W1": ("1wk", "5y", None), "D1": ("1d", "2y", None),
       "H4": ("1h", "180d", "4h"), "H1": ("1h", "60d", None), "M15": ("15m", "30d", None), "M5": ("5m", "30d", None)}
PRIORITY = ["MOM", "EXM", "EXT", "MHV", "CSAK", "RE"]   # priorité de détection (cycle BBMA)

# Codes d'entrée MTF : R = Reentry, E = Extreme, M = MHV. Ordre des TF : D1, H4, H1.
PATTERNS = {
    "REM": [("D1", "RE"), ("H4", "EXT"), ("H1", "MHV")],   # Reentry - Extreme - MHV
    "RRE": [("D1", "RE"), ("H4", "RE"), ("H1", "EXT")],    # Reentry - Reentry - Extreme
    "REE": [("D1", "RE"), ("H4", "EXT"), ("H1", "EXT")],   # Reentry - Extreme - Extreme
}
ORDER = ["REM", "REE", "RRE"]
NAMES = {
    "MOM": "Momentum", "EXM": "Extreme Magic (MA10)", "EXT": "Extreme (MA5)", "MHV": "Market Hilang Volume",
    "CSAK": "Candle Arah Kukuh", "RE": "Reentry",
    "REM": "Reentry D1 · Extreme H4 · MHV H1",
    "RRE": "Reentry D1 · Reentry H4 · Extreme H1",
    "REE": "Reentry D1 · Extreme H4 · Extreme H1",
}

def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)

def detect(df, lb=15):
    """Signal de la dernière bougie : (code, sens) avec sens 'B' (buy) ou 'S' (sell).
    Priorité : MOM > EXM (Extreme Magic, MA10) > EXT (MA5) > MHV > CSAK (Candle Arah) > RE."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    top, low = mid + 2 * sd, mid - 2 * sd
    h5, h10, l5, l10 = wma(h, 5), wma(h, 10), wma(l, 5), wma(l, 10)
    rec = lambda s, n: s.fillna(False).astype(bool).rolling(n, min_periods=1).max().astype(bool)

    mom_b, mom_s = c > top, c < low                                             # MOM (clôture hors BB)
    exm_s, exm_b = h10 > top, l10 < low                                         # Extreme Magic (MA10)
    ext_s, ext_b = h5 > top, l5 < low                                           # Extreme (MA5)
    any_s, any_b = exm_s | ext_s, exm_b | ext_b
    mhv_s = rec(any_s.shift(1), lb) & (h >= top) & (c < top)                    # MHV (fenêtre 15 bougies)
    mhv_b = rec(any_b.shift(1), lb) & (l <= low) & (c > low)
    arah_s = (c < mid) & (c < l5) & (c < l10)                                   # CSAK / Candle Arah
    arah_b = (c > mid) & (c > h5) & (c > h10)
    # tendance mémorisée : dernier MOM / Arah
    trend = pd.Series(np.where(mom_b | arah_b, 1, np.where(mom_s | arah_s, -1, np.nan)), index=c.index).ffill()
    # Zones MA5/MA10
    zone_buy_hi, zone_buy_lo = np.maximum(l5, l10), np.minimum(l5, l10)
    zone_sell_hi, zone_sell_lo = np.maximum(h5, h10), np.minimum(h5, h10)

    # RE strict : bougie précédente HORS zone, bougie actuelle DANS la zone
    re_b = ((trend == 1)
            & (c.shift(1) < zone_buy_lo.shift(1))
            & (c >= zone_buy_lo) & (c <= zone_buy_hi)
            & (l <= zone_buy_hi))
    re_s = ((trend == -1)
            & (c.shift(1) > zone_sell_hi.shift(1))
            & (c >= zone_sell_lo) & (c <= zone_sell_hi)
            & (h >= zone_sell_lo))

    sig = {"MOM": (mom_b, mom_s), "EXM": (exm_b, exm_s), "EXT": (ext_b, ext_s),
           "MHV": (mhv_b, mhv_s), "CSAK": (arah_b, arah_s), "RE": (re_b, re_s)}
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[-1]: return code, "B"
        if s.iloc[-1]: return code, "S"
    return None, None

def mtf_code(cells):
    """cells : {tf: (code, sens)} -> (code MTF, sens) ou (None, None).
    Même sens sur les trois TF requis. EXM est traité comme EXT. Aucun repli, aucun filtre externe."""
    base = lambda x: "EXT" if x == "EXM" else x
    for code in ORDER:
        conds = PATTERNS[code]
        if all(base(cells.get(tf, (None, None))[0]) == sig for tf, sig in conds):
            dirs = {cells[tf][1] for tf, _ in conds}
            if len(dirs) == 1:
                return code, dirs.pop()
    return None, None

def mtf_partial(cells):
    """Code en cours : les 2 premiers TF du motif sont alignés (même sens), le 3e manque encore.
    Retourne {codes, sens, tf, attendu} ou None. Affiché seulement si mtf_code ne trouve rien.
    Un 3e TF présent dans le sens opposé annule le code en cours."""
    base = lambda x: "EXT" if x == "EXM" else x
    found, sens_set, tf_miss, expect = [], set(), None, set()
    for code in ORDER:
        conds = PATTERNS[code]
        (tf1, s1), (tf2, s2), (tf3, s3) = conds
        if not all(base(cells.get(tf, (None, None))[0]) == sig for tf, sig in conds[:2]):
            continue
        dirs = {cells[tf][1] for tf, _ in conds[:2]}
        if len(dirs) != 1:
            continue
        sens = dirs.pop()
        c3, d3 = cells.get(tf3, (None, None))
        if base(c3) == s3 and d3 != sens:      # 3e TF contraire : pas de code en cours
            continue
        found.append(code)
        sens_set.add(sens)
        tf_miss = tf3
        expect.add(s3)
    if not found or len(sens_set) != 1:
        return None
    return {"codes": found, "sens": sens_set.pop(), "tf": tf_miss, "attendu": " / ".join(sorted(expect))}

def confiance_score(cells, tfs, sens_ref):
    """Part des TF scannés ayant un signal dans le même sens que l'entrée (0 à 1). Indicateur, pas une règle."""
    if not tfs or sens_ref is None:
        return None
    n = sum(1 for t in tfs if cells.get(t, (None, None))[1] == sens_ref)
    return n / len(tfs)

def _history(pair, interval, period):
    """Télécharge les bougies (2 essais). Retourne un DataFrame OHLC propre, ou None."""
    for _ in range(2):
        try:
            df = yf.Ticker(sym(pair)).history(period=period, interval=interval)
            if df is not None and len(df):
                df = df[["Open", "High", "Low", "Close"]].dropna()
                if len(df): return df
        except Exception:
            pass
    return None

def _scan_one(pair, tf):
    """((code, sens), ok) pour une paire sur un timeframe."""
    interval, period, rs = TFS[tf]
    df = _history(pair, interval, period)
    if df is None: return (None, None), False
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    # Exclure la dernière bougie en cours de formation (critique sur M5/M15)
    df = df.iloc[:-1]
    if len(df) < 60: return (None, None), False
    try:
        return detect(df), True
    except Exception:
        return (None, None), False

@st.cache_data(ttl=240, show_spinner=False)   # < 5 min : chaque cycle auto refait un vrai scan
def run_scan(pairs, tfs):
    """Scan complet (mis en cache 5 min). Les threads n'appellent aucune fonction Streamlit."""
    jobs = [(p, t) for p in pairs for t in tfs]
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(lambda j: _scan_one(*j), jobs))
    res = {j: r for j, (r, _) in zip(jobs, out)}
    failed = sum(1 for _, ok in out if not ok)
    return res, failed, len(jobs), datetime.now(timezone.utc)

@st.cache_data(ttl=240, show_spinner=False)
def run_progress(pairs):
    """Progression R / E / M par paire, à partir du cache disque de bbma_core. Indépendant des TF sélectionnés."""
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(pairs, ex.map(prog.progress, pairs)))

pairs = list(dict.fromkeys(st.sidebar.text_area(f"Paires (max {MAX_PAIRS})", DEFAULT_PAIRS).upper().split()))
if len(pairs) > MAX_PAIRS:
    st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires.")
    pairs = pairs[:MAX_PAIRS]
tfs = st.sidebar.multiselect("Timeframes", list(TFS), list(TFS))
auto = st.sidebar.checkbox("Scan automatique toutes les 5 min", True)
if st.sidebar.button("Rafraîchir"): st.cache_data.clear()
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
/* Entrée (codes MTF REM / RRE / REE) */
table.bb td.synth{font-weight:700;font-size:13px;min-width:110px}
table.bb td.synth.none{background:#2a2e39;color:#787b86}
table.bb td.synth.buy{background:#26a69a;color:#fff;box-shadow:0 0 14px rgba(38,166,154,.55)}
table.bb td.synth.sell{background:#ef5350;color:#fff;box-shadow:0 0 14px rgba(239,83,80,.55)}
table.bb td.synth.partial{font-weight:700;font-size:13px;background:transparent;outline:1px dashed currentColor;outline-offset:-2px;box-shadow:none}
table.bb td.synth.partial.buy{color:#26a69a}
table.bb td.synth.partial.sell{color:#ef5350}
table.bb td.synth.partial small{font-weight:400;font-size:11px;opacity:1;color:#787b86}
/* Confiance (part des TF scannés dans le même sens) */
table.bb td.conf{font-weight:700;font-size:14px;min-width:70px}
table.bb td.conf.high{background:#26a69a;color:#fff}
table.bb td.conf.mid{background:#2962ff;color:#fff}
table.bb td.conf.low,table.bb td.conf.none{background:#2a2e39;color:#787b86}
</style>"""

ARROW = {"B": "▲", "S": "▼"}

def cell_html(code, d):
    if not code: return '<td class="none">–</td>'
    strong = " strong" if code in ("MOM", "EXM", "EXT") else ""
    tpw = '<span class="tpw">TPW</span>' if code in ("EXM", "EXT") else ""
    tip = NAMES.get(code, code) + (" · Buy" if d == "B" else " · Sell")
    return (f'<td class="{"buy" if d == "B" else "sell"}{strong}" title="{tip}">'
            f'<b>{code}</b> {ARROW[d]}{tpw}</td>')

def mtf_cell(code, d):
    """Cellule 'Entrée' : code MTF (REM / RRE / REE) et sens."""
    if not code: return '<td class="synth none">–</td>'
    cls = "buy" if d == "B" else "sell"
    tip = f"{code} · {NAMES[code]}" + (" · Buy" if d == "B" else " · Sell")
    return f'<td class="synth {cls}" title="{tip}"><b>{code}</b> {ARROW[d]}</td>'

def partial_cell(p):
    """Cellule 'Entrée en cours' : codes possibles, sens, TF manquant (pointillés = pas encore une entrée)."""
    if not p: return '<td class="synth none">–</td>'
    codes = " / ".join(p["codes"])
    d = p["sens"]
    cls = "buy" if d == "B" else "sell"
    tip = f"En cours : {codes} · 2 TF sur 3 alignés · attente {p['tf']} {p['attendu']}"
    return (f'<td class="synth partial {cls}" title="{tip}"><b>{codes}</b> {ARROW[d]}'
            f'<br><small>+{p["tf"]} {p["attendu"]}</small></td>')

def prog_cell(pr):
    """Cellule 'Progression' : lettres déjà validées (R E M), pointillés si incomplet, heure de la dernière étape."""
    if not pr: return '<td class="none">–</td>'
    d = pr["sens"]
    cls = "buy" if d == "B" else "sell"
    letters = " ".join(pr["letters"])
    if pr["expired"]:
        tip = f"Setup expiré · démarré {pr['start']:%d/%m %H:%M} UTC"
        return f'<td class="synth none" title="{tip}"><b>{letters}</b> {ARROW[d]}<br><small>expiré</small></td>'
    done = pr["complete"]
    style = "synth" if done else "synth partial"
    label = done or f"{pr['steps']}/3 en cours"
    tip = (f"{label} · démarré {pr['start']:%d/%m %H:%M} UTC · dernière étape {pr['since']:%d/%m %H:%M} UTC")
    return (f'<td class="{style} {cls}" title="{tip}"><b>{letters}</b> {ARROW[d]}'
            f'<br><small>{label}</small></td>')

def conf_cell(score):
    """Cellule 'Confiance' : pourcentage avec code couleur (vert >= 70%, bleu 40-69%, gris < 40%)."""
    if score is None: return '<td class="conf none">–</td>'
    pct = round(score * 100)
    cls = "high" if pct >= 70 else "mid" if pct >= 40 else "low"
    return f'<td class="conf {cls}">{pct}%</td>'

@st.fragment(run_every=300 if auto else None)   # relance seulement cette partie, toutes les 5 min
def live():
    with st.spinner("Scan en cours..."):
        res, failed, total, ts = run_scan(tuple(pairs), tuple(tfs))
        progs = run_progress(tuple(pairs))
    if failed:
        st.warning(f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
                   "Clique sur Rafraîchir dans quelques instants.")
    if not {"D1", "H4", "H1"} <= set(tfs):
        st.info("Les codes REM / RRE / REE demandent les TF D1, H4 et H1 : sélectionne-les pour voir des entrées.")

    body = []
    for p in pairs:
        cells, tds = {}, []
        for t in tfs:
            code, d = res[(p, t)]
            cells[t] = (code, d)
            tds.append(cell_html(code, d))
        pr = progs.get(p)
        code_e, sens_e = prog.entree(pr)               # une seule source : le moteur de progression
        score = confiance_score(cells, tfs, sens_e) if sens_e else None
        tds.append(mtf_cell(code_e, sens_e))
        tds.append(conf_cell(score))
        tds.append(prog_cell(pr))
        body.append(f'<tr><th class="pair">{html.escape(p)}</th>{"".join(tds)}</tr>')
    head = ('<tr><th class="pair">Pair</th>' + "".join(f"<th>{t}</th>" for t in tfs)
            + '<th class="hmtf">Entrée</th><th class="hmtf">Confiance</th><th class="hmtf">Progression</th></tr>')
    st.markdown(CSS + f'<div class="bb-wrap"><table class="bb">{head}{"".join(body)}</table></div>', unsafe_allow_html=True)
    st.caption("▲ Buy · ▼ Sell · MOM=Momentum · EXM=Extreme Magic (MA10) · EXT=Extreme (MA5) · TPW=TP Wajib (avec EXM/EXT) · "
               "MHV=Market Hilang Volume · CSAK=Candle Arah Kukuh · RE=Reentry (retour depuis l'extérieur de la zone MA5/MA10). "
               "Survole une cellule pour voir le nom du signal.")
    st.caption("Entrée : REM = RE D1 · EXT H4 · MHV H1 ; RRE = RE D1 · RE H4 · EXT H1 ; REE = RE D1 · EXT H4 · EXT H1 "
               "(R = Reentry, E = Extreme, M = MHV). Les trois TF doivent être dans le même sens. EXM compté comme EXT. "
               "Confiance : part des TF scannés dans le même sens que l'entrée (indicateur, pas une règle). "
               "Les bougies non clôturées sont exclues du scan.")
    st.caption("Entrée : uniquement un motif complet (REM, RRE, REE) non expiré. Le motif se lit dans la colonne "
               "Progression, qui est la seule source de calcul.")
    st.caption("Progression : R = RE sur D1, E = EXT sur H4, M = MHV sur H1, et R sur H4 pour RRE. "
               "Pointillés = motif incomplet, gris = expiré. Un setup expire après la fenêtre MAX_AGE "
               "depuis le D1 RE. Un signal de structure contraire le remet à zéro.")
    st.caption(f"Dernier scan : {ts:%H:%M:%S} UTC" + (" · prochain scan automatique dans 5 min" if auto else " · scan automatique désactivé"))

live()
