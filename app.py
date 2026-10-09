import numpy as np, pandas as pd, streamlit as st, yfinance as yf
from concurrent.futures import ThreadPoolExecutor

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

# Codes MTF (colonne de droite) : combinaison de signaux sur D1 / H4 / H1, même sens (B ou S).
PATTERNS = {
    "REM": [("D1", "RE"), ("H4", "EXT"), ("H1", "MHV")],   # Reentry - Extreme - MHV
    "RRE": [("D1", "RE"), ("H4", "RE"), ("H1", "EXT")],    # Reentry - Reentry - Extreme
    "REE": [("D1", "RE"), ("H4", "EXT"), ("H1", "EXT")],   # Reentry - Extreme - Extreme
}
ORDER = ["REM", "REE", "RRE"]

def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)

def detect(df, lb=15):
    """Signal de la dernière bougie : (code, sens) avec sens 'B' (buy) ou 'S' (sell).
    Priorité : MOM > EXM (Extreme Magic, MA10) > EXT (MA5) > MHV > CSAK (Candle Arah) > RE.
    TP Wajib = flag déclenché par EXM/EXT uniquement (affiché à côté du code)."""
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
    # tendance mémorisée (m_lastTrendDir) : dernier MOM / Arah
    trend = pd.Series(np.where(mom_b | arah_b, 1, np.where(mom_s | arah_s, -1, np.nan)), index=c.index).ffill()
    re_s = (trend == -1) & (c < mid) & (h >= np.minimum(h5, h10)) & (c <= np.maximum(h5, h10))   # RE
    re_b = (trend == 1) & (c > mid) & (l <= np.maximum(l5, l10)) & (c >= np.minimum(l5, l10))

    sig = {"MOM": (mom_b, mom_s), "EXM": (exm_b, exm_s), "EXT": (ext_b, ext_s),
           "MHV": (mhv_b, mhv_s), "CSAK": (arah_b, arah_s), "RE": (re_b, re_s)}
    for code in PRIORITY:
        b, s = sig[code]
        if b.iloc[-1]: return code, "B"
        if s.iloc[-1]: return code, "S"
    return None, None

def mtf_code(cells):
    """cells: {tf: (signal, sens)} -> (code MTF, sens) ou (None, None)."""
    for code in ORDER:
        conds = PATTERNS[code]
        base = lambda x: "EXT" if x == "EXM" else x
        if all(base(cells.get(tf, (None, None))[0]) == sig for tf, sig in conds):
            dirs = {cells[tf][1] for tf, _ in conds}
            if len(dirs) == 1: return code, dirs.pop()
    return None, None

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
    if len(df) < 60: return (None, None), False
    try:
        return detect(df), True
    except Exception:
        return (None, None), False

def _d1_trend(pair):
    """Filtre D1 : 'B' si clôture D1 > EMA50, 'S' si <, None si données indisponibles."""
    df = _history(pair, "1d", "2y")
    if df is None or len(df) < 60: return None
    c = df["Close"]
    return "B" if c.iloc[-1] > c.ewm(span=50, adjust=False).mean().iloc[-1] else "S"

@st.cache_data(ttl=300, show_spinner=False)
def run_scan(pairs, tfs):
    """Scan complet (mis en cache 5 min). Les threads n'appellent aucune fonction Streamlit."""
    jobs = [(p, t) for p in pairs for t in tfs]
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(lambda j: _scan_one(*j), jobs))
        trends = list(ex.map(_d1_trend, pairs))
    res = {j: r for j, (r, _) in zip(jobs, out)}
    failed = sum(1 for _, ok in out if not ok)
    return res, dict(zip(pairs, trends)), failed, len(jobs)

pairs = list(dict.fromkeys(st.sidebar.text_area(f"Paires (max {MAX_PAIRS})", DEFAULT_PAIRS).upper().split()))
if len(pairs) > MAX_PAIRS:
    st.sidebar.warning(f"Limité aux {MAX_PAIRS} premières paires.")
    pairs = pairs[:MAX_PAIRS]
tfs = st.sidebar.multiselect("Timeframes", list(TFS), list(TFS))
use_filter = st.sidebar.checkbox("Filtre D1 (EMA50)", True)
if st.sidebar.button("Rafraîchir"): st.cache_data.clear()
if not pairs or not tfs:
    st.info("Choisis au moins une paire et un timeframe dans la barre latérale.")
    st.stop()

with st.spinner("Scan en cours..."):
    res, trends, failed, total = run_scan(tuple(pairs), tuple(tfs))
if failed:
    st.warning(f"{failed}/{total} téléchargements sans données (limite yfinance, marché fermé ou symbole invalide). "
               "Clique sur Rafraîchir dans quelques instants.")

rows = []
for p in pairs:
    row, cells = {"Pair": p}, {}
    for t in tfs:
        code, d = res[(p, t)]
        cells[t] = (code, d)
        row[t] = "None" if not code else f"{code} {'▲' if d == 'B' else '▼'}" + (" TPW" if code in ("EXM", "EXT") else "")
    m, d = mtf_code(cells)
    if m and use_filter:
        tr = trends.get(p)
        if tr and tr != d: m = None   # signal contre la tendance D1 (EMA50) -> ignoré
    row["Code MTF"] = "-" if not m else f"{m} {'▲' if d == 'B' else '▼'}"
    rows.append(row)
df = pd.DataFrame(rows).set_index("Pair")

def color(v):
    if "▲" in str(v): return "color:#22c55e;font-weight:bold"
    if "▼" in str(v): return "color:#ef4444;font-weight:bold"
    return "color:#888"

styler = df.style
styler = styler.map(color) if hasattr(styler, "map") else styler.applymap(color)
h = 38 * (len(df) + 1) + 3
try:
    st.dataframe(styler, width="stretch", height=h)
except TypeError:   # anciennes versions de Streamlit
    st.dataframe(styler, use_container_width=True, height=h)
st.caption("▲ Buy · ▼ Sell · MOM=Momentum · EXM=Extreme Magic (MA10) · EXT=Extreme (MA5) · TPW=TP Wajib (avec EXM/EXT) · MHV · CSAK=Candle Arah · RE=Reentry. "
           "Code MTF : REM = D1 RE + H4 EXT + H1 MHV · RRE = RE + RE + EXT · REE = RE + EXT + EXT (même sens). "
           "Filtre D1 : signal gardé seulement s'il suit la tendance D1 (clôture vs EMA50).")
