"""Entrée par style, à partir du script Analysis_entry_distance.py (à placer à côté de ce module, inchangé).

Les calculs passent par les fonctions du script : add_indicators, get_trend, get_support_resistance,
suggest_entry et build_style_suggestion. Les données viennent de fetch_data du script.
La cellule n'affiche que le biais et la distance ; le détail complet est dans l'infobulle (ordinateur).
Les prix sont affichés avec 5 décimales.
"""
import html

import pandas as pd

import Analysis_entry_distance as ade

TFS_BY_STYLE = {"Scalping": ("M15", "H1"), "Swing": ("H4", "D1")}   # (rapide, lent), comme le script
SCRIPT_KEY = {"M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}      # clés de ade.TIMEFRAMES
ORDER = ["M15", "H1", "H4", "D1"]
NEAR_PCT = 0.3                                                        # comme le script (abs(dist) < 0,3)
SHORT = {"Scalping": "Scal", "Swing": "Swing"}
GREEN, RED, GREY, AMBER = "#26a69a", "#ef5350", "#787b86", "#f0b90b"
DEC = 5


def fmt(x):
    return f"{x:.{DEC}f}"


# ---------- calcul : fonctions du script ----------

def analyze(df):
    """Même logique que ade.analyze_timeframe, sur un DataFrame déjà téléchargé."""
    df = ade.add_indicators(df)
    last = df.iloc[-1]
    trend = ade.get_trend(df)
    support, resistance = ade.get_support_resistance(df)
    notes = ade.suggest_entry(df, trend, support, resistance)
    return {"as_of": df.index[-1], "price": last["Close"], "trend": trend,
            "support": support, "resistance": resistance, "rsi": last["RSI"], "notes": notes}


def analyze_all(frames):
    """{M15,H1,H4,D1: résultat ou None} -> {clé du script: résultat} pour build_style_suggestion."""
    out = {}
    for tf in ORDER:
        df = frames.get(tf)
        out[tf] = analyze(df) if df is not None and len(df) else None
    return out


def style_entry(results, style):
    """Setup chiffré d'un style : mêmes règles que ade.build_style_suggestion."""
    fast_tf, slow_tf = TFS_BY_STYLE[style]
    base = {"style": style, "fast_tf": fast_tf, "slow_tf": slow_tf, "bias": None, "entry": None,
            "stop": None, "target": None, "dist_pct": None, "status": None, "reason": None,
            "script_lines": ade.build_style_suggestion(
                {SCRIPT_KEY[k]: v for k, v in results.items() if v is not None}, style)}
    fast, slow = results.get(fast_tf), results.get(slow_tf)
    if fast is None or slow is None:
        missing = [tf for tf, r in ((fast_tf, fast), (slow_tf, slow)) if r is None]
        return {**base, "status": "missing", "reason": f"données manquantes : {', '.join(missing)}"}

    buf = ade.STOP_BUFFER[style]
    price = fast["price"]
    trends = {fast["trend"], slow["trend"]}
    if trends == {"Uptrend"}:
        bias, entry = "BUY", fast["support"]
        stop, target = entry * (1 - buf), slow["resistance"]
        dist = (price - entry) / price * 100
    elif trends == {"Downtrend"}:
        bias, entry = "SELL", fast["resistance"]
        stop, target = entry * (1 + buf), slow["support"]
        dist = (entry - price) / price * 100
    else:
        return {**base, "status": "no_setup",
                "reason": f"tendances en désaccord ({fast_tf}={fast['trend']}, {slow_tf}={slow['trend']})"}

    status = "at_entry" if abs(dist) < NEAR_PCT else "pending"
    return {**base, "bias": bias, "entry": entry, "stop": stop, "target": target,
            "dist_pct": dist, "status": status}


def entry_by_style(frames):
    results = analyze_all(frames)
    return {style: style_entry(results, style) for style in TFS_BY_STYLE}, results


# ---------- données : fetch_data du script ----------

def ticker_for(pair):
    return ade.SYMBOLS.get(pair, pair + "=X")


def frames_for_pair(pair):
    """{M15, H1, H4, D1} via ade.fetch_data. Un échec donne None pour ce timeframe."""
    ticker, out = ticker_for(pair), {}
    for tf in ORDER:
        cfg = ade.TIMEFRAMES[SCRIPT_KEY[tf]]
        try:
            out[tf] = ade.fetch_data(ticker, cfg["interval"], cfg["range"], cfg["resample"])
        except Exception:
            out[tf] = None
    return out


# ---------- affichage ----------

def tf_lines(tf, r):
    """Bloc d'un timeframe, même présentation que la sortie du script."""
    if r is None:
        return [f"[{SCRIPT_KEY[tf]}] données indisponibles"]
    lines = [f"[{SCRIPT_KEY[tf]}] as of {r['as_of']}",
             f"  Price: {fmt(r['price'])} | Trend: {r['trend']}"]
    line = f"  Support: {fmt(r['support'])}  Resistance: {fmt(r['resistance'])}"
    if not pd.isna(r["rsi"]):
        line += f"  RSI: {r['rsi']:.1f}"
    lines.append(line)
    lines += [f"    - {n}" for n in r["notes"]]
    return lines


def describe_style(r):
    if r["status"] in ("missing", "no_setup"):
        return [f"{SHORT[r['style']]} : {r['reason']}"]
    side = "Achat" if r["bias"] == "BUY" else "Vente"
    state = "prix à l'entrée" if r["status"] == "at_entry" else "pas encore touchée, placer une alerte"
    return [f"{SHORT[r['style']]} {side} ({r['fast_tf']} et {r['slow_tf']}) : "
            f"entrée {fmt(r['entry'])} · stop {fmt(r['stop'])} · cible {fmt(r['target'])} · "
            f"écart {abs(r['dist_pct']):.2f} % · {state}"]


def tooltip_text(setups, results):
    lines = []
    for style in TFS_BY_STYLE:
        lines += describe_style(setups[style])
    lines.append("")
    for tf in ORDER:
        lines += tf_lines(tf, results.get(tf))
    lines.append("")
    lines.append("Trade style suggestions")
    for style in TFS_BY_STYLE:
        lines += setups[style]["script_lines"]
    return "\n".join(lines)


def _tip_attr(text):
    """Infobulle ordinateur : échappement HTML, retours à la ligne en &#10;."""
    return html.escape(text).replace("\n", "&#10;")


def entry_cell(setups, results=None):
    """Cellule « Entrée » : biais et distance ; détail complet dans l'infobulle (ordinateur)."""
    results = results or {}
    tip = _tip_attr(tooltip_text(setups, results))
    lines, at_entry = [], False
    for style in TFS_BY_STYLE:
        r = setups[style]
        if r["bias"] is None:
            continue
        color = GREEN if r["bias"] == "BUY" else RED
        side = "Achat" if r["bias"] == "BUY" else "Vente"
        weight = "700" if r["status"] == "at_entry" else "400"
        at_entry = at_entry or r["status"] == "at_entry"
        lines.append(f'<div style="color:{color};font-weight:{weight}">{SHORT[style]} {side}</div>'
                     f'<div style="color:{color}">{abs(r["dist_pct"]):.1f} %</div>')
    if not lines:
        return f'<td class="lvl none" title="{tip}" style="color:{GREY}">–</td>'
    ring = f";box-shadow:0 0 0 1px {AMBER}" if at_entry else ""
    return (f'<td class="lvl" title="{tip}" style="font-size:0.8em;line-height:1.25{ring}">'
            + "".join(lines) + "</td>")
