"""Distance d'entrée par style de trading, à partir des niveaux S/R (bbma_levels.sr_levels).

Idée reprise de Analysis_entry_distance.py :
  - entrée = support (achat) ou résistance (vente) du timeframe rapide ;
  - cible = niveau opposé du timeframe lent ;
  - stop = entrée décalée de STOP_BUFFER (en pourcentage du prix) ;
  - distance = écart en % entre le prix actuel et l'entrée.
  Biais : achat si les deux timeframes sont en tendance haussière, vente si les deux sont baissières.

Adaptation : les niveaux sont des prix absolus fournis par sr_levels (champ "price"), et la tendance
est fournie par l'appelant (ex. tendance BBMA existante). Ce module ne calcule ni ne modifie aucun signal.
"""

# Timeframes par style : (rapide, lent). Le rapide fixe l'entrée, le lent fixe la cible.
STYLE_TFS = {
    "Scalping": ("M15", "H1"),
    "Swing": ("H4", "D1"),
}
STOP_BUFFER = {"Scalping": 0.003, "Swing": 0.01}   # 0,3 % et 1 % au-delà du niveau (comme la référence)
NEAR_PCT = 0.3                                     # prix à moins de 0,3 % de l'entrée : « à l'entrée »


def build_entry(sr_by_tf, trend_by_tf, style, price):
    """Entrée, stop, cible et distance pour un style donné.

    sr_by_tf    : {timeframe: résultat de sr_levels} ; None si indisponible
    trend_by_tf : {timeframe: "Uptrend" | "Downtrend" | autre}
    price       : prix actuel (dernière clôture fermée du timeframe rapide)

    Retourne un dict. status vaut "at_entry", "pending", "no_setup" ou "missing" ;
    reason explique l'absence de setup."""
    fast_tf, slow_tf = STYLE_TFS[style]
    base = {"style": style, "fast_tf": fast_tf, "slow_tf": slow_tf, "bias": None,
            "entry": None, "stop": None, "target": None, "dist_pct": None,
            "status": None, "reason": None}

    fast, slow = sr_by_tf.get(fast_tf), sr_by_tf.get(slow_tf)
    if fast is None or slow is None:
        missing = [tf for tf, r in ((fast_tf, fast), (slow_tf, slow)) if r is None]
        return {**base, "status": "missing", "reason": f"données manquantes : {', '.join(missing)}"}

    t_fast, t_slow = trend_by_tf.get(fast_tf), trend_by_tf.get(slow_tf)
    if t_fast == "Uptrend" and t_slow == "Uptrend":
        bias = "BUY"
    elif t_fast == "Downtrend" and t_slow == "Downtrend":
        bias = "SELL"
    else:
        return {**base, "status": "no_setup",
                "reason": f"tendances en désaccord ({fast_tf}={t_fast}, {slow_tf}={t_slow})"}

    buffer = STOP_BUFFER[style]
    if bias == "BUY":
        entry_lvl, target_lvl = fast["sup"], slow["res"]
        if entry_lvl is None or target_lvl is None:
            return {**base, "bias": bias, "status": "missing",
                    "reason": f"aucun support {fast_tf} ou aucune résistance {slow_tf} valide"}
        entry = entry_lvl["price"]
        stop = entry * (1 - buffer)
        target = target_lvl["price"]
        dist_pct = (price - entry) / price * 100
    else:
        entry_lvl, target_lvl = fast["res"], slow["sup"]
        if entry_lvl is None or target_lvl is None:
            return {**base, "bias": bias, "status": "missing",
                    "reason": f"aucune résistance {fast_tf} ou aucun support {slow_tf} valide"}
        entry = entry_lvl["price"]
        stop = entry * (1 + buffer)
        target = target_lvl["price"]
        dist_pct = (entry - price) / price * 100

    status = "at_entry" if abs(dist_pct) < NEAR_PCT else "pending"
    return {**base, "bias": bias, "entry": entry, "stop": stop, "target": target,
            "dist_pct": dist_pct, "status": status}


def describe_entry(d, digits=4):
    """Lignes de texte pour l'affichage (même logique que la référence)."""
    if d["status"] in ("missing", "no_setup"):
        return [f"{d['style']} : {d['reason']}"]
    side = "BUY" if d["bias"] == "BUY" else "SELL"
    lines = [f"{d['style']} {side} ({d['fast_tf']} & {d['slow_tf']} même tendance)",
             f"  Entrée ~{d['entry']:.{digits}f}  Stop ~{d['stop']:.{digits}f}  "
             f"Cible ~{d['target']:.{digits}f}"]
    if d["status"] == "at_entry":
        lines.append(f"  Prix déjà à l'entrée ({d['dist_pct']:.2f} % d'écart)")
    else:
        direction = "au-dessus" if d["bias"] == "BUY" else "en dessous"
        lines.append(f"  Prix {abs(d['dist_pct']):.2f} % {direction} de l'entrée, "
                     f"pas encore touchée : placer une alerte")
    return lines
