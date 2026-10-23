"""Comparaison : version précédente (sans pondération, sans contrôle de franchissement, seuil 0,3 ATR)
vs version actuelle, sur les mêmes séries synthétiques."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/claude/bbma")
import bbma_levels as bl

OLD_NEAR = 0.3


def old_sr(df, extra=(), k=3, tol=0.5, near=OLD_NEAR):
    """Règles de la version précédente : nombre de touches seulement, aucun contrôle de franchissement."""
    a = bl._atr(df).iloc[-1]
    close = float(df["Close"].iloc[-1])
    n = len(df)
    zones = [{"price": z["price"], "touches": z["touches"], "src": "pivot"}
             for z in bl._cluster(bl.confirmed_pivots_idx(df, k), tol * a, n)]
    zones += [{"price": p, "touches": 1, "src": s} for p, s, _ in extra]
    above = [z for z in zones if z["price"] > close + near * a]
    below = [z for z in zones if z["price"] < close - near * a]
    on = any(abs(z["price"] - close) <= near * a for z in zones)

    def nearest(c):
        if not c:
            return None
        z = min(c, key=lambda x: abs(x["price"] - close))
        return abs(z["price"] - close) / a
    return {"res": nearest(above), "sup": nearest(below), "on_level": on}


def new_sr(df, extra=()):
    lv = bl.sr_levels(df, extra=extra)
    return {"res": lv["res"]["atr"] if lv["res"] else None,
            "sup": lv["sup"]["atr"] if lv["sup"] else None,
            "on_level": lv["on_level"], "casses": lv["casses"]}


def series(seed, n=1000):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1.0, n))          # ATR ≈ 1
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"Open": c, "High": c + 0.5, "Low": c - 0.5, "Close": c}, index=idx)


rows = []
for seed in range(300):
    df = series(seed)
    # bougies : on regarde la situation à plusieurs instants pour éviter un seul cas
    for end in (400, 600, 800, 1000):
        sub = df.iloc[:end]
        o, nw = old_sr(sub), new_sr(sub)
        rows.append((o, nw))

total = len(rows)
both_res = sum(1 for o, n in rows if o["res"] is not None)
new_res = sum(1 for o, n in rows if n["res"] is not None)
both_sup = sum(1 for o, n in rows if o["sup"] is not None)
new_sup = sum(1 for o, n in rows if n["sup"] is not None)
changed_res = sum(1 for o, n in rows if (o["res"] or -1) != (n["res"] or -1))
changed_sup = sum(1 for o, n in rows if (o["sup"] or -1) != (n["sup"] or -1))
on_old = sum(1 for o, n in rows if o["on_level"])
on_new = sum(1 for o, n in rows if n["on_level"])
casses = [n["casses"] for o, n in rows]
close_old = [o["res"] for o, n in rows if o["res"] is not None and o["res"] < 1.0]
close_new = [n["res"] for o, n in rows if n["res"] is not None and n["res"] < 1.0]

pct = lambda a, b: f"{100 * a / b:.1f} %"
print(f"situations comparées : {total}")
print(f"résistance affichée   : ancienne {pct(both_res, total)} | nouvelle {pct(new_res, total)}")
print(f"support affiché       : ancien {pct(both_sup, total)} | nouveau {pct(new_sup, total)}")
print(f"résistance différente : {pct(changed_res, total)} des cas")
print(f"support différent     : {pct(changed_sup, total)} des cas")
print(f"cadre jaune allumé    : ancien {pct(on_old, total)} | nouveau {pct(on_new, total)}")
print(f"niveaux franchis ignorés : {pct(sum(1 for c in casses if c > 0), total)} des cas (moyenne {np.mean(casses):.2f} par cas)")
print(f"résistance < 1 ATR    : ancienne {pct(len(close_old), total)} | nouvelle {pct(len(close_new), total)}")
