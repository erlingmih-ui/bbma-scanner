"""Vérifie Dukascopy sur vos paires (accès internet requis) : python check_duka.py"""
import sys
import bbma_duka as d
import bbma_pairs as bp

if d.dk is None:
    sys.exit("dukascopy-python non installé : pip install dukascopy-python")
bad = []
for p in bp.PAIR_CHOICES:
    df = d.fetch(p, "5m", "2d") if d.instrument(p) else None
    if df is None:
        print(f"ÉCHEC  {p:8s} {d.instrument(p)}"); bad.append(p)
    else:
        lag = (__import__('pandas').Timestamp.now(tz='UTC') - df.index[-1]).total_seconds() / 60
        print(f"OK     {p:8s} {d.instrument(p)} {len(df)} bougies M5, dernière il y a {lag:.0f} min")
sys.exit(1 if bad else 0)
