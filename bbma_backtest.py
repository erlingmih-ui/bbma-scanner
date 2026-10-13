"""Backtest de la synthèse BBMA et de la colonne Qualité.

Usage (sur votre machine, accès yfinance requis) :
    python bbma_backtest.py EURUSD GBPUSD --base M15 --tfs D1,H4,H1,M15 --months 6

Règles simulées :
  - décision à l'ouverture de chaque bougie du TF de base, avec les bougies fermées uniquement ;
  - entrée à l'ouverture de la première bougie du TF d'entrée retenu par bbma_core.synthese ;
  - stop = SL_ATR x ATR, cible = TP_ATR x ATR, ATR du TF d'entrée ;
  - si stop et cible sont touchés dans la même bougie, le stop compte (hypothèse prudente) ;
  - sortie au bout de HORIZON bougies si ni le stop ni la cible ne sont touchés (TIME) ;
  - un seul trade ouvert à la fois par paire.
Résultat en R : +TP_ATR/SL_ATR pour une cible atteinte, -1 pour un stop, mesure à la sortie pour TIME.
Les fins de mois (MN) sont approximées à 28 jours.
"""
import argparse

import pandas as pd

import bbma_core as core
import bbma_quality as bq
import bbma_session as bs
from bbma_quality_codes import MID_PERIOD

SL_ATR = 1.5          # distance du stop en ATR
TP_ATR = 3.0          # distance de la cible en ATR
HORIZON = 48          # bougies maximum dans un trade (TF d'entrée)
WINDOW = 300          # bougies utilisées pour calculer les signaux à chaque instant
LEN = {"M5": pd.Timedelta(minutes=5), "M15": pd.Timedelta(minutes=15),
       "H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4),
       "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(weeks=1), "MN": pd.Timedelta(days=28)}


def load(pair, tf):
    """Bougies fermées d'un timeframe, rééchantillonnées comme dans bbma_core. None si indisponible."""
    interval, period, rs = core.TFS[tf]
    df = core._history_cached(pair, interval, period)
    if df is None:
        return None
    if tf == "H4":
        return bs.resample_h4(df)
    if rs:
        df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return df


class TFData:
    """Bougies d'un timeframe, avec l'instant de fermeture de chaque bougie pour un accès rapide."""

    def __init__(self, df, tf):
        self.df = df
        self.ends = df.index + LEN[tf]

    def closed_before(self, T):
        """Dernières bougies déjà fermées à l'instant T (ouverture de la bougie de décision)."""
        k = self.ends.searchsorted(T, side="right")
        return self.df.iloc[max(0, k - WINDOW): k]


def features(sub):
    """RSI, distance signée à la MID BB en ATR, et ATR de la dernière bougie fermée."""
    c = sub["Close"]
    mid = c.rolling(MID_PERIOD).mean().iloc[-1]
    a = bq.atr(sub).iloc[-1]
    r = core.rsi(c).iloc[-1]
    dist = None if pd.isna(a) or a <= 0 or pd.isna(mid) else float((c.iloc[-1] - mid) / a)
    feat = {"rsi": None if pd.isna(r) else float(r), "dist_atr": dist}
    return feat, (None if pd.isna(a) else float(a))


def simulate(e, p, sens, entry, a):
    """Simule un trade à partir de la bougie p. Retourne (R, issue, indice de sortie) ou None."""
    if a is None or a <= 0:
        return None
    end = min(p + HORIZON, len(e))
    if end <= p:
        return None
    risk = SL_ATR * a
    sign = 1 if sens == "B" else -1
    sl = entry - sign * risk
    tp = entry + sign * TP_ATR * a
    for j in range(p, end):
        hi, lo = e["High"].iloc[j], e["Low"].iloc[j]
        hit_sl = lo <= sl if sign == 1 else hi >= sl
        hit_tp = hi >= tp if sign == 1 else lo <= tp
        if hit_sl:
            return -1.0, "SL", j
        if hit_tp:
            return TP_ATR / SL_ATR, "TP", j
    j = end - 1
    return sign * (e["Close"].iloc[j] - entry) / risk, "TIME", j


def backtest_pair(pair, base, tfs, start):
    """Liste des trades simulés pour une paire, à partir de l'instant start."""
    frames = {}
    for tf in set(tfs) | {"D1"}:
        df = load(pair, tf)
        if df is None or len(df) < core.MIN_BARS:
            return []
        frames[tf] = TFData(df, tf)

    trades, busy_until = [], None
    for T in frames[base].df.index:
        if T < start or (busy_until is not None and T < busy_until):
            continue

        cells = {}
        avail = []
        for tf in tfs:
            sub = frames[tf].closed_before(T)
            ok = len(sub) >= core.MIN_BARS
            if ok:
                avail.append(tf)
            cells[tf] = core.detect(sub) if ok else (None, None)

        d1 = None
        d1sub = frames["D1"].closed_before(T)
        if len(d1sub) >= core.MIN_BARS:
            c = d1sub["Close"]
            d1 = "B" if c.iloc[-1] > c.ewm(span=50, adjust=False).mean().iloc[-1] else "S"

        tf_e, code_e, sens_e = core.synthese(cells, tfs, True, d1)
        if not sens_e:
            continue

        e = frames[tf_e].df
        p = e.index.searchsorted(T)
        if p >= len(e):
            continue
        esub = frames[tf_e].closed_before(T)
        if len(esub) < core.MIN_BARS:
            continue

        feat, a = features(esub)
        conf = bq.confiance_ponderee(cells, tfs, sens_e, avail)
        q_score, q_grade, _ = bq.qualite_entree(code_e, sens_e, conf, d1, feat)
        kind = bs.entry_kind(cells, tfs, tf_e, code_e)
        q_score, q_grade = bs.cap_grade(q_score, q_grade, kind)
        sim = simulate(e, p, sens_e, float(e["Open"].iloc[p]), a)
        if sim is None:
            continue
        r, outcome, j = sim
        busy_until = e.index[j]
        trades.append({"pair": pair, "time": T, "tf_entry": tf_e, "code": code_e, "sens": sens_e,
                       "grade": q_grade, "score": q_score, "kind": kind, "R": r, "outcome": outcome})
    return trades


def _stats(label, s):
    n = len(s)
    if n == 0:
        return {"grade": label, "n": 0}
    tp = (s["outcome"] == "TP").sum()
    sl = (s["outcome"] == "SL").sum()
    tm = (s["outcome"] == "TIME").sum()
    decided = tp + sl
    return {"grade": label, "n": n,
            "TP %": round(100 * tp / n, 1), "SL %": round(100 * sl / n, 1), "TIME %": round(100 * tm / n, 1),
            "réussite (TP/(TP+SL)) %": round(100 * tp / decided, 1) if decided else None,
            "R moyen": round(s["R"].mean(), 2)}


def summarize(df):
    rows = [_stats(g, df[df["grade"] == g]) for g in ["A", "B", "C"]]
    rows.append(_stats("Tous", df))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Backtest de la synthèse BBMA et de la colonne Qualité.")
    parser.add_argument("pairs", nargs="+", help="paires, ex. EURUSD GBPUSD")
    parser.add_argument("--base", default="M15", help="TF de décision (doit être dans --tfs)")
    parser.add_argument("--tfs", default="D1,H4,H1,M15", help="timeframes séparés par des virgules")
    parser.add_argument("--months", type=int, default=3, help="durée du test en mois")
    parser.add_argument("--csv", default="bbma_backtest_trades.csv", help="fichier de sortie des trades")
    args = parser.parse_args()

    tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    base = args.base.upper()
    if base not in tfs:
        parser.error("--base doit faire partie de --tfs")
    unknown = [t for t in tfs + [base] if t not in core.TFS or t not in LEN]
    if unknown:
        parser.error(f"timeframe non supporté : {', '.join(unknown)}")

    start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=args.months)
    trades = []
    for pair in args.pairs:
        found = backtest_pair(pair.upper(), base, tfs, start)
        print(f"{pair.upper()} : {len(found)} trades")
        trades += found

    if not trades:
        print("Aucun trade simulé. Vérifiez les paires, la connexion ou la durée du test.")
        return
    df = pd.DataFrame(trades)
    df.to_csv(args.csv, index=False)
    print(f"\nTrades enregistrés dans {args.csv}\n")
    print(summarize(df).to_string(index=False))
    print("\nLecture : un grade est utile si son R moyen et son taux de réussite dépassent ceux des grades inférieurs.")
    print("Avec moins de 30 trades par grade, les écarts ne sont pas significatifs.")


if __name__ == "__main__":
    main()
