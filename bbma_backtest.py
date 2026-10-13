"""Backtest de la synthèse BBMA : sorties fidèles à la méthode, comparées aux sorties ATR.

Usage (sur votre machine, accès yfinance requis) :
    python bbma_backtest.py EURUSD GBPUSD --base M15 --tfs D1,H4,H1,M15 --months 6 --mode both

Signaux (identiques dans tous les modes) :
  - décision à l'ouverture de chaque bougie du TF de base, avec les bougies fermées uniquement ;
  - entrée à l'ouverture de la première bougie du TF d'entrée retenu par bbma_core.synthese_ex ;
  - le H4 passe par le chemin unique core.resample_tf -> bbma_session.resample_h4.

Mode « bbma » (sorties de la méthode) :
  - stop derrière l'Extreme / MHV du cycle (core.cycle_anchor) + SL_BUFFER_ATR x ATR ;
    sans cycle actif : extrême des FALLBACK_BARS dernières bougies ;
  - TP1 = MID BB (si elle est devant l'entrée d'au moins MIN_TP1_R x risque ; sinon, après un RE sur
    tendance établie, TP1 = TP1_FALLBACK de la distance vers la bande opposée) ;
  - TP2 = bande de Bollinger opposée ;
  - gestion partielle : PARTIAL du trade soldé à TP1, stop remonté à l'entrée (break-even) pour le reste ;
  - trade ignoré si le risque dépasse MAX_RISK_ATR x ATR ou si la bande opposée n'est pas devant l'entrée.
Mode « atr » (référence) : stop SL_ATR x ATR, cible TP_ATR x ATR.

Hypothèses prudentes : stop et cible touchés dans la même bougie -> le stop compte ; le break-even
n'agit qu'à partir de la bougie suivant TP1 ; sortie TIME après HORIZON bougies ; un seul trade ouvert
à la fois par paire. Résultat en R (1 R = distance entrée -> stop initial).
Les fins de mois (MN) sont approximées à 28 jours.
"""
import argparse

import numpy as np
import pandas as pd

import bbma_core as core
import bbma_quality as bq
import bbma_session as bs
from bbma_quality_codes import MID_PERIOD

# --- Mode ATR (référence) ---
SL_ATR = 1.5          # distance du stop en ATR
TP_ATR = 3.0          # distance de la cible en ATR
# --- Mode BBMA ---
SL_BUFFER_ATR = 0.1   # marge derrière l'Extreme / MHV
FALLBACK_BARS = 10    # extrême des n dernières bougies quand aucun cycle n'est actif
MAX_RISK_ATR = 5.0    # risque maximal accepté (en ATR)
MIN_TP1_R = 0.5       # la MID BB doit être au moins à 0,5 R devant l'entrée pour servir de TP1
TP1_FALLBACK = 0.5    # sinon : TP1 à cette part de la distance vers la bande opposée
PARTIAL = 0.5         # part de la position soldée à TP1
# --- Commun ---
HORIZON = 48          # bougies maximum dans un trade (TF d'entrée)
WINDOW = 300          # bougies utilisées pour calculer les signaux à chaque instant
BB_STD = 2.0
LEN = {"M5": pd.Timedelta(minutes=5), "M15": pd.Timedelta(minutes=15),
       "H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4),
       "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(weeks=1), "MN": pd.Timedelta(days=28)}


def load(pair, tf):
    """Bougies d'un timeframe, rééchantillonnées par le chemin unique du scan. None si indisponible."""
    interval, period, _ = core.TFS[tf]
    df = core._history_cached(pair, interval, period)
    if df is None:
        return None
    return core.resample_tf(df, tf)


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


def bands(sub):
    """(MID, bande haute, bande basse) de la dernière bougie fermée."""
    c = sub["Close"]
    mid = c.rolling(MID_PERIOD).mean().iloc[-1]
    sd = c.rolling(MID_PERIOD).std(ddof=0).iloc[-1]
    return float(mid), float(mid + BB_STD * sd), float(mid - BB_STD * sd)


def simulate(e, p, sens, entry, a):
    """Mode ATR : stop et cible fixes en ATR. Retourne (R, issue, indice de sortie) ou None."""
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


def plan_bbma(esub, sens, entry, a):
    """Plan de trade BBMA : (sl, tp1, tp2, mode_tp1) ou None si le trade n'est pas valide."""
    if a is None or a <= 0:
        return None
    sign = 1 if sens == "B" else -1
    anchor = core.cycle_anchor(esub)
    if anchor is None:
        tail = esub.iloc[-FALLBACK_BARS:]
        anchor = float(tail["Low"].min() if sign == 1 else tail["High"].max())
    sl = anchor - sign * SL_BUFFER_ATR * a
    risk = (entry - sl) * sign
    if risk <= 0 or risk > MAX_RISK_ATR * a:
        return None
    mid, top, low = bands(esub)
    tp2 = top if sign == 1 else low
    if (tp2 - entry) * sign < risk * MIN_TP1_R:        # la bande opposée est trop proche ou déjà dépassée
        return None
    if (mid - entry) * sign >= MIN_TP1_R * risk:
        tp1, mode = mid, "MID"
    else:
        tp1, mode = entry + sign * TP1_FALLBACK * (tp2 - entry), "FALLBACK"
    if (tp1 - entry) * sign >= (tp2 - entry) * sign:   # TP1 ne doit pas dépasser TP2
        tp1, mode = entry + sign * 0.5 * (tp2 - entry), "FALLBACK"
    return sl, tp1, tp2, mode


def simulate_bbma(e, p, sens, entry, sl, tp1, tp2):
    """Mode BBMA : TP1 partiel, puis break-even, puis TP2. Retourne (R, issue, indice de sortie) ou None."""
    end = min(p + HORIZON, len(e))
    if end <= p:
        return None
    sign = 1 if sens == "B" else -1
    risk = (entry - sl) * sign
    r1 = (tp1 - entry) * sign / risk
    r2 = (tp2 - entry) * sign / risk
    stop, got_tp1 = sl, False
    for j in range(p, end):
        hi, lo = e["High"].iloc[j], e["Low"].iloc[j]
        hit_sl = lo <= stop if sign == 1 else hi >= stop
        if hit_sl:                                      # prudent : le stop passe avant les cibles de la même bougie
            if got_tp1:
                return PARTIAL * r1, "TP1+BE", j
            return -1.0, "SL", j
        if not got_tp1 and ((hi >= tp1) if sign == 1 else (lo <= tp1)):
            got_tp1, stop = True, entry                 # break-even effectif dès la bougie suivante
        if got_tp1 and ((hi >= tp2) if sign == 1 else (lo <= tp2)):
            return PARTIAL * r1 + (1 - PARTIAL) * r2, "TP1+TP2", j
    j = end - 1
    rt = sign * (e["Close"].iloc[j] - entry) / risk
    if got_tp1:
        return PARTIAL * r1 + (1 - PARTIAL) * rt, "TP1+TIME", j
    return rt, "TIME", j


def backtest_pair(pair, base, tfs, start, mode="bbma", frames=None, skipped=None):
    """Liste des trades simulés pour une paire, à partir de l'instant start, selon le mode de sortie.
    skipped : liste (optionnelle) qui reçoit les décisions non tradées par le mode BBMA (risque trop grand, bande trop proche) ;
    un même signal persistant y apparaît une fois par bougie de base."""
    if frames is None:
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
        for tf in tfs:
            sub = frames[tf].closed_before(T)
            cells[tf] = core.detect(sub) if len(sub) >= core.MIN_BARS else (None, None)

        d1 = None
        d1sub = frames["D1"].closed_before(T)
        if len(d1sub) >= core.MIN_BARS:
            c = d1sub["Close"]
            d1 = "B" if c.iloc[-1] > c.ewm(span=50, adjust=False).mean().iloc[-1] else "S"

        tf_e, code_e, sens_e, setup = core.synthese_ex(cells, tfs, True, d1)
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
        conf = core.confiance_score(cells, tfs, sens_e)
        q_score, q_grade, _ = bq.qualite_entree(code_e, sens_e, conf, d1, feat)
        kind = bs.entry_kind(cells, tfs, tf_e, code_e)
        q_score, q_grade = bs.cap_grade(q_score, q_grade, kind)
        entry = float(e["Open"].iloc[p])

        extra = {}
        if mode == "atr":
            sim = simulate(e, p, sens_e, entry, a)
        else:
            plan = plan_bbma(esub, sens_e, entry, a)
            if plan is None:
                if skipped is not None:
                    skipped.append({"pair": pair, "time": T, "code": code_e, "setup": setup})
                continue
            sl, tp1, tp2, tp1_mode = plan
            sim = simulate_bbma(e, p, sens_e, entry, sl, tp1, tp2)
            extra = {"sl": sl, "tp1": tp1, "tp2": tp2, "tp1_mode": tp1_mode}
        if sim is None:
            continue
        r, outcome, j = sim
        busy_until = e.index[j]
        trades.append({"pair": pair, "mode": mode, "time": T, "tf_entry": tf_e, "code": code_e, "sens": sens_e,
                       "setup": setup, "grade": q_grade, "score": q_score, "kind": kind,
                       "R": r, "outcome": outcome, **extra})
    return trades


def _stats(label, s):
    n = len(s)
    if n == 0:
        return {"groupe": label, "n": 0}
    win = (s["R"] > 0).sum()
    gains, pertes = s.loc[s["R"] > 0, "R"].sum(), -s.loc[s["R"] < 0, "R"].sum()
    return {"groupe": label, "n": n,
            "gagnants %": round(100 * win / n, 1),
            "SL %": round(100 * (s["outcome"] == "SL").sum() / n, 1),
            "TIME %": round(100 * s["outcome"].str.contains("TIME").sum() / n, 1),
            "R moyen": round(s["R"].mean(), 2),
            "R total": round(s["R"].sum(), 1),
            "facteur de profit": round(gains / pertes, 2) if pertes > 0 else None}


def summarize(df, by="grade"):
    """Tableau de statistiques par groupe (grade par défaut), plus la ligne Tous."""
    order = ["A", "B", "C"] if by == "grade" else sorted(df[by].dropna().unique())
    rows = [_stats(g, df[df[by] == g]) for g in order]
    rows.append(_stats("Tous", df))
    return pd.DataFrame(rows)


def compare(df):
    """Comparaison des modes de sortie sur les mêmes règles de signal : une ligne par mode."""
    return pd.DataFrame([_stats(m, df[df["mode"] == m]) for m in sorted(df["mode"].unique())])


def main():
    global MAX_RISK_ATR
    parser = argparse.ArgumentParser(description="Backtest de la synthèse BBMA (sorties BBMA vs ATR).")
    parser.add_argument("pairs", nargs="+", help="paires, ex. EURUSD GBPUSD")
    parser.add_argument("--base", default="M15", help="TF de décision (doit être dans --tfs)")
    parser.add_argument("--tfs", default="D1,H4,H1,M15", help="timeframes séparés par des virgules")
    parser.add_argument("--months", type=int, default=3, help="durée du test en mois")
    parser.add_argument("--mode", choices=["bbma", "atr", "both"], default="both", help="règle de sortie")
    parser.add_argument("--max-risk", type=float, default=MAX_RISK_ATR,
                        help="mode bbma : risque maximal en ATR (stop derrière l'Extreme au-delà = trade ignoré)")
    parser.add_argument("--csv", default="bbma_backtest_trades.csv", help="fichier de sortie des trades")
    args = parser.parse_args()

    tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    base = args.base.upper()
    if base not in tfs:
        parser.error("--base doit faire partie de --tfs")
    unknown = [t for t in tfs + [base] if t not in core.TFS or t not in LEN]
    if unknown:
        parser.error(f"timeframe non supporté : {', '.join(unknown)}")
    modes = ["bbma", "atr"] if args.mode == "both" else [args.mode]

    start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=args.months)
    MAX_RISK_ATR = args.max_risk
    trades, skipped = [], []
    for pair in args.pairs:
        pair = pair.upper()
        frames = {}
        for tf in set(tfs) | {"D1"}:
            df = load(pair, tf)
            if df is None or len(df) < core.MIN_BARS:
                frames = None
                break
            frames[tf] = TFData(df, tf)
        if frames is None:
            print(f"{pair} : données indisponibles")
            continue
        for m in modes:
            found = backtest_pair(pair, base, tfs, start, m, frames, skipped if m == "bbma" else None)
            print(f"{pair} [{m}] : {len(found)} trades")
            trades += found

    if not trades:
        print("Aucun trade simulé. Vérifiez les paires, la connexion ou la durée du test.")
        return
    df = pd.DataFrame(trades)
    df.to_csv(args.csv, index=False)
    if skipped:
        sk = pd.DataFrame(skipped)
        print("\nDécisions ignorées par le mode bbma (une par bougie de base ; stop trop loin ou bande opposée trop proche) :")
        print(sk.groupby(["setup", "code"]).size().rename("n").reset_index().to_string(index=False))
    print(f"\nTrades enregistrés dans {args.csv}")
    if len(modes) > 1:
        print("\n=== Comparaison BBMA vs ATR (mêmes signaux) ===")
        print(compare(df).to_string(index=False))
    for m in modes:
        print(f"\n=== Mode {m} : par grade ===")
        print(summarize(df[df["mode"] == m]).to_string(index=False))
        print(f"--- Mode {m} : par setup ---")
        print(summarize(df[df["mode"] == m], by="setup").to_string(index=False))
    print("\nLecture : un grade est utile si son R moyen dépasse celui des grades inférieurs.")
    print("Avec moins de 30 trades par ligne, les écarts ne sont pas significatifs. "
          "Le mode BBMA ignore les trades dont le stop dépasse "
          f"{MAX_RISK_ATR} ATR : le nombre de trades diffère donc entre modes.")


if __name__ == "__main__":
    main()
