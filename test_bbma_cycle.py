"""Tests unitaires du moteur BBMA (cycle EXTREME -> MHV -> CSAK -> MOM -> RE).

Données entièrement déterministes. Deux niveaux :
  - tests de machine à états : indicateurs fixés bougie par bougie (_cycle), pour isoler chaque règle ;
  - tests de bout en bout : séquence OHLC complète (_signals), achat, miroir vente, causalité.
Lancer : python -m unittest discover -s tests -v   (depuis le dossier du moteur)
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bbma_core as core  # noqa: E402
import bbma_session as bs  # noqa: E402

# Indicateurs par défaut (sens achat) : MID 100, bande haute 104, bande basse 96,
# MA5/MA10 des plus hauts 102, MA5/MA10 des plus bas 98 (zone RE = 98..100 avec MID).
DEFAULT_IND = dict(MID=100.0, TOP=104.0, LOW=96.0, H5=102.0, H10=102.0, L5=98.0, L10=98.0)


def frame(bars, overrides=None):
    """bars : liste de (O, H, L, C). overrides : {index: {indicateur: valeur}}."""
    overrides = overrides or {}
    n = len(bars)
    f = {k: np.zeros(n) for k in ("O", "H", "L", "C", *DEFAULT_IND)}
    for i, (o, h, l, c) in enumerate(bars):
        f["O"][i], f["H"][i], f["L"][i], f["C"][i] = o, h, l, c
        vals = {**DEFAULT_IND, **overrides.get(i, {})}
        for k, v in vals.items():
            f[k][i] = v
    return f


def cycle(bars, overrides=None, win=core.CSAK_WINDOW):
    return core._cycle(frame(bars, overrides), win)


NEUTRAL = (100.0, 100.5, 99.5, 100.0)
EXT_BAR = (100.0, 100.0, 92.0, 94.0)                  # L5 forcé sous la bande basse
MHV_BAR = (94.0, 97.5, 93.5, 97.0)                    # mèche sous la bande, clôture revenue au-dessus
CSAK_BAR = (97.0, 103.5, 96.5, 103.0)                 # clôture > MID, > MA5/10 hauts
MOM_BAR = (103.0, 106.0, 102.5, 105.0)                # clôture > bande haute
REPULLBACK = (104.0, 104.5, 100.5, 101.0)             # repli baissier, hors zone
RE_BAR = (100.0, 101.5, 98.0, 101.0)                  # mèche basse (57 %) touche la zone, clôture haussière
OVR_EXT = {0: {"L5": 95.0}}                           # EXT à la bougie 0


class TestCycleStateMachine(unittest.TestCase):
    def test_full_buy_cycle_emits_each_stage_in_order(self):
        bars = [EXT_BAR, MHV_BAR, CSAK_BAR, MOM_BAR, REPULLBACK, RE_BAR]
        out = cycle(bars, {0: {"L5": 95.0}, 1: {"L5": 95.0}})
        self.assertEqual(np.flatnonzero(out["MHV"]).tolist(), [1])
        self.assertEqual(np.flatnonzero(out["CSAK"]).tolist(), [2])
        self.assertEqual(np.flatnonzero(out["MOM"]).tolist(), [3])
        self.assertEqual(np.flatnonzero(out["RE"]).tolist(), [5])

    def test_ext_is_not_an_output_signal_but_is_reported_as_alert(self):
        out = cycle([EXT_BAR], {0: {"L5": 95.0}})
        self.assertTrue(out["EXT"][0])
        self.assertFalse(any(out[k][0] for k in ("MHV", "CSAK", "MOM", "RE")))

    def test_double_csak_is_blocked(self):
        bars = [EXT_BAR, MHV_BAR, CSAK_BAR, (103.0, 104.0, 102.8, 103.5)]  # 2e bougie : CSAK aussi
        out = cycle(bars, {0: {"L5": 95.0}, 1: {"L5": 95.0}})
        self.assertEqual(int(out["CSAK"].sum()), 1)
        self.assertFalse(out["CSAK"][3])

    def test_mom_without_csak_is_rejected(self):
        # EXT puis clôture au-delà de la bande haute sans MHV préalable
        bars = [EXT_BAR, (97.0, 106.0, 97.0, 105.0)]           # L=97 > LOW : pas de MHV
        out = cycle(bars, {0: {"L5": 95.0}})
        self.assertFalse(out["MOM"].any())
        self.assertFalse(out["CSAK"].any())

    def test_re_without_mom_is_rejected(self):
        # EXT, MHV, CSAK puis configuration RE : sans MOM, aucun RE
        bars = [EXT_BAR, MHV_BAR, CSAK_BAR, RE_BAR]
        out = cycle(bars, {0: {"L5": 95.0}, 1: {"L5": 95.0}})
        self.assertFalse(out["RE"].any())

    def test_mid_cross_in_wrong_direction_resets_cycle(self):
        bars = [EXT_BAR, MHV_BAR, CSAK_BAR, (101.0, 101.0, 98.5, 99.0), MOM_BAR]  # clôture 99 < MID
        out = cycle(bars, {0: {"L5": 95.0}, 1: {"L5": 95.0}})
        self.assertFalse(out["MOM"].any())

    def test_expired_step_resets_cycle(self):
        bars = [EXT_BAR] + [NEUTRAL] * 19 + [MHV_BAR]          # MHV 20 bougies après l'EXT (> fenêtre 15)
        out = cycle(bars, {0: {"L5": 95.0}, 20: {"L5": 95.0}})
        self.assertFalse(out["MHV"].any())


class TestRentry(unittest.TestCase):
    def _cycle_then(self, mid_bar, overrides_extra=None):
        bars = [EXT_BAR, MHV_BAR, CSAK_BAR, MOM_BAR, REPULLBACK, mid_bar]
        ov = {0: {"L5": 95.0}, 1: {"L5": 95.0}}
        ov.update(overrides_extra or {})
        return cycle(bars, ov)

    def test_valid_re_with_rejection_fires(self):
        self.assertTrue(self._cycle_then(RE_BAR)["RE"][5])

    def test_touch_without_rejection_wick_does_not_fire(self):
        # contact de la zone, bougie haussière, mais mèche basse insignifiante (0.2 sur 3.2)
        no_wick = (98.2, 101.2, 98.0, 101.0)
        self.assertFalse(self._cycle_then(no_wick)["RE"][5])

    def test_touch_and_close_below_zone_does_not_fire(self):
        # le prix traverse la zone et clôture en dessous : pas de rejet
        broke = (99.0, 99.2, 96.5, 97.5)
        self.assertFalse(self._cycle_then(broke)["RE"][5])

    def test_bearish_bar_in_zone_does_not_fire(self):
        bearish = (101.0, 101.5, 98.0, 99.0)                   # clôture < ouverture
        self.assertFalse(self._cycle_then(bearish)["RE"][5])


class TestEndToEnd(unittest.TestCase):
    @staticmethod
    def base(n=40):
        out, prev = [], 100.0
        for i in range(n):
            c = 99.0 if i % 2 == 0 else 101.0
            out.append((prev, max(prev, c) + 0.5, min(prev, c) - 0.5, c))
            prev = c
        return out

    BUY = [(99.0, 99.5, 93.0, 94.0), (94.0, 96.5, 91.0, 96.0), (96.0, 104.0, 95.5, 103.5),
           (103.5, 107.0, 103.0, 106.5), (106.5, 109.5, 106.0, 108.8),
           (108.8, 108.9, 104.5, 105.0), (104.0, 106.0, 101.5, 105.8)]

    @staticmethod
    def mirror(bars):
        return [(200 - o, 200 - l, 200 - h, 200 - c) for (o, h, l, c) in bars]

    @staticmethod
    def make(bars):
        o, h, l, c = zip(*bars)
        idx = pd.date_range("2024-01-01", periods=len(bars), freq="h", tz="UTC")
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)

    def test_buy_end_to_end_sequence(self):
        df = self.make(self.base() + self.BUY)
        sig = core._signals(df)
        fired = [(code, i) for code in ("MHV", "CSAK", "MOM", "RE") for i in np.flatnonzero(sig[code][0].values)]
        base_n = 40
        self.assertIn(("MHV", base_n + 2), fired)
        self.assertIn(("CSAK", base_n + 3), fired)
        self.assertIn(("MOM", base_n + 4), fired)
        self.assertIn(("RE", base_n + 6), fired)
        self.assertEqual(sum(1 for c, _ in fired if c == "CSAK"), 1)

    def test_sell_is_exact_mirror_of_buy(self):
        buy = self.make(self.base() + self.BUY)
        sell = self.make(self.mirror(self.base()) + self.mirror(self.BUY))
        sb, ss = core._signals(buy), core._signals(sell)
        for code in ("MHV", "CSAK", "MOM", "RE"):
            np.testing.assert_array_equal(sb[code][0].values, ss[code][1].values, err_msg=code)

    def test_signals_are_causal_no_lookahead(self):
        df = self.make(self.base() + self.BUY)
        full = core._signals(df)
        for k in (45, 47, 60):
            part = core._signals(df.iloc[:k])
            for code in ("MHV", "CSAK", "MOM", "RE"):
                np.testing.assert_array_equal(full[code][0].values[:k], part[code][0].values, err_msg=f"{code}@{k}")

    def test_deterministic(self):
        df = self.make(self.base() + self.BUY)
        a, b = core._signals(df), core._signals(df)
        for code in a:
            pd.testing.assert_series_equal(a[code][0], b[code][0])


class TestPriorityAndSynthesis(unittest.TestCase):
    def test_re_is_not_masked_by_mom(self):
        sig = {c: (pd.Series([False]), pd.Series([False])) for c in ("MOM", "RE", "CSAK", "MHV", "EXT", "EXM")}
        sig["MOM"] = (pd.Series([True]), pd.Series([False]))
        sig["RE"] = (pd.Series([True]), pd.Series([False]))
        self.assertEqual(core._at(sig, -1), ("RE", "B"))

    def test_ext_and_exm_are_never_entries(self):
        sig = {c: (pd.Series([False]), pd.Series([False])) for c in ("MOM", "RE", "CSAK", "MHV", "EXT", "EXM")}
        sig["EXT"] = (pd.Series([True]), pd.Series([False]))
        sig["EXM"] = (pd.Series([True]), pd.Series([False]))
        self.assertEqual(core._at(sig, -1), (None, None))

    def test_synthese_ignores_ext_on_lower_tf(self):
        cells = {"D1": ("MOM", "B"), "H1": ("EXT", "B")}
        self.assertEqual(core.synthese(cells, ["D1", "H1"], False, None), ("D1", "MOM", "B"))

    def test_d1_filter_applies_to_re_but_not_to_mhv(self):
        cells = {"D1": ("MOM", "B"), "H1": ("MHV", "B")}
        self.assertEqual(core.synthese(cells, ["D1", "H1"], True, "S"), ("H1", "MHV", "B"))
        cells = {"D1": ("MOM", "B"), "H1": ("RE", "B")}
        self.assertEqual(core.synthese(cells, ["D1", "H1"], True, "S"), (None, None, None))
        self.assertEqual(core.synthese(cells, ["D1", "H1"], True, "B"), ("H1", "RE", "B"))

    def test_detect_all_reports_simultaneous_codes(self):
        df = TestEndToEnd.make(TestEndToEnd.base() + TestEndToEnd.BUY)
        self.assertIn(("RE", "B"), core.detect_all(df))


class TestH4SingleSource(unittest.TestCase):
    def test_h4_session_is_utc_and_shared(self):
        self.assertEqual(core.H4_SESSION, "UTC")
        self.assertIs(bs.resample_h4, core.resample_h4)

    def test_closed_bars_h4_matches_utc_resample(self):
        idx = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
        raw = pd.DataFrame({"Open": np.arange(400.0), "High": np.arange(400.0) + 1,
                            "Low": np.arange(400.0) - 1, "Close": np.arange(400.0)}, index=idx)
        data = {("EURUSD", "1h", "180d"): raw}
        mine = core._closed_bars("EURUSD", "H4", data)
        ref = raw.resample("4h").agg(core.AGG).dropna().iloc[:-1]
        pd.testing.assert_frame_equal(mine, ref)
        pd.testing.assert_frame_equal(bs.closed_bars("EURUSD", "H4", data), mine)


if __name__ == "__main__":
    unittest.main()
