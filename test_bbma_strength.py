"""Tests de bbma_strength (force Buy/Sell générale TradingView) sans accès réseau."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bbma_strength as bst  # noqa: E402


class FakeAnalysis:
    def __init__(self, buy, sell, neutral):
        self.summary = {"RECOMMENDATION": "X", "BUY": buy, "SELL": sell, "NEUTRAL": neutral}


class FakeInterval:
    INTERVAL_15_MINUTES = "15m"
    INTERVAL_1_HOUR = "1h"
    INTERVAL_4_HOURS = "4h"
    INTERVAL_1_DAY = "1d"
    INTERVAL_1_WEEK = "1W"


class Patch:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.old = (bst.get_multiple_analysis, bst.Interval)
        bst.get_multiple_analysis, bst.Interval = self.fn, FakeInterval

    def __exit__(self, *a):
        bst.get_multiple_analysis, bst.Interval = self.old


class TestScoring(unittest.TestCase):
    def test_interval_score(self):
        self.assertEqual(bst.interval_score({"BUY": 12, "SELL": 4, "NEUTRAL": 4}), (40.0, 12, 4))
        self.assertIsNone(bst.interval_score({"BUY": 0, "SELL": 0, "NEUTRAL": 0}))
        self.assertIsNone(bst.interval_score(None))

    def test_aggregate_all_buy_and_all_sell(self):
        buy = {tf: (100.0, 10, 0) for tf, _ in bst.INTERVALS}
        sell = {tf: (-100.0, 0, 10) for tf, _ in bst.INTERVALS}
        g = bst.aggregate(buy)
        self.assertEqual((g["score"], g["label"], g["buy_pct"]), (100, "Achat fort", 100))
        g = bst.aggregate(sell)
        self.assertEqual((g["score"], g["label"], g["buy_pct"]), (-100, "Vente forte", 0))

    def test_higher_timeframes_weigh_more(self):
        per = {"M15": (-100.0, 0, 10), "H1": (-100.0, 0, 10), "D1": (100.0, 10, 0), "W1": (100.0, 10, 0)}
        self.assertGreater(bst.aggregate(per)["score"], 0)

    def test_too_few_intervals_gives_none(self):
        per = {"D1": (50.0, 5, 1), "W1": (50.0, 5, 1)}
        self.assertIsNone(bst.aggregate(per))

    def test_gauge_cell_accepts_result(self):
        import bbma_gauge as bg
        g = bst.aggregate({tf: (50.0, 6, 2) for tf, _ in bst.INTERVALS})
        self.assertIn("Achat", bg.gauge_cell(g))
        self.assertIn("–", bg.gauge_cell(None))


class TestFetch(unittest.TestCase):
    def test_fetch_ok_and_routing(self):
        calls = []

        def fake(screener, interval, symbols, timeout=None):
            calls.append((screener, interval, tuple(symbols)))
            return {s: FakeAnalysis(12, 4, 4) for s in symbols}

        with Patch(fake):
            out = bst.fetch_strength(["EURUSD", "XAUUSD", "BTCUSD"])
        for p in ("EURUSD", "XAUUSD", "BTCUSD"):
            self.assertEqual(out[p]["label"], "Achat")
        syms = {s for _, _, ss in calls for s in ss}
        self.assertIn("FX_IDC:EURUSD", syms)
        self.assertIn("OANDA:XAUUSD", syms)
        self.assertIn("BITSTAMP:BTCUSD", syms)
        self.assertIn("crypto", {c[0] for c in calls})

    def test_fallback_to_next_exchange(self):
        def fake(screener, interval, symbols, timeout=None):
            return {s: FakeAnalysis(2, 10, 4) for s in symbols if s.startswith("OANDA:")}

        with Patch(fake):
            out = bst.fetch_strength(["EURUSD"])
        self.assertEqual(out["EURUSD"]["label"], "Vente")

    def test_network_error_gives_none_not_exception(self):
        def fake(**kw):
            raise RuntimeError("réseau")

        with Patch(fake):
            self.assertEqual(bst.fetch_strength(["EURUSD", "GBPUSD"]), {"EURUSD": None, "GBPUSD": None})

    def test_missing_package_gives_none(self):
        old = bst.get_multiple_analysis
        bst.get_multiple_analysis = None
        try:
            self.assertEqual(bst.fetch_strength(["EURUSD"]), {"EURUSD": None})
        finally:
            bst.get_multiple_analysis = old

    def test_unknown_pair_is_none_others_kept(self):
        def fake(screener, interval, symbols, timeout=None):
            return {s: FakeAnalysis(12, 4, 4) for s in symbols if "EURUSD" in s}

        with Patch(fake):
            out = bst.fetch_strength(["EURUSD", "ZZZUSD"])
        self.assertIsNotNone(out["EURUSD"])
        self.assertIsNone(out["ZZZUSD"])


if __name__ == "__main__":
    unittest.main()
