"""Non-régression : l'ancien code (app_original.py, chargé tel quel) doit donner les mêmes résultats
que le nouveau code, à données identiques. yfinance est remplacé par un jeu de données figé."""
import sys, zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bbma_core as core  # noqa: E402

ORIG = ROOT / "app_original.py"


# --- Chargement de l'ancien code -------------------------------------------------
class _StubST:
    """Remplace Streamlit : le décorateur cache_data devient neutre."""
    def set_page_config(self, *a, **k): pass
    def title(self, *a, **k): pass
    def cache_data(self, *a, **k): return lambda f: f


def load_reference():
    src = ORIG.read_text(encoding="utf-8")
    head = src.split("pairs = list(dict.fromkeys", 1)[0]   # tout ce qui précède l'interface
    # Sans cela, l'import réel de streamlit écrase le stub et son cache_data réel
    # renvoie des résultats mis en cache d'un test précédent.
    head = head.replace("streamlit as st, ", "", 1)
    assert "import streamlit" not in head
    ns = {"st": _StubST(), "__name__": "bbma_ref"}
    exec(compile(head, str(ORIG), "exec"), ns)
    return ns


REF = load_reference()


# --- Données figées ---------------------------------------------------------------
N_BARS = {("1mo", "10y"): 120, ("1wk", "5y"): 260, ("1d", "2y"): 500,
          ("1h", "180d"): 4320, ("1h", "60d"): 1440, ("15m", "30d"): 2880, ("5m", "30d"): 8640}
FREQ = {"1mo": "MS", "1wk": "W-MON", "1d": "B", "1h": "h", "15m": "15min", "5m": "5min"}
PAIRS = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURJPY GBPJPY EURGBP XAUUSD BTCUSD".split()

# Pannes simulées : vide, exception, None
FAIL_EMPTY = {("USDJPY=X", "1h", "180d")}
FAIL_EXC = {("GBPJPY=X", "5m", "30d")}
FAIL_NONE = {("EURUSD=X", "1d", "2y")}


@lru_cache(maxsize=None)
def series(symbol, interval, period):
    seed = zlib.crc32(f"{symbol}|{interval}|{period}".encode())
    rng = np.random.default_rng(seed)
    n = N_BARS[(interval, period)]
    vol = rng.uniform(0.0005, 0.004)
    close = 1.1 * np.exp(np.cumsum(rng.standard_normal(n) * vol))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq=FREQ[interval], tz="America/New_York")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1, 1000, n)}, index=idx)


CALLS = []
FAILS = set()


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period, interval):
        CALLS.append((self.symbol, interval, period))
        key = (self.symbol, interval, period)
        if key in FAILS & FAIL_EMPTY:
            return pd.DataFrame()
        if key in FAILS & FAIL_EXC:
            raise RuntimeError("limite de débit simulée")
        if key in FAILS & FAIL_NONE:
            return None
        return series(*key).copy()


@pytest.fixture(autouse=True)
def fake_yf(monkeypatch):
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(core, "PAUSE", 0)
    CALLS.clear()
    FAILS.clear()
    yield


def with_failures(enabled):
    if enabled:
        FAILS.update(FAIL_EMPTY | FAIL_EXC | FAIL_NONE)


# --- Tests -------------------------------------------------------------------------
def test_wma_identique_a_l_original():
    rng = np.random.default_rng(0)
    for n in (5, 10):
        s = pd.Series(rng.normal(1.1, 0.01, 500))
        s.iloc[[3, 200]] = np.nan                         # trous de données
        ref = REF["wma"](s, n)
        new = core.wma(s, n)
        np.testing.assert_allclose(new.to_numpy(), ref.to_numpy(), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("tfs", [
    ["MN", "W1", "D1", "H4", "H1", "M15", "M5"],   # tous les TF
    ["H4", "H1"],                                  # sous-ensemble
    ["M5"],
])
@pytest.mark.parametrize("failures", [False, True])
def test_scan_identique_avec_filtre_d1_d_origine(tfs, failures, monkeypatch):
    """Avec le filtre D1 d'origine, tout doit être strictement identique (refactor + perf)."""
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", False)
    with_failures(failures)
    ref_res, ref_tr, ref_failed, ref_total, _ = REF["run_scan"](tuple(PAIRS), tuple(tfs))
    new_res, new_tr, new_failed, new_total, _ = core.scan(PAIRS, tfs, trend=True)
    assert new_res == ref_res
    assert new_tr == ref_tr
    assert new_failed == ref_failed
    assert new_total == ref_total


def test_scan_sans_filtre_identique_pour_les_tf(monkeypatch):
    """trend=False ne change pas les signaux par TF."""
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", False)
    tfs = ["H4", "H1", "M15", "M5"]
    ref_res, *_ = REF["run_scan"](tuple(PAIRS), tuple(tfs))
    new_res, new_tr, *_ = core.scan(PAIRS, tfs, trend=False)
    assert new_res == ref_res
    assert new_tr == {}


def test_signaux_couverts_par_le_jeu_de_test(monkeypatch):
    """Garde-fou : le jeu figé doit produire des signaux variés, sinon les tests ne prouvent rien."""
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", False)
    res, *_ = core.scan(PAIRS, ["MN", "W1", "D1", "H4", "H1", "M15", "M5"], trend=True)
    codes = {code for code, _ in res.values() if code}
    sens = {s for _, s in res.values() if s}
    print("\nCodes observés :", sorted(codes), "| sens :", sorted(sens))
    assert len(codes) >= 3 and sens == {"B", "S"}


def test_nombre_de_requetes_reduit(monkeypatch):
    """Le D1 n'est plus téléchargé deux fois : 7 requêtes par paire au lieu de 8."""
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", False)
    tfs = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]

    CALLS.clear(); REF["run_scan"](tuple(PAIRS), tuple(tfs)); old = len(CALLS)
    CALLS.clear(); core.scan(PAIRS, tfs, trend=True); new = len(CALLS)
    print(f"\nRequêtes par scan complet : ancien={old}, nouveau={new}")
    assert new == old - len(PAIRS)


def test_filtre_d1_corrige_ne_change_que_la_tendance(monkeypatch):
    """Correction 2 : seul le filtre D1 change. Les signaux par TF restent identiques,
    et seules les entrées BBMA dépendant du filtre peuvent différer."""
    tfs = ["MN", "W1", "D1", "H4", "H1", "M15", "M5"]
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", False)
    old_res, old_tr, *_ = core.scan(PAIRS, tfs, trend=True)
    monkeypatch.setattr(core, "D1_FILTER_CLOSED_ONLY", True)
    new_res, new_tr, *_ = core.scan(PAIRS, tfs, trend=True)

    assert new_res == old_res                       # signaux par TF : inchangés
    changed = [p for p in PAIRS if old_tr.get(p) != new_tr.get(p)]
    print(f"\nTendance D1 modifiée pour : {changed or 'aucune paire'}")
    for p in PAIRS:
        ref_entry = core.synthese({t: old_res[(p, t)] for t in tfs}, tfs, True, old_tr.get(p))
        new_entry = core.synthese({t: new_res[(p, t)] for t in tfs}, tfs, True, new_tr.get(p))
        if ref_entry != new_entry:
            print(f"  {p} : entrée {ref_entry} -> {new_entry}")
