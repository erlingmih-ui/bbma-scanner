"""Test de bout en bout de l'interface : rendu, cache à deux niveaux, scans partiels non mis en cache."""
import sys
from pathlib import Path

import pytest
import streamlit as st
import yfinance
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bbma_core as core  # noqa: E402
import test_golden as g  # noqa: E402  (FakeTicker, série figée, pannes)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(yfinance, "Ticker", g.FakeTicker)
    monkeypatch.setattr(core, "PAUSE", 0)
    g.CALLS.clear()
    g.FAILS.clear()
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def run_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    return at


def test_rendu_sans_exception():
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "bb-wrap" in html and "EURUSD" in html
    assert "Entrée BBMA" in html and "Confiance" in html


def test_second_cycle_sans_telechargement():
    """Deux rendus rapprochés : le second doit être servi par le cache (0 requête)."""
    run_app()
    first = len(g.CALLS)
    assert first > 0
    g.CALLS.clear()
    run_app()
    assert len(g.CALLS) == 0


def test_scan_partiel_non_mis_en_cache():
    """Un téléchargement en échec doit être réessayé au cycle suivant."""
    g.FAILS.update(g.FAIL_EMPTY)
    at = run_app()
    assert not at.exception
    assert any("téléchargements sans données" in w.value for w in at.warning)
    g.CALLS.clear()
    g.FAILS.clear()                      # la source revient
    at = run_app()
    assert not at.exception
    assert len(g.CALLS) > 0              # rescanné, pas servi par le cache
    assert not any("téléchargements sans données" in w.value for w in at.warning)
