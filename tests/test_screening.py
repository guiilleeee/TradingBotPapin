import yaml
import pytest

import equity_universe
import screening

# ---------------------------------------------------------------- file writer

def test_writer_produces_exactly_target_symbols(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out),
        equity_symbols=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        equity_scored=[{"symbol": s, "score": 0.5} for s in "ABCDEFGHIJ"],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    symbols = doc["symbols"]
    assert len(symbols) == 10
    equities = [s for s in symbols if s["asset_class"] == "equity"]
    assert len(equities) == 10

def test_writer_output_is_shaped_for_mains_loader(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"],
        [{"symbol": "AAPL", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    for entry in doc["symbols"]:
        assert set(entry.keys()) == {"symbol", "asset_class"}

def test_writer_only_ever_contributes_the_symbols_key(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"],
        [{"symbol": "AAPL", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(doc.keys()) >= {"symbols"}
    assert "risk" not in doc and "live_execution" not in doc

def test_writer_is_atomic_no_temp_file_left_behind(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"],
        [{"symbol": "AAPL", "score": 0.1}],
    )
    assert out.exists()
    assert not (tmp_path / "symbols.yaml.tmp").exists()

def test_writer_overwrites_a_stale_file_cleanly(tmp_path):
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols:\n  - symbol: OLD\n    asset_class: equity\n", encoding="utf-8")
    screening._write_symbols_file(
        str(out), ["NEW"],
        [{"symbol": "NEW", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "OLD" not in [s["symbol"] for s in doc["symbols"]]
    assert "NEW" in [s["symbol"] for s in doc["symbols"]]

# ------------------------------------------------------------- orchestration

@pytest.fixture
def stub_equity_side(monkeypatch):
    """A healthy, signal-bearing equity universe."""
    universe = [f"SYM{i}" for i in range(30)]
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: universe)
    monkeypatch.setattr(
        equity_universe, "fetch_universe_price_data",
        lambda symbols: {
            s: {"price_change_pct": 1.0, "volume": 1e7} for s in symbols
        },
    )

def test_run_screening_writes_symbols_on_a_healthy_run(tmp_path, stub_equity_side):
    out = tmp_path / "symbols.yaml"
    rc = screening.run_screening(str(out))
    assert rc == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(doc["symbols"]) == 25
    # Market-cap order (the universe's own order) is preserved, not score order.
    assert [e["symbol"] for e in doc["symbols"]] == [f"SYM{i}" for i in range(25)]
    assert all(e["asset_class"] == "equity" for e in doc["symbols"])

def test_run_screening_fails_without_writing_when_equity_universe_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: {"AAPL"})
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: equity}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert "OLD" in out.read_text(encoding="utf-8")

def test_run_screening_leaves_the_file_untouched_on_an_unexpected_exception(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("something in FMP parsing broke")

    monkeypatch.setattr(equity_universe, "build_equity_universe", boom)
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: equity}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert "OLD" in out.read_text(encoding="utf-8")

def test_run_screening_never_touches_the_file_on_first_failed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: set())
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert not out.exists()
