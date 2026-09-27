"""Display-only history: the equity curve's funding start, and "Esborrar historial"."""

import csv
import json
import sqlite3

import pytest

import benchmark
import execution
import main
from logger import EQUITY_CURVE_START_KEY
from tests.test_logger import signal_input, trade_signal


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")


# ---------------------------------------------------------- real vs fallback


def test_read_live_equity_is_none_when_the_read_fails(alpaca_env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(execution.requests, "get", boom)
    assert execution.read_live_equity() is None
    assert execution.fetch_live_equity(1000.0) == 1000.0


def test_read_live_equity_is_none_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    assert execution.read_live_equity() is None


def test_read_live_equity_returns_a_real_zero(alpaca_env, monkeypatch):
    monkeypatch.setattr(execution.requests, "get", lambda *a, **k: FakeResponse({"equity": "0"}))
    assert execution.read_live_equity() == 0.0


# ---------------------------------------------------------- funding marker


@pytest.mark.parametrize("equity", [None, 0.0, 0.01, 9.99])
def test_no_funding_is_recorded_below_the_threshold_or_on_a_failed_read(tmp_logger, equity):
    assert tmp_logger.record_funding_if_first(equity, 10.0) is None
    assert tmp_logger.get_meta(EQUITY_CURVE_START_KEY) is None


def test_the_first_funded_read_sets_the_start_once(tmp_logger):
    first = tmp_logger.record_funding_if_first(10.0, 10.0)
    assert first is not None
    assert tmp_logger.get_meta(EQUITY_CURVE_START_KEY) == first
    # Later deposits, or dropping to zero and refunding, never move it.
    assert tmp_logger.record_funding_if_first(0.0, 10.0) is None
    assert tmp_logger.record_funding_if_first(5000.0, 10.0) is None
    assert tmp_logger.get_meta(EQUITY_CURVE_START_KEY) == first


def test_a_failed_read_in_the_cycle_never_counts_as_funding(tmp_logger, monkeypatch):
    """The fallback ($1,000) is exactly the fake balance the marker hides."""
    monkeypatch.setattr(execution, "read_live_equity", lambda: None)
    assert main._live_equity(tmp_logger, 1000.0, 10.0) == 1000.0
    assert tmp_logger.get_meta(EQUITY_CURVE_START_KEY) is None


def test_the_cycle_marks_funding_on_a_real_balance(tmp_logger, monkeypatch):
    monkeypatch.setattr(execution, "read_live_equity", lambda: 250.0)
    assert main._live_equity(tmp_logger, 1000.0, 10.0) == 250.0
    assert tmp_logger.get_meta(EQUITY_CURVE_START_KEY) is not None


def test_benchmark_json_carries_the_curve_start(tmp_logger, tmp_path):
    assert benchmark.build_series(tmp_logger)["equity_curve_start"] is None
    tmp_logger.record_funding_if_first(100.0, 10.0)
    path = tmp_path / "benchmark.json"
    benchmark.export_benchmark_json(tmp_logger, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["equity_curve_start"] == tmp_logger.get_meta(EQUITY_CURVE_START_KEY)


def test_shipped_config_has_the_funding_threshold():
    import yaml

    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        assert yaml.safe_load(handle)["funding_threshold_usd"] == 10.0


# ---------------------------------------------------------- clear history


def _log_at(tmp_logger, symbol, timestamp):
    row_id = tmp_logger.log_signal(symbol, signal_input(symbol), None, trade_signal(symbol))
    conn = sqlite3.connect(tmp_logger.db_path)
    with conn:
        conn.execute("UPDATE signals SET timestamp = ? WHERE id = ?", (timestamp, row_id))
    conn.close()


def _csv_symbols(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [r["symbol"] for r in csv.DictReader(handle)]


def test_export_leaves_rows_before_the_cutoff_out_of_the_csv_only(tmp_logger, tmp_path, read_signals):
    _log_at(tmp_logger, "OLD", "2026-09-01T10:00:00.123456+00:00")
    _log_at(tmp_logger, "EDGE", "2026-09-20T12:00:00.000001+00:00")
    _log_at(tmp_logger, "NEW", "2026-09-25T10:00:00+00:00")
    path = tmp_path / "signals.csv"

    count = tmp_logger.export_signals_csv(str(path), since="2026-09-20T12:00:00+00:00")

    assert count == 2
    assert _csv_symbols(path) == ["EDGE", "NEW"]
    # Display only: the database keeps every row.
    assert len(read_signals()) == 3


def test_export_without_a_cutoff_is_unchanged(tmp_logger, tmp_path):
    _log_at(tmp_logger, "OLD", "2026-09-01T10:00:00+00:00")
    path = tmp_path / "signals.csv"
    assert tmp_logger.export_signals_csv(str(path)) == 1
    assert tmp_logger.export_signals_csv(str(path), since="not a date") == 1


@pytest.mark.parametrize("content,expected", [
    (None, None),                                              # no file
    ("not json", None),
    ("[]", None),
    ('{"cleared_at": null}', None),
    ('{"cleared_at": "yesterday"}', None),
    ('{"cleared_at": "2026-09-27T10:00:00+00:00"}', "2026-09-27T10:00:00+00:00"),
])
def test_history_cleared_at_reads_the_reset_file_and_never_raises(tmp_path, content, expected):
    path = tmp_path / "history_reset.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert main.history_cleared_at(str(path)) == expected


def test_the_workflow_writes_only_the_reset_file():
    import yaml

    with open(".github/workflows/clear_history.yml", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    steps = "\n".join(s.get("run", "") for s in workflow["jobs"]["clear"]["steps"])
    assert "git add -f docs/history_reset.json" in steps
    assert "trading_bot.db" not in steps and "signals.csv" not in steps
    assert main.DEFAULT_HISTORY_RESET_PATH == "docs/history_reset.json"
