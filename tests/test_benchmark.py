"""SPY benchmark: snapshots per cycle, rebased to the first live trade."""

import json
import sqlite3

import pytest

import benchmark
from models import ExecutionResult, TradeSignal


def _set_timestamps(db_path, table, timestamps):
    conn = sqlite3.connect(db_path)
    ids = [r[0] for r in conn.execute(f"SELECT id FROM {table} ORDER BY id")]
    for row_id, ts in zip(ids, timestamps):
        conn.execute(f"UPDATE {table} SET timestamp = ? WHERE id = ?", (ts, row_id))
    conn.commit()
    conn.close()


def _log_live_buy(tmp_logger, status="success"):
    final = TradeSignal(symbol="AAPL", action="buy", confidence=0.9, position_size_pct=10.0,
                        stop_loss_price=95.0, take_profit_price=115.0, reasoning="x", raw_action="buy")
    result = ExecutionResult(status=status, qty=1.0, fill_price=100.0, message="ok") \
        if status == "success" else ExecutionResult(status=status, message="no")
    tmp_logger.log_signal("AAPL", None, None, final, result, is_live=True)


def test_no_live_trade_yet_means_no_series(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "SPY", 500.0, is_live=True)
    payload = benchmark.build_series(tmp_logger)
    assert payload["first_live_trade_at"] is None
    assert payload["points"] == []


def test_series_is_rebased_to_the_snapshot_at_the_first_live_trade(tmp_logger):
    for equity, price in [(900.0, 480.0), (1000.0, 500.0), (1100.0, 505.0), (1050.0, 525.0)]:
        tmp_logger.record_benchmark_snapshot(equity, "SPY", price, is_live=True)
    _set_timestamps(tmp_logger.db_path, "benchmark_snapshots", [
        "2026-10-01T14:45:00+00:00", "2026-10-02T14:45:00+00:00",
        "2026-10-02T17:30:00+00:00", "2026-10-03T14:45:00+00:00",
    ])
    _log_live_buy(tmp_logger)
    _set_timestamps(tmp_logger.db_path, "signals", ["2026-10-02T14:46:10+00:00"])

    payload = benchmark.build_series(tmp_logger)

    assert payload["baseline"]["equity"] == 1000.0
    assert payload["baseline"]["benchmark_price"] == 500.0
    pts = payload["points"]
    assert [p["t"][:16] for p in pts] == ["2026-10-02T14:45", "2026-10-02T17:30", "2026-10-03T14:45"]
    assert pts[0]["bot_pct"] == 0.0 and pts[0]["benchmark_pct"] == 0.0
    assert pts[1]["bot_pct"] == pytest.approx(10.0)
    assert pts[1]["benchmark_pct"] == pytest.approx(1.0)
    assert pts[2]["bot_pct"] == pytest.approx(5.0)
    assert pts[2]["benchmark_pct"] == pytest.approx(5.0)


def test_failed_live_orders_and_simulation_do_not_start_the_clock(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "SPY", 500.0, is_live=True)
    _log_live_buy(tmp_logger, status="error")
    assert benchmark.build_series(tmp_logger)["first_live_trade_at"] is None


def test_simulation_snapshots_are_excluded(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "SPY", 500.0, is_live=True)
    tmp_logger.record_benchmark_snapshot(9999.0, "SPY", 500.0, is_live=False)
    _log_live_buy(tmp_logger)
    assert all(p["equity"] != 9999.0 for p in benchmark.build_series(tmp_logger)["points"])


def test_snapshots_without_a_spy_price_are_skipped(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "SPY", 500.0, is_live=True)
    tmp_logger.record_benchmark_snapshot(1010.0, "SPY", None, is_live=True)
    _log_live_buy(tmp_logger)
    assert len(benchmark.build_series(tmp_logger)["points"]) == 1


def test_record_snapshot_pairs_equity_with_the_fetched_price(tmp_logger, monkeypatch):
    monkeypatch.setattr(benchmark, "fetch_benchmark_price", lambda symbol="SPY": 512.5)
    benchmark.record_snapshot(tmp_logger, 1234.0, is_live=True)
    snap = tmp_logger.get_benchmark_snapshots(is_live=True)[0]
    assert snap["equity"] == 1234.0 and snap["benchmark_price"] == 512.5


def test_export_writes_the_dashboard_file(tmp_logger, tmp_path):
    path = tmp_path / "benchmark.json"
    benchmark.export_benchmark_json(tmp_logger, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["benchmark_symbol"] == "SPY"
    assert payload["points"] == []


def test_cycle_records_a_snapshot_and_exports(tmp_logger, tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(benchmark, "fetch_benchmark_price", lambda symbol="SPY": 500.0)
    monkeypatch.setattr(main, "_process_symbol", lambda **kw: None)
    config = {
        "symbols": [], "fallback_equity_usd": 1000.0, "db_path": tmp_logger.db_path,
        "csv_path": str(tmp_path / "s.csv"), "positions_path": str(tmp_path / "p.json"),
        "benchmark_path": str(tmp_path / "b.json"),
    }
    main._run_cycle_body(config, is_live=False)
    snaps = tmp_logger.get_benchmark_snapshots(is_live=False)
    assert len(snaps) == 1 and snaps[0]["equity"] == 1000.0
    assert (tmp_path / "b.json").exists()


# ------------------------------------------------- QQQ -> SPY switch (legacy rows)


def _hourly_spy(prices_by_hour):
    import pandas as pd

    idx = pd.DatetimeIndex(list(prices_by_hour), tz="UTC")
    vals = list(prices_by_hour.values())
    return pd.DataFrame({"Open": vals, "Close": [v + 0.5 for v in vals]}, index=idx)


def _legacy_setup(tmp_logger):
    """Three snapshots stored while QQQ was the benchmark, then one stored as SPY."""
    for equity, symbol, price in [(1000.0, "QQQ", 700.0), (1010.0, "QQQ", 707.0),
                                  (1020.0, "QQQ", 714.0), (1030.0, "SPY", 660.0)]:
        tmp_logger.record_benchmark_snapshot(equity, symbol, price, is_live=True)
    _set_timestamps(tmp_logger.db_path, "benchmark_snapshots", [
        "2026-10-01T14:45:00+00:00", "2026-10-02T14:45:00+00:00",
        "2026-10-03T14:45:00+00:00", "2026-10-04T14:45:00+00:00",
    ])
    _log_live_buy(tmp_logger)
    _set_timestamps(tmp_logger.db_path, "signals", ["2026-10-01T14:46:00+00:00"])


def test_legacy_qqq_snapshots_are_priced_from_spy_history(tmp_logger, monkeypatch):
    _legacy_setup(tmp_logger)
    monkeypatch.setattr(benchmark, "fetch_benchmark_history", lambda symbol="SPY": _hourly_spy({
        "2026-10-01T14:00:00+00:00": 600.0, "2026-10-02T14:00:00+00:00": 606.0,
        "2026-10-03T14:00:00+00:00": 612.0,
    }))

    payload = benchmark.build_series(tmp_logger)

    pts = payload["points"]
    assert len(pts) == 4 and payload["benchmark_rebased"] is False
    assert payload["benchmark_backfilled"] is True
    # 14:45 is inside the 14:00 bar, so that bar's open is used.
    assert pts[0]["benchmark_price"] == 600.0 and pts[0]["benchmark_pct"] == 0.0
    assert pts[1]["benchmark_pct"] == pytest.approx(1.0)
    # The recorded SPY price is used as is.
    assert pts[3]["benchmark_price"] == 660.0 and pts[3]["benchmark_pct"] == pytest.approx(10.0)
    # The old QQQ series is kept alongside, rebased to the same baseline.
    assert pts[2]["qqq_pct"] == pytest.approx(2.0) and "qqq_pct" not in pts[3]


def test_without_spy_history_the_line_restarts_at_the_first_spy_snapshot(tmp_logger, monkeypatch):
    _legacy_setup(tmp_logger)
    monkeypatch.setattr(benchmark, "fetch_benchmark_history", lambda symbol="SPY": None)

    payload = benchmark.build_series(tmp_logger)

    assert payload["benchmark_rebased"] is True and payload["benchmark_backfilled"] is False
    assert [p["t"][:10] for p in payload["points"]] == ["2026-10-04"]
    assert payload["points"][0]["benchmark_pct"] == 0.0 and payload["points"][0]["bot_pct"] == 0.0


def test_price_at_uses_close_once_the_bar_has_ended():
    hist = _hourly_spy({"2026-10-02T20:00:00+00:00": 500.0})
    assert benchmark.price_at(hist, "2026-10-02T20:30:00+00:00") == 500.0
    assert benchmark.price_at(hist, "2026-10-03T02:00:00+00:00") == 500.5  # market closed
    assert benchmark.price_at(hist, "2026-10-02T19:00:00+00:00") is None  # before the data
    assert benchmark.price_at(hist, "2026-11-02T02:00:00+00:00") is None  # a month past it
    assert benchmark.price_at(None, "2026-10-02T20:30:00+00:00") is None


def test_price_at_accepts_microsecond_timestamps_on_a_second_resolution_index():
    import pandas as pd

    hist = _hourly_spy({"2026-10-02T14:00:00+00:00": 500.0})
    hist.index = hist.index.as_unit("s")  # what yfinance returns
    assert benchmark.price_at(hist, "2026-10-02T14:45:08.397685+00:00") == 500.0
