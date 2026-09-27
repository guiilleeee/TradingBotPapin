"""QQQ benchmark: snapshots per cycle, rebased to the first live trade."""

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
    tmp_logger.record_benchmark_snapshot(1000.0, "QQQ", 500.0, is_live=True)
    payload = benchmark.build_series(tmp_logger)
    assert payload["first_live_trade_at"] is None
    assert payload["points"] == []


def test_series_is_rebased_to_the_snapshot_at_the_first_live_trade(tmp_logger):
    for equity, price in [(900.0, 480.0), (1000.0, 500.0), (1100.0, 505.0), (1050.0, 525.0)]:
        tmp_logger.record_benchmark_snapshot(equity, "QQQ", price, is_live=True)
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
    tmp_logger.record_benchmark_snapshot(1000.0, "QQQ", 500.0, is_live=True)
    _log_live_buy(tmp_logger, status="error")
    assert benchmark.build_series(tmp_logger)["first_live_trade_at"] is None


def test_simulation_snapshots_are_excluded(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "QQQ", 500.0, is_live=True)
    tmp_logger.record_benchmark_snapshot(9999.0, "QQQ", 500.0, is_live=False)
    _log_live_buy(tmp_logger)
    assert all(p["equity"] != 9999.0 for p in benchmark.build_series(tmp_logger)["points"])


def test_snapshots_without_a_qqq_price_are_skipped(tmp_logger):
    tmp_logger.record_benchmark_snapshot(1000.0, "QQQ", 500.0, is_live=True)
    tmp_logger.record_benchmark_snapshot(1010.0, "QQQ", None, is_live=True)
    _log_live_buy(tmp_logger)
    assert len(benchmark.build_series(tmp_logger)["points"]) == 1


def test_record_snapshot_pairs_equity_with_the_fetched_price(tmp_logger, monkeypatch):
    monkeypatch.setattr(benchmark, "fetch_benchmark_price", lambda symbol="QQQ": 512.5)
    benchmark.record_snapshot(tmp_logger, 1234.0, is_live=True)
    snap = tmp_logger.get_benchmark_snapshots(is_live=True)[0]
    assert snap["equity"] == 1234.0 and snap["benchmark_price"] == 512.5


def test_export_writes_the_dashboard_file(tmp_logger, tmp_path):
    path = tmp_path / "benchmark.json"
    benchmark.export_benchmark_json(tmp_logger, str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["benchmark_symbol"] == "QQQ"
    assert payload["points"] == []


def test_cycle_records_a_snapshot_and_exports(tmp_logger, tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(benchmark, "fetch_benchmark_price", lambda symbol="QQQ": 500.0)
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
