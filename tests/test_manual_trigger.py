"""The dashboard's "Executar analisi ara" button dispatches a cycle restricted to
one arbitrary searched symbol via --trigger-symbols/--trigger-reason=manual. Unlike
volume_watch.py's wake-ups (which only ever fire for symbols already in the
configured watchlist), a manual analysis must be able to target any symbol the
user searched for -- see main.py's trigger_symbols handling in _run_cycle_body.
"""

import main


def _stub_cycle_dependencies(monkeypatch, processed):
    """Stub out everything _run_cycle_body needs beyond the symbol-matching logic
    itself, so this stays a fast, focused unit test rather than an integration test.
    """

    def fake_process_symbol(*, symbol, **kwargs):
        processed.append(symbol)

    monkeypatch.setattr(main, "_process_symbol", fake_process_symbol)


def _config(tmp_logger, tmp_path, symbols):
    # _run_cycle_body exports signals.csv and positions.json at the end of every
    # cycle, defaulting to the real repo-root files when a config doesn't say
    # otherwise -- exactly like tmp_logger's db_path guard above, these must be
    # redirected into tmp_path or a test run overwrites the live dashboard's
    # actual signals.csv / positions.json.
    return {
        "symbols": symbols,
        "fallback_equity_usd": 1000.0,
        "db_path": tmp_logger.db_path,
        "csv_path": str(tmp_path / "signals.csv"),
        "positions_path": str(tmp_path / "positions.json"),
        "benchmark_path": str(tmp_path / "benchmark.json"),
        "live_execution": False,
    }


def test_manual_trigger_processes_a_symbol_outside_the_watchlist(tmp_logger, tmp_path, monkeypatch):
    processed = []
    _stub_cycle_dependencies(monkeypatch, processed)

    config = _config(tmp_logger, tmp_path, [{"symbol": "AAPL", "asset_class": "equity"}])
    main._run_cycle_body(config, is_live=False, trigger_symbols=["TSLA"], trigger_reason="manual")

    assert processed == ["TSLA"]


def test_manual_trigger_still_matches_a_configured_symbol(tmp_logger, tmp_path, monkeypatch):
    processed = []
    _stub_cycle_dependencies(monkeypatch, processed)

    config = _config(tmp_logger, tmp_path, [{"symbol": "AAPL", "asset_class": "equity"}])
    main._run_cycle_body(config, is_live=False, trigger_symbols=["AAPL"], trigger_reason="manual")

    assert processed == ["AAPL"]


def test_wake_trigger_does_not_widen_beyond_the_watchlist(tmp_logger, tmp_path, monkeypatch):
    """The manual-only widening must not leak into wake_buy/wake_sell -- those are
    volume_watch.py's own triggers and must stay scoped to symbols the bot already
    tracks, exactly as before this change."""
    processed = []
    _stub_cycle_dependencies(monkeypatch, processed)

    config = _config(tmp_logger, tmp_path, [{"symbol": "AAPL", "asset_class": "equity"}])
    main._run_cycle_body(config, is_live=False, trigger_symbols=["TSLA"], trigger_reason="wake_buy")

    assert processed == []


def test_scheduled_cycle_goes_through_the_funnel(tmp_logger, tmp_path, monkeypatch):
    """No trigger_symbols at all (the normal cron) runs the funnel over the
    watchlist; with no ranking data (conftest's default stub) the fallback is
    universe order, which for a two-symbol watchlist is both symbols."""
    processed = []
    _stub_cycle_dependencies(monkeypatch, processed)

    config = _config(tmp_logger, tmp_path, [
        {"symbol": "AAPL", "asset_class": "equity"},
        {"symbol": "MSFT", "asset_class": "equity"},
    ])
    main._run_cycle_body(config, is_live=False, trigger_symbols=None, trigger_reason="scheduled")

    assert processed == ["AAPL", "MSFT"]
