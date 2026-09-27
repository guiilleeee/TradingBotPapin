"""ATR-14 and position context reach the model; stops outside 1x-4x ATR are
held; positions going nowhere for N days are closed by the sweep."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import backtest
import data_fetcher
import execution
import main
import risk_manager
from models import ExecutionResult, ExistingPosition, SignalOutput
from tests.cycle_helpers import INDICATORS, buy_signal, record_executions, settings, stub_market


# ------------------------------------------------------------------ ATR


def test_atr_of_a_constant_range_is_that_range():
    n = 60
    close = pd.Series(np.full(n, 100.0))
    df = pd.DataFrame({"Close": close, "High": close + 2.0, "Low": close - 2.0, "Volume": 1.0})
    assert data_fetcher._wilder_atr(df) == pytest.approx(4.0)


def test_atr_counts_gaps_through_the_previous_close():
    close = pd.Series([100.0] * 30 + [110.0] * 30)
    df = pd.DataFrame({"Close": close, "High": close + 1.0, "Low": close - 1.0})
    # the gap bar's true range is 111 - 100 = 11, above its 2.0 high-low range
    assert data_fetcher._wilder_atr(df.iloc[:31]) > 2.0


def test_atr_is_none_without_high_low():
    assert data_fetcher._wilder_atr(pd.DataFrame({"Close": np.arange(60.0)})) is None


def test_compute_indicators_carries_atr():
    close = pd.Series(np.linspace(100, 120, 80))
    df = pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1000.0})
    assert data_fetcher.compute_indicators(df).atr_14 is not None


# -------------------------------------------------------- stop vs ATR band


def _validate(stop, atr=2.0, action="buy"):
    raw = SignalOutput(symbol="AAPL", action=action, confidence=0.9, position_size_pct=5.0,
                       stop_loss_price=stop, take_profit_price=100.0 + (100.0 - stop) * 3,
                       reasoning="prova")
    return risk_manager.validate(raw=raw, current_price=100.0, today_realized_loss_pct=0.0,
                                 circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
                                 max_absolute_position_pct=20.0, min_confidence=0.6,
                                 atr=atr, stop_atr_min=1.0, stop_atr_max=4.0)


@pytest.mark.parametrize("stop, action", [
    (99.0, "hold"),   # 0.5x ATR
    (98.0, "buy"),    # exactly 1x
    (94.0, "buy"),    # 3x
    (92.0, "buy"),    # exactly 4x
    (90.0, "hold"),   # 5x
])
def test_stop_must_sit_between_one_and_four_atr(stop, action):
    result = _validate(stop)
    assert result.action == action
    if action == "hold":
        assert "ATR" in result.override_reason


def test_unknown_atr_skips_the_band():
    assert _validate(99.0, atr=None).action == "buy"


def test_atr_band_never_blocks_a_sell():
    assert _validate(99.5, action="sell").action == "sell"  # 0.25x ATR, a buy would be held


# ---------------------------------------------------- position context


def test_model_sees_days_held_unrealized_pnl_and_atr(tmp_logger, monkeypatch):
    stub_market(monkeypatch, {"AAPL": 110.0})
    monkeypatch.setattr(data_fetcher, "compute_indicators",
                        lambda df: INDICATORS.model_copy(update={"atr_14": 2.5}))
    record_executions(monkeypatch)
    monkeypatch.setattr(execution, "fetch_existing_position",
                        lambda **kw: ExistingPosition(qty=1, avg_entry_price=100.0))
    opened = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    monkeypatch.setattr(tmp_logger, "get_position_opened_at", lambda symbol: opened)
    seen = []

    def model(signal_input, system_prompt):
        seen.append(signal_input)
        return SignalOutput(symbol="AAPL", action="hold", confidence=0.5)

    main._process_symbol(
        symbol="AAPL", config={}, settings=settings(), bot_logger=tmp_logger, generate_signal=model,
        equity=1000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )
    position = seen[0].existing_position
    assert position.days_held == pytest.approx(3.0, abs=0.01)
    assert position.unrealized_pnl_pct == pytest.approx(10.0)
    assert seen[0].technical_indicators.atr_14 == 2.5
    assert seen[0].entry_rules.stop_atr_min == 1.0 and seen[0].entry_rules.stop_atr_max == 4.0


def test_opened_at_comes_from_the_ledger_then_the_last_executed_buy(tmp_logger):
    from models import SignalInput, TradeSignal

    assert tmp_logger.get_position_opened_at("AAPL") is None
    buy = TradeSignal(symbol="AAPL", action="buy", confidence=0.9, position_size_pct=5,
                      stop_loss_price=95, take_profit_price=110, reasoning="x", raw_action="buy")
    tmp_logger.log_signal("AAPL", None, None, buy,
                          ExecutionResult(status="success", qty=1, fill_price=100), is_live=True)
    assert tmp_logger.get_position_opened_at("AAPL") is not None
    tmp_logger.open_simulated_position("MSFT", qty=1, avg_entry_price=100)
    assert tmp_logger.get_position_opened_at("MSFT") is not None


# ------------------------------------------------------------ time exit


SETTINGS = {"enabled": True, "max_holding_days": 10.0, "min_move_pct": 2.0}
NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


@pytest.mark.parametrize("days, price, closes", [
    (11, 101.0, True),    # old and flat
    (9, 101.0, False),    # not old enough
    (11, 103.0, False),   # moved up enough
    (11, 97.5, False),    # moved down enough
])
def test_time_exit_reason(days, price, closes):
    opened = (NOW - timedelta(days=days)).isoformat()
    reason = main.time_exit_reason(opened, 100.0, price, SETTINGS, now=NOW)
    assert (reason is not None) is closes
    if closes:
        assert reason.startswith("Sortida per temps")


def test_time_exit_disabled_or_unknown_open_never_closes():
    old = (NOW - timedelta(days=30)).isoformat()
    assert main.time_exit_reason(old, 100.0, 100.0, {**SETTINGS, "enabled": False}, now=NOW) is None
    assert main.time_exit_reason(None, 100.0, 100.0, SETTINGS, now=NOW) is None


def test_time_exit_settings_parse_config():
    s = main.time_exit_settings({"time_exit": {"max_holding_days": "7", "min_move_pct": "oops"}})
    assert s["max_holding_days"] == 7.0 and s["min_move_pct"] == 2.0 and s["enabled"] is True


def test_sweep_closes_a_stale_ledger_position(tmp_logger, monkeypatch):
    import sqlite3

    tmp_logger.open_simulated_position("AAPL", qty=2, avg_entry_price=100.0,
                                       stop_loss_price=90.0, take_profit_price=130.0)
    old = (datetime.now(timezone.utc) - timedelta(days=12)).isoformat()
    with sqlite3.connect(tmp_logger.db_path) as conn:
        conn.execute("UPDATE simulated_positions SET opened_at = ?", (old,))
    monkeypatch.setattr(data_fetcher, "fetch_ohlcv",
                        lambda symbol, period=None, interval="1d": pd.DataFrame({"Close": [101.0]}))

    sweep = main.sweep_open_positions(tmp_logger, {}, is_live=False, equity_hint=1000.0)

    assert sweep.closed_symbols == {"AAPL"}
    assert sweep.closures[0].reason.startswith("Sortida per temps")
    assert sweep.closures[0].pnl == pytest.approx(2.0)
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(2.0)


def test_live_stale_bracket_position_is_sold_via_execution(tmp_logger, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(days=12)).isoformat()
    monkeypatch.setattr(execution, "fetch_all_live_positions", lambda: {
        "AAPL": ExistingPosition(qty=3, avg_entry_price=100.0),
        "MSFT": ExistingPosition(qty=1, avg_entry_price=100.0),   # not bought by the bot
    })
    monkeypatch.setattr(tmp_logger, "get_position_opened_at",
                        lambda s: old if s == "AAPL" else None)
    monkeypatch.setattr(data_fetcher, "fetch_ohlcv",
                        lambda symbol, period=None, interval="1d": pd.DataFrame({"Close": [100.5]}))
    sells = []

    def fake_execute(signal, current_price, live_equity, is_live, existing_position=None,
                     cash_available=None):
        sells.append((signal.symbol, signal.action, existing_position.qty))
        return ExecutionResult(status="success", qty=3, fill_price=100.4, realized_pnl_usd=1.2,
                               entry_price=100.0)

    monkeypatch.setattr(execution, "execute_trade", fake_execute)

    result = main.sweep_stale_bracketed_positions(tmp_logger, {}, 1000.0, skip=set())

    assert sells == [("AAPL", "sell", 3)]
    assert result.closed_symbols == {"AAPL"}
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(1.2)


def test_backtest_sweep_applies_the_same_time_exit():
    days = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.5}, index=days)
    state = backtest.BacktestState(equity=1000.0)
    state.open_positions["AAPL"] = backtest.OpenPosition(
        symbol="AAPL", qty=1.0, entry_price=100.0, stop_loss_price=90.0,
        take_profit_price=120.0, opened_day=days[0])
    logger = backtest.BacktestLogger(":memory:")

    assert backtest.sweep_positions_for_day(state, {"AAPL": frame}, days[5], logger, SETTINGS) == set()
    assert backtest.sweep_positions_for_day(state, {"AAPL": frame}, days[10], logger, SETTINGS) == {"AAPL"}
    assert state.closed_trades[0].exit_price == pytest.approx(100.5)
