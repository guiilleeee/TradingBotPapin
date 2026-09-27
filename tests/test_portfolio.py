"""Portfolio limits on new buys: count, groups, correlation, gross exposure."""

import numpy as np
import pandas as pd
import pytest
import yaml

import funnel
import main
import portfolio
from models import TradeSignal
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market

CONFIG = {
    "symbols": [{"symbol": "BTC-USD", "asset_class": "crypto"},
                {"symbol": "ETH-USD", "asset_class": "crypto"},
                {"symbol": "SOL-USD", "asset_class": "crypto"}],
    "portfolio": {
        "max_open_positions": 3,
        "max_gross_exposure_pct": 80.0,
        "groups": {
            "crypto": {"asset_class": "crypto", "max_positions": 2},
            "semis": {"symbols": ["NVDA", "AMD", "MU"], "max_positions": 1},
        },
        "correlation": {"threshold": 0.75, "max_correlated_positions": 2},
    },
}
SETTINGS = portfolio.settings_from_config(CONFIG)


def buy(symbol="AAPL", size=10.0, override=None):
    return TradeSignal(symbol=symbol, action="buy", confidence=0.9, position_size_pct=size,
                       stop_loss_price=95.0, take_profit_price=115.0, reasoning="prova",
                       raw_action="buy", override_reason=override)


def check(signal, held, closes=None, equity=10000.0, cfg=CONFIG):
    return portfolio.check_buy(signal, held, equity, portfolio.settings_from_config(cfg), cfg, closes)


def series(returns, seed_price=100.0):
    idx = pd.date_range("2026-06-01", periods=len(returns) + 1, freq="D")
    prices = seed_price * np.cumprod(np.concatenate([[1.0], 1.0 + np.asarray(returns)]))
    return pd.Series(prices, index=idx)


rng = np.random.default_rng(7)
BASE = rng.normal(0, 0.02, 70)
TWIN = BASE + rng.normal(0, 0.003, 70)       # correlated ~0.99 with BASE
OTHER = rng.normal(0, 0.02, 70)              # unrelated


def test_shipped_config_parses_and_has_the_three_groups():
    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    s = portfolio.settings_from_config(cfg)
    assert s["max_open_positions"] > 0
    assert set(s["groups"]) == {"crypto", "semiconductors", "megacap_platforms"}
    assert s["correlation"]["lookback_sessions"] == 60


def test_non_buys_pass_through_untouched():
    hold = buy().model_copy(update={"action": "hold"})
    assert check(hold, {"A": 1, "B": 1, "C": 1}) is hold


def test_position_count_limit_holds_the_buy():
    result = check(buy(), {"MSFT": 100.0, "NFLX": 100.0, "COST": 100.0})
    assert result.action == "hold"
    assert "max_open_positions" in result.override_reason
    assert result.stop_loss_price is None and result.position_size_pct == 0.0


def test_group_cap_by_explicit_list():
    result = check(buy("AMD"), {"NVDA": 100.0})
    assert result.action == "hold"
    assert "group 'semis'" in result.override_reason


def test_group_cap_by_asset_class():
    result = check(buy("SOL-USD"), {"BTC-USD": 100.0, "ETH-USD": 100.0})
    assert result.action == "hold"
    assert "crypto" in result.override_reason


def test_a_symbol_outside_every_group_is_not_group_capped():
    assert check(buy("AAPL"), {"NVDA": 100.0}).action == "buy"


def test_correlated_cluster_over_the_limit_holds_the_buy():
    closes = {"AAPL": series(BASE), "MSFT": series(TWIN), "NFLX": series(TWIN * 1.01)}
    result = check(buy("AAPL"), {"MSFT": 100.0, "NFLX": 100.0}, closes)
    assert result.action == "hold"
    assert "correlated" in result.override_reason
    assert "MSFT" in result.override_reason and "NFLX" in result.override_reason


def test_one_correlated_holding_is_within_a_limit_of_two():
    closes = {"AAPL": series(BASE), "MSFT": series(TWIN), "COST": series(OTHER)}
    assert check(buy("AAPL"), {"MSFT": 100.0, "COST": 100.0}, closes).action == "buy"


def test_too_little_shared_history_is_not_counted_as_correlated():
    closes = {"AAPL": series(BASE[:10]), "MSFT": series(TWIN[:10]), "NFLX": series(TWIN[:10])}
    assert check(buy("AAPL"), {"MSFT": 100.0, "NFLX": 100.0}, closes).action == "buy"


def test_correlation_can_be_switched_off():
    cfg = {**CONFIG, "portfolio": {**CONFIG["portfolio"], "correlation": {"enabled": False}}}
    closes = {"AAPL": series(BASE), "MSFT": series(TWIN), "NFLX": series(TWIN)}
    assert check(buy("AAPL"), {"MSFT": 100.0, "NFLX": 100.0}, closes, cfg=cfg).action == "buy"


def test_gross_exposure_shrinks_the_buy_to_the_room_left():
    # 80% of 10,000 = 8,000 cap; 7,500 held leaves 500 = 5%.
    result = check(buy(size=20.0, override="risk note"), {"MSFT": 7500.0})
    assert result.action == "buy"
    assert result.position_size_pct == pytest.approx(5.0)
    assert result.override_reason.startswith("risk note; portfolio:")


def test_gross_exposure_holds_when_almost_no_room_is_left():
    result = check(buy(size=20.0), {"MSFT": 7950.0})
    assert result.action == "hold"
    assert "gross exposure" in result.override_reason


def test_malformed_settings_keep_their_defaults():
    s = portfolio.settings_from_config({"portfolio": {"max_open_positions": "many",
                                                      "correlation": {"threshold": "high"}}})
    assert s["max_open_positions"] == portfolio.DEFAULTS["max_open_positions"]
    assert s["correlation"]["threshold"] == portfolio.DEFAULTS["correlation"]["threshold"]


# ------------------------------------------------------------ cycle wiring


def _process(tmp_logger, monkeypatch, book, symbol="SOL-USD"):
    stub_market(monkeypatch, {symbol: 100.0})
    calls = record_executions(monkeypatch)
    main._process_symbol(
        symbol=symbol, config=CONFIG, settings=settings(is_live=False), bot_logger=tmp_logger,
        generate_signal=lambda signal_input, system_prompt: buy_signal(symbol, 100.0, stop_pct=0.05),
        equity=10000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
        book=book,
    )
    return calls


def test_cycle_holds_a_buy_the_group_cap_blocks(tmp_logger, read_signals, monkeypatch):
    book = main.CycleBook(cash=5000.0, exposures={"BTC-USD": 100.0, "ETH-USD": 100.0})
    calls = _process(tmp_logger, monkeypatch, book)
    assert calls == []
    assert "group 'crypto'" in read_signals()[0]["override_reason"]


def test_cycle_counts_its_own_fills_toward_the_limits(tmp_logger, monkeypatch):
    book = main.CycleBook(cash=5000.0, exposures={"BTC-USD": 100.0})
    assert len(_process(tmp_logger, monkeypatch, book, "ETH-USD")) == 1
    assert "ETH-USD" in book.exposures
    assert _process(tmp_logger, monkeypatch, book, "SOL-USD") == []  # crypto group now full


def test_unknown_positions_block_every_buy(tmp_logger, read_signals, monkeypatch):
    book = main.CycleBook(cash=5000.0, exposures={}, portfolio_error="Alpaca down")
    assert _process(tmp_logger, monkeypatch, book) == []
    assert "open positions unknown" in read_signals()[0]["override_reason"]


def test_a_wake_cycle_fetches_missing_closes_once(tmp_logger, monkeypatch):
    requested = []

    def fake_fetch(symbols):
        requested.append(sorted(symbols))
        data = funnel.FunnelData()
        data.closes = {"SOL-USD": series(BASE), "NVDA": series(TWIN), "AMD": series(TWIN)}
        return data

    monkeypatch.setattr(funnel, "fetch_funnel_data", fake_fetch)
    book = main.CycleBook(cash=5000.0, exposures={"NVDA": 100.0, "AMD": 100.0})
    calls = _process(tmp_logger, monkeypatch, book)
    assert requested == [["AMD", "NVDA", "SOL-USD"]]
    assert calls == []  # SOL correlated with both holdings: cluster of 3 > 2


def test_simulated_exposures_come_from_the_sweeps_fresh_prices(tmp_logger, monkeypatch):
    import data_fetcher

    tmp_logger.open_simulated_position("AAPL", qty=2.0, avg_entry_price=100.0,
                                       stop_loss_price=90.0, take_profit_price=130.0)
    monkeypatch.setattr(data_fetcher, "fetch_ohlcv",
                        lambda symbol, period=None, interval="1d": pd.DataFrame({"Close": [110.0]}))
    sweep = main.sweep_open_positions(tmp_logger, {}, is_live=False, equity_hint=1000.0)
    book = main.CycleBook()
    main._load_exposures(book, False, sweep)
    assert book.exposures == {"AAPL": pytest.approx(220.0)}
