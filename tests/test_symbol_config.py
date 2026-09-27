"""Per-symbol overrides (DOGE's wider wake thresholds and tighter size cap), asset
class inference, and the Alpaca symbol translation."""

import pytest

import main
import symbol_config
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market

CONFIG = {
    "max_absolute_position_pct": 20.0,
    "wake_trigger": {
        "buy_side": {"price_move_pct_15m": 1.5, "volume_multiple": 2.0},
        "sell_side": {"price_drop_pct_15m": 1.5, "price_drop_pct_1h": 3.0},
        "min_seconds_between_wakes": 3600,
    },
    "symbol_overrides": {
        "DOGE-USD": {
            "wake_trigger": {
                "buy_side": {"price_move_pct_15m": 3.0},
                "sell_side": {"price_drop_pct_15m": 3.0, "price_drop_pct_1h": 6.0},
            },
            "max_absolute_position_pct": 10.0,
        }
    },
    "symbols": [
        {"symbol": "AAPL", "asset_class": "equity"},
        {"symbol": "DOGE-USD", "asset_class": "crypto"},
    ],
}


# ------------------------------------------------------------ wake thresholds


def test_override_replaces_only_the_keys_it_names():
    doge = symbol_config.wake_config(CONFIG, "DOGE-USD")
    assert doge["buy_side"]["price_move_pct_15m"] == 3.0
    # Not named in the override -> inherited from the global block.
    assert doge["buy_side"]["volume_multiple"] == 2.0
    assert doge["sell_side"] == {"price_drop_pct_15m": 3.0, "price_drop_pct_1h": 6.0}
    assert doge["min_seconds_between_wakes"] == 3600


def test_symbol_without_override_gets_the_global_block_unchanged():
    assert symbol_config.wake_config(CONFIG, "AAPL") == CONFIG["wake_trigger"]
    assert symbol_config.wake_config(CONFIG, "BTC-USD") == CONFIG["wake_trigger"]


def test_override_lookup_is_case_insensitive():
    assert symbol_config.wake_config(CONFIG, "doge-usd")["buy_side"]["price_move_pct_15m"] == 3.0


def test_merging_never_mutates_the_global_block():
    symbol_config.wake_config(CONFIG, "DOGE-USD")["buy_side"]["price_move_pct_15m"] = 99.0
    assert CONFIG["wake_trigger"]["buy_side"]["price_move_pct_15m"] == 1.5
    assert CONFIG["symbol_overrides"]["DOGE-USD"]["wake_trigger"]["buy_side"]["price_move_pct_15m"] == 3.0


def test_no_overrides_block_at_all():
    assert symbol_config.wake_config({"wake_trigger": {"a": 1}}, "DOGE-USD") == {"a": 1}
    assert symbol_config.wake_config({}, "DOGE-USD") == {}


# ---------------------------------------------------------------- size cap


def test_doge_cap_is_tightened_to_ten_percent():
    assert symbol_config.max_position_pct(CONFIG, "DOGE-USD", 20.0) == 10.0


def test_other_symbols_keep_the_global_cap():
    assert symbol_config.max_position_pct(CONFIG, "AAPL", 20.0) == 20.0
    assert symbol_config.max_position_pct(CONFIG, "BTC-USD", 20.0) == 20.0


def test_an_override_can_never_raise_the_cap_above_global():
    config = {"symbol_overrides": {"DOGE-USD": {"max_absolute_position_pct": 50.0}}}
    assert symbol_config.max_position_pct(config, "DOGE-USD", 20.0) == 20.0


@pytest.mark.parametrize("bad", ["ten", None, 0, -5, [10]])
def test_a_malformed_cap_override_falls_back_to_global(bad):
    config = {"symbol_overrides": {"DOGE-USD": {"max_absolute_position_pct": bad}}}
    assert symbol_config.max_position_pct(config, "DOGE-USD", 20.0) == 20.0


def _process(symbol, price, tmp_logger, monkeypatch):
    stub_market(monkeypatch, {symbol: price})
    calls = record_executions(monkeypatch)
    main._process_symbol(
        symbol=symbol,
        config=CONFIG,
        settings=settings(is_live=False),
        bot_logger=tmp_logger,
        # 1% stop -> risk sizing wants 100%, so the cap is what decides the size.
        generate_signal=lambda signal_input, system_prompt: buy_signal(symbol, price, stop_pct=0.01),
        equity=1000.0,
        circuit_breaker_loss_pct=3.0,
        max_risk_pct=1.0,
        max_absolute_position_pct=20.0,
        min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )
    return calls


def test_cycle_sizes_doge_at_its_own_ten_percent_cap(tmp_logger, monkeypatch):
    calls = _process("DOGE-USD", 0.25, tmp_logger, monkeypatch)
    assert len(calls) == 1
    assert calls[0]["signal"].position_size_pct == pytest.approx(10.0)
    assert "10.00% absolute position cap" in calls[0]["signal"].override_reason


def test_cycle_sizes_other_symbols_at_the_global_cap(tmp_logger, monkeypatch):
    calls = _process("AAPL", 200.0, tmp_logger, monkeypatch)
    assert calls[0]["signal"].position_size_pct == pytest.approx(20.0)


def test_cycle_tells_the_model_which_asset_class_it_is_looking_at(tmp_logger, monkeypatch):
    seen = []
    stub_market(monkeypatch, {"DOGE-USD": 0.25})
    record_executions(monkeypatch)

    def model(signal_input, system_prompt):
        seen.append(signal_input.asset_class)
        return buy_signal("DOGE-USD", 0.25)

    main._process_symbol(
        symbol="DOGE-USD", config=CONFIG, settings=settings(), bot_logger=tmp_logger,
        generate_signal=model, equity=1000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )
    assert seen == ["crypto"]


# ------------------------------------------------------------ asset class


@pytest.mark.parametrize("symbol,expected", [
    ("BTC-USD", "crypto"), ("doge-usd", "crypto"), ("AAPL", "equity"), ("BRK.B", "equity"),
])
def test_asset_class_inference(symbol, expected):
    assert symbol_config.asset_class(symbol) == expected


def test_explicit_config_asset_class_wins():
    config = {"symbols": [{"symbol": "ODD-USD", "asset_class": "equity"}]}
    assert symbol_config.asset_class("ODD-USD", config) == "equity"


@pytest.mark.parametrize("ours,order,position", [
    ("BTC-USD", "BTC/USD", "BTCUSD"),
    ("DOGE-USD", "DOGE/USD", "DOGEUSD"),
    ("AAPL", "AAPL", "AAPL"),
])
def test_alpaca_symbol_translation(ours, order, position):
    assert symbol_config.alpaca_order_symbol(ours) == order
    assert symbol_config.alpaca_position_symbol(ours) == position


@pytest.mark.parametrize("alpaca,asset_class,ours", [
    ("BTCUSD", "crypto", "BTC-USD"),
    ("BTC/USD", "", "BTC-USD"),
    ("AAPL", "us_equity", "AAPL"),
    # An equity ticker ending in USD must not be mistaken for a crypto pair.
    ("FOOUSD", "us_equity", "FOOUSD"),
])
def test_from_alpaca_symbol(alpaca, asset_class, ours):
    assert symbol_config.from_alpaca_symbol(alpaca, asset_class) == ours
