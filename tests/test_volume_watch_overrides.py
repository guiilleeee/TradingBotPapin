"""volume_watch.run_watch with per-symbol thresholds, the closed-market crypto-only
mode, and wake-ups for crossed stop/take levels on bot-managed exits.

Fully offline: fetch_ohlcv is stubbed, and every config/ledger/positions file lives
in tmp_path.
"""

import time

import pandas as pd
import pytest
import yaml

import volume_watch
from logger import BotLogger


def _bars(last_move_pct, bars=30, volume_last=1000.0):
    """15m bars: flat at 100, then one last bar moved by `last_move_pct`."""
    prices = [100.0] * (bars - 1) + [100.0 * (1 + last_move_pct / 100.0)]
    volumes = [1000.0] * (bars - 1) + [volume_last]
    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=bars, freq="15min")
    return pd.DataFrame({"Close": prices, "Volume": volumes}, index=idx)


def _write_config(tmp_path, symbols, overrides=None, db_path=None):
    config = {
        "symbols": [
            {"symbol": s, "asset_class": "crypto" if s.endswith("-USD") else "equity"}
            for s in symbols
        ],
        "wake_trigger": {
            "buy_side": {"price_move_pct_15m": 1.5, "volume_multiple": 2.0},
            "sell_side": {"price_drop_pct_15m": 1.5, "price_drop_pct_1h": 3.0},
            "min_seconds_between_wakes": 3600,
        },
        "db_path": db_path or str(tmp_path / "none.db"),
    }
    if overrides:
        config["symbol_overrides"] = overrides
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


DOGE_OVERRIDE = {
    "DOGE-USD": {
        "wake_trigger": {
            "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 3.0},
            "sell_side": {"price_drop_pct_15m": 3.0, "price_drop_pct_1h": 6.0},
        }
    }
}


@pytest.fixture
def paths(tmp_path):
    return {
        "wake_state": str(tmp_path / "wake_state.json"),
        "positions": str(tmp_path / "positions.json"),
    }


def test_a_two_percent_spike_wakes_btc_but_not_doge(tmp_path, paths, monkeypatch):
    config_path = _write_config(tmp_path, ["BTC-USD", "DOGE-USD"], DOGE_OVERRIDE)
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(2.0))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_buy"] == ["BTC-USD"]


def test_doge_still_wakes_past_its_own_wider_threshold(tmp_path, paths, monkeypatch):
    config_path = _write_config(tmp_path, ["DOGE-USD"], DOGE_OVERRIDE)
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(3.5))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_buy"] == ["DOGE-USD"]


def test_doge_volume_override_applies_too(tmp_path, paths, monkeypatch):
    """2.5x volume clears the global 2.0x but not DOGE's 3.0x."""
    config_path = _write_config(tmp_path, ["BTC-USD", "DOGE-USD"], DOGE_OVERRIDE)
    monkeypatch.setattr(
        volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(0.0, volume_last=2500.0)
    )

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_buy"] == ["BTC-USD"]


def test_held_doge_sell_side_uses_its_wider_drop_threshold(tmp_path, paths, monkeypatch):
    config_path = _write_config(tmp_path, ["BTC-USD", "DOGE-USD"], DOGE_OVERRIDE)
    (tmp_path / "positions.json").write_text(
        '{"positions": [{"symbol": "BTC-USD", "qty": 1}, {"symbol": "DOGE-USD", "qty": 100}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(-2.0))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_sell"] == ["BTC-USD"]
    assert flagged["wake_buy"] == []


def test_closed_market_checks_crypto_only(tmp_path, paths, monkeypatch):
    monkeypatch.setattr(volume_watch, "_is_market_open", lambda: False)
    config_path = _write_config(tmp_path, ["AAPL", "BTC-USD"])
    fetched = []

    def fake_fetch(symbol, *a, **k):
        fetched.append(symbol)
        return _bars(5.0)

    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", fake_fetch)

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert fetched == ["BTC-USD"]
    assert flagged["wake_buy"] == ["BTC-USD"]


def test_crossed_stop_on_a_managed_exit_wakes_a_sell_even_in_cooldown(tmp_path, paths, monkeypatch):
    db_path = str(tmp_path / "ledger.db")
    BotLogger(db_path).open_simulated_position(
        "ETH-USD", qty=0.5, avg_entry_price=100.0, stop_loss_price=99.5, take_profit_price=110.0
    )
    config_path = _write_config(tmp_path, ["ETH-USD"], db_path=db_path)
    # Woken for a sell 10 minutes ago -- ordinarily still in cooldown.
    volume_watch.save_wake_state({"ETH-USD_sell": time.time() - 600}, paths["wake_state"])
    # -1% puts the price at 99.0, under the 99.5 stop, but well inside the 1.5% drop rule.
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(-1.0))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_sell"] == ["ETH-USD"]


def test_managed_exit_in_cooldown_without_a_crossing_stays_quiet(tmp_path, paths, monkeypatch):
    db_path = str(tmp_path / "ledger.db")
    BotLogger(db_path).open_simulated_position(
        "ETH-USD", qty=0.5, avg_entry_price=100.0, stop_loss_price=90.0, take_profit_price=110.0
    )
    config_path = _write_config(tmp_path, ["ETH-USD"], db_path=db_path)
    volume_watch.save_wake_state({"ETH-USD_sell": time.time() - 600}, paths["wake_state"])
    # A 2% drop would trigger the normal sell rule, but the cooldown holds it back.
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(-2.0))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_sell"] == []


def test_a_held_symbol_outside_the_universe_is_still_watched(tmp_path, paths, monkeypatch):
    config_path = _write_config(tmp_path, ["AAPL"])
    (tmp_path / "positions.json").write_text(
        '{"positions": [{"symbol": "INTC", "qty": 5}]}', encoding="utf-8"
    )
    monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: _bars(-2.0))

    flagged = volume_watch.run_watch(config_path, paths["wake_state"], paths["positions"])

    assert flagged["wake_sell"] == ["INTC"]
