"""Volume watch: threshold triggers, cooldowns, and edge cases.

No test here hits the real network. Every test mocks data_fetcher.fetch_ohlcv
so they run without yfinance/network access.
"""

import json
import time

import pandas as pd
import pytest

import volume_watch


# ----------------------------------------------------------------- helpers


def _make_hourly_df(prices, volumes=None, bars=30):
    """Build a fake hourly OHLCV DataFrame with `bars` rows.

    `prices` is a list of Close values for the last N bars (padded with the
    first value if shorter than `bars`). `volumes` is similar for Volume.
    """
    if len(prices) < bars:
        prices = [prices[0]] * (bars - len(prices)) + prices
    if volumes is None:
        volumes = [1000.0] * bars
    elif len(volumes) < bars:
        volumes = [volumes[0]] * (bars - len(volumes)) + volumes

    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=bars, freq="h")
    return pd.DataFrame(
        {"Close": prices[-bars:], "Volume": volumes[-bars:], "Open": prices[-bars:],
         "High": prices[-bars:], "Low": prices[-bars:]},
        index=idx,
    )


@pytest.fixture
def wake_state_path(tmp_path):
    return str(tmp_path / "wake_state.json")


# ------------------------------------------------------ check_symbol tests


class TestCheckSymbolThresholds:
    """Threshold logic: price move and volume multiple."""

    def test_no_trigger_below_both_thresholds(self, monkeypatch):
        """2% price move, 1.5x volume -- both below defaults (3%, 2.5x)."""
        # Price: 100 -> 102 = 2% move
        prices = [100.0] * 28 + [100.0, 102.0]
        volumes = [1000.0] * 29 + [1500.0]  # 1.5x average
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "sell_side": {"price_drop_pct_15m": 3.0}}, wake_state={}, is_held=True)
        assert result is None

    def test_triggers_on_price_alone(self, monkeypatch):
        """3.1% price move, 1.0x volume -- price alone triggers."""
        prices = [100.0] * 28 + [100.0, 103.1]
        volumes = [1000.0] * 30  # average volume, 1.0x
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is not None
        assert result["symbol"] == "BTC-USD"
        assert any("spiked" in r for r in result["trigger_reasons"])
        

    def test_triggers_on_volume_alone(self, monkeypatch):
        """0.5% price move, 2.6x volume -- volume alone triggers."""
        prices = [100.0] * 28 + [100.0, 100.5]
        volumes = [1000.0] * 29 + [2600.0]  # 2.6x average
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="ETH-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is not None
        assert result["symbol"] == "ETH-USD"
        assert any("volume" in r for r in result["trigger_reasons"])

    def test_triggers_on_both(self, monkeypatch):
        """Both thresholds crossed -- both reasons listed."""
        prices = [100.0] * 28 + [100.0, 104.0]
        volumes = [1000.0] * 29 + [3000.0]
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="SOL-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is not None
        assert len(result["trigger_reasons"]) == 2

    def test_negative_price_move_also_triggers_sell_side(self, monkeypatch):
        """A drop of 3.5% should trigger (absolute value)."""
        prices = [100.0] * 28 + [100.0, 96.5]
        volumes = [1000.0] * 30
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "sell_side": {"price_drop_pct_15m": 3.0}}, wake_state={}, is_held=True)
        assert result is not None
        assert any("dropped" in r for r in result["trigger_reasons"])


class TestCheckSymbolCooldown:
    """Cooldown: min_seconds_between_wakes."""

    def _setup_triggered_df(self, monkeypatch):
        """Helper: set up data that would trigger on price alone."""
        prices = [100.0] * 28 + [100.0, 104.0]
        volumes = [1000.0] * 30
        df = _make_hourly_df(prices, volumes)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

    def test_cooldown_blocks_trigger(self, monkeypatch):
        """Even above threshold, cooldown prevents re-trigger."""
        self._setup_triggered_df(monkeypatch)
        now = time.time()
        wake_state = {"BTC-USD_buy": now - 3600}  # woken 1h ago, cooldown is 2h

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state=wake_state, is_held=False, now=now,
        )
        assert result is None

    def test_cooldown_expired_allows_trigger(self, monkeypatch):
        """Cooldown expired: same thresholds, but enough time has passed."""
        self._setup_triggered_df(monkeypatch)
        now = time.time()
        wake_state = {"BTC-USD_buy": now - 8000}  # woken 8000s ago, cooldown is 7200s

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state=wake_state, is_held=False, now=now,
        )
        assert result is not None

    def test_no_prior_wake_allows_trigger(self, monkeypatch):
        """No previous wake for this symbol -- always eligible."""
        self._setup_triggered_df(monkeypatch)

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is not None


class TestCheckSymbolEdgeCases:
    """Edge cases: data fetch failures, too few bars."""

    def test_data_fetch_failure_returns_none(self, monkeypatch):
        """A failing data fetch is not a crash -- it's a skip."""
        monkeypatch.setattr(
            volume_watch.data_fetcher, "fetch_ohlcv",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("no data")),
        )
        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is None

    def test_single_bar_returns_none(self, monkeypatch):
        """Need at least 2 bars to compute a change."""
        df = _make_hourly_df([100.0], [1000.0], bars=1)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        result = volume_watch.check_symbol(symbol="BTC-USD", wake_config={"min_seconds_between_wakes": 7200, "buy_side": {"price_move_pct_15m": 3.0, "volume_multiple": 2.5}}, wake_state={}, is_held=False)
        assert result is None


# -------------------------------------------------------- wake state tests


class TestWakeState:
    """Persistence of per-symbol cooldown timestamps."""

    def test_load_missing_file_returns_empty(self, tmp_path):
        state = volume_watch.load_wake_state(str(tmp_path / "nonexistent.json"))
        assert state == {}

    def test_round_trip(self, wake_state_path):
        original = {"BTC-USD": 1700000000.0, "ETH-USD": 1700001000.0}
        volume_watch.save_wake_state(original, wake_state_path)
        loaded = volume_watch.load_wake_state(wake_state_path)
        assert loaded == original

    def test_corrupt_file_returns_empty(self, wake_state_path):
        with open(wake_state_path, "w") as f:
            f.write("not valid json {{{{")
        state = volume_watch.load_wake_state(wake_state_path)
        assert state == {}


# --------------------------------------------------------- run_watch tests


class TestRunWatch:
    """Integration-ish tests for run_watch (still fully mocked)."""

    def _write_config(self, tmp_path, symbols, wake_trigger=None):
        import yaml

        config = {
            "symbols": [{"symbol": s, "asset_class": "crypto"} for s in symbols],
        }
        if wake_trigger:
            config["wake_trigger"] = wake_trigger
        path = str(tmp_path / "config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return path

    def test_run_watch_returns_flagged_symbols(self, monkeypatch, tmp_path):
        monkeypatch.setattr(volume_watch, "_is_market_open", lambda: True)
        config_path = self._write_config(tmp_path, ["BTC-USD", "ETH-USD"])
        wake_state_path = str(tmp_path / "wake_state.json")

        # BTC triggers on price, ETH does not
        call_count = {"n": 0}

        def fake_fetch(symbol, *a, **k):
            call_count["n"] += 1
            if symbol == "BTC-USD":
                return _make_hourly_df([100.0] * 28 + [100.0, 104.0], [1000.0] * 30)
            return _make_hourly_df([100.0] * 30, [1000.0] * 30)

        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", fake_fetch)

        flagged = volume_watch.run_watch(config_path, wake_state_path)
        assert "BTC-USD" in flagged["wake_buy"]

        # Verify wake_state was persisted
        state = volume_watch.load_wake_state(wake_state_path)
        assert "BTC-USD_buy" in state
        assert "ETH-USD" not in state

    def test_run_watch_no_triggers(self, monkeypatch, tmp_path):
        monkeypatch.setattr(volume_watch, "_is_market_open", lambda: True)
        config_path = self._write_config(tmp_path, ["BTC-USD"])
        wake_state_path = str(tmp_path / "wake_state.json")

        df = _make_hourly_df([100.0] * 30, [1000.0] * 30)
        monkeypatch.setattr(volume_watch.data_fetcher, "fetch_ohlcv", lambda *a, **k: df)

        flagged = volume_watch.run_watch(config_path, wake_state_path)
        assert flagged == {"wake_buy": [], "wake_sell": []}
