"""Spot crypto on Alpaca: symbol translation, 24/7 trading, no bracket legs."""

import pytest

import execution
from models import ExistingPosition, TradeSignal


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def signal(action="buy", symbol="BTC-USD", size=10.0):
    return TradeSignal(
        symbol=symbol, action=action, confidence=0.9, position_size_pct=size,
        stop_loss_price=58000.0, take_profit_price=66000.0, reasoning="prova", raw_action=action,
    )


@pytest.fixture
def alpaca(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["body"] = json
        return FakeResponse({"id": "c-1", "filled_avg_price": "60010", "filled_qty": "0.00833"})

    monkeypatch.setattr(execution.requests, "post", fake_post)
    return sent


def test_live_crypto_buy_is_a_notional_gtc_order_on_the_slash_symbol(alpaca):
    result = execution.execute_trade(signal(), 60000.0, 5000.0, is_live=True)
    body = alpaca["body"]
    assert result.status == "success"
    assert body["symbol"] == "BTC/USD"
    assert body["notional"] == "500.00"
    assert body["time_in_force"] == "gtc"
    assert "order_class" not in body and "stop_loss" not in body


def test_crypto_entry_is_always_a_bot_managed_exit(alpaca):
    result = execution.execute_trade(signal(), 60000.0, 5000.0, is_live=True)
    assert execution.needs_managed_exit(result)


def test_crypto_buys_while_the_equity_market_is_closed(alpaca, monkeypatch):
    monkeypatch.setattr(execution, "_is_market_open", lambda: False)
    assert execution.execute_trade(signal(), 60000.0, 5000.0, is_live=True).status == "success"


def test_equity_buy_is_still_blocked_while_the_market_is_closed(alpaca, monkeypatch):
    monkeypatch.setattr(execution, "_is_market_open", lambda: False)
    result = execution.execute_trade(signal(symbol="AAPL"), 200.0, 5000.0, is_live=True)
    assert result.status == "skipped"
    assert "market is closed" in result.message


def test_crypto_sell_closes_the_held_quantity(alpaca):
    held = ExistingPosition(qty=0.0123, avg_entry_price=55000.0)
    result = execution.execute_trade(signal("sell"), 60000.0, 5000.0, is_live=True, existing_position=held)
    body = alpaca["body"]
    assert body["side"] == "sell"
    assert body["qty"] == "0.0123"
    assert body["symbol"] == "BTC/USD"
    assert result.realized_pnl_usd == pytest.approx((60010 - 55000) * 0.00833)


def test_crypto_duplicate_buy_and_naked_sell_are_skipped(alpaca):
    held = ExistingPosition(qty=1.0, avg_entry_price=100.0)
    assert execution.execute_trade(signal(), 60000.0, 5000.0, True, held).status == "skipped"
    assert execution.execute_trade(signal("sell"), 60000.0, 5000.0, True, None).status == "skipped"


def test_crypto_simulation_is_a_dry_run_with_no_broker_call(monkeypatch):
    monkeypatch.setattr(
        execution.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call"))
    )
    result = execution.execute_trade(signal(symbol="DOGE-USD"), 0.25, 1000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty == pytest.approx(100.0 / 0.25)


def test_live_crypto_position_lookup_uses_alpacas_unslashed_symbol(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    urls = []

    def fake_get(url, headers=None, timeout=None):
        urls.append(url)
        return FakeResponse({"qty": "0.5", "avg_entry_price": "3000"})

    monkeypatch.setattr(execution.requests, "get", fake_get)
    position = execution.fetch_existing_position("ETH-USD", True, None)
    assert urls[0].endswith("/v2/positions/ETHUSD")
    assert position.qty == 0.5


def test_fetch_all_live_positions_maps_back_to_project_symbols(monkeypatch):
    monkeypatch.setattr(
        execution.requests, "get",
        lambda *a, **k: FakeResponse([
            {"symbol": "AAPL", "asset_class": "us_equity", "qty": "3", "avg_entry_price": "150"},
            {"symbol": "DOGEUSD", "asset_class": "crypto", "qty": "400", "avg_entry_price": "0.2"},
            {"symbol": "MSFT", "asset_class": "us_equity", "qty": "0", "avg_entry_price": "300"},
        ]),
    )
    positions = execution.fetch_all_live_positions()
    assert set(positions) == {"AAPL", "DOGE-USD"}
    assert positions["DOGE-USD"].qty == 400.0


def test_fetch_all_live_positions_raises_on_a_broker_error(monkeypatch):
    monkeypatch.setattr(execution.requests, "get", lambda *a, **k: FakeResponse({}, 500))
    with pytest.raises(RuntimeError):
        execution.fetch_all_live_positions()
