"""Bracket exits reach the circuit breaker; bracket legs are cancelled before a
sell; a tripped breaker still allows sells; buys never exceed cash on hand."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import execution
import main
import notifications
import risk_manager
from models import ExistingPosition, SignalOutput, TradeSignal
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    @property
    def text(self):
        return json.dumps(self._payload)


@pytest.fixture
def live_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    monkeypatch.setattr(execution.time, "sleep", lambda s: None)


def sell(symbol="AAPL"):
    return TradeSignal(symbol=symbol, action="sell", confidence=0.9, position_size_pct=1.0,
                       stop_loss_price=90.0, take_profit_price=120.0, reasoning="prova",
                       raw_action="sell")


def buy(symbol="AAPL", size=20.0):
    return TradeSignal(symbol=symbol, action="buy", confidence=0.9, position_size_pct=size,
                       stop_loss_price=95.0, take_profit_price=115.0, reasoning="prova",
                       raw_action="buy")


# ------------------------------------------------------ cancel before selling


class FakeBroker:
    """Open bracket legs that move to a given status once cancelled."""

    def __init__(self, legs, status_after_cancel="canceled", filled_qty="0", position=None):
        self.legs = {leg["id"]: dict(leg) for leg in legs}
        self.status_after_cancel = status_after_cancel
        self.filled_qty = filled_qty
        self.position = position
        self.deleted = []
        self.posted = []

    def get(self, url, headers=None, params=None, timeout=None):
        if url.endswith("/v2/orders"):
            return FakeResponse([{"id": "parent", "side": "buy", "status": "filled",
                                  "symbol": "AAPL", "legs": list(self.legs.values())}])
        if "/v2/orders/" in url:
            leg = self.legs[url.rsplit("/", 1)[1]]
            return FakeResponse(leg)
        if "/v2/positions/" in url:
            if self.position is None:
                return FakeResponse({}, status_code=404)
            return FakeResponse(self.position)
        raise AssertionError(url)

    def delete(self, url, headers=None, timeout=None):
        order_id = url.rsplit("/", 1)[1]
        self.deleted.append(order_id)
        self.legs[order_id]["status"] = self.status_after_cancel
        self.legs[order_id]["filled_qty"] = self.filled_qty
        return FakeResponse({}, status_code=204)

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted.append(json)
        return FakeResponse({"id": "sell-1", "filled_avg_price": "101", "filled_qty": json["qty"]})


LEGS = [
    {"id": "tp", "side": "sell", "symbol": "AAPL", "type": "limit", "status": "new"},
    {"id": "sl", "side": "sell", "symbol": "AAPL", "type": "stop_limit", "status": "held"},
]


def install(monkeypatch, broker):
    monkeypatch.setattr(execution.requests, "get", broker.get)
    monkeypatch.setattr(execution.requests, "delete", broker.delete)
    monkeypatch.setattr(execution.requests, "post", broker.post)


def test_live_sell_cancels_both_bracket_legs_before_selling(monkeypatch, live_keys):
    broker = FakeBroker(LEGS)
    install(monkeypatch, broker)

    result = execution.execute_trade(sell(), 100.0, 10000.0, is_live=True,
                                     existing_position=ExistingPosition(qty=4, avg_entry_price=90))

    assert sorted(broker.deleted) == ["sl", "tp"]
    assert result.status == "success"
    assert broker.posted and broker.posted[0]["side"] == "sell" and broker.posted[0]["qty"] == "4"


def test_sell_is_not_sent_while_a_leg_refuses_to_cancel(monkeypatch, live_keys):
    broker = FakeBroker(LEGS, status_after_cancel="pending_cancel")
    install(monkeypatch, broker)

    result = execution.execute_trade(sell(), 100.0, 10000.0, is_live=True,
                                     existing_position=ExistingPosition(qty=4, avg_entry_price=90))

    assert result.status == "error"
    assert "still open" in result.message
    assert broker.posted == []


def test_a_leg_that_filled_first_means_no_second_sell(monkeypatch, live_keys):
    broker = FakeBroker(LEGS, status_after_cancel="filled", filled_qty="4", position=None)
    install(monkeypatch, broker)

    result = execution.execute_trade(sell(), 100.0, 10000.0, is_live=True,
                                     existing_position=ExistingPosition(qty=4, avg_entry_price=90))

    assert result.status == "skipped"
    assert broker.posted == []


def test_a_partial_leg_fill_sells_only_the_remainder(monkeypatch, live_keys):
    broker = FakeBroker(LEGS, status_after_cancel="canceled", filled_qty="1",
                        position={"qty": "3", "avg_entry_price": "90"})
    install(monkeypatch, broker)

    execution.execute_trade(sell(), 100.0, 10000.0, is_live=True,
                            existing_position=ExistingPosition(qty=4, avg_entry_price=90))

    assert broker.posted[0]["qty"] == "3"


def test_simulated_sell_never_touches_the_broker(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("simulation must not call Alpaca")

    monkeypatch.setattr(execution.requests, "get", explode)
    monkeypatch.setattr(execution.requests, "delete", explode)
    result = execution.execute_trade(sell(), 100.0, 10000.0, is_live=False,
                                     existing_position=ExistingPosition(qty=4, avg_entry_price=90))
    assert result.status == "dry_run"


# ------------------------------------------------- bracket exits -> pnl table


def closed_bracket(leg_status="filled", leg_type="stop_limit", filled_at=None):
    return {
        "id": "parent-1", "order_class": "bracket", "side": "buy", "symbol": "AAPL",
        "asset_class": "us_equity", "filled_avg_price": "100", "submitted_at": "2026-09-01T14:00:00Z",
        "legs": [
            {"id": "leg-sl", "type": leg_type, "status": leg_status, "filled_qty": "5",
             "filled_avg_price": "94", "filled_at": filled_at or datetime.now(timezone.utc).isoformat()},
            {"id": "leg-tp", "type": "limit", "status": "canceled", "filled_qty": "0",
             "filled_avg_price": None, "filled_at": None},
        ],
    }


def test_fetch_bracket_exit_fills_reads_only_filled_legs(monkeypatch, live_keys):
    monkeypatch.setattr(execution.requests, "get", lambda *a, **k: FakeResponse([closed_bracket()]))
    fills = execution.fetch_bracket_exit_fills("2026-06-01T00:00:00+00:00")
    assert len(fills) == 1
    fill = fills[0]
    assert fill["symbol"] == "AAPL"
    assert fill["kind"] == "stop_loss"
    assert fill["qty"] == 5 and fill["entry_price"] == 100 and fill["exit_price"] == 94


def test_a_leg_still_working_is_not_booked_yet(monkeypatch, live_keys):
    monkeypatch.setattr(execution.requests, "get",
                        lambda *a, **k: FakeResponse([closed_bracket(leg_status="partially_filled")]))
    assert execution.fetch_bracket_exit_fills("2026-06-01T00:00:00+00:00") == []


def test_reconcile_books_a_stop_out_once_and_it_trips_the_breaker(tmp_logger, read_signals, monkeypatch, live_keys):
    monkeypatch.setattr(execution.requests, "get", lambda *a, **k: FakeResponse([closed_bracket()]))
    alerts = []
    monkeypatch.setattr(notifications, "send_auto_close_alert", lambda *a: alerts.append(a))

    first = main.reconcile_bracket_exits(tmp_logger, equity=1000.0)
    second = main.reconcile_bracket_exits(tmp_logger, equity=1000.0)

    assert [c.pnl for c in first] == [pytest.approx(-30.0)]  # (94 - 100) * 5
    assert second == []  # never booked twice
    assert tmp_logger.get_today_realized_loss_pct(1000.0) == pytest.approx(-3.0)
    assert len(alerts) == 1
    rows = read_signals()
    assert len(rows) == 1
    assert rows[0]["final_signal"]["reasoning"].startswith("Stop-loss")
    assert rows[0]["execution_result"]["realized_pnl_usd"] == pytest.approx(-30.0)


def test_an_old_exit_counts_toward_its_own_day_and_is_not_alerted(tmp_logger, monkeypatch, live_keys):
    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    monkeypatch.setattr(execution.requests, "get",
                        lambda *a, **k: FakeResponse([closed_bracket(filled_at=two_days_ago)]))
    alerts = []
    monkeypatch.setattr(notifications, "send_auto_close_alert", lambda *a: alerts.append(a))

    main.reconcile_bracket_exits(tmp_logger, equity=1000.0)

    assert tmp_logger.get_today_realized_loss_pct(1000.0) == pytest.approx(0.0)
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(-30.0)
    assert alerts == []


def test_a_broker_outage_during_reconcile_is_not_fatal(tmp_logger, monkeypatch, live_keys):
    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(execution.requests, "get", boom)
    assert main.reconcile_bracket_exits(tmp_logger, equity=1000.0) == []


# ------------------------------------------- breaker blocks buys, not sells


def test_breaker_still_lets_a_sell_through():
    raw = SignalOutput(symbol="AAPL", action="sell", confidence=0.9, position_size_pct=5.0,
                       stop_loss_price=95.0, take_profit_price=110.0, reasoning="prova")
    result = risk_manager.validate(raw=raw, current_price=100.0, today_realized_loss_pct=-5.0,
                                   circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
                                   max_absolute_position_pct=20.0, min_confidence=0.6)
    assert result.action == "sell"


def _run(tmp_logger, monkeypatch, held, raw):
    stub_market(monkeypatch, {"AAPL": 100.0})
    calls = record_executions(monkeypatch)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: held)
    tmp_logger.record_pnl("MSFT", -50.0)  # -5% of 1000: breaker tripped
    model_calls = []

    def model(signal_input, system_prompt):
        model_calls.append(signal_input.symbol)
        return raw

    main._process_symbol(
        symbol="AAPL", config={}, settings=settings(is_live=False), bot_logger=tmp_logger,
        generate_signal=model, equity=1000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=True),
    )
    return calls, model_calls


def test_tripped_breaker_still_asks_the_model_about_a_held_position(tmp_logger, monkeypatch):
    raw = SignalOutput(symbol="AAPL", action="sell", confidence=0.9, position_size_pct=5.0,
                       stop_loss_price=95.0, take_profit_price=110.0, reasoning="prova")
    calls, model_calls = _run(tmp_logger, monkeypatch, ExistingPosition(qty=2, avg_entry_price=110), raw)
    assert model_calls == ["AAPL"]
    assert [c["signal"].action for c in calls] == ["sell"]


def test_tripped_breaker_skips_the_model_when_nothing_is_held(tmp_logger, monkeypatch):
    calls, model_calls = _run(tmp_logger, monkeypatch, None, buy_signal("AAPL", 100.0))
    assert model_calls == []
    assert calls == []


# ------------------------------------------------------------- cash sizing


def test_budget_is_capped_by_cash():
    assert execution._buy_budget(10000.0, 20.0, None) == pytest.approx(2000.0)
    assert execution._buy_budget(10000.0, 20.0, 500.0) == pytest.approx(500.0)
    assert execution._buy_budget(10000.0, 20.0, -10.0) == 0.0


def test_buy_with_too_little_cash_is_a_clean_skip():
    result = execution.execute_trade(buy(), 100.0, 10000.0, is_live=False, cash_available=0.5)
    assert result.status == "skipped"


def test_whole_share_count_follows_the_cash_cap():
    result = execution.execute_trade(buy(size=20.0), 100.0, 10000.0, is_live=False, cash_available=350.0)
    assert result.qty == pytest.approx(3.0)


def test_live_cash_is_the_smaller_of_cash_and_non_marginable_buying_power(monkeypatch, live_keys):
    monkeypatch.setattr(execution.requests, "get", lambda *a, **k: FakeResponse(
        {"cash": "800.0", "non_marginable_buying_power": "650.5", "equity": "2000"}))
    assert execution.read_live_cash() == pytest.approx(650.5)


def test_simulated_cash_subtracts_the_cost_of_open_positions(tmp_logger):
    tmp_logger.open_simulated_position("AAPL", qty=2.0, avg_entry_price=100.0)
    tmp_logger.record_pnl("MSFT", 25.0)
    assert main.available_cash(tmp_logger, False, 1000.0) == pytest.approx(825.0)


def test_cycle_book_spends_on_buys_and_frees_on_sells():
    book = main.CycleBook(cash=1000.0)
    book.note_fill("AAPL", "buy", 3, 100.0)
    assert book.cash == pytest.approx(700.0)
    book.note_fill("AAPL", "sell", 1, 150.0)
    assert book.cash == pytest.approx(850.0)
    unknown = main.CycleBook(cash=None)
    unknown.note_fill("AAPL", "buy", 3, 100.0)
    assert unknown.cash is None


def test_second_buy_in_a_cycle_sees_the_cash_the_first_one_spent(tmp_logger, monkeypatch):
    stub_market(monkeypatch, {"AAPL": 100.0, "MSFT": 100.0})
    calls = record_executions(monkeypatch)  # fills qty=1 at the current price
    book = main.CycleBook(cash=500.0)
    for symbol in ("AAPL", "MSFT"):
        main._process_symbol(
            symbol=symbol, config={}, settings=settings(is_live=False), bot_logger=tmp_logger,
            generate_signal=lambda signal_input, system_prompt: buy_signal(signal_input.symbol, 100.0, stop_pct=0.02),
            equity=1000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
            max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
            breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
            book=book,
        )
    assert [c["cash_available"] for c in calls] == [pytest.approx(500.0), pytest.approx(400.0)]


def test_open_legs_are_found_in_a_flat_listing_too(monkeypatch, live_keys):
    def fake_get(url, headers=None, params=None, timeout=None):
        if params["nested"] == "true":
            return FakeResponse([])  # filled parent not listed as open
        return FakeResponse([dict(leg) for leg in LEGS])

    monkeypatch.setattr(execution.requests, "get", fake_get)
    assert sorted(o["id"] for o in execution._open_exit_orders("AAPL")) == ["sl", "tp"]
