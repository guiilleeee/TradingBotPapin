
class FakeLogger:
    def __init__(self, position=None):
        self._pos = position
        self.touched = False
    def get_simulated_position(self, symbol):
        self.touched = True
        return self._pos

import json

import pytest

import execution
from models import ExistingPosition, TradeSignal


def signal(action="buy", size=10.0, stop=95.0, take=115.0, symbol="AAPL"):
    return TradeSignal(
        symbol=symbol,
        action=action,
        confidence=0.9,
        position_size_pct=size,
        stop_loss_price=stop,
        take_profit_price=take,
        reasoning="prova",
        raw_action=action,
    )


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """Simulation must work with no keys present at all."""
    for var in (
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "HYPERLIQUID_WALLET_ADDRESS",
        "HYPERLIQUID_PRIVATE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------------- symbols


def test_stop_limit_sits_below_the_stop_when_closing_a_long():
    # The bracket leg that closes a long is a SELL stop, so the limit must be
    # below the stop or it can never fill.
    assert execution._stop_limit_price(100.0, "sell") == pytest.approx(99.0)


def test_stop_limit_sits_above_the_stop_for_a_buy_stop():
    # The mirror case. Hardcoding x0.99 for both sides is how this goes wrong.
    assert execution._stop_limit_price(100.0, "buy") == pytest.approx(101.0)


def test_unknown_exit_side_is_rejected():
    with pytest.raises(ValueError):
        execution._stop_limit_price(100.0, "sideways")


# ------------------------------------------------------------------- guards


def test_buy_is_skipped_when_a_position_already_exists():
    result = execution.execute_trade(
        signal("buy"), 100.0, 1000.0, is_live=False,
        existing_position=ExistingPosition(qty=3.0, avg_entry_price=90.0),
    )
    assert result.status == "skipped"
    assert "already exists" in result.message


def test_sell_with_nothing_held_is_skipped_not_shorted():
    result = execution.execute_trade(
        signal("sell"), 100.0, 1000.0, is_live=False, existing_position=None
    )
    assert result.status == "skipped"
    assert "nothing to sell" in result.message
    assert "short" in result.message


def test_hold_never_reaches_a_venue():
    result = execution.execute_trade(signal("hold"), 100.0, 1000.0, is_live=False)
    assert result.status == "skipped"


# ------------------------------------------------------- credentials gating


def test_simulation_needs_no_credentials():
    result = execution.execute_trade(signal("buy"), 100.0, 10000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty is not None


def test_live_without_alpaca_credentials_errors():
    result = execution.execute_trade(signal("buy"), 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "ALPACA_API_KEY" in result.message


def test_opening_size_comes_from_equity_and_percent():
    # 10% of $10,000 at $100 = 10 whole shares.
    result = execution.execute_trade(signal("buy", size=10.0), 100.0, 10000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty == pytest.approx(10.0)


def test_closing_uses_the_held_quantity_not_a_recomputed_one():
    # A freshly computed size here would be 10 shares; the position is 3.
    held = ExistingPosition(qty=3.0, avg_entry_price=90.0)
    result = execution.execute_trade(
        signal("sell", size=10.0), 100.0, 10000.0, is_live=False, existing_position=held
    )
    assert result.qty == pytest.approx(3.0)
    assert result.realized_pnl_usd == pytest.approx((100.0 - 90.0) * 3.0)


def test_closing_a_fractional_position_sells_the_exact_fraction():
    held = ExistingPosition(qty=0.4237, avg_entry_price=200.0)
    result = execution.execute_trade(
        signal("sell"), 210.0, 10000.0, is_live=False, existing_position=held
    )
    assert result.qty == pytest.approx(0.4237)



def test_small_account_buys_notional_instead_of_skipping():
    result = execution.execute_trade(
        signal=signal(symbol="AAPL", action="buy"),
        existing_position=None,
        current_price=100.0,
        live_equity=500.0,  # 2% is $10.00
        is_live=False,
    )
    assert result.qty == pytest.approx(0.5)


def test_notional_entry_is_flagged_for_a_bot_managed_exit():
    result = execution.execute_trade(
        signal=signal(symbol="MSFT", action="buy"),
        existing_position=None,
        current_price=2000.0,
        live_equity=2500.0,  # 2% is $50.00, so 0.025 MSFT
        is_live=False,
    )
    assert "[managed-exit]" in result.message

def test_whole_share_entry_is_not_flagged_for_a_managed_exit():
    result = execution.execute_trade(signal("buy", size=10.0), 100.0, 10000.0, is_live=False)
    assert execution.needs_managed_exit(result) is False
    assert "bracket" in result.message


def test_budget_below_the_alpaca_minimum_is_a_clean_skip():
    result = execution.execute_trade(signal("buy", size=0.05), 300.0, 1000.0, is_live=False)
    assert result.status == "skipped"
    assert "minimum notional" in result.message


# ------------------------------------------------- live Alpaca request shape


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    @property
    def text(self):
        return json.dumps(self._payload)


@pytest.fixture
def capture_alpaca(monkeypatch):
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["body"] = json
        return FakeResponse({"id": "order-1", "filled_avg_price": "100.5", "filled_qty": "10"})

    monkeypatch.setattr(execution.requests, "post", fake_post)
    return sent


def test_live_whole_share_buy_attaches_a_correctly_directed_bracket(capture_alpaca):
    result = execution.execute_trade(
        signal("buy", size=10.0, stop=95.0, take=115.0), 100.0, 10000.0, is_live=True
    )
    body = capture_alpaca["body"]

    assert result.status == "success"
    assert result.order_id == "order-1"
    assert body["order_class"] == "bracket"
    assert body["qty"] == "10"
    assert body["stop_loss"]["stop_price"] == pytest.approx(95.0)
    # Sell stop -> limit below the stop.
    assert body["stop_loss"]["limit_price"] == pytest.approx(94.05)
    assert body["stop_loss"]["limit_price"] < body["stop_loss"]["stop_price"]
    assert body["take_profit"]["limit_price"] == pytest.approx(115.0)


def test_live_closing_sell_attaches_no_bracket(capture_alpaca):
    held = ExistingPosition(qty=4.0, avg_entry_price=90.0)
    execution.execute_trade(
        signal("sell"), 100.0, 10000.0, is_live=True, existing_position=held
    )
    body = capture_alpaca["body"]

    assert body["side"] == "sell"
    assert body["qty"] == "4"
    assert "order_class" not in body
    assert "stop_loss" not in body
    assert "take_profit" not in body



def test_live_notional_buy_sends_notional_and_day_tif(capture_alpaca):
    execution.execute_trade(
        signal=signal(symbol="AAPL", action="buy"),
        existing_position=None,
        current_price=150.0,
        live_equity=1000.0,  # 10% is $100. AAPL is $150. Fractional.
        is_live=True,
    )
    body = capture_alpaca["body"]
    assert float(body["notional"]) == 100.0
    assert "qty" not in body
    assert body["time_in_force"] == "day"

def test_alpaca_rejection_becomes_an_error_result(monkeypatch):
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")

    class Rejected:
        status_code = 422
        text = "insufficient buying power"

        def json(self):
            return {}

    monkeypatch.setattr(execution.requests, "post", lambda *a, **kw: Rejected())
    result = execution.execute_trade(signal("buy", size=10.0), 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "insufficient buying power" in result.message


def test_an_exploding_venue_returns_an_error_not_a_crash(monkeypatch):
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")

    def boom(*a, **kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(execution.requests, "post", boom)
    result = execution.execute_trade(signal("buy", size=10.0), 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "connection reset" in result.message


def test_execute_trade_redacts_a_secret_embedded_in_an_exploding_venues_message(monkeypatch):
    """execution.execute_trade's ExecutionResult.message is persisted verbatim
    into trading_bot.db (a file this project commits to a now-public repo), so
    any secret an unexpected exception happens to embed must never survive
    into it -- regardless of which internal function raised, and regardless of
    whether that call site remembered to sanitize anything itself.
    """
    fake_secret = "sk-ant-realistic-looking-fake-secret-value-123456"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_secret)
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")

    def boom(*a, **kw):
        # A contrived but representative shape: some unrelated downstream
        # exception whose text happens to embed a credential from the
        # environment (e.g. an SDK echoing a header back in a debug message).
        raise RuntimeError(f"upstream call failed, sent header x-api-key: {fake_secret}")

    monkeypatch.setattr(execution.requests, "post", boom)
    result = execution.execute_trade(signal("buy", size=10.0), 100.0, 10000.0, is_live=True)

    assert result.status == "error"
    assert fake_secret not in result.message
    assert "***REDACTED***" in result.message
    assert "upstream call failed" in result.message  # the non-secret context survives


# ------------------------------------------------------------------- crypto


def spot_market(symbol="BTC/USDC", **overrides):
    """A market dict shaped like a real ccxt Hyperliquid SPOT market."""
    market = {
        "id": "@142", "symbol": symbol, "base": symbol.split("/")[0], "quote": "USDC",
        "type": "spot", "spot": True, "swap": False, "contract": False, "active": True,
        "precision": {"amount": 1e-05, "price": 0.001},
        "limits": {"amount": {"min": None}, "cost": {"min": 10.0}},
    }
    market.update(overrides)
    return market


def perp_market(symbol="BTC/USDC:USDC"):
    """A market dict shaped like a real ccxt Hyperliquid PERPETUAL market."""
    return {
        "id": "BTC", "symbol": symbol, "base": "BTC", "quote": "USDC", "settle": "USDC",
        "type": "swap", "spot": False, "swap": True, "contract": True, "active": True,
        "precision": {"amount": 1e-05, "price": 0.1},
        "limits": {"amount": {"min": None}, "cost": {"min": 10.0}},
    }


class FakeHyperliquid:
    def __init__(self, market=None, order=None, balances=None):
        self._market = market if market is not None else spot_market()
        self._order = order or {"id": "hl-1", "average": 50000.0, "filled": 0.002}
        self._balances = balances or {"total": {}}
        self.orders = []
        self.balance_params = []

    def load_markets(self):
        return {self._market["symbol"]: self._market}

    def market(self, symbol):
        return self._market

    def amount_to_precision(self, symbol, amount):
        return "%.8f" % float(amount)

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.orders.append(
            {"symbol": symbol, "type": type_, "side": side,
             "amount": amount, "price": price, "params": params}
        )
        return self._order

    def fetch_balance(self, params=None):
        self.balance_params.append(params or {})
        return self._balances


# ------------------------------------------------- THE SPOT / NO-LEVERAGE GUARD


def test_simulation_reads_the_ledger_and_never_a_broker(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("simulation must not call a broker")

    monkeypatch.setattr(execution.requests, "get", explode)

    held = ExistingPosition(qty=2.0, avg_entry_price=100.0)
    fake = FakeLogger(position=held)
    assert execution.fetch_existing_position("AAPL", False, fake) is held
    assert execution.fetch_existing_position("BTC-USD", False, fake) is held
    assert fake.touched is True


def test_live_equity_position_comes_from_alpaca(monkeypatch):
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    monkeypatch.setattr(
        execution.requests,
        "get",
        lambda *a, **kw: FakeResponse({"qty": "3", "avg_entry_price": "97.25"}),
    )
    position = execution.fetch_existing_position("AAPL", True, FakeLogger())
    assert position.qty == 3.0
    # Alpaca returns the cost basis natively, so nothing is reconstructed.
    assert position.avg_entry_price == 97.25


def test_live_missing_alpaca_position_is_none(monkeypatch):
    class NotFound:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 must be read as flat, not raised on")

    monkeypatch.setattr(execution.requests, "get", lambda *a, **kw: NotFound())
    assert execution.fetch_existing_position("AAPL", True, FakeLogger()) is None


def test_a_broker_outage_raises_instead_of_reporting_flat(monkeypatch):
    # Returning None on a failed lookup would read as "no position held", which
    # the duplicate-buy guard treats as permission to open one. The symbol has to
    # fail loudly so the cycle skips it instead of doubling up.
    def boom(*a, **kw):
        raise RuntimeError("alpaca unreachable")

    monkeypatch.setattr(execution.requests, "get", boom)
    with pytest.raises(RuntimeError, match="alpaca unreachable"):
        execution.fetch_existing_position("AAPL", True, FakeLogger())


def test_live_equity_falls_back_when_no_venue_answers(monkeypatch):
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")

    def boom(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(execution.requests, "get", boom)
    assert execution.fetch_live_equity(1234.0) == 1234.0


@pytest.mark.parametrize("reported,expected", [("0", 0.0), (0, 0.0), ("0.00", 0.0), ("2500.5", 2500.5)])
def test_live_equity_returns_what_alpaca_reports_even_when_it_is_zero(monkeypatch, reported, expected):
    # An empty account read successfully is $0, not the simulation's fallback.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    monkeypatch.setattr(
        execution.requests, "get",
        lambda *a, **kw: FakeResponse({"equity": reported, "cash": "0", "portfolio_value": "0"}),
    )
    assert execution.fetch_live_equity(1000.0) == expected


@pytest.mark.parametrize("payload", [{}, {"equity": None}, {"equity": "n/a"}, {"equity": "nan"}, []])
def test_a_malformed_account_response_is_a_failed_read(monkeypatch, payload):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    monkeypatch.setattr(execution.requests, "get", lambda *a, **kw: FakeResponse(payload))
    assert execution.fetch_live_equity(1000.0) == 1000.0


def test_live_close_books_pnl_from_the_actual_fill(monkeypatch):
    # The circuit breaker acts on this number, so it must reflect what filled,
    # not the pre-trade price the decision was made at.
    # Realistic-length placeholders, not single characters -- a 1-char "secret"
    # (e.g. "s") gets found and redacted everywhere that letter legitimately
    # appears in ordinary text (sanitize() does an exact literal-value scrub),
    # which is a test-fixture footgun, not a sanitize() bug.
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    monkeypatch.setattr(
        execution.requests,
        "post",
        lambda *a, **kw: FakeResponse({"id": "o", "filled_avg_price": "97.0", "filled_qty": "4"}),
    )

    held = ExistingPosition(qty=4.0, avg_entry_price=100.0)
    result = execution.execute_trade(
        signal("sell"), 105.0, 10000.0, is_live=True, existing_position=held
    )

    assert result.fill_price == pytest.approx(97.0)
    # (97 - 100) * 4 = -12, not the (105 - 100) * 4 = +20 the pre-trade price implies.
    assert result.realized_pnl_usd == pytest.approx(-12.0)


