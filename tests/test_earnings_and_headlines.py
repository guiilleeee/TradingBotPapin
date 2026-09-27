"""Days-to-earnings reaches the model and blocks buys in the blackout window;
headlines older than 72h are dropped and the rest carry their age."""

from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime

import pandas as pd
import pytest

import data_fetcher
import main
import risk_manager
from models import SignalOutput
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def rss(*items):
    body = "".join(
        f"<item><title>{title}</title>"
        + (f"<pubDate>{format_datetime(when)}</pubDate>" if when else "")
        + "</item>"
        for title, when in items
    )
    return f"<rss><channel>{body}</channel></rss>".encode()


def test_headlines_older_than_72h_are_dropped_and_the_rest_show_their_age():
    xml = rss(
        ("Fresh", NOW - timedelta(hours=5)),
        ("Stale", NOW - timedelta(hours=80)),
        ("Undated", None),
        ("Yesterday", NOW - timedelta(hours=30)),
    )
    assert data_fetcher.parse_headlines(xml, now=NOW) == ["[5h ago] Fresh", "[30h ago] Yesterday"]


def test_headline_limit_still_applies():
    xml = rss(*[(f"H{i}", NOW - timedelta(hours=i)) for i in range(10)])
    assert len(data_fetcher.parse_headlines(xml, limit=3, now=NOW)) == 3


def test_headline_fetch_never_raises(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(data_fetcher.requests, "get", boom)
    assert data_fetcher.fetch_headlines("AAPL") == []


@pytest.mark.parametrize("calendar, expected", [
    ({"Earnings Date": [date(2026, 10, 1), date(2026, 10, 3)]}, 4),
    ({"Earnings Date": [date(2026, 9, 20), date(2026, 9, 29)]}, 2),   # past date skipped
    ({"Earnings Date": [date(2026, 9, 27)]}, 0),
    ({"Earnings Date": [date(2026, 9, 1)]}, None),                    # only a past date
    ({"Earnings Date": date(2026, 10, 7)}, 10),                       # scalar
    ({}, None),
    (None, None),
    (pd.DataFrame({0: [pd.Timestamp("2026-10-02")]}, index=["Earnings Date"]), 5),
])
def test_days_to_earnings_parses_the_calendar(calendar, expected):
    assert data_fetcher.days_to_earnings_from_calendar(calendar, today=date(2026, 9, 27)) == expected


def _validate(action="buy", days=None, blackout=2):
    raw = SignalOutput(symbol="AAPL", action=action, confidence=0.9, position_size_pct=5.0,
                       stop_loss_price=95.0, take_profit_price=115.0, reasoning="prova")
    return risk_manager.validate(raw=raw, current_price=100.0, today_realized_loss_pct=0.0,
                                 circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
                                 max_absolute_position_pct=20.0, min_confidence=0.6,
                                 days_to_earnings=days, earnings_blackout_days=blackout)


@pytest.mark.parametrize("days, action", [(0, "hold"), (2, "hold"), (3, "buy"), (None, "buy")])
def test_earnings_blackout_blocks_buys_inside_the_window(days, action):
    result = _validate(days=days)
    assert result.action == action
    if action == "hold":
        assert "earnings" in result.override_reason


def test_earnings_blackout_never_blocks_a_sell():
    assert _validate(action="sell", days=0).action == "sell"


def test_earnings_blackout_of_zero_disables_the_rule():
    assert _validate(days=0, blackout=0).action == "buy"


def _cycle(tmp_logger, monkeypatch, symbol, days, config):
    stub_market(monkeypatch, {symbol: 100.0})
    calls = record_executions(monkeypatch)
    asked = []
    monkeypatch.setattr(data_fetcher, "fetch_days_to_earnings",
                        lambda s, today=None: asked.append(s) or days)
    seen = []

    def model(signal_input, system_prompt):
        seen.append(signal_input)
        return buy_signal(symbol, 100.0, stop_pct=0.02)

    main._process_symbol(
        symbol=symbol, config=config, settings=settings(is_live=False), bot_logger=tmp_logger,
        generate_signal=model, equity=10000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )
    return calls, seen, asked


def test_cycle_passes_days_to_earnings_and_the_rules_to_the_model_and_blocks(tmp_logger, monkeypatch):
    calls, seen, asked = _cycle(tmp_logger, monkeypatch, "AAPL", 1, {"earnings_blackout_days": 2})
    assert asked == ["AAPL"]
    assert seen[0].days_to_earnings == 1
    assert seen[0].entry_rules.earnings_blackout_days == 2
    assert seen[0].entry_rules.min_reward_risk_ratio == 1.5
    assert calls == []


def test_crypto_never_looks_up_earnings(tmp_logger, monkeypatch):
    config = {"symbols": [{"symbol": "BTC-USD", "asset_class": "crypto"}]}
    calls, seen, asked = _cycle(tmp_logger, monkeypatch, "BTC-USD", 0, config)
    assert asked == []
    assert seen[0].days_to_earnings is None
    assert len(calls) == 1
