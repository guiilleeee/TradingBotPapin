"""The bot trades autonomously: no order waits for a human, and alerts only report."""

import pytest
import yaml

import main
import notifications
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market


def test_shipped_config_has_approval_mode_off():
    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["approval_mode"] is False
    assert "approval_timeout_seconds" not in config


def test_a_config_asking_for_approval_refuses_to_trade():
    with pytest.raises(ValueError, match="approval_mode"):
        main._run_cycle_body({"approval_mode": True}, is_live=True)


def test_there_is_no_approval_module_left():
    with pytest.raises(ImportError):
        import approval  # noqa: F401


def _process(tmp_logger, monkeypatch, is_live):
    stub_market(monkeypatch, {"AAPL": 200.0})
    calls = record_executions(monkeypatch)
    alerts = []
    monkeypatch.setattr(notifications, "send_trade_alert", lambda **k: alerts.append(k))
    main._process_symbol(
        symbol="AAPL",
        config={"approval_mode": False},
        settings=settings(is_live=is_live),
        bot_logger=tmp_logger,
        generate_signal=lambda signal_input, system_prompt: buy_signal("AAPL", 200.0, stop_pct=0.02),
        equity=5000.0,
        circuit_breaker_loss_pct=3.0,
        max_risk_pct=1.0,
        max_absolute_position_pct=20.0,
        min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )
    return calls, alerts


@pytest.mark.parametrize("is_live", [True, False])
def test_a_buy_executes_immediately_and_alerts_qty_and_price(tmp_logger, monkeypatch, is_live):
    def no_telegram(*a, **k):
        raise AssertionError("a trade must never talk to Telegram before executing")

    monkeypatch.setattr(notifications, "telegram_call", no_telegram)
    calls, alerts = _process(tmp_logger, monkeypatch, is_live)

    assert len(calls) == 1
    assert len(alerts) == 1
    alert = alerts[0]
    assert set(alert) == {"is_live", "symbol", "action", "qty", "price"}
    assert (alert["symbol"], alert["action"], alert["is_live"]) == ("AAPL", "buy", is_live)
    assert alert["qty"] is not None and alert["price"] > 0
