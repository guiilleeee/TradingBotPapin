import logging

import notifications

TOKEN = "123456789:" + "A" * 35
CHAT = "555000111"


def test_send_alert_dispatches_to_web_push(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_telegram", lambda s, m, c: called.append("telegram"))

    notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": True}, "telegram": {"enabled": False}})

    assert called == ["web"]


def test_send_alert_dispatches_to_telegram_if_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_telegram", lambda s, m, c: called.append("telegram"))

    notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": False}, "telegram": {"enabled": True}})

    assert called == ["telegram"]


def test_failed_push_send_does_not_break_cycle(monkeypatch, caplog):
    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifications, "_send_web_push", boom)
    monkeypatch.setattr(notifications, "_send_telegram", boom)

    with caplog.at_level(logging.ERROR):
        notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": True}, "telegram": {"enabled": True}})

    assert "Web push failed: network down" in caplog.text
    assert "Telegram push failed: network down" in caplog.text


# --------------------------------------------------------- typed alerts


def _must_not_send(*a, **k):
    raise AssertionError("must not send")


def test_typed_alerts_are_a_silent_noop_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notifications.requests, "post", _must_not_send)
    notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 200.0, 0.8, "x")
    notifications.send_auto_close_alert(True, "AAPL", "stop", -5.0)
    notifications.send_cycle_failure_alert(False, "boom")
    notifications.send_circuit_breaker_alert(True, -3.5, 3.0)


def test_a_token_without_a_chat_id_sends_nothing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notifications.requests, "post", _must_not_send)
    notifications.send_cycle_failure_alert(True, "x")


def test_typed_alert_sends_plain_text_to_the_configured_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    sent = []

    class Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True, "result": {"message_id": 1}}

    def fake_post(url, json=None, timeout=None):
        sent.append((url, json))
        return Ok()

    monkeypatch.setattr(notifications.requests, "post", fake_post)
    notifications.send_auto_close_alert(True, "BTC-USD", "Stop-loss activat", -12.5)

    assert len(sent) == 1
    url, payload = sent[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert payload["chat_id"] == CHAT
    assert "parse_mode" not in payload
    assert "BTC-USD" in payload["text"] and "-12.50" in payload["text"]


def test_typed_alert_failure_never_raises(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)

    def boom(*a, **k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(notifications.requests, "post", boom)
    notifications.send_cycle_failure_alert(True, "x")  # must not raise


def test_a_logged_failure_never_contains_the_token(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)

    def boom(url, **k):
        raise RuntimeError(f"Connection refused: {url}")

    monkeypatch.setattr(notifications.requests, "post", boom)
    with caplog.at_level(logging.ERROR):
        notifications.send_cycle_failure_alert(True, "x")
    assert "Telegram push failed" in caplog.text
    assert TOKEN not in caplog.text


def test_every_alert_main_calls_exists():
    """main.py called functions that did not exist (telegram_alerts.*, two missing
    notifications.*) -- an auto-close crashed the whole cycle. Pin the surface."""
    for name in ("send_trade_alert", "send_circuit_breaker_alert", "send_auto_close_alert",
                 "send_cycle_failure_alert", "send_approval_outcome_alert"):
        assert callable(getattr(notifications, name))


def test_malformed_credentials_send_no_alerts(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "not-a-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setattr(notifications.requests, "post", _must_not_send)
    notifications.send_cycle_failure_alert(True, "x")
