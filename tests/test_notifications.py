import logging

import pytest

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
    notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 200.0)
    notifications.send_auto_close_alert(True, "AAPL", 10.0, 195.0)
    notifications.send_cycle_failure_alert(False, "boom")
    notifications.send_circuit_breaker_alert(True, -3.5, 3.0)


def test_a_token_without_a_chat_id_sends_nothing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notifications.requests, "post", _must_not_send)
    notifications.send_cycle_failure_alert(True, "x")


def _capture(monkeypatch):
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
    return sent


def test_typed_alert_sends_plain_text_to_the_configured_chat(monkeypatch):
    sent = _capture(monkeypatch)
    notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 182.30)

    assert len(sent) == 1
    url, payload = sent[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert payload["chat_id"] == CHAT
    assert "parse_mode" not in payload
    assert "reply_markup" not in payload


@pytest.mark.parametrize("call,expected", [
    (lambda: notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 182.30), "BUY 10 AAPL @ $182.30"),
    (lambda: notifications.send_trade_alert(True, "TSLA", "sell", 5, 410.1), "SELL 5 TSLA @ $410.10"),
    (lambda: notifications.send_trade_alert(True, "BTC-USD", "buy", 0.01234, 61234.5),
     "BUY 0.01234 BTC-USD @ $61,234.50"),
    (lambda: notifications.send_trade_alert(True, "DOGE-USD", "buy", 1500, 0.123456),
     "BUY 1500 DOGE-USD @ $0.123456"),
    (lambda: notifications.send_trade_alert(False, "AAPL", "buy", 10.0, 182.30), "[SIM] BUY 10 AAPL @ $182.30"),
    (lambda: notifications.send_trade_alert(True, "AAPL", "buy", None, 182.30), "BUY AAPL @ $182.30"),
    (lambda: notifications.send_auto_close_alert(True, "BTC-USD", 0.01, 57000.0), "SELL 0.01 BTC-USD @ $57,000.00"),
])
def test_trade_alerts_are_one_line_saying_only_what_happened(monkeypatch, call, expected):
    sent = _capture(monkeypatch)
    call()
    assert sent[0][1]["text"] == expected


def test_every_alert_is_a_single_line(monkeypatch):
    sent = _capture(monkeypatch)
    notifications.send_cycle_failure_alert(True, "ValueError: boom\nTraceback line 1\nline 2")
    notifications.send_screening_failure_alert(True, "first\nsecond")
    notifications.send_circuit_breaker_alert(True, -3.5, 3.0)
    notifications.send_screening_complete_alert(True, ["AAPL", "MSFT"])
    notifications.send_volume_wake_alert("AAPL", 182.3, ["volume x3", "price +4%"], "buy")
    texts = [payload["text"] for _, payload in sent]
    assert len(texts) == 5
    assert all("\n" not in t for t in texts)
    assert texts[0] == "CYCLE FAILED: ValueError: boom"


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
                 "send_cycle_failure_alert"):
        assert callable(getattr(notifications, name))
    assert not hasattr(notifications, "send_approval_outcome_alert")


def test_malformed_credentials_send_no_alerts(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "not-a-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setattr(notifications.requests, "post", _must_not_send)
    notifications.send_cycle_failure_alert(True, "x")
