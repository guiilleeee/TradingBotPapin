import logging

import notifications


def test_send_alert_dispatches_to_web_push(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_ntfy", lambda s, m, c: called.append("ntfy"))

    notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": True}, "ntfy": {"enabled": False}})

    assert called == ["web"]


def test_send_alert_dispatches_to_ntfy_if_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_ntfy", lambda s, m, c: called.append("ntfy"))

    notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": False}, "ntfy": {"enabled": True}})

    assert called == ["ntfy"]


def test_failed_push_send_does_not_break_cycle(monkeypatch, caplog):
    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifications, "_send_web_push", boom)
    monkeypatch.setattr(notifications, "_send_ntfy", boom)

    with caplog.at_level(logging.ERROR):
        notifications.send_alert("Subj", "Msg", {"web_push": {"enabled": True}, "ntfy": {"enabled": True}})

    assert "Web push failed: network down" in caplog.text
    assert "ntfy push failed: network down" in caplog.text


# --------------------------------------------------------- typed alerts


def test_typed_alerts_are_a_silent_noop_without_a_topic(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)

    def must_not_send(*a, **k):
        raise AssertionError("no send without NTFY_TOPIC")

    monkeypatch.setattr(notifications.requests, "post", must_not_send)
    notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 200.0, 0.8, "x")
    notifications.send_auto_close_alert(True, "AAPL", "stop", -5.0)
    notifications.send_cycle_failure_alert(False, "boom")
    notifications.send_circuit_breaker_alert(True, -3.5, 3.0)


def test_typed_alert_publishes_to_the_configured_topic(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "tb-0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("NTFY_TOKEN", "tk_testtoken")
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    sent = []

    class Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "x"}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append((url, json, headers))
        return Ok()

    monkeypatch.setattr(notifications.requests, "post", fake_post)
    notifications.send_auto_close_alert(True, "BTC-USD", "Stop-loss activat", -12.5)

    assert len(sent) == 1
    url, payload, headers = sent[0]
    assert url == "https://ntfy.sh"
    assert payload["topic"] == "tb-0123456789abcdef0123456789abcdef"
    assert "BTC-USD" in payload["title"] and "-12.50" in payload["message"]
    assert headers == {"Authorization": "Bearer tk_testtoken"}


def test_typed_alert_failure_never_raises(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "tb-0123456789abcdef0123456789abcdef")

    def boom(*a, **k):
        raise RuntimeError("ntfy down")

    monkeypatch.setattr(notifications.requests, "post", boom)
    notifications.send_cycle_failure_alert(True, "x")  # must not raise


def test_every_alert_main_calls_exists():
    """main.py called functions that did not exist (telegram_alerts.*, two missing
    notifications.*) -- an auto-close crashed the whole cycle. Pin the surface."""
    for name in ("send_trade_alert", "send_circuit_breaker_alert", "send_auto_close_alert",
                 "send_cycle_failure_alert", "send_approval_outcome_alert"):
        assert callable(getattr(notifications, name))


def test_a_weak_topic_sends_no_alerts(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "tradingbot")

    def must_not_send(*a, **k):
        raise AssertionError("weak topic must not be used")

    monkeypatch.setattr(notifications.requests, "post", must_not_send)
    notifications.send_cycle_failure_alert(True, "x")
