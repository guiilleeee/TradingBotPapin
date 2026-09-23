import pytest
import notifications
import logging

def test_send_alert_dispatches_to_web_push(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_telegram", lambda s, m, c: called.append("telegram"))
    
    config = {"web_push": {"enabled": True}, "telegram": {"enabled": False}}
    notifications.send_alert("Subj", "Msg", config)
    
    assert "web" in called
    assert "telegram" not in called

def test_send_alert_dispatches_to_telegram_if_enabled(monkeypatch):
    called = []
    monkeypatch.setattr(notifications, "_send_web_push", lambda s, m, c: called.append("web"))
    monkeypatch.setattr(notifications, "_send_telegram", lambda s, m, c: called.append("telegram"))
    
    config = {"web_push": {"enabled": False}, "telegram": {"enabled": True}}
    notifications.send_alert("Subj", "Msg", config)
    
    assert "web" not in called
    assert "telegram" in called

def test_failed_push_send_does_not_break_cycle(monkeypatch, caplog):
    def boom(*a, **kw):
        raise RuntimeError("network down")
        
    monkeypatch.setattr(notifications, "_send_web_push", boom)
    monkeypatch.setattr(notifications, "_send_telegram", boom)
    
    config = {"web_push": {"enabled": True}, "telegram": {"enabled": True}}
    
    # This should not raise an exception
    with caplog.at_level(logging.ERROR):
        notifications.send_alert("Subj", "Msg", config)
        
    assert "Web push failed: network down" in caplog.text
    assert "Telegram push failed: network down" in caplog.text
