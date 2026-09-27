import json
import logging

import pytest

import notifications
import web_push

SUB = {"endpoint": "https://fcm.googleapis.com/fcm/send/abc123", "keys": {"p256dh": "P" * 87, "auth": "A" * 22}}


@pytest.fixture(scope="module")
def keys():
    return web_push.generate_keys()


@pytest.fixture
def push_env(monkeypatch, keys):
    monkeypatch.setenv("VAPID_PRIVATE_KEY", keys["vapid_private_key"])
    monkeypatch.setenv("PUSH_SUBSCRIPTION_KEY", keys["subscription_private_key"])


def _store(tmp_path, records):
    path = tmp_path / "push_subscriptions.json"
    web_push.write_store(records, str(path))
    return str(path)


class Sender:
    def __init__(self, fail_status=None):
        self.calls = []
        self.fail_status = fail_status

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_status:
            err = Exception("push failed")
            err.response = type("R", (), {"status_code": self.fail_status})()
            raise err


# ------------------------------------------------------------------ keys + crypto


def test_generated_keys_have_the_formats_browsers_and_pywebpush_expect(keys):
    import base64

    from py_vapid import Vapid02

    assert Vapid02.from_string(keys["vapid_private_key"]) is not None
    public = base64.urlsafe_b64decode(keys["vapid_public_key"] + "==")
    assert len(public) == 65 and public[0] == 4  # uncompressed P-256 point


def test_subscription_round_trips_and_the_record_holds_only_ciphertext(keys):
    record = web_push.encrypt_subscription(SUB, keys["subscription_public_key"])
    assert SUB["endpoint"] not in json.dumps(record)
    assert record["id"] == web_push.subscription_id(SUB["endpoint"])
    assert web_push.decrypt_subscription(record, keys["subscription_private_key"]) == SUB


def test_a_record_for_the_wrong_key_does_not_decrypt(keys):
    other = web_push.generate_keys()
    record = web_push.encrypt_subscription(SUB, other["subscription_public_key"])
    with pytest.raises(Exception):
        web_push.decrypt_subscription(record, keys["subscription_private_key"])


@pytest.mark.parametrize("endpoint,ok", [
    ("https://fcm.googleapis.com/fcm/send/x", True),
    ("https://updates.push.services.mozilla.com/wpush/v2/x", True),
    ("https://web.push.apple.com/x", True),
    ("https://db5p.notify.windows.com/w/?token=x", True),
    ("http://fcm.googleapis.com/x", False),
    ("https://evil.example/x", False),
    ("https://fcm.googleapis.com.evil.example/x", False),
    (None, False),
])
def test_only_real_push_services_are_accepted(endpoint, ok):
    assert web_push.endpoint_allowed(endpoint) is ok


def test_a_decrypted_subscription_to_an_arbitrary_host_is_refused(keys):
    record = web_push.encrypt_subscription({**SUB, "endpoint": "https://evil.example/x"},
                                           keys["subscription_public_key"])
    with pytest.raises(ValueError):
        web_push.decrypt_subscription(record, keys["subscription_private_key"])


# ------------------------------------------------------------------ the store


def test_subscribe_then_resubscribe_then_unsubscribe(keys):
    record = web_push.encrypt_subscription(SUB, keys["subscription_public_key"])
    subs = web_push.apply_change([], "push_subscribe", record)
    subs = web_push.apply_change(subs, "push_subscribe", record)  # same device: replaced, not duplicated
    assert len(subs) == 1 and subs[0]["added_at"]
    assert web_push.apply_change(subs, "push_unsubscribe", {"id": record["id"]}) == []


@pytest.mark.parametrize("payload", [
    None, [], {}, {"id": "nothex", "v": 1, "k": "a", "iv": "a", "ct": "a"},
    {"id": "a" * 64, "v": 2, "k": "a", "iv": "a", "ct": "a"},
    {"id": "a" * 64, "v": 1, "k": "not base64!", "iv": "a", "ct": "a"},
    {"id": "a" * 64, "v": 1, "k": "a", "iv": "a", "ct": "a" * 5000},
])
def test_malformed_subscribe_payloads_are_rejected(payload):
    with pytest.raises(ValueError):
        web_push.apply_change([], "push_subscribe", payload)


def test_the_store_is_capped(keys):
    subs = [{"id": f"{i:064x}"} for i in range(web_push.MAX_SUBSCRIPTIONS)]
    record = web_push.encrypt_subscription(SUB, keys["subscription_public_key"])
    with pytest.raises(ValueError, match="already"):
        web_push.apply_change(subs, "push_subscribe", record)


def test_store_cli_reads_the_payload_from_the_environment(tmp_path, monkeypatch, keys):
    record = web_push.encrypt_subscription(SUB, keys["subscription_public_key"])
    path = str(tmp_path / "subs.json")
    monkeypatch.setenv("PAYLOAD", json.dumps(record))
    assert web_push.main(["store", "--action", "push_subscribe", "--path", path]) == 0
    assert [s["id"] for s in web_push.load_store(path)] == [record["id"]]
    monkeypatch.setenv("PAYLOAD", '{"id": "bad"}')
    assert web_push.main(["store", "--action", "push_unsubscribe", "--path", path]) == 1


# ------------------------------------------------------------------ sending


def test_sending_is_a_noop_without_keys(tmp_path):
    sender = Sender()
    assert web_push.send_to_all("t", "b", store_path=str(tmp_path / "none.json"), sender=sender) == 0
    assert sender.calls == []


def test_sends_the_decrypted_subscription_signed_with_vapid(tmp_path, push_env, keys):
    path = _store(tmp_path, [web_push.encrypt_subscription(SUB, keys["subscription_public_key"])])
    sender = Sender()
    assert web_push.send_to_all("TradingBot Papin", "BUY 10 AAPL @ $182.30", store_path=path,
                                dead_path=str(tmp_path / "dead.json"), sender=sender) == 1
    call = sender.calls[0]
    assert call["subscription_info"] == SUB
    assert json.loads(call["data"]) == {"title": "TradingBot Papin", "body": "BUY 10 AAPL @ $182.30"}
    assert call["vapid_private_key"] == keys["vapid_private_key"]
    assert call["vapid_claims"]["sub"].startswith(("https://", "mailto:"))


def test_a_gone_subscription_is_remembered_and_skipped(tmp_path, push_env, keys):
    path = _store(tmp_path, [web_push.encrypt_subscription(SUB, keys["subscription_public_key"])])
    dead = str(tmp_path / "dead.json")
    assert web_push.send_to_all("t", "b", store_path=path, dead_path=dead, sender=Sender(410)) == 0
    later = Sender()
    web_push.send_to_all("t", "b", store_path=path, dead_path=dead, sender=later)
    assert later.calls == []


def test_one_unreadable_record_does_not_block_the_others(tmp_path, push_env, keys):
    good = web_push.encrypt_subscription(SUB, keys["subscription_public_key"])
    bad = {**good, "id": "b" * 64, "ct": "AAAA"}
    path = _store(tmp_path, [bad, good])
    sender = Sender()
    assert web_push.send_to_all("t", "b", store_path=path, dead_path=str(tmp_path / "d.json"), sender=sender) == 1


def test_a_failing_push_service_never_raises(tmp_path, push_env, keys, caplog):
    path = _store(tmp_path, [web_push.encrypt_subscription(SUB, keys["subscription_public_key"])])
    with caplog.at_level(logging.ERROR):
        assert web_push.send_to_all("t", "b", store_path=path, dead_path=str(tmp_path / "d.json"),
                                    sender=Sender(500)) == 0
    assert "Web push failed" in caplog.text
    assert keys["vapid_private_key"] not in caplog.text


# ------------------------------------------------------------------ alerts


def test_alerts_go_to_web_push_even_without_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(web_push, "send_to_all", lambda title, body: sent.append((title, body)))
    notifications.send_trade_alert(True, "AAPL", "buy", 10.0, 182.30)
    assert sent == [("TradingBot Papin", "BUY 10 AAPL @ $182.30")]


def test_a_web_push_crash_never_breaks_an_alert(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(web_push, "send_to_all", boom)
    notifications.send_cycle_failure_alert(True, "x")  # must not raise


def test_the_workflow_only_reads_the_payload_from_env():
    import yaml

    with open(".github/workflows/push_subscriptions.yml", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    job = workflow["jobs"]["store"]
    assert "client_payload" in job["env"]["PAYLOAD"]
    for step in job["steps"]:
        assert "client_payload" not in step.get("run", "")
    assert set(workflow[True]["repository_dispatch"]["types"]) == {"push_subscribe", "push_unsubscribe"}
