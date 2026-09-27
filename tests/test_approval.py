"""Approval mode: the ntfy push approve/reject round-trip, and its gating of the cycle.

No test reaches ntfy: a FakeNtfy stands in for notifications.ntfy_publish /
ntfy_poll, and a fake clock drives the timeout without any real waiting.
"""

import pytest

import approval
import execution
import main
import notifications
from models import ExistingPosition
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market

TOPIC = "tb-test-topic-0123456789abcdef"


@pytest.fixture
def ntfy_env(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", TOPIC)
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    monkeypatch.delenv("NTFY_TOKEN", raising=False)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.001)


class FakeNtfy:
    """Records publishes; `replies` are message bodies that appear on the reply
    topic from the `reply_after_polls`-th poll on. A body "{id}" placeholder is
    filled with the real request id once the request has been published."""

    def __init__(self, replies=(), reply_after_polls=1, fail_publish=False, fail_polls=0, raw=()):
        self.replies = list(replies)
        self.raw = list(raw)
        self.reply_after_polls = reply_after_polls
        self.fail_publish = fail_publish
        self.fail_polls = fail_polls
        self.published = []
        self.polls = []

    @property
    def request_id(self):
        body = self.published[0]["actions"][0]["body"]
        return body.split(":")[0]

    def publish(self, payload):
        if self.fail_publish:
            raise RuntimeError("ntfy unreachable")
        self.published.append(payload)
        return {"id": "m1"}

    def poll(self, topic, since):
        self.polls.append((topic, since))
        if self.fail_polls:
            self.fail_polls -= 1
            raise RuntimeError("poll failed")
        if len(self.polls) < self.reply_after_polls:
            return []
        out = [{"event": "message", "message": b.replace("{id}", self.request_id)} for b in self.replies]
        return self.raw + out


def request(fake, clock, timeout=600):
    return approval.request_approval(
        symbol="AAPL", action="buy", size_pct=10.0, price=200.0, stop_loss=196.0,
        take_profit=212.0, confidence=0.8, reasoning="Tendencia alcista.", equity=5000.0,
        timeout_seconds=timeout, publish=fake.publish, poll=fake.poll,
        clock=clock, wall_clock=lambda: 1_700_000_000.0, sleep=clock.sleep,
    )


# -------------------------------------------------------------- config switch


@pytest.mark.parametrize("value,enabled", [
    (True, True), (False, False), ("true", False), (1, False), (None, False),
])
def test_approval_mode_is_only_on_for_a_literal_true(value, enabled):
    assert approval.approval_enabled({"approval_mode": value}) is enabled


def test_approval_mode_defaults_off():
    assert approval.approval_enabled({}) is False


def test_shipped_config_has_approval_mode_off():
    import yaml

    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["approval_mode"] is False
    assert approval.approval_timeout_seconds(config) == 600


# ------------------------------------------------------------- the push itself


def test_proposal_is_a_max_priority_push_with_two_action_buttons(ntfy_env):
    clock = FakeClock()
    fake = FakeNtfy(replies=["{id}:approve"])
    request(fake, clock)
    push = fake.published[0]

    assert push["topic"] == TOPIC
    assert push["priority"] == 5
    assert "BUY AAPL" in push["title"]
    approve, reject = push["actions"]
    for button, label, verdict in ((approve, "Aprovar", "approve"), (reject, "Rebutjar", "reject")):
        assert button["action"] == "http" and button["method"] == "POST"
        assert button["label"] == label
        assert button["url"] == f"https://ntfy.sh/{TOPIC}-reply"
        assert button["body"] == f"{fake.request_id}:{verdict}"
        assert button["clear"] is True
    for fragment in ("200", "10.00%", "$500.00", "196", "212", "0.80", "Tendencia alcista."):
        assert fragment in push["message"]


def test_buttons_carry_the_access_token_for_a_protected_server(ntfy_env, monkeypatch):
    monkeypatch.setenv("NTFY_SERVER", "https://ntfy.example.org/")
    monkeypatch.setenv("NTFY_TOKEN", "tk_testtoken")
    clock = FakeClock()
    fake = FakeNtfy(replies=["{id}:approve"])
    request(fake, clock)
    button = fake.published[0]["actions"][0]
    assert button["url"] == f"https://ntfy.example.org/{TOPIC}-reply"
    assert button["headers"] == {"Authorization": "Bearer tk_testtoken"}


def test_each_request_gets_a_fresh_id(ntfy_env):
    ids = set()
    for _ in range(3):
        fake = FakeNtfy(replies=["{id}:approve"])
        request(fake, FakeClock())
        ids.add(fake.request_id)
    assert len(ids) == 3


def test_polling_reads_the_reply_topic_since_just_before_the_send(ntfy_env):
    clock = FakeClock()
    fake = FakeNtfy(replies=["{id}:approve"], reply_after_polls=3)
    request(fake, clock)
    assert all(topic == f"{TOPIC}-reply" for topic, _ in fake.polls)
    assert all(since == 1_700_000_000 - approval.SINCE_SKEW_SECONDS for _, since in fake.polls)


# ------------------------------------------------------------- outcomes


def test_approve_tap_approves(ntfy_env):
    clock = FakeClock()
    fake = FakeNtfy(replies=["{id}:approve"], reply_after_polls=4)
    decision = request(fake, clock)
    assert decision.approved is True
    assert decision.outcome == "approved"
    # Follow-up push so the phone shows how it ended.
    assert "APROVADA" in fake.published[-1]["title"]


def test_reject_tap_rejects(ntfy_env):
    decision = request(FakeNtfy(replies=["{id}:reject"]), FakeClock())
    assert decision.approved is False
    assert decision.outcome == "rejected"


def test_approve_and_reject_both_seen_resolves_to_reject(ntfy_env):
    decision = request(FakeNtfy(replies=["{id}:approve", "{id}:reject"]), FakeClock())
    assert decision.outcome == "rejected"


def test_no_answer_times_out_after_the_configured_window(ntfy_env):
    clock = FakeClock()
    fake = FakeNtfy()
    decision = request(fake, clock, timeout=600)
    assert decision.approved is False
    assert decision.outcome == "timeout"
    assert 600 <= clock.now <= 600 + approval.POLL_INTERVAL_SECONDS
    assert "CADUCADA" in fake.published[-1]["title"]


@pytest.mark.parametrize("body", [
    "deadbeefdeadbeef:approve",      # another request's id
    "{id}:APPROVE",                  # not the exact body
    "{id}:approve please",
    "approve",
    "{id}",
    "",
])
def test_anything_but_this_requests_exact_approve_body_is_ignored(ntfy_env, body):
    decision = request(FakeNtfy(replies=[body]), FakeClock(), timeout=60)
    assert decision.outcome == "timeout"


def test_malformed_reply_entries_are_ignored(ntfy_env):
    raw = [None, "garbage", {"event": "message"}, {"message": 12345}, {"event": "keepalive"}]
    decision = request(FakeNtfy(raw=raw), FakeClock(), timeout=60)
    assert decision.outcome == "timeout"


def test_transient_poll_failures_keep_waiting(ntfy_env):
    decision = request(FakeNtfy(replies=["{id}:approve"], fail_polls=3), FakeClock())
    assert decision.approved is True


def test_polls_failing_until_the_deadline_is_a_timeout(ntfy_env):
    decision = request(FakeNtfy(replies=["{id}:approve"], fail_polls=10_000), FakeClock(), timeout=60)
    assert decision.approved is False
    assert decision.outcome == "timeout"


def test_unreachable_ntfy_is_not_an_approval(ntfy_env):
    decision = request(FakeNtfy(fail_publish=True), FakeClock())
    assert decision.approved is False
    assert decision.outcome == "unavailable"


def test_missing_topic_is_not_an_approval(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    fake = FakeNtfy(replies=["{id}:approve"])
    decision = request(fake, FakeClock())
    assert decision.approved is False
    assert decision.outcome == "unavailable"
    assert fake.published == []


def test_ntfy_poll_skips_lines_that_are_not_json(ntfy_env, monkeypatch):
    class Resp:
        text = '{"event":"message","message":"a:approve"}\nnot json\n{"event":"open"}\n'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(notifications.requests, "get", lambda *a, **k: Resp())
    assert notifications.ntfy_poll(f"{TOPIC}-reply", 0) == [{"event": "message", "message": "a:approve"}]


def test_ntfy_poll_raises_on_http_error(ntfy_env, monkeypatch):
    class Resp:
        def raise_for_status(self):
            raise RuntimeError("502")

    monkeypatch.setattr(notifications.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError):
        notifications.ntfy_poll(f"{TOPIC}-reply", 0)

# ------------------------------------------------------------ cycle gating


def _run(tmp_logger, monkeypatch, *, is_live, approval_mode, decision=None, prices=None):
    stub_market(monkeypatch, prices or {"AAPL": 200.0})
    calls = record_executions(monkeypatch)
    requests_made = []

    def fake_request(**kwargs):
        requests_made.append(kwargs)
        return decision

    monkeypatch.setattr(approval, "request_approval", fake_request)
    monkeypatch.setattr(notifications, "send_approval_outcome_alert", lambda *a, **k: None)

    config = {"approval_mode": approval_mode, "approval_timeout_seconds": 600}
    main._process_symbol(
        symbol="AAPL",
        config=config,
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
    return calls, requests_made


def _last_execution(read_signals):
    return read_signals()[-1]["execution_result"]


def test_approval_off_executes_immediately(tmp_logger, monkeypatch):
    calls, asked = _run(tmp_logger, monkeypatch, is_live=True, approval_mode=False)
    assert len(calls) == 1
    assert asked == []


def test_simulation_is_never_gated(tmp_logger, monkeypatch):
    calls, asked = _run(tmp_logger, monkeypatch, is_live=False, approval_mode=True)
    assert len(calls) == 1
    assert asked == []


def test_approved_live_order_executes(tmp_logger, read_signals, monkeypatch):
    calls, asked = _run(
        tmp_logger, monkeypatch, is_live=True, approval_mode=True,
        decision=approval.ApprovalDecision(True, "approved"),
    )
    assert len(asked) == 1
    assert asked[0]["symbol"] == "AAPL" and asked[0]["action"] == "buy"
    assert asked[0]["timeout_seconds"] == 600
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ["rejected", "timeout", "unavailable"])
def test_unapproved_live_order_is_skipped_and_logged(tmp_logger, read_signals, monkeypatch, outcome):
    calls, _ = _run(
        tmp_logger, monkeypatch, is_live=True, approval_mode=True,
        decision=approval.ApprovalDecision(False, outcome, "detail"),
    )
    assert calls == []
    result = _last_execution(read_signals)
    assert result["status"] == "skipped"
    assert f"approval {outcome}" in result["message"]
    # Nothing was bought, so nothing is on the ledger.
    assert tmp_logger.get_simulated_position("AAPL") is None


def test_approved_order_uses_the_rechecked_price(tmp_logger, monkeypatch):
    calls, _ = _run(
        tmp_logger, monkeypatch, is_live=True, approval_mode=True,
        decision=approval.ApprovalDecision(True, "approved"),
        prices={"AAPL": [200.0, 201.0]},
    )
    assert calls[0]["price"] == 201.0


def test_approved_but_stop_already_crossed_is_skipped(tmp_logger, read_signals, monkeypatch):
    # Stop is 196 (2% under 200); by the time the tap arrives the price is 195.
    calls, _ = _run(
        tmp_logger, monkeypatch, is_live=True, approval_mode=True,
        decision=approval.ApprovalDecision(True, "approved"),
        prices={"AAPL": [200.0, 195.0]},
    )
    assert calls == []
    result = _last_execution(read_signals)
    assert result["status"] == "skipped"
    assert "no longer valid" in result["message"]


def test_a_hold_never_asks_for_approval(tmp_logger, monkeypatch):
    stub_market(monkeypatch, {"AAPL": 200.0})
    record_executions(monkeypatch)
    monkeypatch.setattr(
        approval, "request_approval",
        lambda **k: (_ for _ in ()).throw(AssertionError("hold must not ask")),
    )
    from models import SignalOutput

    main._process_symbol(
        symbol="AAPL", config={"approval_mode": True}, settings=settings(is_live=True),
        bot_logger=tmp_logger,
        generate_signal=lambda si, system_prompt: SignalOutput(symbol="AAPL", action="hold", confidence=0.3),
        equity=5000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )


def test_automatic_stop_exits_are_never_gated(tmp_logger, monkeypatch):
    """The sweep's live managed exit goes straight to execution, approval mode or not."""
    import pandas as pd

    from models import ExecutionResult

    tmp_logger.open_simulated_position(
        "BTC-USD", qty=0.01, avg_entry_price=60000.0, stop_loss_price=58000.0, take_profit_price=66000.0
    )
    monkeypatch.setattr(main.data_fetcher, "fetch_ohlcv", lambda s, **k: pd.DataFrame({"Close": [57000.0]}))
    monkeypatch.setattr(
        approval, "request_approval",
        lambda **k: (_ for _ in ()).throw(AssertionError("stop exits must not wait")),
    )
    sent = []

    def fake_execute(signal, current_price, live_equity, is_live, existing_position=None):
        sent.append(signal.action)
        return ExecutionResult(status="success", qty=0.01, fill_price=57000.0, message="ok")

    monkeypatch.setattr(execution, "execute_trade", fake_execute)

    sweep = main.sweep_open_positions(tmp_logger, {"approval_mode": True}, is_live=True, equity_hint=5000.0)

    assert sent == ["sell"]
    assert sweep.closed_symbols == {"BTC-USD"}


def _positions_in_turn(monkeypatch, *positions):
    """fetch_existing_position returns these in order (the last one repeats)."""
    queue = list(positions)

    def fake(**kwargs):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(execution, "fetch_existing_position", fake)


def _run_approved(tmp_logger, monkeypatch, decision_fn):
    monkeypatch.setattr(approval, "request_approval", decision_fn)
    main._process_symbol(
        symbol="AAPL", config={"approval_mode": True}, settings=settings(is_live=True),
        bot_logger=tmp_logger,
        generate_signal=lambda si, system_prompt: buy_signal("AAPL", 200.0, stop_pct=0.02),
        equity=5000.0, circuit_breaker_loss_pct=3.0, max_risk_pct=1.0,
        max_absolute_position_pct=20.0, min_reward_risk_ratio=1.5,
        breaker_tracker=main.CircuitBreakerTracker(already_tripped_at_cycle_start=False),
    )


def test_approved_order_is_judged_against_the_position_as_it_is_after_the_wait(tmp_logger, monkeypatch):
    """Another cycle bought AAPL while this one waited for the tap: execution must
    see that holding (so its duplicate-buy guard skips), not the pre-wait flat."""
    stub_market(monkeypatch, {"AAPL": 200.0})
    calls = record_executions(monkeypatch)
    bought_meanwhile = ExistingPosition(qty=5.0, avg_entry_price=199.0)
    _positions_in_turn(monkeypatch, None, bought_meanwhile)

    _run_approved(tmp_logger, monkeypatch, lambda **k: approval.ApprovalDecision(True, "approved"))

    assert calls[0]["existing_position"] == bought_meanwhile


def test_approved_but_position_recheck_fails_is_skipped(tmp_logger, read_signals, monkeypatch):
    stub_market(monkeypatch, {"AAPL": 200.0})
    calls = record_executions(monkeypatch)
    _positions_in_turn(monkeypatch, None, RuntimeError("Alpaca down"))

    _run_approved(tmp_logger, monkeypatch, lambda **k: approval.ApprovalDecision(True, "approved"))

    assert calls == []
    result = read_signals()[-1]["execution_result"]
    assert result["status"] == "skipped"
    assert "position re-check failed" in result["message"]


def test_a_cancelled_job_during_the_wait_places_no_order_and_writes_nothing(
    tmp_logger, read_signals, monkeypatch
):
    """GitHub cancels a job with SIGINT, which Python raises as KeyboardInterrupt
    inside the wait. It must propagate (nothing here catches BaseException) with
    no order submitted and no signal row or ledger change written."""
    stub_market(monkeypatch, {"AAPL": 200.0})
    calls = record_executions(monkeypatch)

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_approved(tmp_logger, monkeypatch, interrupted)

    assert calls == []
    assert read_signals() == []
    assert tmp_logger.get_all_simulated_positions() == []


@pytest.mark.parametrize("topic", ["tradingbot", "tb-short", "tb-0123456789abcdef-with space", "tb/0123456789abcdef0123456789", "x" * 65])
def test_a_weak_or_invalid_topic_is_refused_for_approval(monkeypatch, topic):
    monkeypatch.setenv("NTFY_TOPIC", topic)
    fake = FakeNtfy(replies=["{id}:approve"])
    decision = request(fake, FakeClock())
    assert decision.approved is False
    assert decision.outcome == "unavailable"
    assert fake.published == []


def test_the_documented_topic_recipe_is_strong():
    import secrets

    assert notifications.topic_is_strong("tb-" + secrets.token_hex(16))
