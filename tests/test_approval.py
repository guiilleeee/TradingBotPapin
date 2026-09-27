"""Approval mode: the Telegram inline-button approve/reject round-trip, and its gating of the cycle.

No test reaches Telegram: a FakeTelegram stands in for notifications.telegram_call,
and a fake clock drives the timeout without any real waiting.
"""

import pytest

import approval
import execution
import main
import notifications
from models import ExistingPosition
from tests.cycle_helpers import buy_signal, record_executions, settings, stub_market

TOKEN = "123456789:" + "A" * 35
CHAT = "555000111"


@pytest.fixture
def tg_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.001)


def tap(data, chat=CHAT, sender=CHAT, update_id=100):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cq{update_id}",
            "from": {"id": int(sender)},
            "message": {"message_id": 42, "chat": {"id": int(chat)}},
            "data": data,
        },
    }


class FakeTelegram:
    """Records every Bot API call. `taps` are updates returned by getUpdates from
    the `reply_after_polls`-th poll on; a "{id}" placeholder in their callback
    data is filled with the real request id once the proposal has been sent."""

    def __init__(self, taps=(), reply_after_polls=1, fail_send=False, fail_polls=0, raw=()):
        self.taps = list(taps)
        self.raw = list(raw)
        self.reply_after_polls = reply_after_polls
        self.fail_send = fail_send
        self.fail_polls = fail_polls
        self.calls = []

    def of(self, method):
        return [payload for m, payload in self.calls if m == method]

    @property
    def proposal(self):
        return self.of("sendMessage")[0]

    @property
    def request_id(self):
        return self.proposal["reply_markup"]["inline_keyboard"][0][0]["callback_data"].split(":")[0]

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        if method == "sendMessage":
            if self.fail_send:
                raise RuntimeError("telegram unreachable")
            return {"message_id": 42}
        if method == "getUpdates":
            if self.fail_polls:
                self.fail_polls -= 1
                raise RuntimeError("poll failed")
            if len(self.of("getUpdates")) < self.reply_after_polls:
                return []
            out = []
            for t in self.taps:
                t = {**t, "callback_query": {**t["callback_query"]}}
                t["callback_query"]["data"] = str(t["callback_query"]["data"]).replace("{id}", self.request_id)
                out.append(t)
            return self.raw + out
        return True


def request(fake, clock, timeout=600):
    return approval.request_approval(
        symbol="AAPL", action="buy", size_pct=10.0, price=200.0, stop_loss=196.0,
        take_profit=212.0, confidence=0.8, reasoning="Tendencia alcista.", equity=5000.0,
        timeout_seconds=timeout, call=fake, clock=clock, sleep=clock.sleep,
    )


# -------------------------------------------------------------- config switch


@pytest.mark.parametrize("value,enabled", [
    (True, True), (False, False), ("true", False), (1, False), (None, False),
])
def test_approval_mode_is_only_on_for_a_literal_true(value, enabled):
    assert approval.approval_enabled({"approval_mode": value}) is enabled


def test_approval_mode_defaults_off():
    assert approval.approval_enabled({}) is False


def test_shipped_config_has_approval_mode_on():
    import yaml

    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["approval_mode"] is True
    assert approval.approval_timeout_seconds(config) == 600


# ------------------------------------------------------------- the message itself


def test_proposal_is_a_plain_text_message_with_two_inline_buttons(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve")])
    request(fake, FakeClock())
    msg = fake.proposal

    assert msg["chat_id"] == CHAT
    assert "parse_mode" not in msg
    assert "BUY AAPL" in msg["text"]
    approve, reject = msg["reply_markup"]["inline_keyboard"][0]
    assert approve == {"text": "Aprovar", "callback_data": f"{fake.request_id}:approve"}
    assert reject == {"text": "Rebutjar", "callback_data": f"{fake.request_id}:reject"}
    assert len(approve["callback_data"].encode()) <= 64  # Telegram's limit
    for fragment in ("200", "10.00%", "$500.00", "196", "212", "0.80", "Tendencia alcista."):
        assert fragment in msg["text"]


def test_each_request_gets_a_fresh_id(tg_env):
    ids = set()
    for _ in range(3):
        fake = FakeTelegram(taps=[tap("{id}:approve")])
        request(fake, FakeClock())
        ids.add(fake.request_id)
    assert len(ids) == 3


def test_polling_asks_only_for_button_taps_and_confirms_what_it_has_seen(tg_env):
    fake = FakeTelegram(raw=[tap("stale:approve", update_id=7)], reply_after_polls=1)
    request(fake, FakeClock(), timeout=15)
    polls = fake.of("getUpdates")
    assert all(p["allowed_updates"] == ["callback_query"] for p in polls)
    assert "offset" not in polls[0]
    assert all(p["offset"] == 8 for p in polls[1:])


# ------------------------------------------------------------- outcomes


def test_approve_tap_approves(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve")], reply_after_polls=4)
    decision = request(fake, FakeClock())
    assert decision.approved is True
    assert decision.outcome == "approved"
    # Spinner stopped, buttons removed, and a follow-up shows how it ended.
    assert fake.of("answerCallbackQuery") == [{"callback_query_id": "cq100"}]
    assert fake.of("editMessageReplyMarkup")[0]["message_id"] == 42
    assert "APROVADA" in fake.of("sendMessage")[-1]["text"]


def test_reject_tap_rejects(tg_env):
    decision = request(FakeTelegram(taps=[tap("{id}:reject")]), FakeClock())
    assert decision.approved is False
    assert decision.outcome == "rejected"


def test_approve_and_reject_both_seen_resolves_to_reject(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve", update_id=1), tap("{id}:reject", update_id=2)])
    assert request(fake, FakeClock()).outcome == "rejected"


def test_no_answer_times_out_after_the_configured_window(tg_env):
    clock = FakeClock()
    fake = FakeTelegram()
    decision = request(fake, clock, timeout=600)
    assert decision.approved is False
    assert decision.outcome == "timeout"
    assert 600 <= clock.now <= 600 + approval.POLL_INTERVAL_SECONDS
    assert "CADUCADA" in fake.of("sendMessage")[-1]["text"]


@pytest.mark.parametrize("data", [
    "deadbeefdeadbeef:approve",      # another request's id
    "{id}:APPROVE",                  # not the exact data
    "{id}:approve please",
    "approve",
    "{id}",
    "",
])
def test_anything_but_this_requests_exact_approve_data_is_ignored(tg_env, data):
    decision = request(FakeTelegram(taps=[tap(data)]), FakeClock(), timeout=60)
    assert decision.outcome == "timeout"


def test_a_tap_from_another_chat_is_ignored(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve", chat="999", sender="999")])
    assert request(fake, FakeClock(), timeout=60).outcome == "timeout"


def test_a_tap_from_another_user_in_a_private_chat_is_ignored(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve", sender="999")])
    assert request(fake, FakeClock(), timeout=60).outcome == "timeout"


def test_in_a_group_chat_any_member_can_answer(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    fake = FakeTelegram(taps=[tap("{id}:approve", chat="-1001234567890", sender="999")])
    assert request(fake, FakeClock()).approved is True


def test_malformed_updates_are_ignored(tg_env):
    raw = [
        None, "garbage", {}, {"update_id": "x"}, {"callback_query": None},
        {"callback_query": {"data": "x"}},
        {"callback_query": {"message": {"chat": "nope"}, "data": "x"}},
        {"update_id": True, "callback_query": {"message": {"chat": {"id": int(CHAT)}}, "from": {}, "data": 5}},
    ]
    decision = request(FakeTelegram(raw=raw), FakeClock(), timeout=60)
    assert decision.outcome == "timeout"


def test_a_non_list_getupdates_result_is_ignored(tg_env):
    class Weird(FakeTelegram):
        def __call__(self, method, payload):
            if method == "getUpdates":
                self.calls.append((method, payload))
                return {"not": "a list"}
            return super().__call__(method, payload)

    assert request(Weird(), FakeClock(), timeout=30).outcome == "timeout"


def test_transient_poll_failures_keep_waiting(tg_env):
    decision = request(FakeTelegram(taps=[tap("{id}:approve")], fail_polls=3), FakeClock())
    assert decision.approved is True


def test_polls_failing_until_the_deadline_is_a_timeout(tg_env):
    fake = FakeTelegram(taps=[tap("{id}:approve")], fail_polls=10_000)
    decision = request(fake, FakeClock(), timeout=60)
    assert decision.approved is False
    assert decision.outcome == "timeout"


def test_unreachable_telegram_is_not_an_approval(tg_env):
    decision = request(FakeTelegram(fail_send=True), FakeClock())
    assert decision.approved is False
    assert decision.outcome == "unavailable"


def test_close_out_failures_do_not_change_the_decision(tg_env):
    class Flaky(FakeTelegram):
        def __call__(self, method, payload):
            if method in ("answerCallbackQuery", "editMessageReplyMarkup") or (
                method == "sendMessage" and self.of("sendMessage")
            ):
                self.calls.append((method, payload))
                raise RuntimeError("down")
            return super().__call__(method, payload)

    assert request(Flaky(taps=[tap("{id}:approve")]), FakeClock()).approved is True


@pytest.mark.parametrize("var", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
def test_missing_credentials_are_not_an_approval(tg_env, monkeypatch, var):
    monkeypatch.delenv(var, raising=False)
    fake = FakeTelegram(taps=[tap("{id}:approve")])
    decision = request(fake, FakeClock())
    assert decision.approved is False
    assert decision.outcome == "unavailable"
    assert fake.calls == []


@pytest.mark.parametrize("token,chat", [
    ("not-a-token", CHAT),
    ("123:short", CHAT),
    (TOKEN, "@mychannel"),
    (TOKEN, "12 34"),
])
def test_malformed_credentials_are_refused_for_approval(monkeypatch, token, chat):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", chat)
    fake = FakeTelegram(taps=[tap("{id}:approve")])
    decision = request(fake, FakeClock())
    assert decision.outcome == "unavailable"
    assert fake.calls == []


# --------------------------------------------------------- telegram_call itself


class _Resp:
    def __init__(self, body, status_error=None):
        self._body = body
        self._err = status_error

    def raise_for_status(self):
        if self._err:
            raise self._err

    def json(self):
        return self._body


def test_telegram_call_posts_to_the_bot_api_and_returns_the_result(tg_env, monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None):
        seen.append((url, json))
        return _Resp({"ok": True, "result": [{"update_id": 1}]})

    monkeypatch.setattr(notifications.requests, "post", fake_post)
    assert notifications.telegram_call("getUpdates", {"timeout": 0}) == [{"update_id": 1}]
    assert seen == [(f"https://api.telegram.org/bot{TOKEN}/getUpdates", {"timeout": 0})]


def test_telegram_call_raises_when_ok_is_not_true(tg_env, monkeypatch):
    monkeypatch.setattr(
        notifications.requests, "post",
        lambda *a, **k: _Resp({"ok": False, "description": "Conflict: webhook is active"}),
    )
    with pytest.raises(RuntimeError, match="webhook"):
        notifications.telegram_call("getUpdates", {})


def test_telegram_call_raises_on_http_error(tg_env, monkeypatch):
    monkeypatch.setattr(notifications.requests, "post", lambda *a, **k: _Resp({}, RuntimeError("502")))
    with pytest.raises(RuntimeError):
        notifications.telegram_call("getUpdates", {})


def test_a_send_error_never_leaks_the_token_into_the_decision(tg_env):
    class Leaky(FakeTelegram):
        def __call__(self, method, payload):
            raise RuntimeError(f"Max retries exceeded with url: /bot{TOKEN}/sendMessage")

    decision = request(Leaky(), FakeClock())
    assert decision.outcome == "unavailable"
    assert TOKEN not in decision.detail

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
