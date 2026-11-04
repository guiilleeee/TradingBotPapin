"""A cycle that analysed symbols but logged no signal must fail loudly: non-zero
exit code, a Telegram alert, and a "failed" entry in job_status.json carrying the
first error. (Anthropic credit ran out once and every cycle still reported ok.)"""

import main
import job_status
import notifications

from test_manual_trigger import _config

SYMBOLS = [{"symbol": "AAPL", "asset_class": "equity"}, {"symbol": "MSFT", "asset_class": "equity"}]


def _run(tmp_logger, tmp_path, monkeypatch, process):
    alerts = []
    monkeypatch.setattr(main, "_process_symbol", process)
    monkeypatch.setattr(notifications, "send_cycle_failure_alert", lambda live, msg: alerts.append(msg))
    rc = main._run_cycle_body(_config(tmp_logger, tmp_path, SYMBOLS), is_live=False)
    return rc, alerts


def test_every_symbol_failing_returns_nonzero_and_alerts_with_the_first_error(
        tmp_logger, tmp_path, monkeypatch, capsys):
    def boom(*, symbol, **kwargs):
        raise RuntimeError(f"credit balance is too low ({symbol})")

    rc, alerts = _run(tmp_logger, tmp_path, monkeypatch, boom)

    assert rc == 1
    assert len(alerts) == 1
    assert "2/2 symbols failed" in alerts[0] and "AAPL: RuntimeError: credit balance is too low" in alerts[0]
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith("CYCLE FAILED:")


def test_the_failure_is_recorded_in_job_status_with_the_first_error(
        tmp_logger, tmp_path, monkeypatch, capsys):
    def boom(*, symbol, **kwargs):
        raise RuntimeError("credit balance is too low")

    rc, _ = _run(tmp_logger, tmp_path, monkeypatch, boom)
    log = capsys.readouterr().out
    path = str(tmp_path / "job_status.json")
    run = job_status.record_run(path, "cycle", "2026-11-05T14:45:00+00:00", rc, "cycle", log)

    assert run["status"] == "failed"
    assert "credit balance is too low" in run["error"]


def test_zero_logged_signals_is_a_failure_even_without_an_exception(tmp_logger, tmp_path, monkeypatch):
    rc, alerts = _run(tmp_logger, tmp_path, monkeypatch, lambda *, symbol, **kw: None)
    assert rc == 1 and alerts


def test_one_good_symbol_keeps_the_cycle_ok(tmp_logger, tmp_path, monkeypatch):
    def process(*, symbol, bot_logger, **kwargs):
        if symbol == "AAPL":
            raise RuntimeError("one bad symbol")
        bot_logger.log_signal(symbol, None, None, None)

    rc, alerts = _run(tmp_logger, tmp_path, monkeypatch, process)
    assert rc == 0 and alerts == []


def test_a_cycle_with_nothing_to_analyse_is_still_ok(tmp_logger, tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(main, "_process_symbol", lambda **kw: None)
    monkeypatch.setattr(notifications, "send_cycle_failure_alert", lambda live, msg: alerts.append(msg))
    rc = main._run_cycle_body(_config(tmp_logger, tmp_path, SYMBOLS), is_live=False,
                              trigger_symbols=["NOPE"], trigger_reason="wake_buy")
    assert rc == 0 and alerts == []
