"""Weekly live report: trades rebuilt from broker fills + logged signals."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import live_report
import notifications
from models import ExecutionResult, TradeSignal

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
CONFIG = {"symbols": [{"symbol": "AAPL", "asset_class": "equity"},
                      {"symbol": "BTC-USD", "asset_class": "crypto"}]}


def iso(days_ago, hours=0):
    return (NOW - timedelta(days=days_ago, hours=hours)).isoformat()


def fill(order_id, symbol, side, qty, price, days_ago, hours=0):
    return {"id": f"{order_id}-{qty}-{price}", "order_id": order_id, "symbol": symbol,
            "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": iso(days_ago, hours), "type": "fill"}


def log_buy(logger, symbol, order_id, confidence=0.7, stop=95.0, trigger="scheduled",
            held_by=None):
    final = TradeSignal(symbol=symbol, action="hold" if held_by else "buy", confidence=confidence,
                        position_size_pct=0.0 if held_by else 10.0,
                        stop_loss_price=None if held_by else stop,
                        take_profit_price=None if held_by else 115.0,
                        reasoning="prova", raw_action="buy", override_reason=held_by)
    result = None if held_by else ExecutionResult(status="success", order_id=order_id, qty=1,
                                                  fill_price=100.0)
    return logger.log_signal(symbol, None, None, final, result, is_live=True, trigger_reason=trigger)


def backdate(logger, row_id, when):
    import sqlite3

    with sqlite3.connect(logger.db_path) as conn:
        conn.execute("UPDATE signals SET timestamp = ? WHERE id = ?", (when, row_id))


def test_fifo_pairing_with_partial_fills_and_exit_reasons(tmp_logger):
    # AAPL: bought in two partial fills, stopped out by a bracket leg.
    backdate(tmp_logger, log_buy(tmp_logger, "AAPL", "buy-1", confidence=0.8), iso(3))
    tmp_logger.record_broker_exit({"leg_id": "leg-sl", "symbol": "AAPL", "kind": "stop_loss",
                                   "filled_at": iso(1), "qty": 4, "entry_price": 100.0,
                                   "exit_price": 94.0})
    # BTC: bought by a wake-up cycle, closed by the model's own sell.
    backdate(tmp_logger, log_buy(tmp_logger, "BTC-USD", "buy-2", trigger="wake_buy", stop=90.0), iso(2))
    sell = TradeSignal(symbol="BTC-USD", action="sell", confidence=0.8, position_size_pct=1.0,
                       stop_loss_price=90, take_profit_price=120, reasoning="x", raw_action="sell")
    tmp_logger.log_signal("BTC-USD", None, None, sell,
                          ExecutionResult(status="success", order_id="sell-2", qty=0.5,
                                          fill_price=110.0), is_live=True)
    fills = [
        fill("buy-1", "AAPL", "buy", 3, 100.0, 3),
        fill("buy-1", "AAPL", "buy", 1, 100.0, 3),
        fill("buy-2", "BTC/USD", "buy", 0.5, 100.0, 2),
        fill("leg-sl", "AAPL", "sell", 4, 94.0, 1),
        fill("sell-2", "BTCUSD", "sell", 0.5, 110.0, 0, hours=2),
    ]

    report = live_report.build_report(fills, tmp_logger.db_path, CONFIG, now=NOW)

    trades = {t["symbol"]: t for t in report["week"]["trades"]}
    aapl, btc = trades["AAPL"], trades["BTC-USD"]
    assert aapl["qty"] == 4 and aapl["realized_pnl_usd"] == pytest.approx(-24.0)
    assert aapl["close_reason"] == "stop_loss"
    assert aapl["entry_confidence"] == 0.8
    assert aapl["r_multiple"] == pytest.approx(-1.2)          # (94-100)/(100-95)
    assert btc["asset_class"] == "crypto"
    assert btc["close_reason"] == "model_sell"
    assert btc["trigger_reason"] == "wake_buy"
    assert btc["realized_pnl_usd"] == pytest.approx(5.0)
    summary = report["week"]["summary"]
    assert summary["total_closed_trades"] == 2 and summary["win_rate_pct"] == pytest.approx(50.0)
    assert report["open_lots"] == 0


def test_a_sell_split_across_two_lots_and_an_open_lot_left_over(tmp_logger):
    fills = [
        fill("b1", "AAPL", "buy", 2, 100.0, 5),
        fill("b2", "AAPL", "buy", 2, 110.0, 4),
        fill("s1", "AAPL", "sell", 3, 120.0, 1),
    ]
    report = live_report.build_report(fills, tmp_logger.db_path, CONFIG, now=NOW)
    pnl = sorted(t["realized_pnl_usd"] for t in report["week"]["trades"])
    assert pnl == [pytest.approx(10.0), pytest.approx(40.0)]  # 1 @110->120, 2 @100->120
    assert report["open_lots"] == 1
    assert all(t["close_reason"] == "unknown" for t in report["week"]["trades"])


def test_a_sell_with_no_known_entry_is_not_a_trade(tmp_logger):
    report = live_report.build_report([fill("s1", "AAPL", "sell", 3, 120.0, 1)],
                                      tmp_logger.db_path, CONFIG, now=NOW)
    assert report["all_time"]["summary"]["total_closed_trades"] == 0


def test_week_and_all_time_windows(tmp_logger):
    fills = [
        fill("b1", "AAPL", "buy", 1, 100.0, 30), fill("s1", "AAPL", "sell", 1, 90.0, 20),
        fill("b2", "AAPL", "buy", 1, 100.0, 5), fill("s2", "AAPL", "sell", 1, 105.0, 2),
    ]
    report = live_report.build_report(fills, tmp_logger.db_path, CONFIG, now=NOW)
    assert report["week"]["summary"]["total_closed_trades"] == 1
    assert report["all_time"]["summary"]["total_closed_trades"] == 2
    assert "trades" not in report["all_time"]


def test_automatic_exit_rows_classify_the_close(tmp_logger):
    tmp_logger.log_auto_close_signal("AAPL", "Sortida per temps: 11 dies...", 101.0, 1, 1.0,
                                     1000.0, is_live=True, entry_price=100.0, order_id="s1")
    fills = [fill("b1", "AAPL", "buy", 1, 100.0, 12), fill("s1", "AAPL", "sell", 1, 101.0, 1)]
    report = live_report.build_report(fills, tmp_logger.db_path, CONFIG, now=NOW)
    assert report["week"]["trades"][0]["close_reason"] == "time_exit"


def test_override_stats_count_holds_by_rule_and_resizes(tmp_logger):
    log_buy(tmp_logger, "AAPL", None, held_by="confidence 0.55 is below the 0.60 minimum for this mode")
    log_buy(tmp_logger, "MSFT", None, held_by="portfolio: group 'semis' already holds 2 (AMD, NVDA), limit 2")
    log_buy(tmp_logger, "NFLX", None,
            held_by="reward:risk 1.20 is below the 1.50 minimum (...); stop-loss 99 is 0.50x ATR from price, under the 1.00x minimum")
    rid = log_buy(tmp_logger, "COST", "o-9")
    import sqlite3
    with sqlite3.connect(tmp_logger.db_path) as conn:
        conn.execute("UPDATE signals SET override_reason = ? WHERE id = ?",
                     ("risk-based size 50.00% clamped to the 20.00% absolute position cap", rid))

    rows, _ = live_report._load_signal_rows(tmp_logger.db_path)
    stats = live_report.override_stats(rows, None)

    assert stats["proposed_buys"] == 4
    assert stats["executed_buys"] == 1
    assert stats["held_by_rule"] == {"confidence": 1, "portfolio_group": 1, "reward_risk": 1, "atr_band": 1}
    assert stats["resized_by_rule"] == {"absolute_position_cap": 1}


def test_telegram_text_flags_a_small_sample(tmp_logger):
    fills = [fill("b1", "AAPL", "buy", 1, 100.0, 5), fill("s1", "AAPL", "sell", 1, 105.0, 2)]
    report = live_report.build_report(fills, tmp_logger.db_path, CONFIG, now=NOW)
    subject, body = live_report.format_telegram(report)
    assert subject.startswith("WEEKLY REPORT: 1 trades")
    assert "treat rates as noise" in body


def test_run_writes_json_and_sends(tmp_path, monkeypatch):
    db = tmp_path / "bot.db"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"live_execution: true\ndb_path: {db.as_posix()}\nsymbols: []\n", encoding="utf-8")
    monkeypatch.setattr(live_report, "fetch_fill_activities", lambda after: [])
    sent = []
    monkeypatch.setattr(notifications, "send_weekly_report", lambda s, b: sent.append(s))
    out = tmp_path / "docs" / "live_report.json"

    assert live_report.run(str(cfg), str(out)) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "live" and data["week"]["summary"]["total_closed_trades"] == 0
    assert len(sent) == 1


def test_run_skips_in_simulation(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("live_execution: false\nsymbols: []\n", encoding="utf-8")
    monkeypatch.setattr(live_report, "fetch_fill_activities",
                        lambda after: (_ for _ in ()).throw(AssertionError("no broker in sim")))
    assert live_report.run(str(cfg), str(tmp_path / "r.json")) == 0
    assert not (tmp_path / "r.json").exists()


def test_fill_activities_paginate(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-alpaca-secret")
    pages = [[{"id": f"a{i}"} for i in range(100)], [{"id": "last"}]]
    tokens = []

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, headers=None, params=None, timeout=None):
        tokens.append(params.get("page_token"))
        return Resp(pages[len(tokens) - 1])

    monkeypatch.setattr(live_report.requests, "get", fake_get)
    assert len(live_report.fetch_fill_activities("2026-01-01T00:00:00+00:00")) == 101
    assert tokens == [None, "a99"]


def test_weekly_notification_never_raises(monkeypatch):
    monkeypatch.setattr(notifications, "telegram_configured", lambda: True)
    monkeypatch.setattr(notifications, "_send_telegram",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    notifications.send_weekly_report("WEEKLY REPORT", "body")
