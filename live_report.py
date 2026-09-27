"""Weekly live-results report: what the bot's real trades actually did.

The pnl table can't answer this on its own -- it has no link from a close back
to the decision that opened it. So trades are rebuilt from the broker's side:

  1. Alpaca's FILL activities (every buy and sell that actually executed),
     grouped per order and paired FIFO per symbol into closed trades.
  2. Each entry is joined to the logged buy signal by broker order id (falling
     back to the nearest executed buy for that symbol) for confidence, trigger
     and planned stop.
  3. Each exit is classified by the signal row that carries its order id --
     bracket leg, sweep exit, time exit, or the model's own sell.

Then diagnose_backtest_loss's analysis runs on the result, unchanged: win rate,
reward:risk achieved, confidence calibration, breakdowns by asset class and
close reason -- plus breakdown by trigger and how often each risk rule turned
a proposed buy into a hold.

Output: docs/live_report.json for the dashboard's "Informe" tab, a Telegram
message, and a one-line web push. Read-only against the broker and the DB.

  python live_report.py --config config.yaml [--output docs/live_report.json] [--no-send]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

import diagnose_backtest_loss as diag
import execution
import notifications
import symbol_config
from logger import EQUITY_CURVE_START_KEY, BotLogger, parse_utc
from mode import resolve_is_live

DEFAULT_OUTPUT_PATH = "docs/live_report.json"
REPORT_WINDOW = timedelta(days=7)
# With no recorded funding date, look back this far for fills.
DEFAULT_HISTORY = timedelta(days=365)
# A buy signal this close to a fill is taken as the decision behind it when no
# order id links them (rows logged before order ids were kept on every row).
ENTRY_MATCH_WINDOW = timedelta(hours=2)
EXIT_MATCH_WINDOW = timedelta(hours=1)
QTY_EPSILON = 1e-9
# Below this many closed trades, every rate in the report is mostly noise.
MIN_MEANINGFUL_SAMPLE = 30


@dataclass
class LiveTrade(diag.ClosedTrade):
    trigger_reason: str = "unknown"
    r_multiple: Optional[float] = None
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None


# ------------------------------------------------------------------ broker


def fetch_fill_activities(after_iso: str) -> List[Dict[str, Any]]:
    """Every FILL activity after `after_iso`, oldest first. Raises on failure."""
    execution._require_credentials()
    out: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    for _page in range(200):
        params: Dict[str, Any] = {"after": after_iso, "direction": "asc", "page_size": 100}
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(
            f"{execution.ALPACA_BASE_URL}/v2/account/activities/FILL",
            headers=execution._alpaca_headers(),
            params=params,
            timeout=execution.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        page = resp.json() or []
        out.extend(page)
        if len(page) < 100:
            break
        page_token = str(page[-1].get("id") or "")
        if not page_token:
            break
    return out


# ------------------------------------------------------------ reconstruction


def _project_symbol(raw: str, crypto_bases: Dict[str, str]) -> str:
    sym = str(raw or "").upper()
    if "/" in sym:
        return symbol_config.from_alpaca_symbol(sym)
    return crypto_bases.get(sym, sym)


def _orders_from_fills(fills: List[Dict[str, Any]], crypto_bases: Dict[str, str]) -> List[Dict[str, Any]]:
    """Group partial fills into one record per order: total qty, VWAP, first fill time."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for fill in fills:
        side = str(fill.get("side", ""))
        if side not in ("buy", "sell"):
            continue  # sell_short never happens here; ignore anything else
        try:
            qty = float(fill["qty"])
            price = float(fill["price"])
        except (KeyError, TypeError, ValueError):
            continue
        when = parse_utc(fill.get("transaction_time"))
        if qty <= 0 or price <= 0 or when is None:
            continue
        order_id = str(fill.get("order_id") or fill.get("id"))
        rec = grouped.setdefault(order_id, {
            "order_id": order_id, "side": side,
            "symbol": _project_symbol(fill.get("symbol"), crypto_bases),
            "qty": 0.0, "notional": 0.0, "time": when,
        })
        rec["qty"] += qty
        rec["notional"] += qty * price
        rec["time"] = min(rec["time"], when)
    orders = []
    for rec in grouped.values():
        rec["price"] = rec["notional"] / rec["qty"]
        orders.append(rec)
    orders.sort(key=lambda r: (r["time"], r["order_id"]))
    return orders


def _load_signal_rows(db_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Live signal rows with decoded blobs, and broker_exits' leg id -> kind."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, timestamp, symbol, final_signal, execution_result, override_reason, "
            "is_live, trigger_reason FROM signals WHERE is_live = 1 ORDER BY id"
        ).fetchall()
        exits = conn.execute("SELECT leg_order_id, kind FROM broker_exits").fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        item = dict(row)
        for key in ("final_signal", "execution_result"):
            try:
                item[key] = json.loads(item[key]) if item[key] else {}
            except (TypeError, ValueError):
                item[key] = {}
        item["time"] = parse_utc(item["timestamp"])
        out.append(item)
    out_exits = {str(r["leg_order_id"]): str(r["kind"]) for r in exits}
    return out, out_exits


def classify_exit_reasoning(reasoning: str) -> str:
    if reasoning.startswith("Stop-loss"):
        return "stop_loss"
    if reasoning.startswith("Take-profit"):
        return "take_profit"
    if reasoning.startswith("Sortida per temps"):
        return "time_exit"
    return "auto_close_unknown"


def _nearest(rows: List[Dict[str, Any]], symbol: str, when: datetime, window: timedelta,
             before_only: bool) -> Optional[Dict[str, Any]]:
    best, best_gap = None, None
    for row in rows:
        if row["symbol"].upper() != symbol.upper() or row["time"] is None:
            continue
        gap = (when - row["time"]).total_seconds()
        if before_only and gap < -60:  # a minute of clock skew either way
            continue
        if abs(gap) > window.total_seconds():
            continue
        if best_gap is None or abs(gap) < best_gap:
            best, best_gap = row, abs(gap)
    return best


def reconstruct_trades(
    orders: List[Dict[str, Any]],
    signal_rows: List[Dict[str, Any]],
    broker_exit_kinds: Dict[str, str],
    config: Dict[str, Any],
) -> Tuple[List[LiveTrade], int]:
    """FIFO-pair buy and sell orders per symbol. Returns (closed trades, open lots)."""
    by_order = {
        str(r["execution_result"].get("order_id")): r
        for r in signal_rows if r["execution_result"].get("order_id")
    }
    executed_buys = [
        r for r in signal_rows
        if r["final_signal"].get("action") == "buy"
        and r["execution_result"].get("status") == "success"
    ]
    auto_exits = [r for r in signal_rows if r.get("override_reason") == "automatic exit"]

    lots: Dict[str, deque] = defaultdict(deque)
    trades: List[LiveTrade] = []
    for order in orders:
        symbol = order["symbol"]
        if order["side"] == "buy":
            entry_row = by_order.get(order["order_id"]) or _nearest(
                executed_buys, symbol, order["time"], ENTRY_MATCH_WINDOW, before_only=False)
            lots[symbol].append({**order, "remaining": order["qty"], "entry_row": entry_row})
            continue

        exit_reason = _exit_reason(order, by_order, broker_exit_kinds, auto_exits)
        remaining = order["qty"]
        while remaining > QTY_EPSILON and lots[symbol]:
            lot = lots[symbol][0]
            take = min(remaining, lot["remaining"])
            trades.append(_trade(symbol, lot, order, take, exit_reason, config))
            lot["remaining"] -= take
            remaining -= take
            if lot["remaining"] <= QTY_EPSILON:
                lots[symbol].popleft()
        # A sell with no lot left is a position opened before the window (or
        # outside the bot); it has no known entry, so it is not a trade here.

    open_lots = sum(len(q) for q in lots.values())
    return trades, open_lots


def _exit_reason(order, by_order, broker_exit_kinds, auto_exits) -> str:
    if order["order_id"] in broker_exit_kinds:
        return broker_exit_kinds[order["order_id"]]
    row = by_order.get(order["order_id"])
    if row is None:
        row = _nearest(auto_exits, order["symbol"], order["time"], EXIT_MATCH_WINDOW, before_only=False)
    if row is None:
        return "unknown"
    if row.get("override_reason") == "automatic exit":
        return classify_exit_reasoning(str(row["final_signal"].get("reasoning") or ""))
    return "model_sell"


def _trade(symbol, lot, order, qty, exit_reason, config) -> LiveTrade:
    entry_row = lot["entry_row"] or {}
    final = entry_row.get("final_signal") or {}
    entry, exit_price = lot["price"], order["price"]
    stop = final.get("stop_loss_price")
    r_multiple = None
    if stop is not None and float(stop) < entry:
        r_multiple = (exit_price - entry) / (entry - float(stop))
    return LiveTrade(
        symbol=symbol,
        asset_class=symbol_config.asset_class(symbol, config),
        opened_date=lot["time"].isoformat(),
        closed_date=order["time"].isoformat(),
        holding_days=round((order["time"] - lot["time"]).total_seconds() / 86400.0, 2),
        qty=qty,
        entry_price=entry,
        exit_price=exit_price,
        realized_pnl_usd=(exit_price - entry) * qty,
        entry_confidence=final.get("confidence"),
        close_reason=exit_reason,
        trigger_reason=str(entry_row.get("trigger_reason") or "unknown"),
        r_multiple=r_multiple,
        entry_order_id=lot["order_id"],
        exit_order_id=order["order_id"],
    )


# -------------------------------------------------------------- rule overrides


# (substring in one override clause, rule name), after the two prefix checks in
# classify_override. First match wins.
OVERRIDE_RULES = [
    ("circuit breaker", "circuit_breaker"),
    ("pre-earnings blackout", "earnings_blackout"),
    ("x ATR", "atr_band"),
    ("credible risk boundary", "min_stop_distance"),
    ("wrong side", "levels_wrong_side"),
    ("missing required", "levels_missing"),
    ("non-positive position_size_pct", "bad_size"),
    ("portfolio: open positions unknown", "portfolio_unknown"),
    ("max_open_positions", "portfolio_max_positions"),
    ("portfolio: group", "portfolio_group"),
    ("portfolio: returns correlated", "portfolio_correlation"),
    ("gross exposure", "portfolio_exposure"),
    ("clamped", "absolute_position_cap"),
]


def classify_override(clause: str) -> str:
    if clause.startswith("confidence "):
        return "confidence"
    if clause.startswith("reward:risk"):
        return "reward_risk"
    for needle, name in OVERRIDE_RULES:
        if needle in clause:
            return name
    return "other"


def override_stats(signal_rows: List[Dict[str, Any]], start: Optional[datetime]) -> Dict[str, Any]:
    """How the model's buy proposals fared: executed, held (by which rule), resized."""
    proposed = executed = 0
    held: Dict[str, int] = defaultdict(int)
    resized: Dict[str, int] = defaultdict(int)
    for row in signal_rows:
        if start is not None and (row["time"] is None or row["time"] < start):
            continue
        final = row["final_signal"]
        if final.get("raw_action") != "buy":
            continue
        proposed += 1
        clauses = [c.strip() for c in str(row.get("override_reason") or "").split("; ") if c.strip()]
        if final.get("action") == "hold":
            for clause in clauses:
                if "clamped" in clause or "cut to" in clause:
                    continue
                held[classify_override(clause)] += 1
        else:
            if row["execution_result"].get("status") == "success":
                executed += 1
            for clause in clauses:
                if "clamped" in clause or "cut to" in clause:
                    resized[classify_override(clause)] += 1
    return {
        "proposed_buys": proposed,
        "executed_buys": executed,
        "held_by_rule": dict(sorted(held.items(), key=lambda kv: -kv[1])),
        "resized_by_rule": dict(sorted(resized.items(), key=lambda kv: -kv[1])),
    }


# ------------------------------------------------------------------ report


def _period(trades: List[LiveTrade], signal_rows, start: Optional[datetime]) -> Dict[str, Any]:
    in_period = [t for t in trades if start is None or parse_utc(t.closed_date) >= start]
    rs = [t.r_multiple for t in in_period if t.r_multiple is not None]
    summary = diag.realized_risk_reward(in_period)
    summary["avg_r_multiple"] = sum(rs) / len(rs) if rs else None
    summary["avg_holding_days"] = (
        sum(t.holding_days for t in in_period) / len(in_period) if in_period else None
    )
    return {
        "start": start.isoformat() if start else None,
        "summary": summary,
        "by_asset_class": diag.breakdown_by_asset_class(in_period),
        "by_trigger": diag._group_stats(in_period, "trigger_reason"),
        "by_close_reason": diag.breakdown_by_close_reason(in_period),
        "confidence_vs_outcome": diag.confidence_vs_outcome(in_period),
        "overrides": override_stats(signal_rows, start),
        "trades": [asdict(t) for t in sorted(in_period, key=lambda t: t.closed_date, reverse=True)],
    }


def build_report(
    fills: List[Dict[str, Any]],
    db_path: str,
    config: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    crypto_bases = {
        symbol_config.alpaca_position_symbol(e["symbol"]).upper(): e["symbol"]
        for e in config.get("symbols") or []
        if isinstance(e, dict) and symbol_config.is_crypto(e["symbol"], config)
    }
    signal_rows, broker_exit_kinds = _load_signal_rows(db_path)
    orders = _orders_from_fills(fills, crypto_bases)
    trades, open_lots = reconstruct_trades(orders, signal_rows, broker_exit_kinds, config)
    week = _period(trades, signal_rows, now - REPORT_WINDOW)
    all_time = _period(trades, signal_rows, None)
    all_time.pop("trades")  # the week's list is what the dashboard shows
    return {
        "generated_at": now.isoformat(),
        "mode": "live",
        "fills_read": len(fills),
        "open_lots": open_lots,
        "min_meaningful_sample": MIN_MEANINGFUL_SAMPLE,
        "week": week,
        "all_time": all_time,
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def _num(value: Optional[float], spec: str = "+.2f") -> str:
    return "n/a" if value is None else format(value, spec)


def format_telegram(report: Dict[str, Any]) -> Tuple[str, str]:
    """(subject, body) for the weekly Telegram message."""
    week, total = report["week"]["summary"], report["all_time"]["summary"]
    subject = (f"WEEKLY REPORT: {week['total_closed_trades']} trades, "
               f"P&L ${week['total_realized_pnl_usd']:+.2f}")
    lines = [
        f"This week: win rate {_pct(week['win_rate_pct'])}, "
        f"avg R {_num(week['avg_r_multiple'])}, win/loss ratio {_num(week['risk_reward_ratio'], '.2f')}",
        f"All time: {total['total_closed_trades']} trades, P&L ${total['total_realized_pnl_usd']:+.2f}, "
        f"win rate {_pct(total['win_rate_pct'])}, avg R {_num(total['avg_r_multiple'])}",
    ]
    if total["total_closed_trades"] < report["min_meaningful_sample"]:
        lines.append(f"(fewer than {report['min_meaningful_sample']} closed trades: treat rates as noise)")
    exits = report["week"]["by_close_reason"]
    if exits:
        lines.append("Exits: " + ", ".join(f"{r['close_reason']} {r['count']}" for r in exits))
    triggers = report["week"]["by_trigger"]
    if triggers:
        lines.append("By trigger: " + ", ".join(
            f"{r['name']} {r['count']} (${r['total_pnl_usd']:+.0f})" for r in triggers))
    ov = report["week"]["overrides"]
    lines.append(f"Buy proposals: {ov['proposed_buys']}, executed {ov['executed_buys']}")
    if ov["held_by_rule"]:
        lines.append("Held by: " + ", ".join(f"{k} {v}" for k, v in ov["held_by_rule"].items()))
    corr = report["all_time"]["confidence_vs_outcome"]["confidence_vs_pnl_correlation"]
    lines.append(f"Confidence vs P&L correlation (all time): {_num(corr, '+.2f')}")
    return subject, "\n".join(lines)


def write_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)


def run(config_path: str, output_path: str, send: bool = True) -> int:
    import main as main_module

    config = main_module.load_config(config_path)
    if not resolve_is_live(config):
        print("live_report: live_execution is off; the report covers real trades only. Skipping.")
        return 0
    db_path = config.get("db_path", "trading_bot.db")
    bot_logger = BotLogger(db_path)
    funded = parse_utc(bot_logger.get_meta(EQUITY_CURVE_START_KEY) or "")
    after = (funded or datetime.now(timezone.utc) - DEFAULT_HISTORY) - timedelta(days=1)

    fills = fetch_fill_activities(after.isoformat())
    report = build_report(fills, db_path, config)
    write_report(report, output_path)
    subject, body = format_telegram(report)
    print(subject)
    print(body)
    if send:
        notifications.send_weekly_report(subject, body)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly live-results report.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-send", action="store_true", help="write the JSON only; no Telegram/push")
    args = parser.parse_args()
    return run(args.config, args.output, send=not args.no_send)


if __name__ == "__main__":
    raise SystemExit(main())
