"""QQQ benchmark: the bot's cumulative return against buy-and-hold Nasdaq-100.

Every cycle records one snapshot -- the equity main.py just computed, and QQQ's
latest close fetched at the same moment -- into `benchmark_snapshots`. The
export then rebases both series to 0% at the bot's first live trade, so the
dashboard's Resum tab can plot "what the bot did" against "what simply holding
QQQ would have done" over exactly the same window.

Baseline: the last live snapshot taken at or before the first live fill (the
start of that cycle -- equity before the trade, QQQ at the same instant). If no
snapshot predates it, the first one after it.

Caveat, stated once here and on the dashboard: account equity moves with
deposits and withdrawals as well as trading. A deposit shows up as bot
"return". This is a like-for-like comparison only while no money moves in or out.

Soft-fail throughout, like every other dashboard export: a QQQ fetch failure
records the snapshot with a null price (skipped by the export) and never
fails the cycle.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import data_fetcher
from logger import EQUITY_CURVE_START_KEY, BotLogger, utc_now_iso

BENCHMARK_SYMBOL = "QQQ"
DEFAULT_BENCHMARK_PATH = "benchmark.json"


def fetch_benchmark_price(symbol: str = BENCHMARK_SYMBOL) -> Optional[float]:
    """Latest QQQ close (the live price mid-session). None on any failure."""
    try:
        return data_fetcher.latest_price(data_fetcher.fetch_ohlcv(symbol, period="5d"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [benchmark] {symbol} price unavailable ({type(exc).__name__}: {exc})")
        return None


def record_snapshot(bot_logger: BotLogger, equity: float, is_live: bool) -> None:
    price = fetch_benchmark_price()
    bot_logger.record_benchmark_snapshot(equity, BENCHMARK_SYMBOL, price, is_live)


def build_series(bot_logger: BotLogger) -> Dict[str, Any]:
    first_trade = bot_logger.get_first_live_trade_timestamp()
    payload: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "benchmark_symbol": BENCHMARK_SYMBOL,
        "first_live_trade_at": first_trade,
        # The dashboard's equity curve starts here (the account's first real funding).
        "equity_curve_start": bot_logger.get_meta(EQUITY_CURVE_START_KEY),
        "baseline": None,
        "points": [],
    }
    if first_trade is None:
        return payload

    snapshots = [
        s for s in bot_logger.get_benchmark_snapshots(is_live=True)
        if s["benchmark_price"] and s["equity"] and s["equity"] > 0
    ]
    if not snapshots:
        return payload

    before = [s for s in snapshots if s["timestamp"] <= first_trade]
    baseline = before[-1] if before else snapshots[0]
    base_equity = float(baseline["equity"])
    base_price = float(baseline["benchmark_price"])

    points: List[Dict[str, Any]] = []
    for snap in snapshots:
        if snap["timestamp"] < baseline["timestamp"]:
            continue
        points.append({
            "t": snap["timestamp"],
            "equity": float(snap["equity"]),
            "benchmark_price": float(snap["benchmark_price"]),
            "bot_pct": (float(snap["equity"]) / base_equity - 1.0) * 100.0,
            "benchmark_pct": (float(snap["benchmark_price"]) / base_price - 1.0) * 100.0,
        })

    payload["baseline"] = {
        "t": baseline["timestamp"],
        "equity": base_equity,
        "benchmark_price": base_price,
    }
    payload["points"] = points
    return payload


def export_benchmark_json(bot_logger: BotLogger, path: str = DEFAULT_BENCHMARK_PATH) -> int:
    payload = build_series(bot_logger)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return len(payload["points"])
