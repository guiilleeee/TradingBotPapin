"""SPY benchmark: the bot's cumulative return against buy-and-hold the S&P 500.

Every cycle records one snapshot -- the equity main.py just computed, and SPY's
latest close fetched at the same moment -- into `benchmark_snapshots`. The
export then rebases both series to 0% at the bot's first live trade, so the
dashboard's Resum tab can plot "what the bot did" against "what simply holding
SPY would have done" over exactly the same window.

The benchmark used to be QQQ (the universe was Nasdaq-heavy; it is now the S&P 500).
Snapshots already stored with benchmark_symbol = 'QQQ' are never rewritten. At
export time their SPY price is looked up from SPY's hourly history at the
snapshot's timestamp, so the comparison covers the whole period. If that history
cannot be fetched, those snapshots are skipped and the SPY line starts (and is
rebased) at the first snapshot that carries a SPY price; the payload says so in
`benchmark_rebased` and the dashboard repeats it in its footnote. The old QQQ
series stays in each point as `qqq_pct`.

Baseline: the last live snapshot taken at or before the first live fill (the
start of that cycle -- equity before the trade, SPY at the same instant). If no
snapshot predates it, the first one after it.

Caveat, stated once here and on the dashboard: account equity moves with
deposits and withdrawals as well as trading. A deposit shows up as bot
"return". This is a like-for-like comparison only while no money moves in or out.

Soft-fail throughout, like every other dashboard export: a SPY fetch failure
records the snapshot with a null price (skipped by the export) and never
fails the cycle.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Dict, List, Optional

import data_fetcher
from logger import EQUITY_CURVE_START_KEY, BotLogger, parse_utc, utc_now_iso

BENCHMARK_SYMBOL = "SPY"
LEGACY_SYMBOL = "QQQ"  # the previous benchmark; its stored snapshots stay as they are
DEFAULT_BENCHMARK_PATH = "benchmark.json"


def fetch_benchmark_price(symbol: str = BENCHMARK_SYMBOL) -> Optional[float]:
    """Latest close of the benchmark (the live price mid-session). None on any failure."""
    try:
        return data_fetcher.latest_price(data_fetcher.fetch_ohlcv(symbol, period="5d"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [benchmark] {symbol} price unavailable ({type(exc).__name__}: {exc})")
        return None


def record_snapshot(bot_logger: BotLogger, equity: float, is_live: bool) -> None:
    price = fetch_benchmark_price()
    bot_logger.record_benchmark_snapshot(equity, BENCHMARK_SYMBOL, price, is_live)


def fetch_benchmark_history(symbol: str = BENCHMARK_SYMBOL) -> Any:
    """Hourly bars covering the account's whole life so far. None on any failure."""
    try:
        return data_fetcher.fetch_ohlcv(symbol, period="6mo", interval="1h")
    except Exception as exc:  # noqa: BLE001
        print(f"  [benchmark] {symbol} history unavailable ({type(exc).__name__}: {exc})")
        return None


def price_at(history: Any, timestamp: str) -> Optional[float]:
    """The benchmark's price at `timestamp` from hourly bars, or None.

    Inside a bar: that bar's open (the price at its start, at most an hour old).
    After the bar has ended (market closed, or the next bar is missing): its
    close, i.e. the last traded price. Older than a week of bars: no answer.
    """
    try:
        when = parse_utc(timestamp)
        if history is None or len(history) == 0 or when is None:
            return None
        index = history.index
        index = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
        # yfinance's index can be second-resolution; a microsecond timestamp would not compare.
        when = when.replace(microsecond=0)
        pos = index.searchsorted(when, side="right") - 1
        if pos < 0:
            return None
        start = index[pos].to_pydatetime()
        if when - start > timedelta(days=7):
            return None
        row = history.iloc[pos]
        price = float(row["Open"]) if when - start < timedelta(hours=1) else float(row["Close"])
        return price if price > 0 else None
    except Exception:  # noqa: BLE001
        return None


def build_series(bot_logger: BotLogger) -> Dict[str, Any]:
    first_trade = bot_logger.get_first_live_trade_timestamp()
    payload: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "benchmark_symbol": BENCHMARK_SYMBOL,
        "legacy_benchmark_symbol": LEGACY_SYMBOL,
        "first_live_trade_at": first_trade,
        # The dashboard's equity curve starts here (the account's first real funding).
        "equity_curve_start": bot_logger.get_meta(EQUITY_CURVE_START_KEY),
        "baseline": None,
        # True when SPY could not be priced at the first live trade, so the
        # benchmark line starts (at 0%) later than the bot line.
        "benchmark_rebased": False,
        # True when part of the SPY line comes from hourly history instead of
        # prices recorded at the time (snapshots stored while QQQ was benchmark).
        "benchmark_backfilled": False,
        "points": [],
    }
    if first_trade is None:
        return payload

    snapshots = [
        s for s in bot_logger.get_benchmark_snapshots(is_live=True)
        if s["equity"] and s["equity"] > 0
    ]
    needs_history = any(s.get("benchmark_symbol") != BENCHMARK_SYMBOL for s in snapshots)
    history = fetch_benchmark_history(BENCHMARK_SYMBOL) if needs_history else None

    priced: List[Dict[str, Any]] = []
    backfilled = False
    for snap in snapshots:
        symbol, recorded = snap.get("benchmark_symbol"), snap.get("benchmark_price")
        spy = None
        if symbol == BENCHMARK_SYMBOL:
            spy = float(recorded) if recorded else None
        else:
            spy = price_at(history, snap["timestamp"])
            backfilled = backfilled or spy is not None
        if spy is None:
            continue
        priced.append({
            **snap, "spy": spy,
            "qqq": float(recorded) if symbol == LEGACY_SYMBOL and recorded else None,
        })
    if not priced:
        return payload

    before = [s for s in priced if s["timestamp"] <= first_trade]
    baseline = before[-1] if before else priced[0]
    base_equity = float(baseline["equity"])
    base_price = float(baseline["spy"])
    base_qqq = baseline["qqq"]

    points: List[Dict[str, Any]] = []
    for snap in priced:
        if snap["timestamp"] < baseline["timestamp"]:
            continue
        point = {
            "t": snap["timestamp"],
            "equity": float(snap["equity"]),
            "benchmark_price": snap["spy"],
            "bot_pct": (float(snap["equity"]) / base_equity - 1.0) * 100.0,
            "benchmark_pct": (snap["spy"] / base_price - 1.0) * 100.0,
        }
        if base_qqq and snap["qqq"]:
            point["qqq_pct"] = (snap["qqq"] / base_qqq - 1.0) * 100.0
        points.append(point)

    payload["baseline"] = {
        "t": baseline["timestamp"],
        "equity": base_equity,
        "benchmark_price": base_price,
    }
    payload["benchmark_rebased"] = not before
    payload["benchmark_backfilled"] = backfilled
    payload["points"] = points
    return payload


def export_benchmark_json(bot_logger: BotLogger, path: str = DEFAULT_BENCHMARK_PATH) -> int:
    payload = build_series(bot_logger)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return len(payload["points"])
