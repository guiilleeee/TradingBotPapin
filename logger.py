"""SQLite persistence: signal audit trail, realised P&L, and the bot-managed ledger.

Three tables and no ORM. Every timestamp written here is UTC, and every read that
filters by day must use UTC too -- mixing the two silently misaligns the day
boundary depending on which machine the cycle happens to run on.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Set

from models import ExecutionResult, ExistingPosition, SignalInput, SignalOutput, TradeSignal

DEFAULT_DB_PATH = "trading_bot.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    signal_input     TEXT,
    raw_output       TEXT,
    final_signal     TEXT,
    override_reason  TEXT,
    execution_result TEXT,
    is_live          INTEGER
);

CREATE TABLE IF NOT EXISTS pnl (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    realized_pnl_usd REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS simulated_positions (
    symbol            TEXT PRIMARY KEY,
    qty               REAL NOT NULL,
    avg_entry_price   REAL NOT NULL,
    stop_loss_price   REAL,
    take_profit_price REAL,
    opened_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    equity           REAL NOT NULL,
    benchmark_symbol TEXT NOT NULL,
    benchmark_price  REAL,
    is_live          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint          TEXT PRIMARY KEY,
    auth              TEXT NOT NULL,
    p256dh            TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Live bracket exit legs already booked into pnl, so a leg is never booked twice.
CREATE TABLE IF NOT EXISTS broker_exits (
    leg_order_id     TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    kind             TEXT NOT NULL,
    filled_at        TEXT NOT NULL,
    qty              REAL NOT NULL,
    entry_price      REAL NOT NULL,
    exit_price       REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL
);
"""

EQUITY_CURVE_START_KEY = "equity_curve_start"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: Any) -> Optional[datetime]:
    """An ISO-8601 timestamp as an aware UTC datetime, or None if it isn't one."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_day_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _dump(value: Any) -> Optional[str]:
    """JSON-serialise a Pydantic model, a dict, or None for a blob column."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), default=str)
    return json.dumps(value, default=str)


def _load(blob: Any) -> Dict[str, Any]:
    try:
        return json.loads(blob) if blob else {}
    except (TypeError, ValueError):
        return {}


class BotLogger:
    """Owns the SQLite file. One connection per call, so nothing stays locked.

    Note on `simulated_positions`: in simulation it holds every open position. In
    live it holds only positions the bot has to exit itself -- fractional/notional
    entries the broker will not accept bracket legs for. Same table, same sweep.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_is_live_column(conn)
            self._migrate_trigger_reason_column(conn)

    def _migrate_is_live_column(self, conn: sqlite3.Connection) -> None:
        """Add `signals.is_live`, once, for a db created before this column existed.

        CREATE TABLE IF NOT EXISTS never adds a column to an existing table, so a
        db from before the dashboard's SIMULACIO/REAL row labeling needed this has
        to be migrated explicitly. Every row written before this migration reads
        back with is_live NULL -- an honest "mode unknown for this old row", never
        guessed.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(signals)")}
        if "is_live" not in columns:
            conn.execute("ALTER TABLE signals ADD COLUMN is_live INTEGER")

    def _migrate_trigger_reason_column(self, conn: sqlite3.Connection) -> None:
        """Add `signals.trigger_reason`, once, for a db created before volume-wake.

        Same pattern as _migrate_is_live_column. Old rows read back with
        trigger_reason NULL -- an honest "reason unknown for this old row".
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(signals)")}
        if "trigger_reason" not in columns:
            conn.execute("ALTER TABLE signals ADD COLUMN trigger_reason TEXT")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ signals

    def log_signal(
        self,
        symbol: str,
        signal_input: Optional[SignalInput],
        raw_output: Optional[SignalOutput],
        final_signal: Optional[TradeSignal],
        execution_result: Optional[ExecutionResult] = None,
        is_live: Optional[bool] = None,
        trigger_reason: Optional[str] = None,
    ) -> int:
        """`is_live` is optional (defaults to None, i.e. "mode unknown") purely so
        every existing call site that predates this field keeps working unchanged.
        main.py always passes it explicitly -- see run_cycle -- so it is only ever
        None for rows logged before this field existed, or via a caller that has
        deliberately chosen not to record it.

        `trigger_reason` distinguishes scheduled cycles from volume-wake cycles.
        None for rows logged before volume-wake existed.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO signals (timestamp, symbol, signal_input, raw_output, "
                "final_signal, override_reason, execution_result, is_live, trigger_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_now_iso(),
                    symbol,
                    _dump(signal_input),
                    _dump(raw_output),
                    _dump(final_signal),
                    final_signal.override_reason if final_signal else None,
                    _dump(execution_result),
                    None if is_live is None else int(is_live),
                    trigger_reason,
                ),
            )
            return int(cur.lastrowid or 0)

    def symbols_signalled_today(self) -> Set[str]:
        """Upper-cased symbols with any signal row since 00:00 UTC today.

        funnel.prefilter uses this to promote a pool symbol at most once per day.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM signals WHERE timestamp >= ?",
                (_utc_day_start_iso(),),
            ).fetchall()
        return {str(row["symbol"]).upper() for row in rows}

    def log_auto_close_signal(
        self,
        symbol: str,
        reason: str,
        price: float,
        qty: float,
        pnl: float,
        equity: float,
        is_live: Optional[bool] = None,
        entry_price: Optional[float] = None,
        timestamp: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> int:
        """Write a synthetic signal row for a stop-loss / take-profit auto-close.

        `timestamp` defaults to now; a broker-side exit found later passes its fill time.
        `order_id` is the broker order that closed it, when there was one -- the
        weekly report (live_report.py) matches fills back to their reason by it.

        The dashboard renders every row through one code path, so these blobs must
        carry the same keys a model-driven row does. Two in particular:
        account_equity_usd, which the dashboard reads off the newest row for
        "Patrimoni total", and reasoning, which rendered as "undefined" without one.
        Both were omitted in an earlier build; hence the asserts below.
        """
        signal_input: Dict[str, Any] = {
            "symbol": symbol,
            "current_price": price,
            "account_equity_usd": equity,
            "existing_position": {"qty": qty, "avg_entry_price": entry_price},
            "technical_indicators": None,
            "recent_headlines": [],
            "synthetic": True,
        }
        decision: Dict[str, Any] = {
            "symbol": symbol,
            "action": "sell",
            "confidence": 1.0,
            "position_size_pct": 0.0,
            "stop_loss_price": None,
            "take_profit_price": None,
            "reasoning": reason,
        }
        final_signal = dict(decision, override_reason="automatic exit", raw_action="sell")
        execution_result = {
            "status": "success",
            "order_id": order_id,
            "fill_price": price,
            "message": reason,
            "realized_pnl_usd": pnl,
            "qty": qty,
            "entry_price": entry_price,
        }

        assert signal_input["account_equity_usd"] is not None, "synthetic row needs equity"
        assert final_signal["reasoning"], "synthetic row needs reasoning"

        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO signals (timestamp, symbol, signal_input, raw_output, "
                "final_signal, override_reason, execution_result, is_live) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp or utc_now_iso(),
                    symbol,
                    json.dumps(signal_input),
                    json.dumps(decision),
                    json.dumps(final_signal),
                    "automatic exit",
                    json.dumps(execution_result),
                    None if is_live is None else int(is_live),
                ),
            )
            return int(cur.lastrowid or 0)

    def get_last_buy_price(self, symbol: str) -> Optional[float]:
        """Cost basis for `symbol` taken from the most recent logged buy.

        Needed for OKX, whose spot balance carries no entry price. This is only
        correct because the duplicate-buy guard in execution.py keeps at most one
        open buy per symbol at a time, so "the last buy" is the open position.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT final_signal, signal_input, execution_result FROM signals "
                "WHERE symbol = ? ORDER BY id DESC LIMIT 200",
                (symbol,),
            ).fetchall()

        for row in rows:
            final = _load(row["final_signal"])
            if final.get("action") != "buy":
                continue

            execution = _load(row["execution_result"])
            if execution.get("status") not in ("success", "dry_run"):
                continue

            fill = execution.get("fill_price")
            if fill:
                return float(fill)

            entered = _load(row["signal_input"]).get("current_price")
            if entered:
                return float(entered)

        return None

    def get_position_opened_at(self, symbol: str) -> Optional[str]:
        """When the currently open position in `symbol` was opened, or None.

        The ledger's opened_at when the bot manages the exit; otherwise the most
        recent buy that actually executed (at most one open buy per symbol, see
        get_last_buy_price), which covers live bracket entries.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT opened_at FROM simulated_positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            if row is not None and row["opened_at"]:
                return str(row["opened_at"])
            rows = conn.execute(
                "SELECT timestamp, final_signal, execution_result FROM signals "
                "WHERE symbol = ? ORDER BY id DESC LIMIT 200",
                (symbol,),
            ).fetchall()
        for row in rows:
            if _load(row["final_signal"]).get("action") != "buy":
                continue
            if _load(row["execution_result"]).get("status") in ("success", "dry_run"):
                return str(row["timestamp"])
        return None

    # ------------------------------------------------------ benchmark snapshots

    def record_benchmark_snapshot(
        self,
        equity: float,
        benchmark_symbol: str,
        benchmark_price: Optional[float],
        is_live: bool,
    ) -> None:
        """One (equity, benchmark price) pair, taken at the same instant each cycle."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO benchmark_snapshots (timestamp, equity, benchmark_symbol, "
                "benchmark_price, is_live) VALUES (?, ?, ?, ?, ?)",
                (utc_now_iso(), float(equity), benchmark_symbol,
                 None if benchmark_price is None else float(benchmark_price), int(bool(is_live))),
            )

    def get_benchmark_snapshots(self, is_live: bool = True) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, equity, benchmark_symbol, benchmark_price FROM "
                "benchmark_snapshots WHERE is_live = ? ORDER BY timestamp, id",
                (int(bool(is_live)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_first_live_trade_timestamp(self) -> Optional[str]:
        """Timestamp of the first live buy/sell that actually filled, or None."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, final_signal, execution_result FROM signals "
                "WHERE is_live = 1 ORDER BY id"
            ).fetchall()
        for row in rows:
            if _load(row["final_signal"]).get("action") not in ("buy", "sell"):
                continue
            if _load(row["execution_result"]).get("status") == "success":
                return str(row["timestamp"])
        return None

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_meta_once(self, key: str, value: str) -> bool:
        """Write `key` only if it has never been set. True if this call set it."""
        with self._conn() as conn:
            cur = conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            return cur.rowcount == 1

    def record_funding_if_first(self, real_equity: Optional[float], threshold_usd: float) -> Optional[str]:
        """Mark where the displayed equity curve starts: the first *real* balance
        read at or above `threshold_usd`. Set once, never moved. Returns the new
        timestamp if this call set it, else None.

        `real_equity` must be None when the read failed -- a fallback figure is
        exactly the fake balance this marker exists to hide.
        """
        if real_equity is None or real_equity < threshold_usd:
            return None
        now = utc_now_iso()
        return now if self.set_meta_once(EQUITY_CURVE_START_KEY, now) else None

    # -------------------------------------------------------- push subscriptions

    def save_push_subscription(self, endpoint: str, auth: str, p256dh: str) -> None:
        """Save a new Web Push subscription."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO push_subscriptions (endpoint, auth, p256dh, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (endpoint, auth, p256dh, utc_now_iso()),
            )

    def delete_push_subscription(self, endpoint: str) -> None:
        """Remove an invalid or unsubscribed Web Push subscription."""
        with self._conn() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))

    def get_all_push_subscriptions(self) -> List[Dict[str, str]]:
        """Return all active Web Push subscriptions."""
        with self._conn() as conn:
            rows = conn.execute("SELECT endpoint, auth, p256dh FROM push_subscriptions").fetchall()
            return [{"endpoint": r["endpoint"], "auth": r["auth"], "p256dh": r["p256dh"]} for r in rows]

    # ---------------------------------------------------------------------- pnl

    def record_pnl(self, symbol: str, amount: float, timestamp: Optional[str] = None) -> None:
        """Book a realised P&L amount.

        Must be called on every close, live and simulated. An earlier build defined
        this method and never invoked it anywhere, which left the circuit breaker
        permanently inert: get_today_realized_loss_pct always summed an empty table.

        `timestamp` is when the close actually happened, for a close discovered
        after the fact (a broker-side bracket exit); it decides which UTC day's
        circuit breaker the amount counts toward. Defaults to now.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO pnl (timestamp, symbol, realized_pnl_usd) VALUES (?, ?, ?)",
                (timestamp or utc_now_iso(), symbol, float(amount)),
            )

    def record_broker_exit(self, fill: Dict[str, Any]) -> Optional[float]:
        """Book one filled bracket leg (execution.fetch_bracket_exit_fills item).

        Returns the realised P&L if this leg was new, or None if it was already
        booked. The dedupe row and the pnl row go in one transaction, so a crash
        between them can neither lose the loss nor count it twice.
        """
        pnl = (float(fill["exit_price"]) - float(fill["entry_price"])) * float(fill["qty"])
        filled_at = parse_utc(fill["filled_at"])
        stamp = filled_at.isoformat() if filled_at else utc_now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO broker_exits (leg_order_id, symbol, kind, filled_at, qty, "
                "entry_price, exit_price, realized_pnl_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fill["leg_id"], fill["symbol"], fill["kind"], stamp, float(fill["qty"]),
                 float(fill["entry_price"]), float(fill["exit_price"]), pnl),
            )
            if cur.rowcount == 0:
                return None
            conn.execute(
                "INSERT INTO pnl (timestamp, symbol, realized_pnl_usd) VALUES (?, ?, ?)",
                (stamp, fill["symbol"], pnl),
            )
        return pnl

    def get_today_realized_loss_pct(self, equity: float) -> float:
        """Today's realised P&L as a percent of equity. Negative means a loss.

        UTC day boundary, matching how the timestamps are written. Local time here
        would misalign the day depending on where the runner happens to live.
        """
        if equity <= 0:
            return 0.0
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0.0) AS total FROM pnl WHERE timestamp >= ?",
                (_utc_day_start_iso(),),
            ).fetchone()
        return (float(row["total"]) / equity) * 100.0

    def get_all_time_realized_pnl(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0.0) AS total FROM pnl"
            ).fetchone()
        return float(row["total"])

    # --------------------------------------------------------- simulated ledger

    def get_simulated_position(self, symbol: str) -> Optional[ExistingPosition]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT qty, avg_entry_price FROM simulated_positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None or float(row["qty"]) <= 0:
            return None
        return ExistingPosition(
            qty=float(row["qty"]), avg_entry_price=float(row["avg_entry_price"])
        )

    def get_all_simulated_positions(self) -> List[Dict[str, Any]]:
        """Full rows, including the exit levels the sweep needs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, qty, avg_entry_price, stop_loss_price, take_profit_price, "
                "opened_at FROM simulated_positions ORDER BY symbol"
            ).fetchall()
        return [dict(row) for row in rows]

    def open_simulated_position(
        self,
        symbol: str,
        qty: float,
        avg_entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO simulated_positions (symbol, qty, avg_entry_price, "
                "stop_loss_price, take_profit_price, opened_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, "
                "avg_entry_price=excluded.avg_entry_price, "
                "stop_loss_price=excluded.stop_loss_price, "
                "take_profit_price=excluded.take_profit_price, opened_at=excluded.opened_at",
                (
                    symbol,
                    float(qty),
                    float(avg_entry_price),
                    stop_loss_price,
                    take_profit_price,
                    utc_now_iso(),
                ),
            )

    def close_simulated_position(self, symbol: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM simulated_positions WHERE symbol = ?", (symbol,))

    # ------------------------------------------------------------------- export

    def export_signals_csv(self, path: str, limit: int = 500, since: Optional[str] = None) -> int:
        """Flatten the newest `limit` signal rows into the dashboard's CSV.

        `since` (the dashboard's "history cleared at") leaves older rows out of
        the file only; the database keeps every row.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        cutoff = parse_utc(since) if since else None
        if cutoff is not None:
            rows = [r for r in rows if (parse_utc(r["timestamp"]) or cutoff) >= cutoff]

        fields = [
            "id",
            "timestamp",
            "symbol",
            "action",
            "raw_action",
            "confidence",
            "position_size_pct",
            "stop_loss_price",
            "take_profit_price",
            "current_price",
            "account_equity_usd",
            "override_reason",
            "execution_status",
            "fill_price",
            "entry_price",
            "qty",
            "realized_pnl_usd",
            "reasoning",
            "recent_headlines",
            "market_positioning",
            "mode",
            "trigger_reason",
        ]

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in reversed(rows):
                final = _load(row["final_signal"])
                inp = _load(row["signal_input"])
                execution = _load(row["execution_result"])
                writer.writerow(
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "symbol": row["symbol"],
                        "action": final.get("action"),
                        "raw_action": final.get("raw_action"),
                        "confidence": final.get("confidence"),
                        "position_size_pct": final.get("position_size_pct"),
                        "stop_loss_price": final.get("stop_loss_price"),
                        "take_profit_price": final.get("take_profit_price"),
                        "current_price": inp.get("current_price"),
                        "account_equity_usd": inp.get("account_equity_usd"),
                        "override_reason": row["override_reason"],
                        "execution_status": execution.get("status"),
                        "fill_price": execution.get("fill_price"),
                        "entry_price": execution.get("entry_price"),
                        "qty": execution.get("qty"),
                        "realized_pnl_usd": execution.get("realized_pnl_usd"),
                        "reasoning": final.get("reasoning"),
                        # json.dumps, not the raw list/None -- CSV has no native list
                        # type, and the dashboard's Analisi tab JSON.parses this back.
                        "recent_headlines": json.dumps(inp.get("recent_headlines") or []),
                        "market_positioning": inp.get("market_positioning"),
                        # Plain text, no emoji -- the dashboard applies its own color
                        # coding. NULL (row logged before this field existed, or a
                        # caller that chose not to record it) exports as "", never a
                        # guessed mode -- see BotLogger.log_signal's is_live docstring.
                        "mode": (
                            "REAL" if row["is_live"] == 1
                            else "SIMULACIO" if row["is_live"] == 0
                            else ""
                        ),
                        # "scheduled" for normal 8h cycles, "volume_wake" for
                        # off-schedule cycles triggered by volume_watch.py.
                        # NULL (pre-existing rows) exports as "".
                        "trigger_reason": row["trigger_reason"] or "",
                    }
                )
        return len(rows)
