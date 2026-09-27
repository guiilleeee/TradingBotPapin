"""Order routing for Alpaca: US equities, plus a fixed set of spot crypto pairs.

Three rules run through everything in this module:

  * Real broker APIs are only ever touched when `is_live` is True. Simulation never
    authenticates and never places an order -- it runs the same sizing and the same
    guards, then returns a dry_run result.
  * The system is spot-only. It opens longs and closes them. It never shorts, so a
    sell with nothing held is a skip, not an order.
  * No leverage, no margin, no options, ever. Only plain equity orders and plain
    spot crypto orders are placed.

Crypto differs from equities in exactly three ways here, all in `_execute_crypto`:
it trades 24/7 (no market-hours gate), Alpaca accepts no bracket legs on it (so
every crypto entry is a bot-managed exit swept by main.py), and its symbols are
translated from the project's "BTC-USD" form to Alpaca's "BTC/USD" / "BTCUSD".
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

import symbol_config
from models import ExecutionResult, ExistingPosition, TradeSignal
from secrets_redaction import sanitize

# Live endpoint by default. This whole path is already gated behind an explicit
# `live_execution: true`, so silently routing a "live" run to paper would be its
# own kind of wrong. Point ALPACA_BASE_URL at https://paper-api.alpaca.markets to
# rehearse against paper with live_execution on.
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Alpaca rejects notional orders below $1.
ALPACA_MIN_NOTIONAL_USD = 1.0
# Limit offset for a bracket's stop_loss leg, as a fraction of the stop price.
STOP_LIMIT_BUFFER = 0.01
HTTP_TIMEOUT = 20.0

# Marks a fill whose exit the bot has to manage itself, because no bracket could be
# attached (a notional/fractional equity entry). main.py registers these in the
# ledger so the per-cycle sweep can close them.
MANAGED_EXIT_MARKER = "[managed-exit]"

# A bracket's exit legs hold the position's shares, so Alpaca rejects a separate
# market sell for them ("insufficient qty available"). Before a live equity sell
# the legs are cancelled, and cancellation is asynchronous: poll until each leg
# reports a terminal status, and never send the sell while one is still live.
CANCEL_CONFIRM_ATTEMPTS = 10
CANCEL_CONFIRM_INTERVAL_SECONDS = 0.5
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}


def needs_managed_exit(result: ExecutionResult) -> bool:
    """True when this fill has no broker-side stop and the sweep must cover it."""
    return MANAGED_EXIT_MARKER in (result.message or "")


# ----------------------------------------------------------------- credentials


def _alpaca_credentials() -> Tuple[Optional[str], Optional[str]]:
    return os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")


def _alpaca_headers() -> Dict[str, str]:
    key, secret = _alpaca_credentials()
    return {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
        "Content-Type": "application/json",
    }


# ----------------------------------------------------------- market hours


def _is_market_open() -> bool:
    """Check if NYSE/NASDAQ is currently open (weekdays 9:30-16:00 ET).

    Uses the Alpaca clock endpoint when credentials are available (authoritative),
    falls back to a simple timezone-based check otherwise. The fallback does not
    account for holidays, but erring on the side of attempting a trade that Alpaca
    will reject with a clear error is better than silently skipping a valid
    opportunity because our holiday list is stale.
    """
    key, secret = _alpaca_credentials()
    if key and secret:
        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/v2/clock",
                headers=_alpaca_headers(),
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("is_open", False)
        except Exception:
            pass  # fall through to local check

    # Local fallback: simple weekday + time-of-day check in US/Eastern.
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Weekday: Monday=0, Sunday=6
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


# ------------------------------------------------------------------- positions


def fetch_existing_position(
    symbol: str,
    is_live: bool,
    bot_logger: Any,
    asset_class: Optional[str] = None,
) -> Optional[ExistingPosition]:
    """Current holding for `symbol`, from the broker in live and the ledger in sim.

    `asset_class` is accepted for callers that already know it (position_metrics.py
    passes it); the broker symbol is derived from `symbol` itself either way. An
    earlier signature without this parameter made every live position lookup from
    position_metrics.py raise TypeError, silently emptying positions.json in live.
    """
    if not is_live:
        # No broker call at all. The simulated ledger is the whole truth here.
        return bot_logger.get_simulated_position(symbol)
    return _fetch_alpaca_position(symbol)


def fetch_all_live_positions() -> Dict[str, ExistingPosition]:
    """Every open Alpaca position, keyed by this project's symbol form ("BTC-USD").

    Raises on a broker failure, for the same reason _fetch_alpaca_position does:
    an empty dict means "flat everywhere", and callers act on that.
    """
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions",
        headers=_alpaca_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    out: Dict[str, ExistingPosition] = {}
    for row in resp.json() or []:
        qty = float(row.get("qty", 0.0) or 0.0)
        avg = float(row.get("avg_entry_price", 0.0) or 0.0)
        if qty <= 0 or avg <= 0:
            continue
        symbol = symbol_config.from_alpaca_symbol(
            str(row.get("symbol", "")), str(row.get("asset_class", ""))
        )
        out[symbol] = ExistingPosition(qty=qty, avg_entry_price=avg)
    return out


def fetch_live_exposures() -> Dict[str, float]:
    """Dollar exposure (market value) of every open Alpaca position, keyed like
    fetch_all_live_positions. Raises on a broker failure."""
    _require_credentials()
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions",
        headers=_alpaca_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    out: Dict[str, float] = {}
    for row in resp.json() or []:
        qty = float(row.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        value = row.get("market_value")
        if value in (None, ""):
            value = qty * float(row.get("avg_entry_price", 0.0) or 0.0)
        symbol = symbol_config.from_alpaca_symbol(
            str(row.get("symbol", "")), str(row.get("asset_class", ""))
        )
        out[symbol] = abs(float(value))
    return out


def _fetch_alpaca_position(symbol: str) -> Optional[ExistingPosition]:
    """Live equity position, or None if Alpaca says there genuinely isn't one.

    A failed lookup deliberately raises rather than returning None. "None" here
    means "no position held", which the duplicate-buy guard reads as permission
    to open one -- so swallowing a broker outage would let the bot double up on a
    position it already has. The caller skips the symbol instead.
    """
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions/{symbol_config.alpaca_position_symbol(symbol)}",
        headers=_alpaca_headers(),
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code == 404:
        return None  # Alpaca's explicit "flat in this symbol"
    resp.raise_for_status()

    data = resp.json()
    qty = float(data.get("qty", 0.0))
    # Alpaca returns the cost basis natively, so no reconstruction is needed.
    avg = float(data.get("avg_entry_price", 0.0))
    if qty <= 0 or avg <= 0:
        return None
    return ExistingPosition(qty=qty, avg_entry_price=avg)


def read_live_equity() -> Optional[float]:
    """Total account equity from Alpaca, or None if it could not be read.

    A successful read is returned as-is, including 0: an empty account must
    show as empty, not as the simulation's starting balance.
    """
    key, secret = _alpaca_credentials()
    if not (key and secret):
        return None
    try:
        resp = requests.get(
            f"{ALPACA_BASE_URL}/v2/account", headers=_alpaca_headers(), timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        equity = float(resp.json()["equity"])
    except Exception:
        return None
    return equity if math.isfinite(equity) else None


def fetch_live_equity(fallback: float) -> float:
    """`read_live_equity`, or `fallback` only if the read fails."""
    equity = read_live_equity()
    return fallback if equity is None else equity


def read_live_cash() -> Optional[float]:
    """Cash a new buy can actually spend, or None if it could not be read.

    The smaller of `cash` and `non_marginable_buying_power` -- the project never
    uses margin, and crypto buys are limited by the latter anyway.
    """
    key, secret = _alpaca_credentials()
    if not (key and secret):
        return None
    try:
        resp = requests.get(
            f"{ALPACA_BASE_URL}/v2/account", headers=_alpaca_headers(), timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    values = []
    for field_name in ("cash", "non_marginable_buying_power"):
        try:
            value = float(data[field_name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else None


def _require_credentials() -> None:
    """Fail fast, before any network call, when there is nothing to authenticate with."""
    key, secret = _alpaca_credentials()
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing")


# ----------------------------------------------------------- bracket exit legs


def _open_exit_orders(symbol: str) -> List[Dict[str, Any]]:
    """Every open sell order on `symbol` -- in practice a bracket's take-profit
    and stop-loss legs. Raises on a broker failure: "no open orders" would let
    the sell go out against shares the legs still hold."""
    order_symbol = symbol_config.alpaca_position_symbol(symbol)
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=_alpaca_headers(),
        params={"status": "open", "symbols": order_symbol, "nested": "true", "limit": 500},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    found: Dict[str, Dict[str, Any]] = {}
    for order in resp.json() or []:
        # nested=true rolls legs up under their parent; a flat listing returns
        # them as top-level rows. Accept either shape.
        for row in [order] + list(order.get("legs") or []):
            if (
                row.get("id")
                and row.get("side") == "sell"
                and str(row.get("symbol", order_symbol)).upper() == order_symbol.upper()
                and row.get("status") not in TERMINAL_ORDER_STATUSES
            ):
                found[str(row["id"])] = row
    return list(found.values())


def fetch_bracket_exit_fills(after_iso: str) -> List[Dict[str, Any]]:
    """Every bracket exit leg that has filled, for brackets submitted after `after_iso`.

    A whole-share entry carries its stop and target at the broker, so when one
    of them fills no code of ours runs -- this is how those exits are found.
    `after` filters on the parent's submission time, not the leg's fill time,
    so the caller passes a window wide enough to cover the longest hold.
    Raises on a broker failure; the caller decides what that means.

    Each item: leg_id, symbol, kind ("take_profit" | "stop_loss"), qty,
    entry_price, exit_price, filled_at (ISO).
    """
    _require_credentials()
    out: List[Dict[str, Any]] = []
    cursor = after_iso
    for _page in range(20):  # 20 x 500 orders is far beyond this account's volume
        resp = requests.get(
            f"{ALPACA_BASE_URL}/v2/orders",
            headers=_alpaca_headers(),
            params={"status": "closed", "nested": "true", "direction": "asc",
                    "limit": 500, "after": cursor},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        orders = resp.json() or []
        for parent in orders:
            if parent.get("order_class") != "bracket" or parent.get("side") != "buy":
                continue
            try:
                entry = float(parent.get("filled_avg_price") or 0.0)
            except (TypeError, ValueError):
                continue
            if entry <= 0:
                continue
            for leg in parent.get("legs") or []:
                try:
                    qty = float(leg.get("filled_qty") or 0.0)
                    exit_price = float(leg.get("filled_avg_price") or 0.0)
                except (TypeError, ValueError):
                    continue
                # Terminal only: a partially filled leg that is still working
                # would be booked now and its remainder never.
                if (
                    qty <= 0 or exit_price <= 0 or not leg.get("id") or not leg.get("filled_at")
                    or leg.get("status") not in TERMINAL_ORDER_STATUSES
                ):
                    continue
                out.append({
                    "leg_id": str(leg["id"]),
                    "symbol": symbol_config.from_alpaca_symbol(
                        str(parent.get("symbol", "")), str(parent.get("asset_class", ""))
                    ),
                    "kind": "take_profit" if leg.get("type") == "limit" else "stop_loss",
                    "qty": qty,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "filled_at": str(leg["filled_at"]),
                })
        if len(orders) < 500:
            break
        cursor = str(orders[-1].get("submitted_at") or "")
        if not cursor:
            break
    return out


def cancel_exit_orders(symbol: str) -> Tuple[bool, bool, str]:
    """Cancel `symbol`'s open exit legs and wait for Alpaca to confirm.

    Returns (ok, leg_filled, message). ok=False means a leg is still live and
    the caller must not sell. leg_filled=True means a leg filled before the
    cancel landed, so the broker already sold some or all of the position.
    """
    orders = _open_exit_orders(symbol)
    if not orders:
        return True, False, "no open exit orders"

    for order in orders:
        # 204 on success; 422 when the order is already filled/cancelled, which
        # the status poll below sorts out.
        requests.delete(
            f"{ALPACA_BASE_URL}/v2/orders/{order['id']}",
            headers=_alpaca_headers(),
            timeout=HTTP_TIMEOUT,
        )

    pending = {str(o["id"]) for o in orders}
    leg_filled = False
    for attempt in range(CANCEL_CONFIRM_ATTEMPTS):
        for order_id in sorted(pending):
            resp = requests.get(
                f"{ALPACA_BASE_URL}/v2/orders/{order_id}",
                headers=_alpaca_headers(),
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            status = str(data.get("status", ""))
            if status in TERMINAL_ORDER_STATUSES:
                pending.discard(order_id)
                if float(data.get("filled_qty") or 0.0) > 0:
                    leg_filled = True
        if not pending:
            return True, leg_filled, f"cancelled {len(orders)} exit order(s)"
        if attempt < CANCEL_CONFIRM_ATTEMPTS - 1:
            time.sleep(CANCEL_CONFIRM_INTERVAL_SECONDS)

    return False, leg_filled, (
        f"{len(pending)} exit order(s) still open after cancelling; sell not sent"
    )


# ------------------------------------------------------------------- guardrails


def _guard(
    signal: TradeSignal, existing_position: Optional[ExistingPosition]
) -> Optional[ExecutionResult]:
    """Duplicate-buy and naked-sell checks. Returns a skip result, or None to proceed."""
    held = existing_position.qty if existing_position else 0.0

    if signal.action == "buy" and held > 0:
        return ExecutionResult(
            status="skipped",
            message=f"position in {signal.symbol} already exists ({held:g} held); not adding",
        )

    if signal.action == "sell" and held <= 0:
        return ExecutionResult(
            status="skipped",
            message=f"nothing to sell in {signal.symbol}; spot-only, so not opening a short",
        )

    return None


def _stop_limit_price(stop_price: float, exit_side: str) -> float:
    """Limit price for a bracket's stop_loss leg, offset in the direction that works.

    The leg that closes a long is a SELL stop, so its limit must sit BELOW the stop
    (x0.99) or it can never fill. The mirror case -- a BUY stop closing a short --
    needs the limit ABOVE (x1.01). We only ever open longs, but hardcoding one
    multiplier for both sides is precisely how that ends up silently wrong.
    """
    if exit_side == "sell":
        return round(stop_price * (1.0 - STOP_LIMIT_BUFFER), 2)
    if exit_side == "buy":
        return round(stop_price * (1.0 + STOP_LIMIT_BUFFER), 2)
    raise ValueError(f"unknown exit side {exit_side!r}")


# --------------------------------------------------------------------- equities


def _buy_budget(equity: float, size_pct: float, cash_available: Optional[float]) -> float:
    """Dollars for a new long: the risk-sized share of equity, never more than
    the cash left. Equity counts open positions, so sizing off it alone let
    several buys in one cycle add up to more cash than the account holds."""
    budget = equity * (size_pct / 100.0)
    if cash_available is not None:
        budget = min(budget, max(float(cash_available), 0.0))
    return budget


def _execute_equity(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition],
    cash_available: Optional[float] = None,
) -> ExecutionResult:
    key, secret = _alpaca_credentials()

    if signal.action not in ("buy", "sell", "hold"):
        return ExecutionResult(
            status="skipped", message=f"{signal.symbol}: unsupported action {signal.action}. Only spot buys and sells are allowed."
        )

    if is_live and not (key and secret):
        return ExecutionResult(
            status="error", message="ALPACA_API_KEY / ALPACA_API_SECRET missing; cannot trade live"
        )

    # Market-hours gating: buys outside market hours skip cleanly.
    # Sells (closing positions) are allowed to queue as market-on-open.
    if signal.action == "buy" and not _is_market_open():
        return ExecutionResult(
            status="skipped",
            message=(
                f"{signal.symbol}: market is closed; buy orders are not placed outside "
                "NYSE/NASDAQ trading hours (weekdays 9:30-16:00 ET)"
            ),
        )

    blocked = _guard(signal, existing_position)
    if blocked:
        return blocked

    if signal.action == "sell":
        # Closing uses what is actually held. Recomputing a size here would try to
        # sell an amount unrelated to the position.
        assert existing_position is not None  # guaranteed by _guard
        qty = existing_position.qty
        if is_live:
            ok, leg_filled, cancel_note = cancel_exit_orders(signal.symbol)
            if not ok:
                return ExecutionResult(status="error", message=f"{signal.symbol}: {cancel_note}")
            if leg_filled:
                # A leg beat the cancel: the broker sold some or all of it already
                # (reconcile_bracket_exits books that P&L). Sell only what is left.
                remaining = _fetch_alpaca_position(signal.symbol)
                if remaining is None:
                    return ExecutionResult(
                        status="skipped",
                        message=f"{signal.symbol}: a bracket exit filled first; position already closed",
                    )
                qty = remaining.qty
        body: Dict[str, Any] = {
            "symbol": signal.symbol,
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "qty": _format_qty(qty),
        }
        # No bracket on a closing order: it is liquidating, not opening exposure.
        realized = (current_price - existing_position.avg_entry_price) * qty
        return _submit_alpaca(
            body,
            is_live=is_live,
            qty=qty,
            fill_price=current_price,
            realized_pnl_usd=realized,
            entry_price=existing_position.avg_entry_price,
            note=f"close {qty:g} {signal.symbol}",
        )

    # --- buy ---------------------------------------------------------------
    budget_usd = _buy_budget(live_equity, signal.position_size_pct, cash_available)
    whole_shares = math.floor(budget_usd / current_price) if current_price > 0 else 0

    if whole_shares >= 1:
        # Whole shares support bracket legs, so the stop lives at the broker and
        # survives the bot being offline.
        body = {
            "symbol": signal.symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "qty": str(whole_shares),
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(signal.take_profit_price or 0.0), 2)},
            "stop_loss": {
                "stop_price": round(float(signal.stop_loss_price or 0.0), 2),
                # Exit side is a sell, because we are opening a long.
                "limit_price": _stop_limit_price(float(signal.stop_loss_price or 0.0), "sell"),
            },
        }
        return _submit_alpaca(
            body,
            is_live=is_live,
            qty=float(whole_shares),
            fill_price=current_price,
            note=f"open {whole_shares} {signal.symbol} with bracket",
        )

    # Sub-one-share budget. Whole-share-only buying computes to 0 shares here and
    # silently skips every equity trade on a small account -- the exact bug this
    # build exists to not repeat. Use a notional order instead.
    if budget_usd < ALPACA_MIN_NOTIONAL_USD:
        return ExecutionResult(
            status="skipped",
            message=(
                f"{signal.symbol}: budget ${budget_usd:.2f} is below Alpaca's "
                f"${ALPACA_MIN_NOTIONAL_USD:.2f} minimum notional"
            ),
        )

    # Verified against Alpaca's current docs: fractional/notional orders support
    # market, limit, stop and stop_limit with time_in_force=day only, and cannot
    # carry bracket legs. So the entry goes in bare and the exit is bot-managed
    # via the ledger sweep -- better than skipping the trade outright.
    body = {
        "symbol": signal.symbol,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": f"{budget_usd:.2f}",
    }
    qty_est = budget_usd / current_price
    return _submit_alpaca(
        body,
        is_live=is_live,
        qty=qty_est,
        fill_price=current_price,
        note=(
            f"open ${budget_usd:.2f} notional of {signal.symbol} "
            f"(~{qty_est:.6f} sh); no bracket possible on a notional order {MANAGED_EXIT_MARKER}"
        ),
    )


# ----------------------------------------------------------------------- crypto


def _execute_crypto(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition],
    cash_available: Optional[float] = None,
) -> ExecutionResult:
    """Spot crypto on Alpaca. 24/7, so no market-hours gate; no bracket legs exist
    for crypto orders, so every entry is marked for a bot-managed exit.
    """
    key, secret = _alpaca_credentials()

    if signal.action not in ("buy", "sell"):
        return ExecutionResult(
            status="skipped",
            message=f"{signal.symbol}: unsupported action {signal.action}. Only spot buys and sells are allowed.",
        )

    if is_live and not (key and secret):
        return ExecutionResult(
            status="error", message="ALPACA_API_KEY / ALPACA_API_SECRET missing; cannot trade live"
        )

    blocked = _guard(signal, existing_position)
    if blocked:
        return blocked

    order_symbol = symbol_config.alpaca_order_symbol(signal.symbol)

    if signal.action == "sell":
        assert existing_position is not None  # guaranteed by _guard
        qty = existing_position.qty
        body: Dict[str, Any] = {
            "symbol": order_symbol,
            "side": "sell",
            "type": "market",
            "time_in_force": "gtc",
            "qty": _format_qty(qty),
        }
        realized = (current_price - existing_position.avg_entry_price) * qty
        return _submit_alpaca(
            body,
            is_live=is_live,
            qty=qty,
            fill_price=current_price,
            realized_pnl_usd=realized,
            entry_price=existing_position.avg_entry_price,
            note=f"close {qty:g} {signal.symbol}",
        )

    budget_usd = _buy_budget(live_equity, signal.position_size_pct, cash_available)
    if budget_usd < ALPACA_MIN_NOTIONAL_USD:
        return ExecutionResult(
            status="skipped",
            message=(
                f"{signal.symbol}: budget ${budget_usd:.2f} is below Alpaca's "
                f"${ALPACA_MIN_NOTIONAL_USD:.2f} minimum notional"
            ),
        )

    body = {
        "symbol": order_symbol,
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": f"{budget_usd:.2f}",
    }
    qty_est = budget_usd / current_price
    return _submit_alpaca(
        body,
        is_live=is_live,
        qty=qty_est,
        fill_price=current_price,
        note=(
            f"open ${budget_usd:.2f} notional of {signal.symbol} "
            f"(~{qty_est:.8f}); crypto takes no bracket legs {MANAGED_EXIT_MARKER}"
        ),
    )


def _format_qty(qty: float) -> str:
    """Alpaca accepts up to 9 decimals; trim trailing zeros so whole lots stay clean."""
    return f"{qty:.9f}".rstrip("0").rstrip(".")


def _submit_alpaca(
    body: Dict[str, Any],
    is_live: bool,
    qty: float,
    fill_price: float,
    note: str,
    realized_pnl_usd: Optional[float] = None,
    entry_price: Optional[float] = None,
) -> ExecutionResult:
    if not is_live:
        return ExecutionResult(
            status="dry_run",
            message=f"[sim] {note}",
            qty=qty,
            fill_price=fill_price,
            realized_pnl_usd=realized_pnl_usd,
            entry_price=entry_price,
        )

    resp = requests.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=_alpaca_headers(),
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code >= 400:
        return ExecutionResult(
            status="error", message=f"Alpaca rejected the order ({resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    filled_price = data.get("filled_avg_price")
    filled_qty = data.get("filled_qty")

    actual_price = float(filled_price) if filled_price else fill_price
    actual_qty = float(filled_qty) if filled_qty and float(filled_qty) > 0 else qty

    # Recompute realised P&L from what actually filled, not from the pre-trade
    # price. This number feeds the circuit breaker, so an estimate is not enough.
    if entry_price is not None:
        realized_pnl_usd = (actual_price - entry_price) * actual_qty

    return ExecutionResult(
        status="success",
        order_id=str(data.get("id")) if data.get("id") else None,
        fill_price=actual_price,
        qty=actual_qty,
        realized_pnl_usd=realized_pnl_usd,
        entry_price=entry_price,
        message=note,
    )


# ------------------------------------------------------------------ entry point


def execute_trade(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition] = None,
    cash_available: Optional[float] = None,
) -> ExecutionResult:
    """Route one signal to Alpaca, with `.message` guaranteed secret-free.

    The actual routing lives in `_execute_trade`; this is the one choke point
    every path through it returns through, so `.message` -- persisted verbatim
    into trading_bot.db (a file this project commits to a now-public repo) --
    is sanitized here exactly once, regardless of which internal function or
    except clause built it, and regardless of any new message-building call
    site added inside `_execute_trade` in the future. Scrubbing at every
    individual call site instead would silently stop covering the next one
    someone adds -- the same fragility a full-system audit flagged in the
    first place.
    """
    result = _execute_trade(
        signal, current_price, live_equity, is_live, existing_position, cash_available
    )
    if result.message:
        result = result.model_copy(update={"message": sanitize(result.message)})
    return result


def _execute_trade(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition] = None,
    cash_available: Optional[float] = None,
) -> ExecutionResult:
    """Route one signal to Alpaca.

    `cash_available` caps a buy's budget (see _buy_budget); None leaves the
    equity-based size alone.

    Wrapped end to end: a catastrophic failure on one symbol returns an error
    result for that symbol and never takes the rest of the cycle down with it.
    """
    try:
        if signal.action == "hold":
            return ExecutionResult(status="skipped", message="hold; nothing to execute")

        if symbol_config.is_crypto(signal.symbol):
            return _execute_crypto(
                signal, current_price, live_equity, is_live, existing_position, cash_available
            )
        return _execute_equity(
            signal, current_price, live_equity, is_live, existing_position, cash_available
        )
    except Exception as exc:  # noqa: BLE001 - one symbol must not kill the cycle
        return ExecutionResult(
            status="error", message=f"{signal.symbol}: execution failed: {type(exc).__name__}: {exc}"
        )
