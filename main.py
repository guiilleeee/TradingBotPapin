"""One trading cycle, end to end.

Run order matters and is deliberate:

  1. resolve the mode (live vs simulation) -- structurally, not by config accident
  2. sweep open bot-managed positions for stop/take crossings, booking realised P&L
  3. compute equity from what is left open, so a swept position is never counted
     as both realised and unrealised
  4. snapshot equity + QQQ for the benchmark
  5. choose symbols: explicit --trigger-symbols, or the funnel over the universe
     (every held symbol plus the top-N ranked candidates -- see funnel.py)
  6. per symbol: circuit breaker, data, model, risk manager, [approval], execution, log
  7. export the dashboard CSV, positions.json, benchmark.json
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import approval
import benchmark
import data_fetcher
import execution
import funnel
import notifications
import position_metrics
import risk_manager
import symbol_config
from logger import BotLogger
from mode import ModeSettings, resolve_is_live, resolve_mode_settings
from models import ExecutionResult, ExistingPosition, SignalInput, TradeSignal

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_SYMBOLS_PATH = "symbols.yaml"


# --------------------------------------------------------------------- config


def default_symbols_path(config_path: str) -> str:
    """Where symbols.yaml lives for a given config.yaml, absent an explicit override.

    Scoped to `config_path`'s own directory rather than a bare "symbols.yaml"
    resolved against the current working directory -- a fixed cwd-relative
    default meant every test using an isolated tmp_path config was actually
    picking up the real, committed repo-root symbols.yaml in CI (whatever cwd
    happened to be pytest's), silently overriding each test's own symbol list.
    """
    directory = os.path.dirname(config_path)
    return os.path.join(directory, DEFAULT_SYMBOLS_PATH) if directory else DEFAULT_SYMBOLS_PATH


def load_config(
    path: str = DEFAULT_CONFIG_PATH, symbols_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load config.yaml, then let symbols.yaml (if present) override its symbol list.

    This is a one-way, single-key merge on purpose: symbols_path can only ever
    replace the `symbols` key, never anything else. The weekly screening job that
    writes symbols.yaml has no way to touch risk parameters, thresholds, or
    provider settings even if its own logic were somehow wrong -- that guarantee
    holds structurally here, not by convention in screening.py.

    symbols.yaml only ever carries the screened *equities*. The fixed crypto
    entries (asset_class: crypto) in config.yaml are kept alongside whatever it
    supplies -- the weekly screen rotates stocks, never the crypto set.

    A missing or empty symbols.yaml is not an error: config.yaml's own `symbols`
    (hand-tuned, checked into the repo) stands in for it, so the 4h cycle never
    ends up with nothing to trade because the weekly job hasn't run yet, or broke.

    `symbols_path` defaults to `default_symbols_path(path)` -- i.e. scoped next
    to whichever config.yaml is actually in use -- rather than a fixed cwd-relative
    path, so an isolated (e.g. tmp_path-based) config can never pick up the real
    project's committed symbols.yaml just because pytest's cwd happens to be the
    repo root.
    """
    import yaml  # imported here so importing this module needs no config deps

    if symbols_path is None:
        symbols_path = default_symbols_path(path)

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if os.path.exists(symbols_path):
        with open(symbols_path, "r", encoding="utf-8") as handle:
            screened = yaml.safe_load(handle) or {}
        screened_symbols = screened.get("symbols")
        if screened_symbols:
            fixed_crypto = [
                entry for entry in config.get("symbols", []) or []
                if isinstance(entry, dict)
                and str(entry.get("asset_class", "")).lower() == symbol_config.CRYPTO
            ]
            screened_names = {
                (e["symbol"] if isinstance(e, dict) else str(e)).upper() for e in screened_symbols
            }
            config["symbols"] = list(screened_symbols) + [
                e for e in fixed_crypto if e["symbol"].upper() not in screened_names
            ]

    return config


def infer_asset_class(symbol: str, config: Dict[str, Any]) -> str:
    """Asset class for a symbol: "crypto" for the fixed "-USD" pairs, else "equity"."""
    return symbol_config.asset_class(symbol, config)


def get_provider(config: Dict[str, Any]) -> Tuple[str, Any]:
    """Return (name, generate_signal) for the configured provider."""
    name = str(config.get("signal_provider", "claude")).strip().lower()
    if name == "gemini":
        import signal_generator_gemini

        return name, signal_generator_gemini.generate_signal
    if name == "claude":
        import signal_generator

        return name, signal_generator.generate_signal
    raise ValueError(f"unknown signal_provider {name!r}; expected 'claude' or 'gemini'")


# ---------------------------------------------------------------------- sweep


@dataclass
class Closure:
    symbol: str
    reason: str
    price: float
    qty: float
    pnl: float
    entry_price: float


@dataclass
class SweepResult:
    closed_symbols: Set[str] = field(default_factory=set)
    closures: List[Closure] = field(default_factory=list)
    unrealized_pnl: float = 0.0


@dataclass
class CircuitBreakerTracker:
    """Tracks whether the circuit breaker alert has already been sent.

    "Tripping" is the transition from not-tripped to tripped, not the tripped
    state itself. `already_tripped_at_cycle_start` is computed once, before any
    symbol is processed, from the same persisted realised-P&L data every other
    circuit-breaker check already reads -- so a breaker still tripped from an
    earlier cycle *today* (UTC) correctly sends no new alert, while a breaker
    that trips for the first time mid-cycle sends exactly one.
    """

    already_tripped_at_cycle_start: bool
    alerted_this_cycle: bool = False

    def note_tripped(self, is_live: bool, today_loss_pct: float, threshold_pct: float) -> None:
        if self.already_tripped_at_cycle_start or self.alerted_this_cycle:
            return
        notifications.send_circuit_breaker_alert(is_live, today_loss_pct, threshold_pct)
        self.alerted_this_cycle = True


def sweep_open_positions(
    bot_logger: BotLogger,
    config: Dict[str, Any],
    is_live: bool,
    equity_hint: float,
) -> SweepResult:
    """Close any bot-managed position whose price has crossed its stop or target.

    One pass, before any new symbol is considered. It does two jobs at once on
    purpose: a position it closes books its P&L as realised via `record_pnl`, and
    only the positions it leaves open contribute to `unrealized_pnl`. Splitting
    these into two passes is how the same dollar ends up counted twice.

    Synthetic log rows are not written here -- the caller writes them once equity
    is known, so every row carries a real "Patrimoni total" value.
    """
    result = SweepResult()

    for row in bot_logger.get_all_simulated_positions():
        symbol = str(row["symbol"])
        qty = float(row["qty"])
        entry = float(row["avg_entry_price"])
        stop = row["stop_loss_price"]
        take = row["take_profit_price"]

        if qty <= 0:
            bot_logger.close_simulated_position(symbol)
            continue

        try:
            price = data_fetcher.latest_price(data_fetcher.fetch_ohlcv(symbol))
        except Exception as exc:  # noqa: BLE001
            print(f"  [sweep] {symbol}: price unavailable ({exc}); leaving position open")
            result.unrealized_pnl += 0.0
            continue

        # Long-only, so a stop is crossed from above and a target from below.
        # If a gap crosses both in one bar, assume the worse of the two.
        hit_stop = stop is not None and price <= float(stop)
        hit_take = take is not None and price >= float(take)

        if not (hit_stop or hit_take):
            result.unrealized_pnl += (price - entry) * qty
            continue

        if hit_stop:
            reason = (
                f"Stop-loss activat: el preu {price:.6g} ha creuat el nivell "
                f"{float(stop):.6g}. Posicio tancada automaticament."
            )
        else:
            reason = (
                f"Take-profit assolit: el preu {price:.6g} ha superat l'objectiu "
                f"{float(take):.6g}. Posicio tancada automaticament."
            )

        fill = price
        if is_live:
            # Live managed exits place a real closing order. If it does not fill,
            # the position stays on the books and stays unrealised.
            exit_signal = TradeSignal(
                symbol=symbol,
                action="sell",
                confidence=1.0,
                position_size_pct=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reasoning=reason,
                override_reason="automatic exit",
                raw_action="sell",
            )
            exec_result = execution.execute_trade(
                signal=exit_signal,
                current_price=price,
                live_equity=equity_hint,
                is_live=True,
                existing_position=ExistingPosition(qty=qty, avg_entry_price=entry),
            )
            if exec_result.status != "success":
                print(f"  [sweep] {symbol}: managed exit did not fill ({exec_result.message})")
                result.unrealized_pnl += (price - entry) * qty
                continue
            fill = float(exec_result.fill_price or price)
            qty = float(exec_result.qty or qty)

        pnl = (fill - entry) * qty
        bot_logger.record_pnl(symbol, pnl)
        bot_logger.close_simulated_position(symbol)
        result.closed_symbols.add(symbol)
        result.closures.append(
            Closure(symbol=symbol, reason=reason, price=fill, qty=qty, pnl=pnl, entry_price=entry)
        )
        print(f"  [sweep] {symbol}: closed at {fill:.6g}, P&L {pnl:+.2f} USD")

    return result


# ----------------------------------------------------------------------- cycle


def run_cycle(
    config_path: str = DEFAULT_CONFIG_PATH,
    symbols_path: Optional[str] = None,
    trigger_symbols: Optional[List[str]] = None,
    trigger_reason: str = "scheduled",
) -> int:
    # Safe default for the failure alert's mode label if resolution itself
    # fails before is_live is known -- mirrors mode.py's own philosophy
    # ("failing safe is the point"), applied here to a notification rather
    # than a trading decision. Updated below the instant the real value is
    # known, so a failure any time after that point alerts with the true mode.
    is_live = False
    try:
        config = load_config(config_path, symbols_path)
        is_live = resolve_is_live(config)
        return _run_cycle_body(config, is_live, trigger_symbols, trigger_reason)
    except Exception as exc:
        # A cycle-level failure -- distinct from a single symbol's
        # _process_symbol call raising, which is already caught in the
        # per-symbol loop below and never reaches this level.
        notifications.send_cycle_failure_alert(is_live, f"{type(exc).__name__}: {exc}")
        raise


def _run_cycle_body(
    config: Dict[str, Any],
    is_live: bool,
    trigger_symbols: Optional[List[str]] = None,
    trigger_reason: str = "scheduled",
) -> int:
    settings: ModeSettings = resolve_mode_settings(is_live, config)

    bot_logger = BotLogger(config.get("db_path", "trading_bot.db"))
    provider_name, generate_signal = get_provider(config)

    fallback_equity = float(config.get("fallback_equity_usd", 1000.0))
    circuit_breaker_loss_pct = float(config.get("circuit_breaker_loss_pct", 3.0))
    max_risk_pct = float(config.get("max_risk_pct", 1.0))
    max_absolute_position_pct = float(config.get("max_absolute_position_pct", 20.0))
    min_reward_risk_ratio = float(
        config.get("min_reward_risk_ratio", risk_manager.DEFAULT_MIN_REWARD_RISK_RATIO)
    )

    print(f"=== TradingBot cycle | mode={settings.label} | provider={provider_name}"
          f" | trigger={trigger_reason} ===")
    print(f"min_confidence={settings.min_confidence:.2f}  max_risk={max_risk_pct:.2f}%  "
          f"cap={max_absolute_position_pct:.2f}%  breaker=-{circuit_breaker_loss_pct:.2f}%  "
          f"min_reward_risk={min_reward_risk_ratio:.2f}")

    # --- equity + sweep ---------------------------------------------------
    if is_live:
        equity = execution.fetch_live_equity(fallback_equity)
        sweep = sweep_open_positions(bot_logger, config, is_live, equity)
        if sweep.closures:
            equity = execution.fetch_live_equity(fallback_equity)
    else:
        sweep = sweep_open_positions(bot_logger, config, is_live, fallback_equity)
        # Realised P&L already includes anything the sweep just booked, and only
        # still-open positions contributed unrealised. No dollar is counted twice.
        equity = (
            fallback_equity
            + bot_logger.get_all_time_realized_pnl()
            + sweep.unrealized_pnl
        )

    equity = max(equity, 0.01)  # keep downstream gt=0 validators satisfiable

    for closure in sweep.closures:
        bot_logger.log_auto_close_signal(
            symbol=closure.symbol,
            reason=closure.reason,
            price=closure.price,
            qty=closure.qty,
            pnl=closure.pnl,
            equity=equity,
            is_live=is_live,
            entry_price=closure.entry_price,
        )
        notifications.send_auto_close_alert(is_live, closure.symbol, closure.reason, closure.pnl)

    print(f"Equity: ${equity:,.2f}  (auto-closed this cycle: "
          f"{sorted(sweep.closed_symbols) or 'none'})")

    # Same instant as the equity figure above, before any trade this cycle moves it.
    try:
        benchmark.record_snapshot(bot_logger, equity, is_live)
    except Exception as exc:  # noqa: BLE001 - dashboard data never fails a cycle
        print(f"benchmark snapshot failed (non-fatal): {type(exc).__name__}: {exc}")

    # The breaker's state as of *before* any symbol in this cycle is processed.
    # Passed into _process_symbol via the tracker so the alert fires exactly
    # once, on the transition into the tripped state -- never for a breaker
    # already tripped entering this cycle, never once per symbol afterward.
    breaker_tracker = CircuitBreakerTracker(
        already_tripped_at_cycle_start=(
            bot_logger.get_today_realized_loss_pct(equity) <= -abs(circuit_breaker_loss_pct)
        )
    )

    # --- per symbol -------------------------------------------------------
    configured_symbols = config.get("symbols", []) or []

    # When --trigger-symbols is set (volume-wake), restrict to just those symbols.
    # They still go through the exact same path: sweep, circuit breaker, data,
    # model, risk_manager.validate, execute_trade. No shortcuts.
    if trigger_symbols:
        trigger_set = {s.upper() for s in trigger_symbols}
        matched = [
            entry for entry in configured_symbols
            if (entry["symbol"] if isinstance(entry, dict) else str(entry)).upper() in trigger_set
        ]
        if trigger_reason in ("manual", "wake_sell"):
            # A manual, dashboard-triggered analysis (the Cercador tab's "Executar
            # analisi ara") is allowed to target any symbol the user searched for,
            # not just one already in the configured watchlist. wake_sell only
            # ever fires for a symbol already held, which may have rotated out of
            # the universe since it was bought -- it must still be reviewable.
            # wake_buy stays scoped to the watchlist. Synthesize entries for the rest.
            matched_upper = {
                (entry["symbol"] if isinstance(entry, dict) else str(entry)).upper()
                for entry in matched
            }
            for sym in sorted(trigger_set - matched_upper):
                matched.append({"symbol": sym, "asset_class": symbol_config.asset_class(sym, config)})
        configured_symbols = matched
        if not configured_symbols:
            print(f"  No configured symbols match --trigger-symbols {trigger_symbols}")
    elif funnel.funnel_enabled(config):
        configured_symbols = _funnel_symbols(config, configured_symbols, is_live, bot_logger)

    for entry in configured_symbols:
        symbol = entry["symbol"] if isinstance(entry, dict) else str(entry)

        if symbol in sweep.closed_symbols:
            print(f"- {symbol}: auto-closed this cycle, not re-evaluating")
            continue

        try:
            _process_symbol(
                symbol=symbol,
                config=config,
                settings=settings,
                bot_logger=bot_logger,
                generate_signal=generate_signal,
                equity=equity,
                circuit_breaker_loss_pct=circuit_breaker_loss_pct,
                max_risk_pct=max_risk_pct,
                max_absolute_position_pct=max_absolute_position_pct,
                min_reward_risk_ratio=min_reward_risk_ratio,
                breaker_tracker=breaker_tracker,
                trigger_reason=trigger_reason,
            )
        except Exception as exc:  # noqa: BLE001 - one symbol never kills the cycle
            print(f"- {symbol}: FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)

    # --- export -----------------------------------------------------------
    csv_path = config.get("csv_path", "signals.csv")
    rows = bot_logger.export_signals_csv(csv_path)
    print(f"Exported {rows} signal rows to {csv_path}")

    # Holdings-list data for the dashboard: open positions only, never the full
    # screening universe. Soft-fail like every other dashboard-facing export here --
    # a broken return calculation or a broker hiccup must not fail the cycle itself.
    try:
        positions_path = config.get("positions_path", position_metrics.DEFAULT_POSITIONS_PATH)
        position_rows = position_metrics.compute_position_metrics(
            bot_logger, config, is_live, infer_asset_class
        )
        position_metrics.export_positions_json(position_rows, positions_path, is_live=is_live)
        print(f"Exported {len(position_rows)} open positions to {positions_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"positions.json export failed (non-fatal): {type(exc).__name__}: {exc}")

    try:
        benchmark_path = config.get("benchmark_path", benchmark.DEFAULT_BENCHMARK_PATH)
        points = benchmark.export_benchmark_json(bot_logger, benchmark_path)
        print(f"Exported {points} benchmark points to {benchmark_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark.json export failed (non-fatal): {type(exc).__name__}: {exc}")

    return 0


def _held_symbols(is_live: bool, bot_logger: BotLogger) -> List[str]:
    """Every symbol with an open position, universe member or not.

    Live asks the broker for the full list (so a position in a symbol the weekly
    screen has since rotated out is still reviewed) and adds the bot-managed
    ledger. If the broker can't be reached, the ledger alone stands in -- this
    cycle may then skip reviewing a bracket-protected equity, which the broker's
    own stop still covers.
    """
    held = {str(r["symbol"]) for r in bot_logger.get_all_simulated_positions() if float(r["qty"]) > 0}
    if is_live:
        try:
            held |= set(execution.fetch_all_live_positions().keys())
        except Exception as exc:  # noqa: BLE001
            print(f"  [funnel] live position list unavailable ({type(exc).__name__}: {exc}); "
                  "using the bot-managed ledger only")
    return sorted(held)


def _funnel_symbols(
    config: Dict[str, Any],
    configured_symbols: List[Any],
    is_live: bool,
    bot_logger: BotLogger,
) -> List[Dict[str, Any]]:
    """Rank the universe locally (no LLM) and keep held + top-N. See funnel.py."""
    universe = [e["symbol"] if isinstance(e, dict) else str(e) for e in configured_symbols]
    entries = {s.upper(): e for s, e in zip(universe, configured_symbols)}
    top_n = funnel.top_n_from_config(config)
    held = _held_symbols(is_live, bot_logger)

    # A new equity entry while NYSE is closed would be skipped by execution
    # anyway, so don't pay the model to propose one; crypto trades 24/7.
    equities_open = execution._is_market_open()

    def eligible(symbol: str) -> bool:
        return equities_open or symbol_config.is_crypto(symbol, config)

    data = funnel.fetch_funnel_data(universe)
    result = funnel.select(universe, held, data, top_n=top_n, eligible=eligible)

    print(f"Funnel: {len(universe)} ranked locally, {len(data)} with data; "
          f"equity market {'open' if equities_open else 'closed (crypto-only candidates)'}")
    for row in result.scored[: top_n + 4]:
        if row["has_data"]:
            print(f"  {row['symbol']:<9} score {row['score']:.3f}  vol x{row['volume_ratio']:.2f}  "
                  f"move {row['last_return_pct']:+.2f}% ({row['move_z']:.1f} sigma)")
    print(f"  held (always reviewed): {result.held or 'none'}")
    print(f"  top {top_n} candidates:   {result.candidates or 'none'}")

    selected = []
    for symbol in result.selected:
        entry = entries.get(symbol.upper())
        if entry is None:
            entry = {"symbol": symbol, "asset_class": symbol_config.asset_class(symbol, config)}
        selected.append(entry)
    return selected


def _process_symbol(
    symbol: str,
    config: Dict[str, Any],
    settings: ModeSettings,
    bot_logger: BotLogger,
    generate_signal: Any,
    equity: float,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
    min_reward_risk_ratio: float,
    breaker_tracker: CircuitBreakerTracker,
    trigger_reason: str = "scheduled",
) -> None:
    is_live = settings.is_live
    max_absolute_position_pct = symbol_config.max_position_pct(
        config, symbol, max_absolute_position_pct
    )

    df = data_fetcher.fetch_ohlcv(symbol)
    indicators = data_fetcher.compute_indicators(df)
    current_price = data_fetcher.latest_price(df)

    existing_position: Optional[ExistingPosition] = execution.fetch_existing_position(
        symbol=symbol, is_live=is_live, bot_logger=bot_logger
    )

    signal_input = SignalInput(
        symbol=symbol,
        asset_class=symbol_config.asset_class(symbol, config),
        current_price=current_price,
        account_equity_usd=equity,
        existing_position=existing_position,
        technical_indicators=indicators,
        recent_headlines=data_fetcher.fetch_headlines(symbol),
    )

    # Recomputed per symbol so a loss taken earlier in this cycle can still trip
    # the breaker for the symbols that follow.
    today_loss_pct = bot_logger.get_today_realized_loss_pct(equity)

    if today_loss_pct <= -abs(circuit_breaker_loss_pct):
        # Breaker is already tripped, so skip the model call entirely -- there is
        # no decision it could return that we would act on, and it costs money.
        # The alert itself only actually sends once -- see CircuitBreakerTracker.
        breaker_tracker.note_tripped(is_live, today_loss_pct, circuit_breaker_loss_pct)
        blocked = TradeSignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            position_size_pct=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reasoning=(
                f"Circuit breaker actiu: la perdua realitzada d'avui ({today_loss_pct:.2f}%) "
                f"supera el limit de -{abs(circuit_breaker_loss_pct):.2f}%. "
                "No s'ha consultat el model."
            ),
            override_reason=(
                f"circuit breaker tripped at {today_loss_pct:.2f}%; model call skipped"
            ),
            raw_action="hold",
        )
        bot_logger.log_signal(symbol, signal_input, None, blocked, None, is_live=is_live, trigger_reason=trigger_reason)
        print(f"- {symbol}: circuit breaker tripped ({today_loss_pct:.2f}%); skipped")
        return

    raw = generate_signal(signal_input, system_prompt=settings.system_prompt)

    final = risk_manager.validate(
        raw=raw,
        current_price=current_price,
        today_realized_loss_pct=today_loss_pct,
        circuit_breaker_loss_pct=circuit_breaker_loss_pct,
        max_risk_pct=max_risk_pct,
        max_absolute_position_pct=max_absolute_position_pct,
        min_confidence=settings.min_confidence,
        min_reward_risk_ratio=min_reward_risk_ratio,
    )

    exec_result = None
    if final.action != "hold" and is_live and approval.approval_enabled(config):
        final, current_price, exec_result = _approval_gate(
            symbol=symbol,
            raw=raw,
            final=final,
            current_price=current_price,
            config=config,
            equity=equity,
            is_live=is_live,
            today_loss_pct=today_loss_pct,
            circuit_breaker_loss_pct=circuit_breaker_loss_pct,
            max_risk_pct=max_risk_pct,
            max_absolute_position_pct=max_absolute_position_pct,
            min_confidence=settings.min_confidence,
            min_reward_risk_ratio=min_reward_risk_ratio,
        )
        if exec_result is None:
            # Re-read the holding too, not just the price: another cycle (a
            # wake-up running alongside this one) may have bought or sold this
            # symbol during the wait, and the duplicate-buy / naked-sell guards
            # must judge the position as it is now. A failed lookup is a skip.
            try:
                existing_position = execution.fetch_existing_position(
                    symbol=symbol, is_live=is_live, bot_logger=bot_logger
                )
            except Exception as exc:  # noqa: BLE001
                exec_result = ExecutionResult(
                    status="skipped",
                    message=f"approved, but the position re-check failed ({type(exc).__name__}); "
                            "order not submitted",
                )

    if final.action != "hold" and exec_result is None:
        exec_result = execution.execute_trade(
            signal=final,
            current_price=current_price,
            live_equity=equity,
            is_live=is_live,
            existing_position=existing_position,
        )

    bot_logger.log_signal(symbol, signal_input, raw, final, exec_result, is_live=is_live, trigger_reason=trigger_reason)

    if exec_result and exec_result.status in ("success", "dry_run"):
        if exec_result.realized_pnl_usd is not None:
            # Without this call the circuit breaker never sees a loss and is inert.
            bot_logger.record_pnl(symbol, float(exec_result.realized_pnl_usd))

        _update_ledger(bot_logger, final, exec_result, current_price, is_live)

        notifications.send_trade_alert(
            is_live=is_live,
            symbol=symbol,
            action=final.action,
            size_pct=final.position_size_pct,
            price=float(exec_result.fill_price or current_price),
            confidence=final.confidence,
            reasoning=final.reasoning,
        )

    print(
        f"- {symbol}: {final.raw_action} -> {final.action} "
        f"(conf {final.confidence:.2f}, size {final.position_size_pct:.2f}%)"
        + (f" | override: {final.override_reason}" if final.override_reason else "")
        + (f" | exec: {exec_result.status} - {exec_result.message}" if exec_result else "")
    )


def _approval_gate(
    *,
    symbol: str,
    raw: Any,
    final: TradeSignal,
    current_price: float,
    config: Dict[str, Any],
    equity: float,
    is_live: bool,
    today_loss_pct: float,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
    min_confidence: float,
    min_reward_risk_ratio: float,
) -> Tuple[TradeSignal, float, Optional[ExecutionResult]]:
    """Hold a live order until a human approves it (approval.py).

    Returns (final, current_price, exec_result). A non-None exec_result means
    "do not execute; log this skip instead". On approval the price is re-fetched
    and the model's raw signal re-validated against it: up to ten minutes can
    pass, and a stop the price has already crossed must not go out as a bracket.
    """
    timeout = approval.approval_timeout_seconds(config)
    print(f"- {symbol}: {final.action} awaiting approval (up to {int(timeout)}s)...")
    decision = approval.request_approval(
        symbol=symbol,
        action=final.action,
        size_pct=final.position_size_pct,
        price=current_price,
        stop_loss=final.stop_loss_price,
        take_profit=final.take_profit_price,
        confidence=final.confidence,
        reasoning=final.reasoning,
        equity=equity,
        timeout_seconds=timeout,
    )
    if not decision.approved:
        notifications.send_approval_outcome_alert(is_live, symbol, final.action, decision.outcome)
        return final, current_price, ExecutionResult(
            status="skipped",
            message=f"approval {decision.outcome}: {decision.detail}; order not submitted",
        )

    try:
        fresh_price = data_fetcher.latest_price(data_fetcher.fetch_ohlcv(symbol))
    except Exception as exc:  # noqa: BLE001 - no fresh price, no order
        return final, current_price, ExecutionResult(
            status="skipped",
            message=f"approved, but the price re-check failed ({type(exc).__name__}); order not submitted",
        )

    revalidated = risk_manager.validate(
        raw=raw,
        current_price=fresh_price,
        today_realized_loss_pct=today_loss_pct,
        circuit_breaker_loss_pct=circuit_breaker_loss_pct,
        max_risk_pct=max_risk_pct,
        max_absolute_position_pct=max_absolute_position_pct,
        min_confidence=min_confidence,
        min_reward_risk_ratio=min_reward_risk_ratio,
    )
    if revalidated.action != final.action:
        return revalidated, fresh_price, ExecutionResult(
            status="skipped",
            message=(
                f"approved, but no longer valid at the re-checked price {fresh_price:.6g}: "
                f"{revalidated.override_reason}; order not submitted"
            ),
        )
    return revalidated, fresh_price, None


def _update_ledger(
    bot_logger: BotLogger,
    final: TradeSignal,
    exec_result: Any,
    current_price: float,
    is_live: bool,
) -> None:
    """Keep the bot-managed ledger in step with what just happened.

    In simulation it tracks every position. In live it tracks only fills with no
    broker-side bracket, which are the ones the sweep has to exit.
    """
    if final.action == "sell":
        bot_logger.close_simulated_position(final.symbol)
        return

    if final.action != "buy":
        return

    if is_live and not execution.needs_managed_exit(exec_result):
        # A live bracket order already carries its own stop and target at the
        # broker; tracking it here would double up the exit.
        return

    bot_logger.open_simulated_position(
        symbol=final.symbol,
        qty=float(exec_result.qty or 0.0),
        avg_entry_price=float(exec_result.fill_price or current_price),
        stop_loss_price=final.stop_loss_price,
        take_profit_price=final.take_profit_price,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TradingBot cycle.")
    parser.add_argument("--config", default=os.environ.get("BOT_CONFIG", DEFAULT_CONFIG_PATH))
    parser.add_argument("--symbols", default=os.environ.get("BOT_SYMBOLS"))
    parser.add_argument(
        "--trigger-symbols",
        default=None,
        help="Comma-separated list of symbols to restrict this cycle to (e.g. AAPL,MSFT). "
             "Used by volume_watch.py to trigger a cycle for specific symbols only.",
    )
    parser.add_argument(
        "--trigger-reason",
        default="scheduled",
        help="Why this cycle was triggered: 'scheduled' (normal 8h cron), 'wake_buy' "
             "(off-schedule, buy-side trigger), 'wake_sell' (off-schedule, sell-side "
             "trigger), or 'manual' (dashboard-triggered one-off analysis).",
    )
    args = parser.parse_args()

    trigger_symbols = None
    if args.trigger_symbols:
        trigger_symbols = [s.strip() for s in args.trigger_symbols.split(",") if s.strip()]

    return run_cycle(
        args.config,
        args.symbols,
        trigger_symbols=trigger_symbols,
        trigger_reason=args.trigger_reason,
    )


if __name__ == "__main__":
    raise SystemExit(main())
