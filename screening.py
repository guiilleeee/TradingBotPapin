"""Weekly symbol screening: the top-25 Nasdaq-100 equities by market cap.

This job never trades and never touches the model. It only decides what the
trading cycle gets to *analyse* -- the AI's own judgment (system prompt rules 4/5:
no volume confirmation, conflicting signals -> hold) and risk_manager.py remain
the only things that can turn a candidate into an actual order. Nothing written
here can widen or bypass either.

Equities only. The universe is the 25 largest non-financial Nasdaq-100
companies by market cap (equity_universe.build_equity_universe), written in
market-cap order. The five crypto pairs are fixed in config.yaml and are never
rotated here -- main.load_config keeps them alongside this file's equities.

The per-cycle choice of *which* of these the model actually analyses is not made
here; funnel.py ranks the whole universe locally at every cycle. The weekly
volume/momentum scores below are informational (logged into symbols.yaml).

Blast radius: this entire module can fail in any way and the trading cycle is
unaffected -- `main.load_config` falls back to whatever `symbols.yaml` (or
`config.yaml`) already has. `run_screening` enforces that on the writing side:
it only ever replaces `symbols.yaml` after producing a complete, valid result,
and never partially or emptily overwrites it.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

import equity_universe
import notifications
from secrets_redaction import sanitize

DEFAULT_OUTPUT_PATH = "symbols.yaml"
DEFAULT_CONFIG_PATH_FOR_ALERTS = "config.yaml"

# All 25 are kept; funnel.py picks per cycle.
EQUITY_COUNT = equity_universe.TARGET_UNIVERSE_SIZE


# ------------------------------------------------------------------ output


def _entries(symbols: List[str], asset_class: str) -> List[Dict[str, str]]:
    return [{"symbol": s, "asset_class": asset_class} for s in symbols]


def _current_is_live_for_alerts(config_path: str = DEFAULT_CONFIG_PATH_FOR_ALERTS) -> bool:
    """Best-effort is_live, read only to label this run's alerts.

    Screening itself has no live/sim behavior of its own -- it does the same
    universe-building and scoring regardless of live_execution. This exists
    purely so its alerts carry the same unmistakable mode label every other
    alert in this project carries. Defaults to False (simulation) on any
    failure to read or parse config.yaml.
    """
    try:
        import yaml

        import mode

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return mode.resolve_is_live(config)
    except Exception:
        return False


def run_screening(
    output_path: str = DEFAULT_OUTPUT_PATH,
    config_path: str = DEFAULT_CONFIG_PATH_FOR_ALERTS,
) -> int:
    """Build the equity-only symbol slate and write it, or leave the existing file alone.

    Returns 0 on a complete, valid write; 1 on anything short of that. A
    non-zero return must never come with a partial or empty write -- the whole
    point of writing to a temp path first is that a crash midway through
    leaves last week's symbols.yaml exactly as it was.
    """
    is_live = _current_is_live_for_alerts(config_path)

    def fail(reason: str) -> int:
        print(f"FAILED: {reason} Leaving {output_path} untouched.")
        notifications.send_screening_failure_alert(is_live, reason)
        return 1

    try:
        print(f"=== Weekly symbol screening (Nasdaq-100 top {EQUITY_COUNT}) ===")

        print("Building equity universe (Nasdaq-100, by market cap)...")
        equity_pool = equity_universe.build_equity_universe()
        print(f"  universe: {len(equity_pool)} symbols")
        if len(equity_pool) < EQUITY_COUNT:
            return fail(
                f"equity universe has only {len(equity_pool)} symbols, need at "
                f"least {EQUITY_COUNT}."
            )

        print("  fetching universe-wide volume/momentum (one batched yfinance call)...")
        price_data = equity_universe.fetch_universe_price_data(sorted(equity_pool))
        print(f"  usable price data for {len(price_data)}/{len(equity_pool)} universe symbols")
        equity_scored = equity_universe.score_equities(set(equity_pool), price_data)
        # Market-cap order, not score order: every one of these is kept, and the
        # order is what the funnel falls back to on a score tie.
        equity_symbols = list(equity_pool)[:EQUITY_COUNT]
        signal_count = sum(1 for r in equity_scored if r["has_signal"])
        print(f"  {signal_count} symbols carried real volume/momentum signal this week")
        print(f"  selected: {equity_symbols}")

    except Exception as exc:  # noqa: BLE001 - any failure here must not touch the file
        summary = sanitize(f"{type(exc).__name__}: {exc}")
        print(f"FAILED: screening raised {summary}")
        print(sanitize(traceback.format_exc()))
        print(f"Leaving {output_path} untouched -- the trading cycle keeps last week's list.")
        notifications.send_screening_failure_alert(is_live, summary)
        return 1

    _write_symbols_file(output_path, equity_symbols, equity_scored)
    print(f"Wrote {output_path}: {len(equity_symbols)} equity symbols")
    notifications.send_screening_complete_alert(is_live, equity_symbols)
    return 0


def _write_symbols_file(
    output_path: str,
    equity_symbols: List[str],
    equity_scored: List[Dict[str, Any]],
) -> None:
    """Atomic write: a temp file plus a rename, so a crash mid-write can never
    leave symbols.yaml half-written or truncated.
    """
    import os
    import yaml

    equity_by_symbol = {r["symbol"]: r for r in equity_scored}

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": _entries(equity_symbols, "equity"),
        # Informational only -- main.py's loader reads nothing but "symbols"
        # above, so nothing here can ever affect what the trading cycle trades.
        "scores": {
            "equity": {
                s: {"score": round(equity_by_symbol[s]["score"], 4)}
                for s in equity_symbols
                if s in equity_by_symbol
            },
        },
    }

    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Generated weekly by screening.py -- do not hand-edit, it is "
            "overwritten every run.\n"
            "# Contributes ONLY the `symbols` list to the trading cycle; every risk "
            "parameter, threshold, and provider setting stays in config.yaml.\n"
        )
        yaml.safe_dump(document, handle, sort_keys=False)
    os.replace(tmp_path, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the weekly symbol screen.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH_FOR_ALERTS)
    args = parser.parse_args()
    return run_screening(args.output, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
