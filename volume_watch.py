"""Volume/price watcher: cheap, no-model polling between the 8h scheduled cycles.

Runs every 15 minutes (volume_watch.yml, same cron as refresh_positions.yml).
Bidirectional Wake-up logic:
- Check current positions from positions.json.
- If we hold a symbol, check for sell-side triggers (price drops).
- If we don't hold a symbol, check for buy-side triggers (price spikes / volume multiples).
- Market-hours aware: aborts cleanly outside market hours.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import data_fetcher
from execution import _is_market_open

# --------------------------------------------------------------------------- config

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_WAKE_STATE_PATH = "wake_state.json"
DEFAULT_POSITIONS_PATH = "positions.json"


# ---------------------------------------------------------------------- wake state

def load_wake_state(path: str = DEFAULT_WAKE_STATE_PATH) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict):
            return {k: float(v) for k, v in state.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {}


def save_wake_state(state: Dict[str, float], path: str = DEFAULT_WAKE_STATE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_held_symbols(positions_path: str = DEFAULT_POSITIONS_PATH) -> set[str]:
    """Return a set of symbols currently held according to positions.json."""
    if not os.path.exists(positions_path):
        return set()
    try:
        with open(positions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle dict wrapping `{"positions": [...]}` (the actual format) or direct array
        positions = data.get("positions", []) if isinstance(data, dict) else data
        return {str(p.get("symbol")) for p in positions if p.get("qty", 0) > 0}
    except Exception:
        return set()


# ----------------------------------------------------------------------- analysis


def check_symbol(
    symbol: str,
    wake_config: Dict[str, Any],
    wake_state: Dict[str, float],
    is_held: bool,
    now: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    now = now if now is not None else time.time()
    min_seconds = float(wake_config.get("min_seconds_between_wakes", 3600))
    direction = "sell" if is_held else "buy"
    state_key = f"{symbol}_{direction}"

    if (now - wake_state.get(state_key, 0.0)) < min_seconds:
        return None

    try:
        # We need 15m intervals for short-term spikes, and 1h intervals for 1h drops.
        # It's cheaper to just fetch 15m and aggregate, or fetch both. yfinance 15m gives us both.
        df = data_fetcher.fetch_ohlcv(symbol, period="5d", interval="15m")
    except Exception as exc:
        print(f"  [volume_watch] {symbol}: data unavailable ({exc}); skipping")
        return None

    if len(df) < 5:  # Need at least 5 x 15m bars to cover >1h
        return None

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else None

    current_price = float(close.iloc[-1])
    price_15m_ago = float(close.iloc[-2])
    price_1h_ago = float(close.iloc[-5]) # 4 bars ago is 1h ago if 15m interval

    if price_15m_ago == 0 or price_1h_ago == 0:
        return None

    price_move_pct_15m = (current_price - price_15m_ago) / price_15m_ago * 100.0
    price_move_pct_1h = (current_price - price_1h_ago) / price_1h_ago * 100.0

    # Volume multiple vs 24h average
    volume_multiple = 0.0
    if volume is not None and len(volume) > 26:
        current_vol = float(volume.iloc[-1])
        # approx 26 bars of 15m is roughly 1 trading day (6.5 hours). 
        avg_24h = float(volume.iloc[-27:-1].mean())
        if avg_24h > 0:
            volume_multiple = current_vol / avg_24h

    reasons = []

    if is_held:
        # Sell-side triggers
        sell_conf = wake_config.get("sell_side", {})
        drop_15m_thr = float(sell_conf.get("price_drop_pct_15m", 1.5))
        drop_1h_thr = float(sell_conf.get("price_drop_pct_1h", 3.0))

        if price_move_pct_15m <= -drop_15m_thr:
            reasons.append(f"dropped {abs(price_move_pct_15m):.1f}% in 15m")
        if price_move_pct_1h <= -drop_1h_thr:
            reasons.append(f"dropped {abs(price_move_pct_1h):.1f}% in 1h")
    else:
        # Buy-side triggers
        buy_conf = wake_config.get("buy_side", {})
        spike_15m_thr = float(buy_conf.get("price_move_pct_15m", 1.5))
        vol_mult_thr = float(buy_conf.get("volume_multiple", 2.0))

        if price_move_pct_15m >= spike_15m_thr:
            reasons.append(f"spiked {price_move_pct_15m:.1f}% in 15m")
        if volume_multiple >= vol_mult_thr:
            reasons.append(f"volume {volume_multiple:.1f}x average")

    if not reasons:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "state_key": state_key,
        "trigger_reasons": reasons,
    }


def run_watch(
    config_path: str = DEFAULT_CONFIG_PATH,
    wake_state_path: str = DEFAULT_WAKE_STATE_PATH,
    positions_path: str = DEFAULT_POSITIONS_PATH,
    now: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Check all configured symbols and return categorized flagged symbols.
    Returns: {"wake_buy": ["AAPL", ...], "wake_sell": ["MSFT", ...]}
    """
    if not _is_market_open():
        print("Market is closed. Skipping volume watch.")
        return {}

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    wake_config = config.get("wake_trigger", {}) or {}
    now = now if now is not None else time.time()
    wake_state = load_wake_state(wake_state_path)
    held_symbols = get_held_symbols(positions_path)

    print("=== Bidirectional Volume/Price Watch ===")

    flagged = {"wake_buy": [], "wake_sell": []}
    
    for entry in config.get("symbols", []) or []:
        symbol = entry["symbol"] if isinstance(entry, dict) else str(entry)
        is_held = symbol in held_symbols

        result = check_symbol(
            symbol=symbol,
            wake_config=wake_config,
            wake_state=wake_state,
            is_held=is_held,
            now=now,
        )

        if result is not None:
            print(f"  {symbol}: TRIGGERED {result['direction'].upper()} ({', '.join(result['trigger_reasons'])})")
            flagged[f"wake_{result['direction']}"].append(symbol)
            wake_state[result['state_key']] = now
        else:
            print(f"  {symbol}: no trigger")

    if any(flagged.values()):
        save_wake_state(wake_state, wake_state_path)

    return flagged


# ------------------------------------------------------------------ entry point


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Bidirectional watcher")
    parser.add_argument("--config", default=os.environ.get("BOT_CONFIG", DEFAULT_CONFIG_PATH))
    parser.add_argument("--wake-state", default=DEFAULT_WAKE_STATE_PATH)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS_PATH)
    args = parser.parse_args()

    flagged = run_watch(args.config, args.wake_state, args.positions)

    if not flagged.get("wake_buy") and not flagged.get("wake_sell"):
        print("No symbols flagged. Done.")
        return 0

    import main as main_module
    
    # Process sells first for capital efficiency, then buys
    ret = 0
    for reason in ["wake_sell", "wake_buy"]:
        symbols = flagged.get(reason, [])
        if not symbols:
            continue
            
        symbols_list = ",".join(symbols)
        print(f"\n=== Triggering cycle for: {symbols_list} ({reason}) ===\n")

        sys.argv = [
            "main.py",
            "--config", args.config,
            "--trigger-symbols", symbols_list,
            "--trigger-reason", reason,
        ]
        rc = main_module.main()
        if rc != 0:
            ret = rc

    return ret

if __name__ == "__main__":
    raise SystemExit(main())
