"""Per-symbol facts and overrides: asset class, broker symbol, wake thresholds, size cap.

One place answers "what is different about this symbol?" so volume_watch.py,
main.py, and execution.py cannot drift apart on it.

Overrides come from config.yaml's `symbol_overrides` block only. main.load_config
never lets symbols.yaml touch anything but the `symbols` list, so the weekly
screening job has no path to widen a threshold or a cap here either.

The size-cap override can only ever tighten: an override above the global
`max_absolute_position_pct` is clamped back down to it. That keeps the risk
layer's one-way guarantee (a rule can make a trade smaller, never larger) intact
even against a typo in config.yaml.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

CRYPTO = "crypto"
EQUITY = "equity"

# yfinance names crypto pairs "BTC-USD"; that is the canonical form everywhere in
# this project (config, ledger, signals). Only execution.py translates it for Alpaca.
_CRYPTO_QUOTE_SUFFIX = "-USD"


def _entry_symbol(entry: Any) -> str:
    return entry["symbol"] if isinstance(entry, dict) else str(entry)


def asset_class(symbol: str, config: Mapping[str, Any] | None = None) -> str:
    """"crypto" or "equity". An explicit asset_class on the config entry wins;
    otherwise a "-USD" suffix means crypto (the only crypto form this project uses).
    """
    for entry in (config or {}).get("symbols", []) or []:
        if isinstance(entry, dict) and _entry_symbol(entry).upper() == symbol.upper():
            declared = str(entry.get("asset_class", "")).strip().lower()
            if declared in (CRYPTO, EQUITY):
                return declared
    return CRYPTO if symbol.upper().endswith(_CRYPTO_QUOTE_SUFFIX) else EQUITY


def is_crypto(symbol: str, config: Mapping[str, Any] | None = None) -> bool:
    return asset_class(symbol, config) == CRYPTO


def alpaca_order_symbol(symbol: str) -> str:
    """Alpaca's order symbol: "BTC/USD" for crypto, unchanged for equities."""
    if symbol.upper().endswith(_CRYPTO_QUOTE_SUFFIX):
        return symbol.upper()[: -len(_CRYPTO_QUOTE_SUFFIX)] + "/USD"
    return symbol


def alpaca_position_symbol(symbol: str) -> str:
    """Alpaca's position symbol: "BTCUSD" for crypto (no separator), unchanged otherwise."""
    if symbol.upper().endswith(_CRYPTO_QUOTE_SUFFIX):
        return symbol.upper()[: -len(_CRYPTO_QUOTE_SUFFIX)] + "USD"
    return symbol


def from_alpaca_symbol(alpaca_symbol: str, alpaca_asset_class: str = "") -> str:
    """Inverse of the two above: "BTCUSD" / "BTC/USD" (crypto) -> "BTC-USD"."""
    sym = alpaca_symbol.upper()
    if "/" in sym:
        base, _, quote = sym.partition("/")
        return f"{base}-{quote}"
    if alpaca_asset_class.lower() == CRYPTO and sym.endswith("USD") and len(sym) > 3:
        return sym[:-3] + "-USD"
    return alpaca_symbol


# ------------------------------------------------------------------ overrides


def _overrides_for(config: Mapping[str, Any], symbol: str) -> Dict[str, Any]:
    overrides = (config or {}).get("symbol_overrides") or {}
    for key, value in overrides.items():
        if str(key).upper() == symbol.upper() and isinstance(value, dict):
            return value
    return {}


def _deep_merge(base: Dict[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def wake_config(config: Mapping[str, Any], symbol: str) -> Dict[str, Any]:
    """The global wake_trigger block with this symbol's overrides merged on top.

    A symbol with no override gets the global block unchanged; an override only
    replaces the keys it names (e.g. DOGE's buy_side.price_move_pct_15m) and
    inherits everything else.
    """
    base = dict((config or {}).get("wake_trigger") or {})
    patch = _overrides_for(config, symbol).get("wake_trigger") or {}
    return _deep_merge(base, patch) if patch else base


def max_position_pct(config: Mapping[str, Any], symbol: str, global_cap: float) -> float:
    """This symbol's absolute position cap -- never above the global one."""
    override = _overrides_for(config, symbol).get("max_absolute_position_pct")
    if override is None:
        return float(global_cap)
    try:
        value = float(override)
    except (TypeError, ValueError):
        return float(global_cap)
    if value <= 0:
        return float(global_cap)
    return min(value, float(global_cap))
