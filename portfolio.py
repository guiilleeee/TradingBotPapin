"""Portfolio-level limits on new buys: position count, gross exposure, and
concentration (static groups plus measured return correlation).

risk_manager.py judges one trade on its own. This module judges it against what
is already held, and follows the same contract: it can only shrink a buy or turn
it into a hold, never enlarge anything, and it never touches a sell or a hold.

Rules, in order (the first one that blocks wins; the exposure rule may shrink):

  1. max_open_positions      -- held count already at the limit -> hold.
  2. groups                  -- the symbol's group (crypto, semiconductors, ...)
                                already holds max_positions -> hold.
  3. correlation             -- 60-session daily-return correlation against every
                                held symbol; if the cluster of positions correlated
                                at >= threshold would exceed max_correlated_positions
                                (counting this buy) -> hold. A pair with too little
                                overlapping history is not counted.
  4. max_gross_exposure_pct  -- the buy is shrunk to whatever room is left under
                                the cap; below min_position_pct of equity -> hold.

The closes for rule 3 come from the funnel's own batched download
(funnel.FunnelData.closes), so a scheduled cycle fetches nothing extra.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

import symbol_config
from models import TradeSignal

DEFAULTS: Dict[str, Any] = {
    "max_open_positions": 6,
    "max_gross_exposure_pct": 80.0,
    # A buy the exposure cap would shrink below this is not worth placing.
    "min_position_pct": 1.0,
    "groups": {},
    "correlation": {
        "enabled": True,
        "lookback_sessions": 60,
        "threshold": 0.75,
        "max_correlated_positions": 2,
        "min_overlap_sessions": 30,
    },
}


def settings_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """`portfolio` merged over DEFAULTS. A malformed value keeps its default."""
    raw = (config or {}).get("portfolio") or {}
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if not isinstance(raw, dict):
        return out
    for key in ("max_open_positions",):
        try:
            out[key] = max(int(raw.get(key, out[key])), 0)
        except (TypeError, ValueError):
            pass
    for key in ("max_gross_exposure_pct", "min_position_pct"):
        try:
            out[key] = max(float(raw.get(key, out[key])), 0.0)
        except (TypeError, ValueError):
            pass
    if isinstance(raw.get("groups"), dict):
        out["groups"] = raw["groups"]
    corr = raw.get("correlation")
    if isinstance(corr, dict):
        merged = dict(DEFAULTS["correlation"])
        merged["enabled"] = corr.get("enabled", True) is not False
        for key, cast in (("lookback_sessions", int), ("threshold", float),
                          ("max_correlated_positions", int), ("min_overlap_sessions", int)):
            try:
                merged[key] = cast(corr.get(key, merged[key]))
            except (TypeError, ValueError):
                pass
        out["correlation"] = merged
    return out


def groups_for(symbol: str, settings: Mapping[str, Any], config: Mapping[str, Any]) -> List[str]:
    """Every configured group `symbol` belongs to, by explicit list or asset class."""
    names = []
    upper = symbol.upper()
    for name, group in (settings.get("groups") or {}).items():
        if not isinstance(group, dict):
            continue
        members = {str(s).upper() for s in group.get("symbols") or []}
        by_class = group.get("asset_class")
        if upper in members or (by_class and symbol_config.asset_class(symbol, config) == by_class):
            names.append(str(name))
    return names


def return_correlation(
    a: pd.Series, b: pd.Series, lookback: int, min_overlap: int
) -> Optional[float]:
    """Correlation of daily returns over the last `lookback` shared sessions, or
    None with fewer than `min_overlap` of them."""
    ra = a.dropna().astype(float).pct_change()
    rb = b.dropna().astype(float).pct_change()
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna().tail(lookback)
    if len(joined) < max(min_overlap, 2):
        return None
    value = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return None if pd.isna(value) else float(value)


def _to_hold(signal: TradeSignal, reason: str) -> TradeSignal:
    reasons = [r for r in (signal.override_reason, reason) if r]
    return signal.model_copy(update={
        "action": "hold",
        "position_size_pct": 0.0,
        "stop_loss_price": None,
        "take_profit_price": None,
        "override_reason": "; ".join(reasons),
    })


def check_buy(
    signal: TradeSignal,
    held: Mapping[str, float],
    equity: float,
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
    closes: Optional[Mapping[str, pd.Series]] = None,
) -> TradeSignal:
    """Apply the portfolio rules to a buy that already passed risk_manager.

    `held` maps each open position's symbol to its current dollar exposure.
    Anything other than a buy passes through untouched.
    """
    if signal.action != "buy":
        return signal
    symbol = signal.symbol
    held = {s: float(v) for s, v in held.items() if s.upper() != symbol.upper()}

    # 1. Position count.
    limit = int(settings["max_open_positions"])
    if limit and len(held) >= limit:
        return _to_hold(signal, f"portfolio: {len(held)} positions open, at the "
                                f"max_open_positions limit of {limit}")

    # 2. Static groups.
    for name in groups_for(symbol, settings, config):
        group = settings["groups"][name]
        try:
            cap = int(group.get("max_positions"))
        except (TypeError, ValueError):
            continue
        members = [s for s in held if name in groups_for(s, settings, config)]
        if len(members) >= cap:
            return _to_hold(signal, f"portfolio: group '{name}' already holds "
                                    f"{len(members)} ({', '.join(sorted(members))}), limit {cap}")

    # 3. Measured correlation.
    corr = settings["correlation"]
    if corr["enabled"] and held and closes:
        mine = closes.get(symbol)
        if mine is not None:
            clustered = []
            for other in sorted(held):
                theirs = closes.get(other)
                if theirs is None:
                    continue
                value = return_correlation(mine, theirs, corr["lookback_sessions"],
                                           corr["min_overlap_sessions"])
                if value is not None and value >= corr["threshold"]:
                    clustered.append(f"{other} {value:.2f}")
            if len(clustered) + 1 > int(corr["max_correlated_positions"]):
                return _to_hold(signal, f"portfolio: returns correlated >= {corr['threshold']:.2f} "
                                        f"with {', '.join(clustered)}; limit "
                                        f"{corr['max_correlated_positions']} correlated positions")

    # 4. Gross exposure: shrink to the room left, or hold if too little remains.
    if equity > 0:
        cap_usd = equity * float(settings["max_gross_exposure_pct"]) / 100.0
        room_pct = max(cap_usd - sum(held.values()), 0.0) / equity * 100.0
        if signal.position_size_pct > room_pct:
            if room_pct < float(settings["min_position_pct"]):
                return _to_hold(signal, f"portfolio: gross exposure at the "
                                        f"{settings['max_gross_exposure_pct']:.0f}% cap "
                                        f"({room_pct:.2f}% of equity left)")
            reasons = [r for r in (signal.override_reason, (
                f"portfolio: size {signal.position_size_pct:.2f}% cut to {room_pct:.2f}% to stay "
                f"under the {settings['max_gross_exposure_pct']:.0f}% gross exposure cap")) if r]
            return signal.model_copy(update={"position_size_pct": room_pct,
                                             "override_reason": "; ".join(reasons)})
    return signal
