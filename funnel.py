"""Per-cycle funnel: rank the whole universe locally, send only the best few to the model.

A full-universe cycle (scheduled, or a manual workflow_dispatch of trading_bot.yml)
runs this before any model call. It is free -- one batched yfinance download, no
LLM -- and it decides which symbols are worth paying Claude to look at:

  * every symbol with an open position (always; a held position is never left
    unreviewed because it ranked low), plus
  * the top `top_n` (default 6) non-held symbols by score.

Wake-up cycles (wake_buy / wake_sell) and single-symbol manual analyses name their
symbols explicitly and never pass through here.

Scoring is cross-asset on purpose. The weekly equity screen ranks raw volume, which
is fine inside one asset class but meaningless across two: crypto volume is quoted
in dollars and would outrank every stock every time. So both inputs here are
relative to the symbol's own recent history:

  volume_ratio -- the larger of the last two daily volumes over the average of the
                  20 sessions before them. Taking the larger of two means a
                  still-forming daily bar (today, mid-session) can't hide a spike
                  that yesterday's complete bar already shows.
  move_z       -- |last daily return| over the standard deviation of the prior 20
                  daily returns: "how unusual is today's move *for this symbol*",
                  so a routine 3% crypto day doesn't outrank a 2-sigma stock move.

score = 0.6 * percentile(volume_ratio) + 0.4 * percentile(move_z) -- the same
weights and percentile ranking as equity_universe.score_equities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set

import pandas as pd

from equity_universe import MOMENTUM_WEIGHT, VOLUME_WEIGHT, percentile_ranks

DEFAULT_TOP_N = 6
FETCH_PERIOD = "3mo"
LOOKBACK_SESSIONS = 20
# Two recent bars plus the lookback plus one prior close for the first return.
MIN_BARS = LOOKBACK_SESSIONS + 3


@dataclass
class FunnelResult:
    held: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    scored: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def selected(self) -> List[str]:
        """Held first (a sell frees capital for the buys after it), then candidates."""
        return self.held + [s for s in self.candidates if s not in self.held]


def top_n_from_config(config: Dict[str, Any]) -> int:
    try:
        value = int(((config or {}).get("funnel") or {}).get("top_n", DEFAULT_TOP_N))
    except (TypeError, ValueError):
        return DEFAULT_TOP_N
    return max(value, 0)


def funnel_enabled(config: Dict[str, Any]) -> bool:
    return ((config or {}).get("funnel") or {}).get("enabled", True) is not False


def metrics_from_frame(close: pd.Series, volume: pd.Series) -> Dict[str, float] | None:
    """volume_ratio and move_z for one symbol's daily bars, or None if too short."""
    close = close.dropna().astype(float)
    volume = volume.reindex(close.index).fillna(0.0).astype(float)
    if len(close) < MIN_BARS:
        return None

    returns = close.pct_change().dropna()
    last_return = float(returns.iloc[-1])
    prior_returns = returns.iloc[-(LOOKBACK_SESSIONS + 1):-1]
    sigma = float(prior_returns.std()) if len(prior_returns) >= 2 else 0.0
    move_z = abs(last_return) / sigma if sigma > 0 else 0.0

    recent_volume = float(volume.iloc[-2:].max())
    base_volume = float(volume.iloc[-(LOOKBACK_SESSIONS + 2):-2].mean())
    volume_ratio = recent_volume / base_volume if base_volume > 0 else 0.0

    return {"volume_ratio": volume_ratio, "move_z": move_z, "last_return_pct": last_return * 100.0}


def fetch_funnel_data(symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """One batched yfinance download for the whole universe. Never raises."""
    symbols = list(symbols)
    if not symbols:
        return {}
    try:
        import yfinance as yf

        df = yf.download(
            symbols,
            period=FETCH_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [funnel] price download failed ({type(exc).__name__}: {exc})")
        return {}
    if df is None or df.empty:
        return {}

    if not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat([df], keys=symbols[:1], axis=1)

    out: Dict[str, Dict[str, float]] = {}
    present = {c[0] for c in df.columns}
    for symbol in symbols:
        if symbol not in present:
            continue
        try:
            metrics = metrics_from_frame(df[symbol]["Close"], df[symbol]["Volume"])
        except Exception:  # noqa: BLE001 - one bad symbol never sinks the ranking
            continue
        if metrics is not None:
            out[symbol] = metrics
    return out


def score_symbols(universe: Sequence[str], data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Every universe symbol, best first. Symbols without data score 0 and sort
    after every symbol with data; ties keep universe order (market-cap order for
    equities), so the ranking is deterministic.
    """
    volume_pct = percentile_ranks({s: d["volume_ratio"] for s, d in data.items() if s in universe})
    move_pct = percentile_ranks({s: d["move_z"] for s, d in data.items() if s in universe})

    rows = []
    for position, symbol in enumerate(universe):
        has_data = symbol in data
        score = (
            VOLUME_WEIGHT * volume_pct.get(symbol, 0.0) + MOMENTUM_WEIGHT * move_pct.get(symbol, 0.0)
            if has_data else 0.0
        )
        rows.append({
            "symbol": symbol,
            "score": score,
            "has_data": has_data,
            "position": position,
            **(data.get(symbol) or {}),
        })
    rows.sort(key=lambda r: (not r["has_data"], -r["score"], r["position"]))
    return rows


def select(
    universe: Sequence[str],
    held: Iterable[str],
    data: Dict[str, Dict[str, float]],
    top_n: int = DEFAULT_TOP_N,
    eligible: Callable[[str], bool] = lambda _symbol: True,
) -> FunnelResult:
    """Held symbols (all of them, universe or not) plus the top_n eligible others.

    `eligible` filters new-entry candidates only -- main.py uses it to drop
    equities while NYSE is closed, since a buy there would be skipped anyway.
    Held symbols bypass it: a position must always be reviewable.
    """
    held_set: Set[str] = {s.upper() for s in held}
    held_ordered = [s for s in universe if s.upper() in held_set]
    held_ordered += sorted(s for s in held if s.upper() not in {u.upper() for u in universe})

    scored = score_symbols(universe, data)
    candidates = [
        r["symbol"] for r in scored
        if r["symbol"].upper() not in held_set and eligible(r["symbol"])
    ][: max(top_n, 0)]

    return FunnelResult(held=held_ordered, candidates=candidates, scored=scored)
