"""Code-enforced risk layer.

The model proposes; this module disposes. Every rule here can only ever make a
trade smaller or turn it into a hold -- there is no path through this function
that makes a position larger than the caller's caps allow.

Deliberately knows nothing about live vs simulation: `min_confidence` arrives
already resolved by the caller. One responsibility, one signature.
"""

from __future__ import annotations

from typing import List, Optional

from models import SignalOutput, TradeSignal

# A stop this close to the entry is not a risk boundary, it is noise. Sizing off it
# would divide by a near-zero number and demand an enormous position.
MIN_STOP_DISTANCE_PCT = 0.003  # 0.3%

# Below this reward:risk, a setup cannot be profitable at any win rate this
# strategy has actually shown -- the diagnostic on the historical-screening
# backtest found a realised 1.18 against a 37.5% win rate that needed ~1.67 to
# break even. 1.5 is deliberately a bit under that break-even figure rather
# than tuned to it exactly: the win rate itself may improve once the worst
# setups stop qualifying at all, and this is a hypothesis to re-test, not a
# number picked to match one past sample. Config-driven (min_reward_risk_ratio
# in config.yaml); this default only protects callers that predate the
# parameter, e.g. existing tests -- every real caller reads it from config.
DEFAULT_MIN_REWARD_RISK_RATIO = 1.5


def validate(
    raw: SignalOutput,
    current_price: float,
    today_realized_loss_pct: float,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
    min_confidence: float,
    min_reward_risk_ratio: float = DEFAULT_MIN_REWARD_RISK_RATIO,
    days_to_earnings: Optional[int] = None,
    earnings_blackout_days: int = 0,
    atr: Optional[float] = None,
    stop_atr_min: float = 0.0,
    stop_atr_max: float = 0.0,
) -> TradeSignal:
    """Apply the risk rules in order and return the signal execution may act on.

    Args:
        raw: the model's unmodified output.
        current_price: last close, used for the stop-distance calculation.
        today_realized_loss_pct: today's realised P&L as a percentage of equity.
            Negative means a loss.
        circuit_breaker_loss_pct: positive magnitude at which trading halts for the day.
        max_risk_pct: percent of equity to risk on one trade (e.g. 1.0 for 1%).
        max_absolute_position_pct: hard cap on position size as a percent of equity.
        min_confidence: already resolved for the current mode by the caller.
        min_reward_risk_ratio: minimum acceptable reward:risk (take-profit distance
            over stop-loss distance, both from current_price). Structural, not part
            of the live/simulation confidence-threshold split -- applies identically
            in both modes.
        days_to_earnings: calendar days to the next earnings report, or None
            when unknown -- an unknown date never blocks.
        earnings_blackout_days: a buy with days_to_earnings in [0, N] is held;
            a bracket held through an earnings gap can lose far past its stop.
            0 disables the rule.
        atr: the symbol's ATR-14 in price units, or None when unknown.
        stop_atr_min / stop_atr_max: a buy's stop distance must lie within
            [min, max] x atr. Tighter is inside normal daily noise; wider sizes
            the position down to almost nothing. 0 disables either bound, and
            an unknown ATR skips the rule.
    """
    reasons: List[str] = []

    raw_action = raw.action
    action = raw.action
    size = raw.position_size_pct
    stop: Optional[float] = raw.stop_loss_price
    take: Optional[float] = raw.take_profit_price

    # 1. Circuit breaker. Checked first so nothing else can talk us past it.
    #    Blocks new buys only: a sell reduces exposure, and on a day bad enough
    #    to trip the breaker, closing a losing position is a decision the model
    #    must still be able to make.
    breaker = abs(circuit_breaker_loss_pct)
    if action == "buy" and today_realized_loss_pct <= -breaker:
        reasons.append(
            f"circuit breaker: today's realised P&L {today_realized_loss_pct:.2f}% is at or "
            f"beyond the -{breaker:.2f}% daily limit"
        )
        action = "hold"

    # 1b. Earnings blackout. Buy only: exiting ahead of earnings is fine.
    if (
        action == "buy"
        and earnings_blackout_days > 0
        and days_to_earnings is not None
        and 0 <= days_to_earnings <= earnings_blackout_days
    ):
        reasons.append(
            f"earnings in {days_to_earnings} day(s), inside the "
            f"{earnings_blackout_days}-day pre-earnings blackout"
        )
        action = "hold"

    # 2. Confidence threshold.
    if action != "hold" and raw.confidence < min_confidence:
        reasons.append(
            f"confidence {raw.confidence:.2f} is below the {min_confidence:.2f} minimum for this mode"
        )
        action = "hold"

    # 3. Clearly broken model sizing. This rejects nonsense output only -- the real
    #    number comes from rule 4 below and overwrites whatever the model suggested.
    if action in ("buy", "sell") and size <= 0:
        reasons.append(
            f"model returned a non-positive position_size_pct ({size}) for a {action}"
        )
        action = "hold"

    # 4. Risk-based sizing from the stop distance.
    if action in ("buy", "sell") and stop is not None:
        if current_price <= 0:
            reasons.append(f"current_price {current_price} is not positive; cannot size the trade")
            action = "hold"
        elif action == "buy" and (stop >= current_price or (take is not None and take <= current_price)):
            # Every distance below is an abs(), so without this a long whose stop
            # sits above the price (or target below it) sized and passed like a
            # sane one -- and went to the broker as an inverted bracket. It is
            # also exactly what a long looks like after the price has already
            # fallen through the stop.
            reasons.append(
                f"buy levels are on the wrong side of price {current_price:.6g} "
                f"(stop-loss {stop:.6g} must be below it"
                + (f", take-profit {take:.6g} above it)" if take is not None else ")")
            )
            action = "hold"
        else:
            stop_distance_pct = abs(current_price - stop) / current_price
            if stop_distance_pct < MIN_STOP_DISTANCE_PCT:
                reasons.append(
                    f"stop-loss {stop:.6g} sits {stop_distance_pct * 100:.3f}% from price "
                    f"{current_price:.6g}, under the {MIN_STOP_DISTANCE_PCT * 100:.1f}% minimum "
                    "to be a credible risk boundary"
                )
                action = "hold"
            else:
                # 4b. Reward:risk floor. Buy only, deliberately -- a sell's
                # stop_loss_price/take_profit_price are schema-required (every
                # buy/sell must carry both, per prompts.py's hard rule 2) but
                # never actually used to manage anything once a sell executes:
                # _update_ledger closes the position from the ledger's own
                # qty/entry price the instant action == "sell" and never reads
                # either field again. Applying this floor to a sell would risk
                # trapping the bot in a position the model has already decided
                # to exit, based on numbers that don't describe anything real.
                # Reward:risk is an entry question; only checkable once a valid
                # stop distance exists and a take-profit is actually present
                # (a missing take_profit is rejected on its own by rule 5
                # below, never treated as a reward:risk failure here). Placed
                # before sizing is computed: a setup this rule rejects should
                # never have a size computed for it in the first place, even
                # though rule 6's catch-all zeroing would discard it either way.
                if action == "buy" and take is not None:
                    risk_distance = abs(current_price - stop)
                    reward_distance = abs(take - current_price)
                    reward_risk_ratio = reward_distance / risk_distance
                    if reward_risk_ratio < min_reward_risk_ratio:
                        reasons.append(
                            f"reward:risk {reward_risk_ratio:.2f} is below the "
                            f"{min_reward_risk_ratio:.2f} minimum (take-profit {take:.6g}, "
                            f"stop-loss {stop:.6g}, price {current_price:.6g})"
                        )
                        action = "hold"

                # 4c. Stop distance in ATR units. Buy only, for the same
                # reason as 4b: a sell's levels manage nothing.
                if action == "buy" and atr is not None and atr > 0:
                    atr_multiple = abs(current_price - stop) / atr
                    if stop_atr_min > 0 and atr_multiple < stop_atr_min:
                        reasons.append(
                            f"stop-loss {stop:.6g} is {atr_multiple:.2f}x ATR from price, under "
                            f"the {stop_atr_min:.2f}x minimum (inside normal daily noise)"
                        )
                        action = "hold"
                    elif stop_atr_max > 0 and atr_multiple > stop_atr_max:
                        reasons.append(
                            f"stop-loss {stop:.6g} is {atr_multiple:.2f}x ATR from price, over "
                            f"the {stop_atr_max:.2f}x maximum"
                        )
                        action = "hold"

                if action in ("buy", "sell"):
                    computed = max_risk_pct / stop_distance_pct
                    if computed > max_absolute_position_pct:
                        reasons.append(
                            f"risk-based size {computed:.2f}% clamped to the "
                            f"{max_absolute_position_pct:.2f}% absolute position cap"
                        )
                        computed = max_absolute_position_pct
                    size = computed

    # 5. Missing exit levels.
    if action in ("buy", "sell"):
        missing = [
            name
            for name, value in (("stop_loss_price", stop), ("take_profit_price", take))
            if value is None
        ]
        if missing:
            reasons.append(f"{action} is missing required {' and '.join(missing)}")
            action = "hold"

    # 6. Catch-all zeroing. Any route to hold lands here, including a hold the model
    #    produced on its own -- this is not per-rule cleanup.
    if action == "hold":
        size = 0.0
        stop = None
        take = None

    return TradeSignal(
        symbol=raw.symbol,
        action=action,
        confidence=raw.confidence,
        position_size_pct=size,
        stop_loss_price=stop,
        take_profit_price=take,
        reasoning=raw.reasoning,
        override_reason="; ".join(reasons) if reasons else None,
        raw_action=raw_action,
    )
