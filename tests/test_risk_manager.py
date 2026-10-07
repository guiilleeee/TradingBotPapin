import pytest

import risk_manager
from models import SignalOutput

DEFAULTS = dict(
    current_price=100.0,
    today_realized_loss_pct=0.0,
    circuit_breaker_loss_pct=3.0,
    max_risk_pct=1.0,
    max_absolute_position_pct=20.0,
    min_confidence=0.65,
)


def buy(**overrides):
    base = dict(
        symbol="AAPL",
        action="buy",
        confidence=0.9,
        position_size_pct=10.0,
        stop_loss_price=95.0,
        take_profit_price=115.0,
        reasoning="prova",
    )
    base.update(overrides)
    return SignalOutput(**base)


def validate(raw, **overrides):
    kwargs = dict(DEFAULTS)
    kwargs.update(overrides)
    return risk_manager.validate(raw=raw, **kwargs)


def test_clean_buy_is_sized_from_the_stop_distance():
    # 5% stop distance, risking 1% of equity -> a 20% position.
    result = validate(buy())
    assert result.action == "buy"
    assert result.raw_action == "buy"
    assert result.position_size_pct == pytest.approx(20.0)
    assert result.override_reason is None


def test_size_ignores_whatever_the_model_suggested():
    # 10% stop distance -> 10% position, regardless of the model's 77%.
    result = validate(buy(position_size_pct=77.0, stop_loss_price=90.0))
    assert result.position_size_pct == pytest.approx(10.0)


def test_size_is_clamped_to_the_absolute_cap():
    # 2% stop distance would demand 50%; the cap is 20%.
    result = validate(buy(stop_loss_price=98.0))
    assert result.action == "buy"
    assert result.position_size_pct == pytest.approx(20.0)
    assert "clamped" in result.override_reason


def test_circuit_breaker_overrides_to_hold():
    result = validate(buy(), today_realized_loss_pct=-3.5)
    assert result.action == "hold"
    assert result.raw_action == "buy"
    assert "circuit breaker" in result.override_reason
    assert result.position_size_pct == 0.0
    assert result.stop_loss_price is None
    assert result.take_profit_price is None


def test_circuit_breaker_fires_exactly_at_the_limit():
    assert validate(buy(), today_realized_loss_pct=-3.0).action == "hold"
    assert validate(buy(), today_realized_loss_pct=-2.99).action == "buy"


def test_circuit_breaker_does_not_touch_a_hold():
    result = validate(buy(action="hold"), today_realized_loss_pct=-9.0)
    assert result.action == "hold"
    assert result.override_reason is None


def test_low_confidence_overrides_to_hold():
    result = validate(buy(confidence=0.5))
    assert result.action == "hold"
    assert "confidence" in result.override_reason


def test_confidence_threshold_is_the_one_passed_in():
    # Same signal, two thresholds. The risk manager has no opinion about mode.
    assert validate(buy(confidence=0.5), min_confidence=0.65).action == "hold"
    assert validate(buy(confidence=0.5), min_confidence=0.40).action == "buy"


def test_non_positive_model_size_overrides_to_hold():
    result = validate(buy(position_size_pct=0.0))
    assert result.action == "hold"
    assert "non-positive position_size_pct" in result.override_reason


def test_stop_too_close_overrides_to_hold():
    # 0.2% away: below the 0.3% floor, so not a credible risk boundary.
    result = validate(buy(stop_loss_price=99.8))
    assert result.action == "hold"
    assert "credible risk boundary" in result.override_reason
    assert result.position_size_pct == 0.0


def test_stop_just_above_the_floor_is_allowed():
    result = validate(buy(stop_loss_price=99.6))  # 0.4%
    assert result.action == "buy"
    assert result.position_size_pct == pytest.approx(20.0)  # clamped from 250%


def test_missing_take_profit_overrides_to_hold():
    result = validate(buy(take_profit_price=None))
    assert result.action == "hold"
    assert "take_profit_price" in result.override_reason


def test_missing_both_levels_names_both():
    raw = buy(stop_loss_price=None, take_profit_price=None)
    result = validate(raw)
    assert result.action == "hold"
    assert "stop_loss_price" in result.override_reason
    assert "take_profit_price" in result.override_reason


# ---------------------------------------------------------- reward:risk floor


def test_reward_risk_below_the_minimum_overrides_to_hold():
    # price=100, stop=95 (risk 5), take=105 (reward 5) -> ratio 1.0, below the
    # 1.5 default minimum.
    result = validate(buy(stop_loss_price=95.0, take_profit_price=105.0))
    assert result.action == "hold"
    assert result.position_size_pct == 0.0
    assert "reward:risk 1.00" in result.override_reason
    assert "1.50 minimum" in result.override_reason


def test_reward_risk_at_exactly_the_minimum_is_allowed():
    # risk 5, reward 7.5 -> ratio exactly 1.5.
    result = validate(buy(stop_loss_price=95.0, take_profit_price=107.5))
    assert result.action == "buy"


def test_reward_risk_above_the_minimum_is_unaffected():
    result = validate(buy())  # default: risk 5, reward 15 -> ratio 3.0
    assert result.action == "buy"
    assert result.override_reason is None or "reward:risk" not in result.override_reason


def test_reward_risk_minimum_is_configurable():
    # Same 1.0 ratio as the first test, but with a lower minimum that permits it.
    result = validate(buy(stop_loss_price=95.0, take_profit_price=105.0), min_reward_risk_ratio=1.0)
    assert result.action == "buy"


def test_reward_risk_floor_never_applies_to_a_sell():
    # A sell's stop/take are schema-required but never used to manage anything
    # once the sell executes -- this floor must never trap the bot in a
    # position the model has already decided to exit.
    result = validate(buy(action="sell", stop_loss_price=95.0, take_profit_price=105.0))
    assert result.action == "sell"


def test_reward_risk_floor_never_fires_on_top_of_an_already_rejected_trade():
    # Confidence already forced this to hold before rule 4 even runs -- the
    # reward:risk check must never additionally fire (and must never be the
    # reason reported) on a trade another rule already killed.
    result = validate(
        buy(confidence=0.1, stop_loss_price=95.0, take_profit_price=105.0), min_confidence=0.65
    )
    assert result.action == "hold"
    assert "reward:risk" not in result.override_reason


def test_sell_is_sized_and_validated_the_same_way():
    result = validate(buy(action="sell"))
    assert result.action == "sell"
    assert result.raw_action == "sell"
    assert result.position_size_pct == pytest.approx(20.0)


def test_hold_from_the_model_is_zeroed_and_not_flagged():
    result = validate(buy(action="hold"))
    assert result.action == "hold"
    assert result.raw_action == "hold"
    assert result.position_size_pct == 0.0
    assert result.stop_loss_price is None
    assert result.override_reason is None


def test_multiple_rules_are_all_reported():
    result = validate(buy(confidence=0.1), today_realized_loss_pct=-5.0)
    assert result.action == "hold"
    assert "circuit breaker" in result.override_reason


def test_reasoning_and_confidence_survive_an_override():
    raw = buy(confidence=0.2, reasoning="analisi en catala")
    result = validate(raw)
    assert result.reasoning == "analisi en catala"
    assert result.confidence == 0.2  # the model's honest number is preserved


def test_non_positive_price_cannot_produce_a_trade():
    result = validate(buy(), current_price=0.0)
    assert result.action == "hold"
    assert "not positive" in result.override_reason


def test_buy_with_stop_above_price_is_held():
    result = validate(buy(stop_loss_price=101.0, take_profit_price=115.0))
    assert result.action == "hold"
    assert "wrong side" in result.override_reason
    assert result.position_size_pct == 0.0


def test_buy_with_take_profit_below_price_is_held():
    result = validate(buy(stop_loss_price=95.0, take_profit_price=99.0))
    assert result.action == "hold"
    assert "wrong side" in result.override_reason


def test_sell_levels_are_not_side_checked():
    """A sell's levels are never used once it executes (see rule 4b's comment)."""
    raw = buy(action="sell", stop_loss_price=105.0, take_profit_price=90.0)
    assert validate(raw).action == "sell"


# ----------------------------------------------- confidence-scaled sizing

# Live-like settings: threshold 0.58, floor 10%, cap 25%, risk backstop 3%.
SCALED = dict(
    min_confidence=0.58, max_risk_pct=3.0, max_absolute_position_pct=25.0,
    min_position_size_pct=10.0, full_size_confidence=0.90,
)


@pytest.mark.parametrize("confidence, expected", [
    (0.58, 10.0),    # at the threshold: the floor
    (0.65, 13.28125),
    (0.74, 17.5),    # halfway up the ramp
    (0.85, 22.65625),
    (0.90, 25.0),    # full size
    (0.97, 25.0),    # flat above it
])
def test_size_rises_linearly_with_confidence(confidence, expected):
    # 5% stop: the old rule would give 3 / 0.05 = 60%, clamped to the cap.
    result = validate(buy(confidence=confidence), **SCALED)
    assert result.action == "buy"
    assert result.position_size_pct == pytest.approx(expected)


def test_scaled_size_ignores_the_stop_distance_until_the_risk_backstop_binds():
    tight = validate(buy(confidence=0.74, stop_loss_price=97.0), **SCALED)  # 3% stop
    wide = validate(buy(confidence=0.74, stop_loss_price=92.0), **SCALED)   # 8% stop
    assert tight.position_size_pct == pytest.approx(17.5)
    assert wide.position_size_pct == pytest.approx(17.5)


def test_risk_backstop_still_caps_a_full_confidence_buy():
    # 20% stop at 3% max risk -> 15% position, below the 25% confidence size.
    result = validate(buy(confidence=0.95, stop_loss_price=80.0, take_profit_price=140.0), **SCALED)
    assert result.position_size_pct == pytest.approx(15.0)
    assert result.position_size_pct * 0.20 <= 3.0 + 1e-9


def test_floor_is_pulled_down_by_a_tighter_per_symbol_cap():
    result = validate(buy(confidence=0.58), **{**SCALED, "max_absolute_position_pct": 8.0})
    assert result.position_size_pct == pytest.approx(8.0)


def test_scaling_is_off_without_min_position_size_pct():
    legacy = {k: v for k, v in SCALED.items() if k not in ("min_position_size_pct", "full_size_confidence")}
    result = validate(buy(confidence=0.60), **legacy)
    assert result.position_size_pct == pytest.approx(25.0)  # 60% clamped to the cap


def test_scaling_never_changes_a_sell_or_a_rejection():
    assert validate(buy(confidence=0.50), **SCALED).action == "hold"
    sell = validate(buy(action="sell", confidence=0.60), **SCALED)
    assert sell.action == "sell"
