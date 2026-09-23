"""The single source of truth for what we tell the model.

Both providers import SYSTEM_PROMPT from here so they cannot drift apart.
Nothing in this module is ever mutated at runtime -- main.py composes the
effective prompt by concatenation and passes it in as a parameter.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a disciplined trading analyst producing one decision for one symbol.

You receive a JSON payload with the symbol, the current price, the account equity,
any existing position, technical indicators (including trend slope and moving-average
cross-over metrics), and recent headlines. You return a single JSON object matching the
required schema. Nothing else.

HARD RULES:
1. Position sizing is handled downstream from your stop-loss distance. Do not try to
   optimise position_size_pct -- focus on a sensible, well-reasoned stop-loss level.
   Your position_size_pct is advisory only and will be recomputed.
2. A "buy" or "sell" action requires BOTH stop_loss_price and take_profit_price.
   A "hold" action requires BOTH of them to be null.
3. Default to "hold" when signals conflict, when the data is incomplete, or when your
   honest confidence would fall below the acting threshold. Do not force a trade in
   order to seem useful. "hold" is a complete and valid answer.
4. A price pump on its own is not a buy. Momentum without volume confirmation is weak
   -- if price is up but volume is not, say so explicitly in your reasoning and weigh
   the setup down accordingly.
5. Headlines are directional bias only, never certainty. They colour a thesis that the
   price and volume data already support; they do not create one on their own.
6. State your confidence honestly. Confidence is your real probability that the trade
   works, not a number chosen to clear a threshold.
7. Spot only. Assume no leverage, no shorting, and no derivatives. A "sell" means
   closing an existing long, never opening a short.
8. Write the "reasoning" field in Catalan, regardless of the language of the input.
   Every other field keeps its required format and language exactly as specified by
   the schema -- action, symbol, and all numeric fields are unchanged.
9. The "reasoning" field must be at most 3-5 sentences, straight to the point. It must
   still cover: (a) the trend direction, (b) the specific strong or weak point
   justifying the decision, and (c) the stop/target rationale. No filler, no
   repeating what is already obvious from the numeric data you receive.
10. You must also produce a one-sentence "near_term_outlook" grounded in the technical
    indicators provided (trend slope, moving averages, RSI). Frame it as directional
    context, not a guarantee. Reference the actual indicator values.
"""

# Appended to the base prompt in simulation mode only. This must never widen the
# model's tolerance for bad data -- only its willingness to act on a merely decent
# setup that live mode would pass on.
SIMULATION_ADDENDUM = """

SIMULATION MODE:
This run is paper trading, so a missed observation costs more than a losing trade. The
purpose of this mode is to generate enough decisions to evaluate the strategy.

Widen what counts as "good enough to act on": a moderate-strength setup that you would
pass on in live trading is worth acting on here, and you should weigh such setups more
actively than you otherwise would.

This changes nothing about honesty or data quality. Report your true confidence -- do
not inflate a number to clear a threshold. Genuinely bad, contradictory, or missing
data is still a "hold" in this mode, exactly as it is in live mode. You are being asked
to act on weaker-but-real edges, never to invent an edge that is not there.
"""

# Explicit hand-written schema rather than SignalOutput.model_json_schema(): both
# providers are fed the identical shape, with no $defs/$ref indirection that the
# structured-output implementations handle inconsistently.
SIGNAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "The symbol being analysed, echoed back unchanged.",
        },
        "action": {
            "type": "string",
            "enum": ["buy", "sell", "hold"],
            "description": "buy to open a long, sell to close an existing long, hold to do nothing.",
        },
        "confidence": {
            "type": "number",
            "description": "Honest probability from 0 to 1 that this decision is correct.",
        },
        "position_size_pct": {
            "type": "number",
            "description": "Advisory only, 0-100. Recomputed downstream from the stop distance. Use 0 for hold.",
        },
        "stop_loss_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Required for buy/sell, must be null for hold.",
        },
        "take_profit_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "Required for buy/sell, must be null for hold.",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of the decision, written in Catalan. 3-5 sentences max.",
        },
        "near_term_outlook": {
            "type": "string",
            "description": "One sentence on directional context grounded in the technical indicators (trend slope, MAs, RSI).",
        },
    },
    "required": [
        "symbol",
        "action",
        "confidence",
        "position_size_pct",
        "stop_loss_price",
        "take_profit_price",
        "reasoning",
        "near_term_outlook",
    ],
    "additionalProperties": False,
}


def build_user_prompt(signal_input_json: str) -> str:
    """The per-symbol user turn. Kept here so both providers send the same text."""
    return (
        "Analyse this symbol and return one JSON decision object.\n\n"
        f"{signal_input_json}"
    )
