"""The funnel: rank all 30 locally, send the model only held + top 6.

Unit level (funnel.py) and cycle level (main._run_cycle_body picks exactly the
funnel's symbols for scheduled runs, never for wake/manual runs).
"""

import numpy as np
import pandas as pd
import pytest

import execution
import funnel
import main

EQUITIES = [f"EQ{i:02d}" for i in range(25)]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
UNIVERSE = EQUITIES + CRYPTO


def data_with_scores(order):
    """Fixture metrics where `order` (best first) is the unambiguous ranking:
    both volume_ratio and move_z strictly decrease along it."""
    n = len(order)
    return {s: {"volume_ratio": float(n - i), "move_z": float(n - i), "last_return_pct": 1.0}
            for i, s in enumerate(order)}


# -------------------------------------------------------------- metrics


def test_metrics_measure_volume_and_move_against_the_symbols_own_history():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, 40)
    returns[-1] = 0.05  # a ~5-sigma day
    close = pd.Series(100 * np.cumprod(1 + returns))
    volume = pd.Series([1_000.0] * 39 + [3_000.0])

    m = funnel.metrics_from_frame(close, volume)

    assert m["volume_ratio"] == pytest.approx(3.0)
    assert m["move_z"] > 3.0
    assert m["last_return_pct"] == pytest.approx(5.0)


def test_a_partial_today_bar_does_not_hide_yesterdays_spike():
    close = pd.Series(np.linspace(100, 101, 40))
    volume = pd.Series([1_000.0] * 38 + [4_000.0, 200.0])  # spike yesterday, thin today
    assert funnel.metrics_from_frame(close, volume)["volume_ratio"] == pytest.approx(4.0)


def test_too_little_history_yields_no_metrics():
    assert funnel.metrics_from_frame(pd.Series([1.0] * 10), pd.Series([1.0] * 10)) is None


def test_scale_is_irrelevant_so_crypto_dollar_volume_cannot_dominate():
    """Two identical-shaped histories, one with 1e9x the volume, score the same."""
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 40)))
    vol = pd.Series(rng.uniform(900, 1100, 40))
    a = funnel.metrics_from_frame(close, vol)
    b = funnel.metrics_from_frame(close, vol * 1e9)
    assert a["volume_ratio"] == pytest.approx(b["volume_ratio"])


# ------------------------------------------------------------- selection


def test_top_six_are_the_six_best_scores():
    order = list(reversed(UNIVERSE))  # DOGE-USD best ... EQ00 worst
    result = funnel.select(UNIVERSE, held=[], data=data_with_scores(order), top_n=6)
    assert result.candidates == order[:6]
    assert result.selected == order[:6]


def test_held_symbols_are_always_sent_and_never_use_a_candidate_slot():
    order = list(reversed(UNIVERSE))
    held = ["EQ00", "EQ01"]  # the two worst scorers
    result = funnel.select(UNIVERSE, held=held, data=data_with_scores(order), top_n=6)

    assert result.held == ["EQ00", "EQ01"]
    assert len(result.candidates) == 6
    assert not set(result.candidates) & set(held)
    assert len(result.selected) == 8
    # Held first, so a sell frees capital before the buys.
    assert result.selected[:2] == ["EQ00", "EQ01"]


def test_a_top_ranked_held_symbol_does_not_shrink_the_candidate_list():
    order = list(reversed(UNIVERSE))
    result = funnel.select(UNIVERSE, held=[order[0]], data=data_with_scores(order), top_n=6)
    assert result.candidates == order[1:7]
    assert len(result.selected) == 7


def test_a_held_symbol_outside_the_universe_is_still_sent():
    result = funnel.select(UNIVERSE, held=["INTC"], data=data_with_scores(UNIVERSE), top_n=6)
    assert "INTC" in result.selected
    assert len(result.selected) == 7


def test_eligibility_filters_candidates_but_never_held_symbols():
    """NYSE closed: only crypto may be a new candidate, a held equity is still reviewed."""
    result = funnel.select(
        UNIVERSE, held=["EQ05"], data=data_with_scores(UNIVERSE), top_n=6,
        eligible=lambda s: s.endswith("-USD"),
    )
    assert result.held == ["EQ05"]
    assert set(result.candidates) == set(CRYPTO)  # only 5 exist, so fewer than 6
    assert len(result.candidates) == 5


def test_symbols_without_data_rank_after_every_symbol_with_data():
    data = data_with_scores(["EQ20", "EQ21"])
    result = funnel.select(UNIVERSE, held=[], data=data, top_n=6)
    assert result.candidates[:2] == ["EQ20", "EQ21"]
    # Backfill is universe order (market-cap order), deterministic.
    assert result.candidates[2:] == ["EQ00", "EQ01", "EQ02", "EQ03"]


def test_no_data_at_all_falls_back_to_universe_order():
    result = funnel.select(UNIVERSE, held=[], data={}, top_n=6)
    assert result.candidates == UNIVERSE[:6]


def test_ranking_is_deterministic_on_ties():
    tied = {s: {"volume_ratio": 1.0, "move_z": 1.0, "last_return_pct": 0.0} for s in UNIVERSE}
    first = funnel.select(UNIVERSE, held=[], data=tied, top_n=6).candidates
    second = funnel.select(UNIVERSE, held=[], data=tied, top_n=6).candidates
    assert first == second == UNIVERSE[:6]


@pytest.mark.parametrize("config,expected", [
    ({}, 6), ({"funnel": {"top_n": 3}}, 3), ({"funnel": {"top_n": "x"}}, 6), ({"funnel": {"top_n": -1}}, 0),
])
def test_top_n_from_config(config, expected):
    assert funnel.top_n_from_config(config) == expected


# ------------------------------------------------------- cycle integration


def _config(tmp_logger, tmp_path, **extra):
    config = {
        "symbols": [{"symbol": s, "asset_class": "crypto" if s.endswith("-USD") else "equity"}
                    for s in UNIVERSE],
        "fallback_equity_usd": 1000.0,
        "db_path": tmp_logger.db_path,
        "csv_path": str(tmp_path / "signals.csv"),
        "positions_path": str(tmp_path / "positions.json"),
        "benchmark_path": str(tmp_path / "benchmark.json"),
        "live_execution": False,
    }
    config.update(extra)
    return config


@pytest.fixture
def processed(monkeypatch):
    seen = []
    monkeypatch.setattr(main, "_process_symbol", lambda *, symbol, **kw: seen.append(symbol))
    monkeypatch.setattr(main, "get_provider", lambda config: ("stub", None))
    # The end-of-cycle positions export prices holdings over the network; out of scope here.
    monkeypatch.setattr(main.position_metrics, "compute_position_metrics", lambda *a, **k: [])
    # The sweep prices ledger positions; a flat 100 crosses no stop/take here.
    monkeypatch.setattr(
        main.data_fetcher, "fetch_ohlcv",
        lambda symbol, period=None, interval="1d": pd.DataFrame({"Close": [100.0]}),
    )
    return seen


def test_scheduled_cycle_sends_only_held_plus_top_six(tmp_logger, tmp_path, monkeypatch, processed):
    order = list(reversed(UNIVERSE))
    monkeypatch.setattr(funnel, "fetch_funnel_data", lambda symbols: data_with_scores(order))
    tmp_logger.open_simulated_position("EQ00", qty=1.0, avg_entry_price=100.0)

    main._run_cycle_body(_config(tmp_logger, tmp_path), is_live=False)

    assert processed == ["EQ00"] + order[:6]


def test_scheduled_cycle_with_equity_market_closed_proposes_only_crypto(
    tmp_logger, tmp_path, monkeypatch, processed
):
    monkeypatch.setattr(execution, "_is_market_open", lambda: False)
    monkeypatch.setattr(funnel, "fetch_funnel_data", lambda symbols: data_with_scores(UNIVERSE))
    tmp_logger.open_simulated_position("EQ03", qty=1.0, avg_entry_price=100.0)

    main._run_cycle_body(_config(tmp_logger, tmp_path), is_live=False)

    assert processed == ["EQ03"] + CRYPTO


def test_live_cycle_reads_held_symbols_from_the_broker(tmp_logger, tmp_path, monkeypatch, processed):
    from models import ExistingPosition

    monkeypatch.setattr(execution, "read_live_equity", lambda: 5000.0)
    monkeypatch.setattr(
        execution, "fetch_all_live_positions",
        lambda: {"EQ24": ExistingPosition(qty=2.0, avg_entry_price=50.0)},
    )
    monkeypatch.setattr(funnel, "fetch_funnel_data", lambda symbols: data_with_scores(UNIVERSE))

    main._run_cycle_body(_config(tmp_logger, tmp_path), is_live=True)

    assert processed == ["EQ24"] + UNIVERSE[:6]


def test_live_cycle_survives_a_broker_outage_on_the_position_list(
    tmp_logger, tmp_path, monkeypatch, processed
):
    monkeypatch.setattr(execution, "read_live_equity", lambda: 5000.0)

    def boom():
        raise RuntimeError("Alpaca down")

    monkeypatch.setattr(execution, "fetch_all_live_positions", boom)
    monkeypatch.setattr(funnel, "fetch_funnel_data", lambda symbols: data_with_scores(UNIVERSE))

    main._run_cycle_body(_config(tmp_logger, tmp_path), is_live=True)

    assert processed == UNIVERSE[:6]


@pytest.mark.parametrize("reason", ["wake_buy", "wake_sell", "manual"])
def test_targeted_cycles_bypass_the_funnel(tmp_logger, tmp_path, monkeypatch, processed, reason):
    def must_not_run(symbols):
        raise AssertionError("funnel must not run for a targeted cycle")

    monkeypatch.setattr(funnel, "fetch_funnel_data", must_not_run)

    main._run_cycle_body(
        _config(tmp_logger, tmp_path), is_live=False,
        trigger_symbols=["DOGE-USD"], trigger_reason=reason,
    )

    assert processed == ["DOGE-USD"]


def test_funnel_can_be_switched_off(tmp_logger, tmp_path, monkeypatch, processed):
    main._run_cycle_body(_config(tmp_logger, tmp_path, funnel={"enabled": False}), is_live=False)
    assert processed == UNIVERSE
