"""Universe assembly: top-25 by market cap from the screen, plus the 5 fixed crypto
pairs that symbols.yaml can never rotate out; and the shipped config itself."""

import yaml

import equity_universe
import main

FIXED_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]


def test_shipped_config_is_25_equities_plus_the_5_fixed_crypto():
    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    equities = [e["symbol"] for e in config["symbols"] if e["asset_class"] == "equity"]
    crypto = [e["symbol"] for e in config["symbols"] if e["asset_class"] == "crypto"]
    assert len(equities) == 25 and len(set(equities)) == 25
    assert crypto == FIXED_CRYPTO
    doge = config["symbol_overrides"]["DOGE-USD"]
    assert doge["max_absolute_position_pct"] == 10.0
    assert doge["wake_trigger"]["buy_side"]["price_move_pct_15m"] == 3.0
    assert config["funnel"]["top_n"] == 6


def test_screened_symbols_replace_equities_but_keep_fixed_crypto(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"symbols": (
        [{"symbol": "OLD", "asset_class": "equity"}]
        + [{"symbol": s, "asset_class": "crypto"} for s in FIXED_CRYPTO]
    )}), encoding="utf-8")
    (tmp_path / "symbols.yaml").write_text(yaml.safe_dump({"symbols": [
        {"symbol": "NEW1", "asset_class": "equity"}, {"symbol": "NEW2", "asset_class": "equity"},
    ]}), encoding="utf-8")

    config = main.load_config(str(config_path))

    assert [e["symbol"] for e in config["symbols"]] == ["NEW1", "NEW2"] + FIXED_CRYPTO


def test_without_symbols_yaml_the_config_list_stands(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"symbols": [{"symbol": "AAPL", "asset_class": "equity"}]}),
                           encoding="utf-8")
    assert main.load_config(str(config_path))["symbols"] == [{"symbol": "AAPL", "asset_class": "equity"}]


# ---------------------------------------------------------- market-cap ranking


def _rows(caps):
    return [{"symbol": s, "mcap": c} for s, c in caps.items()]


def test_rank_by_market_cap_orders_largest_first():
    rows = _rows({"A": 1.0, "B": 3.0, "C": 2.0})
    assert equity_universe.rank_by_market_cap(rows, 2, market_cap_lookup=lambda s: 0.0) == ["B", "C"]


def test_missing_market_caps_are_filled_from_the_lookup():
    rows = _rows({"A": 0.0, "B": 3.0, "C": 0.0})
    lookup = {"A": 5.0, "C": 1.0}.get
    assert equity_universe.rank_by_market_cap(rows, 3, market_cap_lookup=lookup) == ["A", "B", "C"]


def test_too_few_real_market_caps_publishes_nothing():
    """All-zero caps used to 'sort' into FMP's arbitrary row order; now it's a failure."""
    rows = _rows({"A": 0.0, "B": 0.0, "C": 0.0})
    assert equity_universe.rank_by_market_cap(rows, 2, market_cap_lookup=lambda s: 0.0) == []


def test_secondary_share_class_does_not_take_a_slot():
    rows = _rows({"GOOGL": 10.0, "GOOG": 9.9, "X": 1.0})
    assert equity_universe.rank_by_market_cap(rows, 2, market_cap_lookup=lambda s: 0.0) == ["GOOGL", "X"]


def test_target_universe_size_is_25():
    assert equity_universe.TARGET_UNIVERSE_SIZE == 25
