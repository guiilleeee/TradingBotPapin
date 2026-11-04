"""Universe assembly: top-29 by market cap from the screen, equities only (any crypto
entry is dropped by the loader); and the shipped config itself."""

import yaml

import equity_universe
import main

FIXED_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]


def test_shipped_config_is_29_equities_and_no_crypto():
    with open(main.DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    equities = [e["symbol"] for e in config["symbols"]]
    assert len(equities) == 29 and len(set(equities)) == 29
    assert all(e["asset_class"] == "equity" for e in config["symbols"])
    assert not any(s.endswith("-USD") for s in equities)
    assert "crypto" not in config["portfolio"]["groups"]
    assert not any(k.endswith("-USD") for k in (config.get("symbol_overrides") or {}))
    assert config["funnel"]["top_n"] == 6


def test_shipped_symbols_yaml_is_29_equities_and_no_crypto_key():
    with open("symbols.yaml", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    assert len(doc["symbols"]) == 29
    assert all(e["asset_class"] == "equity" for e in doc["symbols"])
    assert "crypto" not in doc.get("scores", {})


def test_loaded_universe_has_no_crypto():
    config = main.load_config(main.DEFAULT_CONFIG_PATH)
    names = [e["symbol"] for e in config["symbols"]]
    assert len(names) == 29
    assert not any(s.endswith("-USD") for s in names)


def test_crypto_entries_are_dropped_from_both_files(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"symbols": (
        [{"symbol": "OLD", "asset_class": "equity"}]
        + [{"symbol": s, "asset_class": "crypto"} for s in FIXED_CRYPTO]
    )}), encoding="utf-8")
    assert [e["symbol"] for e in main.load_config(str(config_path))["symbols"]] == ["OLD"]

    (tmp_path / "symbols.yaml").write_text(yaml.safe_dump({"symbols": [
        {"symbol": "NEW1", "asset_class": "equity"}, {"symbol": "BTC-USD", "asset_class": "crypto"},
    ]}), encoding="utf-8")
    assert [e["symbol"] for e in main.load_config(str(config_path))["symbols"]] == ["NEW1"]


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


def test_target_universe_size_is_29():
    assert equity_universe.TARGET_UNIVERSE_SIZE == 29


def test_symbols_yaml_pool_reaches_the_prefilter_and_nothing_else(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"max_risk_pct": 1.0, "symbols": [
        {"symbol": "OLD", "asset_class": "equity"}]}), encoding="utf-8")
    (tmp_path / "symbols.yaml").write_text(yaml.safe_dump({
        "symbols": [{"symbol": "NEW1", "asset_class": "equity"}],
        "pool": ["new1", "pool2", ""],
        "max_risk_pct": 50.0,
    }), encoding="utf-8")

    config = main.load_config(str(config_path))

    assert config["prefilter_pool"] == ["NEW1", "POOL2"]
    assert config["max_risk_pct"] == 1.0


def test_rank_by_market_cap_keep_all_returns_the_whole_ranking():
    rows = [{"symbol": s, "mcap": m} for s, m in [("A", 1.0), ("B", 3.0), ("C", 2.0)]]
    assert equity_universe.rank_by_market_cap(
        rows, 2, market_cap_lookup=lambda s: 0.0, keep_all=True) == ["B", "C", "A"]
    assert equity_universe.rank_by_market_cap(
        rows[:1], 2, market_cap_lookup=lambda s: 0.0, keep_all=True) == []
