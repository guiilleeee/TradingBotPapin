"""FMP integration, Nasdaq-list fallback, yfinance batch pricing, and equity
scoring -- all offline.

No test here touches the network. FMP calls are mocked with response shapes
documented on FMP's own doc pages (verified via search, since every FMP
endpoint -- including a request that would just prove a path exists -- requires
a real key, and none is available in this environment); the Nasdaq-list
fallback is exercised against payloads shaped like a live
api.nasdaq.com/api/quote/list-type/nasdaq100 response, not the live site. `fetch_universe_price_data`'s shape
(MultiIndex `(symbol, field)` columns from `group_by="ticker"`, an all-NaN
column for a delisted-shaped symbol, an empty-list call raising inside
pandas) was verified live against yfinance 1.7.0 and the real S&P 500 list
before writing these fakes -- see equity_universe.py's docstring.
"""

import pandas as pd
import pytest
import requests

import equity_universe


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(equity_universe.requests, "get", boom)


@pytest.fixture(autouse=True)
def fmp_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # The real code paces calls with a small delay; tests should not pay for it.
    monkeypatch.setattr(equity_universe.time, "sleep", lambda *_: None)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise equity_universe.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    @property
    def text(self):
        return self._payload


def mock_get(monkeypatch, by_path):
    """route requests.get(url, ...) to a canned response keyed by URL suffix.

    A value is a payload (HTTP 200), a FakeResponse, or an exception to raise.
    """

    def fake_get(url, params=None, timeout=None, headers=None):
        for suffix, payload in by_path.items():
            if url.endswith(suffix):
                if isinstance(payload, Exception):
                    raise payload
                return payload if isinstance(payload, FakeResponse) else FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(equity_universe.requests, "get", fake_get)


# ------------------------------------------------------- nasdaq-100 fallback


class RestrictedResponse(FakeResponse):
    """FMP's 402 on a plan that doesn't include the endpoint -- with the key in
    the URL, exactly as a real requests.HTTPError would carry it."""

    def __init__(self):
        super().__init__({"Error Message": "Restricted Endpoint"}, status=402)

    def raise_for_status(self):
        raise requests.exceptions.HTTPError(
            "402 Client Error: Payment Required for url: "
            "https://financialmodelingprep.com/stable/nasdaq-constituent?apikey=test-key"
        )


FMP_RESTRICTED = {
    "/stable/nasdaq-constituent": RestrictedResponse(),
    "/api/v3/nasdaq_constituent": RestrictedResponse(),
}


def nasdaq_payload(rows):
    """The shape api.nasdaq.com/api/quote/list-type/nasdaq100 returns (verified live)."""
    return {"data": {"totalrecords": len(rows), "data": {"rows": rows}}}


def nasdaq_rows(n=100):
    # SYM0 is the largest; market caps formatted the way Nasdaq sends them.
    return [
        {"symbol": f"SYM{i}", "sector": "", "marketCap": f"{(n - i) * 1_000_000_000:,}"}
        for i in range(n)
    ]


@pytest.fixture
def no_yfinance(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("market caps were all provided; yfinance must not be hit")

    monkeypatch.setattr(equity_universe.yf, "Ticker", explode)


def test_fmp_restricted_falls_back_to_nasdaq_list(monkeypatch, capsys, no_yfinance):
    mock_get(monkeypatch, {**FMP_RESTRICTED, "nasdaq100": nasdaq_payload(nasdaq_rows())})

    top = equity_universe.fetch_nasdaq100_top(25)

    assert top == [f"SYM{i}" for i in range(25)]
    out = capsys.readouterr().out
    # Each FMP failure is logged with its real reason, and the key never leaks.
    assert "/stable/nasdaq-constituent failed" in out
    assert "/api/v3/nasdaq_constituent failed" in out
    assert "402" in out
    assert "test-key" not in out


def test_fmp_implausibly_large_list_falls_back_instead_of_giving_up(monkeypatch, capsys, no_yfinance):
    huge = [{"symbol": f"X{i}", "marketCap": 1e9} for i in range(500)]
    mock_get(monkeypatch, {
        "/stable/nasdaq-constituent": huge,
        "/api/v3/nasdaq_constituent": huge,
        "nasdaq100": nasdaq_payload(nasdaq_rows()),
    })

    assert equity_universe.fetch_nasdaq100_top(25)[0] == "SYM0"
    assert "500 rows" in capsys.readouterr().out


def test_missing_fmp_key_still_gets_a_universe_from_nasdaq(monkeypatch, capsys, no_yfinance):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    mock_get(monkeypatch, {"nasdaq100": nasdaq_payload(nasdaq_rows())})

    assert len(equity_universe.build_equity_pool()) == 100
    assert "FMP_API_KEY is not set" in capsys.readouterr().out


def test_every_source_failing_returns_empty_without_raising(monkeypatch, capsys):
    mock_get(monkeypatch, {**FMP_RESTRICTED, "nasdaq100": ConnectionError("nasdaq.com is down")})

    assert equity_universe.fetch_nasdaq100_top(25) == []
    out = capsys.readouterr().out
    assert "402" in out
    assert "nasdaq.com failed: nasdaq.com is down" in out
    assert "every source failed" in out


def test_nasdaq_http_error_also_degrades_to_empty(monkeypatch, capsys):
    mock_get(monkeypatch, {**FMP_RESTRICTED, "nasdaq100": FakeResponse({}, status=403)})
    assert equity_universe.build_equity_universe() == []
    assert "nasdaq.com failed: HTTP 403" in capsys.readouterr().out


def test_nasdaq_parser_normalizes_dedupes_and_parses_market_cap(monkeypatch):
    rows = nasdaq_rows(98) + [
        {"symbol": " brk.b ", "marketCap": "1,000"},
        {"symbol": "BRK/B", "marketCap": "2,000"},  # same issuer, other spelling
        {"symbol": "NOCAP", "marketCap": "NA"},
        {"symbol": "", "marketCap": "5"},
    ]
    mock_get(monkeypatch, {"nasdaq100": nasdaq_payload(rows)})

    by_symbol = {r["symbol"]: r["mcap"] for r in equity_universe._fetch_nasdaq100_from_nasdaq_api()}

    assert by_symbol["BRK-B"] == 1000.0  # first spelling wins
    assert "BRK/B" not in by_symbol and "brk.b" not in by_symbol
    assert by_symbol["NOCAP"] == 0.0  # left for rank_by_market_cap's yfinance fill
    assert by_symbol["SYM0"] == 98_000_000_000.0
    assert "" not in by_symbol
    assert len(by_symbol) == 100


@pytest.mark.parametrize("payload", [
    {"data": None},                      # shape changed
    {"data": {"data": {"rows": "x"}}},   # rows not a list
    nasdaq_payload([]),                  # empty
    nasdaq_payload(nasdaq_rows(40)),     # partial list -> would be a wrong top 25
])
def test_nasdaq_parser_raises_rather_than_return_a_wrong_universe(monkeypatch, payload):
    mock_get(monkeypatch, {"nasdaq100": payload})
    with pytest.raises(RuntimeError, match="nasdaq.com"):
        equity_universe._fetch_nasdaq100_from_nasdaq_api()


# ------------------------------------------------------------------- api key


def test_missing_api_key_raises_a_named_error(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(equity_universe.FMPError, match="FMP_API_KEY"):
        equity_universe._get("/stable/sp-500")


def test_fmp_exception_never_carries_the_literal_api_key_however_it_fails(monkeypatch):
    """FMP's key travels as a query-string parameter, unlike every other
    credential this project holds -- a real requests.HTTPError's default
    string form embeds the full request URL, key included. Whatever _get
    raises must never carry it, wherever that exception is eventually printed
    or logged (now, or after some future change to a caller).
    """
    fake_key = "fmp-real-secret-key-abc123"
    monkeypatch.setenv("FMP_API_KEY", fake_key)

    class ExplodingResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "500 Server Error: Internal Server Error for url: "
                f"https://financialmodelingprep.com/stable/sp-500?apikey={fake_key}"
            )

    monkeypatch.setattr(equity_universe.requests, "get", lambda *a, **kw: ExplodingResponse())

    with pytest.raises(equity_universe.FMPError) as excinfo:
        equity_universe._get("/stable/sp-500")

    message = str(excinfo.value)
    assert fake_key not in message
    assert "REDACTED" in message


def fake_price_frame(data):
    """{symbol: {"Close": [...], "Volume": [...]}} -> a MultiIndex DataFrame
    shaped exactly like `yf.download(..., group_by="ticker")`'s real return
    value (verified live -- see equity_universe.py's docstring): top-level
    columns are symbols, second level is OHLCV field.
    """
    frames = {symbol: pd.DataFrame(cols) for symbol, cols in data.items()}
    return pd.concat(frames, axis=1)


def test_price_data_computes_change_from_the_last_two_closes_and_latest_volume(monkeypatch):
    frame = fake_price_frame({
        "AAPL": {"Close": [100.0, 110.0], "Volume": [1_000_000.0, 2_000_000.0]},
    })
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL"])

    assert data["AAPL"]["price_change_pct"] == pytest.approx(10.0)
    assert data["AAPL"]["volume"] == pytest.approx(2_000_000.0)


def test_price_data_never_calls_download_for_an_empty_symbol_list(monkeypatch):
    # yf.download([]) raises inside pandas.concat rather than returning an
    # empty frame -- confirmed live -- so an empty list must short-circuit
    # before ever reaching yf.download at all.
    def explode(*a, **kw):
        raise AssertionError("must not call yf.download with no symbols")

    monkeypatch.setattr(equity_universe.yf, "download", explode)
    assert equity_universe.fetch_universe_price_data([]) == {}


def test_price_data_drops_a_symbol_with_all_nan_data(monkeypatch):
    # The real shape of a delisted/bad ticker mixed into a batch download:
    # its columns exist but are entirely NaN (confirmed live).
    import numpy as np

    frame = fake_price_frame({
        "AAPL": {"Close": [100.0, 110.0], "Volume": [1_000_000.0, 2_000_000.0]},
        "DEADTICKER": {"Close": [np.nan, np.nan], "Volume": [np.nan, np.nan]},
    })
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL", "DEADTICKER"])

    assert "AAPL" in data
    assert "DEADTICKER" not in data


def test_price_data_drops_a_symbol_missing_from_the_result_entirely(monkeypatch):
    frame = fake_price_frame({"AAPL": {"Close": [100.0, 110.0], "Volume": [1e6, 2e6]}})
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL", "NOTINRESULT"])

    assert "NOTINRESULT" not in data


def test_price_data_needs_at_least_two_trading_days(monkeypatch):
    frame = fake_price_frame({"AAPL": {"Close": [100.0], "Volume": [1e6]}})
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_degrades_to_empty_on_a_download_exception(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("yahoo is down")

    monkeypatch.setattr(equity_universe.yf, "download", boom)
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_returns_empty_on_a_completely_empty_frame(monkeypatch):
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: pd.DataFrame())
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_is_a_single_batched_call_not_one_per_symbol(monkeypatch):
    calls = []

    def fake_download(symbols, **kw):
        calls.append(list(symbols))
        return fake_price_frame({s: {"Close": [100.0, 101.0], "Volume": [1e6, 1e6]} for s in symbols})

    monkeypatch.setattr(equity_universe.yf, "download", fake_download)
    symbols = [f"SYM{i}" for i in range(50)]
    equity_universe.fetch_universe_price_data(symbols)

    assert len(calls) == 1
    assert calls[0] == symbols


def test_percentile_ranks_of_an_empty_input():
    assert equity_universe.percentile_ranks({}) == {}


def test_percentile_ranks_span_zero_to_one():
    ranks = equity_universe.percentile_ranks({"A": 10.0, "B": 30.0, "C": 20.0})
    assert ranks["A"] == 0.0
    assert ranks["C"] == 0.5
    assert ranks["B"] == 1.0


def test_percentile_rank_of_a_single_value_is_one():
    assert equity_universe.percentile_ranks({"A": 5.0}) == {"A": 1.0}


# --------------------------------------------------------------------- scoring


def price_data(**per_symbol):
    """{"AAPL": (change_pct, volume), ...} -> the price_data shape score_equities takes."""
    return {
        symbol: {"price_change_pct": change, "volume": volume}
        for symbol, (change, volume) in per_symbol.items()
    }


def test_volume_below_the_floor_is_not_counted_towards_volume_signal():
    universe = {"AAPL"}
    data = price_data(AAPL=(0.0, 1.0))  # real momentum (flat), but a tiny volume
    scored = equity_universe.score_equities(universe, data)
    assert scored[0]["volume"] is None  # excluded by the floor
    # Still has_signal, though: a real (if zero) price change is itself signal
    # now -- this is the core behavioural change over the old FMP-list gate.
    assert scored[0]["has_signal"] is True


def test_a_symbol_outside_the_universe_is_ignored():
    universe = {"AAPL"}
    data = price_data(TSLA=(5.0, 1e8))
    scored = equity_universe.score_equities(universe, data)
    assert len(scored) == 1
    assert scored[0]["symbol"] == "AAPL"
    assert scored[0]["has_signal"] is False


def test_a_symbol_with_no_price_data_at_all_has_no_signal():
    # The one case that should now be rare (a delisted/failed ticker in the
    # batch, not "wasn't on someone else's movers list").
    universe = {"AAPL", "MSFT"}
    data = price_data(AAPL=(1.2, 5e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["has_signal"] is True
    assert by_symbol["MSFT"]["has_signal"] is False


def test_volume_and_momentum_both_contribute():
    universe = {"AAPL", "MSFT", "GOOG"}
    data = price_data(AAPL=(0.5, 5e7), MSFT=(0.5, 1e7), GOOG=(9.0, 5e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["has_signal"] is True
    assert by_symbol["MSFT"]["has_signal"] is True
    assert by_symbol["GOOG"]["has_signal"] is True
    # AAPL and MSFT have identical (small) momentum; AAPL's higher volume rank
    # must be what puts it ahead, proving the volume half of the blend counts.
    assert by_symbol["AAPL"]["score"] > by_symbol["MSFT"]["score"]


def test_a_loser_contributes_momentum_by_magnitude_not_sign():
    universe = {"AAPL", "MSFT"}
    data = price_data(AAPL=(5.0, 1e7), MSFT=(-5.0, 1e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["score"] == by_symbol["MSFT"]["score"]


def test_every_symbol_with_real_data_carries_momentum_even_without_a_big_move():
    # The core fix: momentum is no longer gated on "was this on a whole-market
    # gainers/losers list" -- any real price change, however small, is a real
    # relative-momentum data point once percentile-ranked against the universe.
    universe = {"AAPL", "MSFT", "GOOG"}
    data = price_data(AAPL=(0.01, 1e7), MSFT=(0.02, 1e7), GOOG=(0.03, 1e7))
    scored = equity_universe.score_equities(universe, data)
    assert all(r["has_signal"] for r in scored)
    assert all(r["momentum_pct"] is not None for r in scored)


def test_scoring_is_deterministic_across_repeated_calls():
    universe = {"AAPL", "MSFT", "GOOG", "TSLA"}
    data = price_data(AAPL=(1.0, 5e7), MSFT=(-2.0, 3e7), GOOG=(0.5, 2e7), TSLA=(4.0, 9e7))
    first = equity_universe.score_equities(universe, data)
    second = equity_universe.score_equities(universe, data)
    assert [r["symbol"] for r in first] == [r["symbol"] for r in second]


# --------------------------------------------------------------- selection


def test_select_top_prefers_signal_bearing_candidates():
    scored = [
        {"symbol": "NOSIGNAL", "score": 0.0, "has_signal": False},
        {"symbol": "SIGNAL", "score": 0.3, "has_signal": True},
    ]
    assert equity_universe.select_top_equities(scored, 1) == ["SIGNAL"]


def test_select_top_backfills_deterministically_when_short_on_signal():
    # Only one symbol has real signal this week; the rest must still be filled,
    # not left short of the requested count.
    scored = [
        {"symbol": "A", "score": 0.0, "has_signal": False},
        {"symbol": "B", "score": 0.9, "has_signal": True},
        {"symbol": "C", "score": 0.0, "has_signal": False},
    ]
    result = equity_universe.select_top_equities(scored, 3)
    assert len(result) == 3
    assert result[0] == "B"
    assert set(result) == {"A", "B", "C"}


def test_select_top_never_exceeds_the_pool_size():
    scored = [{"symbol": "A", "score": 0.0, "has_signal": False}]
    assert equity_universe.select_top_equities(scored, 5) == ["A"]
