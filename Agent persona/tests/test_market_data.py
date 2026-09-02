import pytest

from data.market_data import PriceStore, horizon_to_sessions


@pytest.fixture
def store(tmp_path):
    """A price store backed by hand-written cache files - no network."""
    s = PriceStore(cache_dir=tmp_path)
    (tmp_path / "AAA.csv").write_text(
        "date,close\n2016-01-04,100.0\n2016-01-05,101.0\n2016-01-06,99.0\n2016-01-08,103.0\n"
    )
    return s


def test_close_on_falls_back_to_the_last_known_price(store):
    assert store.close_on("AAA", "2016-01-05") == 101.0
    assert store.close_on("AAA", "2016-01-07") == 99.0     # holiday: last close carries
    assert store.close_on("AAA", "2015-12-31") is None     # before history starts


def test_forward_return_is_close_to_close_in_bps(store):
    assert store.forward_return_bps("AAA", "2016-01-04", 1) == pytest.approx(100.0)
    assert store.forward_return_bps("AAA", "2016-01-05", 1) == pytest.approx(-198.02, abs=0.1)


def test_forward_return_is_none_past_the_end_of_history(store):
    assert store.forward_return_bps("AAA", "2016-01-08", 1) is None


def test_sessions_are_bounded_inclusively(store):
    assert store.sessions("AAA", "2016-01-05", "2016-01-06") == ["2016-01-05", "2016-01-06"]
    assert len(store.sessions("AAA")) == 4


def test_next_session_walks_both_directions(store):
    assert store.next_session("AAA", "2016-01-05", 1) == "2016-01-06"
    assert store.next_session("AAA", "2016-01-05", -1) == "2016-01-04"
    assert store.next_session("AAA", "2016-01-04", -1) is None


def test_horizon_mapping():
    assert horizon_to_sessions("1d") == 1
    assert horizon_to_sessions("5d") == 5
    assert horizon_to_sessions("nonsense") == 1


@pytest.fixture
def intraday(tmp_path):
    store = PriceStore(cache_dir=tmp_path)
    directory = tmp_path / "intraday"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "AAA_15m.csv").write_text(
        "timestamp,close\n"
        "2026-09-02T15:00:00Z,100.0\n2026-09-02T15:15:00Z,101.0\n2026-09-02T15:30:00Z,99.0\n"
    )
    return store


def test_price_at_uses_the_last_bar_at_or_before_the_stamp(intraday):
    assert intraday.price_at("AAA", "2026-09-02T15:20:00Z") == 101.0
    assert intraday.price_at("AAA", "2026-09-02T14:00:00Z") is None


def test_intraday_forward_return_is_measured_in_bars(intraday):
    assert intraday.forward_return_bps_intraday("AAA", "2026-09-02T15:00:00Z", 1) == pytest.approx(100.0)
    assert intraday.forward_return_bps_intraday("AAA", "2026-09-02T15:15:00Z", 1) == pytest.approx(-198.02, abs=0.1)


def test_intraday_return_is_none_past_the_last_bar(intraday):
    assert intraday.forward_return_bps_intraday("AAA", "2026-09-02T15:30:00Z", 1) is None


def test_intraday_horizons_are_recognised():
    from data.market_data import horizon_to_bars, is_intraday
    assert is_intraday("15m") and not is_intraday("1d")
    assert horizon_to_bars("15m") == 1 and horizon_to_bars("1h") == 4
