import pytest

from sim.execution import ExecutionConfig, MarketFillVenue, summarize_fills
from sim.portfolio import Portfolio

PRICES = {"QQQ": 100.0, "TLT": 50.0}


def order(side, ticker, qty, order_type="MARKET", limit_price=None):
    return {"ticker": ticker, "side": side, "qty": qty, "order_type": order_type,
            "limit_price": limit_price, "rationale": "", "news_refs": []}


def venue(**kw):
    """Costs and caps off unless a test is specifically about one of them."""
    kw.setdefault("max_position_pct_equity", 1.0)
    cfg = ExecutionConfig(cost_bps=0.0, slippage_bps=0.0, max_order_pct_equity=1.0, **kw)
    v = MarketFillVenue(cfg)
    v.start_cycle()
    return v


def run(v, orders, book=None):
    book = book or Portfolio(10000.0)
    fills = v.execute({"a": orders}, PRICES, {"a": book}, "T")
    return fills, book


def test_a_valid_buy_fills_and_moves_cash_and_shares():
    (fill,), book = run(venue(), [order("BUY", "QQQ", 10)])
    assert fill["status"] == "FILLED"
    assert book.positions["QQQ"]["qty"] == 10
    assert book.cash == pytest.approx(9000.0)


def test_buying_more_than_the_cash_allows_is_a_partial_fill():
    """Half the book is already in shares, so cash binds before the equity cap."""
    book = Portfolio(5000.0)
    book.apply_fill("QQQ", 50, 100.0, 0.0)     # equity 10k, cash 0... top it back up
    book.cash = 5000.0
    # position cap off, so cash is the only binding constraint
    (fill,), book = run(venue(max_position_pct_equity=0.0), [order("BUY", "QQQ", 500)], book=book)
    assert fill["status"] == "PARTIAL"
    assert fill["filled_qty"] == 50            # $5,000 of cash at $100
    assert fill["reason"] == "insufficient cash"
    assert book.cash >= 0


def test_the_tighter_of_cash_and_the_equity_cap_wins():
    (fill,), _ = run(venue(), [order("BUY", "QQQ", 500)])
    assert fill["status"] == "PARTIAL"
    assert fill["filled_qty"] == 100           # 100% equity cap and cash coincide here


def test_selling_shares_you_do_not_hold_is_rejected():
    (fill,), book = run(venue(), [order("SELL", "TLT", 5)])
    assert fill["status"] == "REJECTED"
    assert "insufficient shares" in fill["reason"]
    assert book.positions == {}


def test_shorting_is_allowed_when_configured():
    (fill,), book = run(venue(allow_short=True), [order("SELL", "TLT", 5)])
    assert fill["status"] == "FILLED"
    assert book.positions["TLT"]["qty"] == -5


def test_repeated_buys_cannot_exceed_the_position_limit():
    """Per-order caps alone let an agent build a whole-book position one slice at a time."""
    v = venue(max_position_pct_equity=0.3)
    book = Portfolio(10000.0)
    v.execute({"a": [order("BUY", "QQQ", 20)]}, PRICES, {"a": book}, "T")   # 20% of equity
    (second,) = v.execute({"a": [order("BUY", "QQQ", 20)]}, PRICES, {"a": book}, "T")
    assert second["status"] == "PARTIAL"
    assert "position capped" in second["reason"]
    assert book.positions["QQQ"]["qty"] == pytest.approx(30.0)


def test_the_position_limit_does_not_block_selling():
    v = venue(max_position_pct_equity=0.1)
    book = Portfolio(10000.0)
    book.apply_fill("QQQ", 50, 100.0, 0.0)
    (fill,) = v.execute({"a": [order("SELL", "QQQ", 50)]}, PRICES, {"a": book}, "T")
    assert fill["status"] == "FILLED"


def test_orders_are_capped_at_a_share_of_equity():
    v = MarketFillVenue(ExecutionConfig(cost_bps=0.0, slippage_bps=0.0,
                                        max_order_pct_equity=0.25,
                                        max_position_pct_equity=1.0))
    v.start_cycle()
    (fill,), _ = run(v, [order("BUY", "QQQ", 100)])
    assert fill["filled_qty"] == 25          # 25% of 10k equity at $100
    assert "capped" in fill["reason"]


def test_slippage_is_charged_against_the_taker():
    v = MarketFillVenue(ExecutionConfig(cost_bps=0.0, slippage_bps=100.0,
                                        max_order_pct_equity=1.0,
                                        max_position_pct_equity=1.0))
    v.start_cycle()
    buy_fills = v.execute({"a": [order("BUY", "QQQ", 1)]}, PRICES, {"a": Portfolio(10000.0)}, "T")
    seller = Portfolio(10000.0)
    seller.apply_fill("QQQ", 10, 100.0, 0.0)
    sell_fills = v.execute({"a": [order("SELL", "QQQ", 1)]}, PRICES, {"a": seller}, "T")
    assert buy_fills[0]["price"] == pytest.approx(101.0)
    assert sell_fills[0]["price"] == pytest.approx(99.0)


def test_commission_is_charged_on_notional():
    v = MarketFillVenue(ExecutionConfig(cost_bps=10.0, slippage_bps=0.0,
                                        max_order_pct_equity=1.0,
                                        max_position_pct_equity=1.0))
    v.start_cycle()
    (fill,), book = run(v, [order("BUY", "QQQ", 10)])
    assert fill["cost"] == pytest.approx(1.0)
    assert book.cash == pytest.approx(8999.0)


def test_cooldown_blocks_a_second_trade_in_the_same_ticker():
    v = MarketFillVenue(ExecutionConfig(cost_bps=0.0, slippage_bps=0.0,
                                        max_order_pct_equity=1.0,
                                        max_position_pct_equity=1.0, cooldown_cycles=2))
    book = Portfolio(10000.0)
    v.start_cycle()
    first = v.execute({"a": [order("BUY", "QQQ", 5)]}, PRICES, {"a": book}, "T")
    v.start_cycle()
    second = v.execute({"a": [order("BUY", "QQQ", 5)]}, PRICES, {"a": book}, "T")
    assert first[0]["status"] == "FILLED"
    assert second[0]["status"] == "REJECTED"
    assert "cooldown" in second[0]["reason"]


def test_a_limit_order_away_from_the_market_does_not_fill():
    (fill,), _ = run(venue(), [order("BUY", "QQQ", 5, "LIMIT", 90.0)])
    assert fill["status"] == "REJECTED"
    assert "limit not marketable" in fill["reason"]


def test_a_marketable_limit_order_fills():
    (fill,), _ = run(venue(), [order("BUY", "QQQ", 5, "LIMIT", 110.0)])
    assert fill["status"] == "FILLED"


def test_unknown_ticker_is_rejected_with_a_reason():
    (fill,), _ = run(venue(), [order("BUY", "ZZZ", 5)])
    assert fill["status"] == "REJECTED"
    assert fill["reason"] == "no price for ticker"


def test_fractional_shares_round_down_to_whole():
    (fill,), _ = run(venue(), [order("BUY", "QQQ", 5.7)])
    assert fill["filled_qty"] == 5


def test_agents_trade_their_own_books_only():
    v = venue()
    a, b = Portfolio(10000.0), Portfolio(10000.0)
    v.execute({"a": [order("BUY", "QQQ", 10)]}, PRICES, {"a": a, "b": b}, "T")
    assert a.positions and not b.positions
    assert b.cash == 10000.0


def test_summary_counts_statuses_and_reasons():
    v = venue()
    fills = v.execute({"a": [order("BUY", "QQQ", 10), order("SELL", "TLT", 1)]},
                      PRICES, {"a": Portfolio(10000.0)}, "T")
    summary = summarize_fills(fills)
    assert summary["n_orders"] == 2
    assert summary["fill_rate"] == 0.5
    assert summary["by_status"]["REJECTED"] == 1


def test_an_order_missing_optional_fields_still_executes():
    """Hand-built orders that skip normalization must not crash the venue."""
    bare = {"side": "BUY", "ticker": "QQQ", "qty": 5, "rationale": "", "news_refs": []}
    fills, book = run(venue(), [bare])
    assert fills[0]["status"] == "FILLED"
    assert book.positions["QQQ"]["qty"] == 5


def test_junk_entries_are_ignored_rather_than_raised():
    fills, _ = run(venue(), [{"nonsense": True}, order("BUY", "QQQ", 1)])
    assert len(fills) == 1 and fills[0]["status"] == "FILLED"
