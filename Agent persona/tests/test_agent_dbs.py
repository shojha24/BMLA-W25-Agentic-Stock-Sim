import pytest

from sim.actions_db import ActionsDB
from sim.assets_db import AssetsDB
from sim.portfolio import Portfolio


def fill(agent, ticker="QQQ", status="FILLED", side="BUY", notional=-1000.0):
    return {"agent_id": agent, "timestamp": "2016-01-04T20:00:00Z", "ticker": ticker,
            "side": side, "requested_qty": 10.0, "filled_qty": 10.0 if status != "REJECTED" else 0.0,
            "price": 100.0, "notional": notional if status != "REJECTED" else 0.0,
            "cost": 0.1, "status": status, "reason": "" if status == "FILLED" else "insufficient cash"}


@pytest.fixture
def actions(tmp_path):
    db = ActionsDB(tmp_path / "actions.sqlite")
    db.record("r1", 1, "2016-01-04", [fill("macro"), fill("quant", "TLT", "REJECTED")],
              rationales={"macro|QQQ": "risk-on"})
    db.record("r1", 2, "2016-01-05", [fill("macro", "GLD"), fill("quant", "QQQ")])
    return db


def test_last_trades_are_newest_first_and_agent_scoped(actions):
    rows = actions.last_trades("r1", "macro")
    assert [r["ticker"] for r in rows] == ["GLD", "QQQ"]
    assert all(r["agent_id"] == "macro" for r in rows)


def test_rejected_orders_are_hidden_from_last_trades_by_default(actions):
    assert actions.last_trades("r1", "quant") == [] or \
        all(r["status"] != "REJECTED" for r in actions.last_trades("r1", "quant"))
    assert any(r["status"] == "REJECTED" for r in actions.last_trades("r1", "quant", filled_only=False))


def test_peer_actions_exclude_the_asking_agent(actions):
    rows = actions.peer_actions("r1", "macro")
    assert rows and all(r["agent_id"] != "macro" for r in rows)


def test_peer_actions_can_be_scoped_to_one_cycle(actions):
    assert all(r["cycle"] == 1 for r in actions.peer_actions("r1", "macro", cycle=1))


def test_rationale_is_stored_with_the_action(actions):
    row = [r for r in actions.last_trades("r1", "macro", limit=5) if r["ticker"] == "QQQ"][0]
    assert row["rationale"] == "risk-on"


def test_trades_on_day_feeds_the_reflection_pipeline(actions):
    assert [r["ticker"] for r in actions.trades_on_day("r1", "macro", "2016-01-04")] == ["QQQ"]


def test_summary_does_not_count_rejected_notional(actions):
    summary = actions.summary("r1")
    assert summary["quant"]["by_status"]["REJECTED"] == 1
    assert summary["quant"]["traded_notional"] == 1000.0     # only the filled one


def test_runs_are_isolated_from_each_other(actions):
    assert actions.last_trades("other_run", "macro") == []


def test_assets_db_snapshots_and_reloads_a_book(tmp_path):
    db = AssetsDB(tmp_path / "assets.sqlite")
    book = Portfolio(10000.0)
    book.apply_fill("QQQ", 10, 100.0, 0.0)
    db.snapshot("r1", 1, "T1", "macro", book, {"QQQ": 100.0})
    book.apply_fill("QQQ", 5, 110.0, 0.0)
    db.snapshot("r1", 2, "T2", "macro", book, {"QQQ": 110.0})

    latest = db.latest("r1", "macro")
    assert latest["cycle"] == 2
    assert latest["positions"]["QQQ"]["qty"] == 15
    assert len(db.equity_curve("r1", "macro")) == 2


def test_assets_are_private_per_agent(tmp_path):
    db = AssetsDB(tmp_path / "assets.sqlite")
    db.snapshot("r1", 1, "T", "macro", Portfolio(10000.0), {})
    assert db.latest("r1", "quant") is None
