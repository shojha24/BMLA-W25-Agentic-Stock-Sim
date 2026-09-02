"""Public DB: every action every agent took.

Public on purpose - this is what lets an agent be shown what the others did
(and what it did last cycle) in a later brief, and what makes a run auditable
after the fact. One row per order, including the ones the venue refused.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.types import Fill

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    cycle         INTEGER NOT NULL,
    timestamp     TEXT NOT NULL,
    day           TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL,
    requested_qty REAL NOT NULL,
    filled_qty    REAL NOT NULL,
    price         REAL NOT NULL,
    notional      REAL NOT NULL,
    cost          REAL NOT NULL,
    status        TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    rationale     TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_actions_agent ON actions(run_id, agent_id, id);
CREATE INDEX IF NOT EXISTS idx_actions_cycle ON actions(run_id, cycle);
"""


class ActionsDB:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def record(self, run_id: str, cycle: int, day: str, fills: Sequence[Fill],
               rationales: Optional[Dict[str, str]] = None) -> int:
        rationales = rationales or {}
        rows = [
            (run_id, cycle, f["timestamp"], day, f["agent_id"], f["ticker"], f["side"],
             f["requested_qty"], f["filled_qty"], f["price"], f["notional"], f["cost"],
             f["status"], f["reason"], rationales.get(f"{f['agent_id']}|{f['ticker']}", ""))
            for f in fills
        ]
        if not rows:
            return 0
        conn = self._conn()
        conn.executemany(
            "INSERT INTO actions (run_id, cycle, timestamp, day, agent_id, ticker, side, "
            "requested_qty, filled_qty, price, notional, cost, status, reason, rationale) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        return len(rows)

    # ---------------- reads ----------------

    @staticmethod
    def _rows(cursor) -> List[Dict[str, Any]]:
        return [dict(r) for r in cursor.fetchall()]

    def last_trades(self, run_id: str, agent_id: str, limit: int = 5,
                    filled_only: bool = True) -> List[Dict[str, Any]]:
        """Most recent first - the 'Your last trades' line of the brief."""
        query = ("SELECT * FROM actions WHERE run_id=? AND agent_id=? "
                 + ("AND status != 'REJECTED' " if filled_only else "")
                 + "ORDER BY id DESC LIMIT ?")
        return self._rows(self._conn().execute(query, (run_id, agent_id, limit)))

    def peer_actions(self, run_id: str, exclude_agent: str, cycle: Optional[int] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """What everyone else did - the public half of the store."""
        if cycle is None:
            return self._rows(self._conn().execute(
                "SELECT * FROM actions WHERE run_id=? AND agent_id != ? ORDER BY id DESC LIMIT ?",
                (run_id, exclude_agent, limit)))
        return self._rows(self._conn().execute(
            "SELECT * FROM actions WHERE run_id=? AND agent_id != ? AND cycle=? ORDER BY id DESC LIMIT ?",
            (run_id, exclude_agent, cycle, limit)))

    def trades_on_day(self, run_id: str, agent_id: str, day: str) -> List[Dict[str, Any]]:
        """A day's trades for one agent - the input to the reflection pipeline."""
        return self._rows(self._conn().execute(
            "SELECT * FROM actions WHERE run_id=? AND agent_id=? AND day=? ORDER BY id",
            (run_id, agent_id, day)))

    def summary(self, run_id: str) -> Dict[str, Any]:
        rows = self._rows(self._conn().execute(
            "SELECT agent_id, status, COUNT(*) n, "
            "SUM(CASE WHEN status != 'REJECTED' THEN ABS(notional) ELSE 0 END) notional "
            "FROM actions WHERE run_id=? GROUP BY agent_id, status", (run_id,)))
        out: Dict[str, Any] = {}
        for row in rows:
            agent = out.setdefault(row["agent_id"], {"orders": 0, "by_status": {}, "traded_notional": 0.0})
            agent["orders"] += row["n"]
            agent["by_status"][row["status"]] = row["n"]
            agent["traded_notional"] = round(agent["traded_notional"] + (row["notional"] or 0.0), 2)
        return out
