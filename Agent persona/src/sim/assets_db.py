"""Private DB: each agent's own book.

The whiteboard's `Ag. 1: {"Liq.": ..., "AAPL": ...}` - one row per agent per
cycle, so an agent can be shown its own balance and history without seeing
anyone else's, and so a run can be resumed or replayed.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim.portfolio import Portfolio

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_assets (
    run_id     TEXT NOT NULL,
    cycle      INTEGER NOT NULL,
    timestamp  TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    cash       REAL NOT NULL,
    equity     REAL NOT NULL,
    positions  TEXT NOT NULL,
    PRIMARY KEY (run_id, cycle, agent_id)
);
"""


class AssetsDB:
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

    def snapshot(self, run_id: str, cycle: int, timestamp: str, agent_id: str,
                 book: Portfolio, prices: Dict[str, float]) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO agent_assets "
            "(run_id, cycle, timestamp, agent_id, cash, equity, positions) VALUES (?,?,?,?,?,?,?)",
            (run_id, cycle, timestamp, agent_id, round(book.cash, 2),
             round(book.equity(prices), 2), json.dumps(book.positions)))
        conn.commit()

    def latest(self, run_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM agent_assets WHERE run_id=? AND agent_id=? ORDER BY cycle DESC LIMIT 1",
            (run_id, agent_id)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["positions"] = json.loads(out["positions"])
        return out

    def history(self, run_id: str, agent_id: str) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT cycle, timestamp, cash, equity FROM agent_assets "
            "WHERE run_id=? AND agent_id=? ORDER BY cycle", (run_id, agent_id)).fetchall()
        return [dict(r) for r in rows]

    def equity_curve(self, run_id: str, agent_id: str) -> List[float]:
        return [r["equity"] for r in self.history(run_id, agent_id)]
