"""The single-cycle inspector must keep running the same pipeline as the simulator.

It drifted once already: after briefs and orders were added to the engine,
main.py was still running the old raw-digest path.
"""
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
MAIN = SRC / "main.py"


def run_main(*args, timeout=180):
    result = subprocess.run(
        [sys.executable, str(MAIN), "--mode", "mock", "--no-rag", *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_one_cycle_runs_the_whole_pipeline():
    out = run_main("--task", "roundtable")
    for key in ("segment", "historical_context", "consensus", "agent_outputs",
                "fills", "execution", "books"):
        assert key in out, f"main.py no longer produces {key}"


def test_agents_are_given_briefs_and_their_own_books():
    out = run_main("--task", "roundtable")
    assert out["books"]                       # one book per agent
    assert all(o["checks"].get("brief") for o in out["agent_outputs"])


def test_the_brief_can_be_switched_off():
    out = run_main("--task", "roundtable", "--no-briefs")
    assert all(not o["checks"].get("brief") for o in out["agent_outputs"])


def test_execution_can_be_skipped():
    out = run_main("--task", "roundtable", "--no-execute")
    assert out["fills"] == []


def test_panel_mode_runs_a_single_round():
    out = run_main("--task", "panel")
    assert all(o["checks"]["round"] == 1 for o in out["agent_outputs"])


def test_rag_status_reports_a_mode():
    result = subprocess.run([sys.executable, str(MAIN), "--task", "rag-status"],
                            capture_output=True, text=True, timeout=180,
                            cwd=str(Path(__file__).resolve().parents[2]))
    assert result.returncode == 0
    assert "mode" in json.loads(result.stdout)
