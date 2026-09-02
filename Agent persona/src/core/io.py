from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p


def append_jsonl(path: str | Path, record: Any) -> Path:
    """One line per simulation cycle - the run log the evaluator reads."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def load_env(extra_paths: Optional[List[Path]] = None) -> Dict[str, bool]:
    """Load .env from the repo root and from 'Agent persona/'.

    Tolerates the legacy format where the file held a bare API key with no
    'VAR=' prefix, which python-dotenv silently ignores.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore

    root = Path(__file__).resolve().parents[3]
    paths = [root / ".env", root / "Agent persona" / ".env"] + list(extra_paths or [])

    seen: Dict[str, bool] = {}
    for path in paths:
        if not path.exists():
            seen[str(path)] = False
            continue
        if load_dotenv is not None:
            load_dotenv(path, override=False)
        _load_bare_key(path)
        seen[str(path)] = True
    return seen


def _load_bare_key(path: Path) -> None:
    """Rescue a .env whose only line is a naked key value."""
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return
    for line in lines:
        if not line or line.startswith("#") or "=" in line:
            continue
        if line.startswith("sk-or-") and not os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = line
        elif line.startswith("AIza") and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = line
