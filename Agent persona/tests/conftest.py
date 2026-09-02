import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture
def digest():
    return {
        "timestamp": "2026-01-17T19:53:00Z",
        "news_digest": [
            {
                "news_id": "news_1", "time": "2026-01-17T19:45:00Z", "source": "MockWire",
                "headline": "CPI hotter-than-expected; yields jump",
                "summary": "Inflation surprised to the upside.",
                "tickers_mentioned": ["SPY", "TLT"],
                "macro_tags": ["CPI", "YIELDS"], "sentiment": "BEARISH", "confidence": 0.8,
            },
            {
                "news_id": "news_2", "time": "2026-01-17T19:49:00Z", "source": "MockWire",
                "headline": "Megacap tech earnings beat",
                "summary": "Margins beat and capex guidance raised.",
                "tickers_mentioned": ["QQQ"],
                "macro_tags": ["EARNINGS"], "sentiment": "BULLISH", "confidence": 0.7,
            },
        ],
    }


@pytest.fixture
def state():
    return {
        "cash_usd": 10000.0,
        "positions": {"QQQ": {"qty": 10, "avg_price": 390.0}},
        "prices": {"SPY": 520.0, "QQQ": 400.0, "TLT": 95.0, "GLD": 190.0},
    }
