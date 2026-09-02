"""Daily price data with an on-disk cache.

Uses the public Yahoo Finance chart endpoint: no API key, no extra dependency.
Everything downstream asks two questions - "what did this close at?" and "what
happened next?" - so those are the only two things this exposes.
"""
from __future__ import annotations

import csv
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; agentic-stock-sim/1.0)"}


def _epoch(day: str) -> int:
    return int(datetime.strptime(day[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


class PriceStore:
    def __init__(self, cache_dir: Optional[Path] = None, request_pause_s: float = 0.4):
        root = Path(__file__).resolve().parents[3]
        self.cache_dir = Path(cache_dir or root / "dataset" / "prices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_pause_s = request_pause_s
        self._series: Dict[str, Dict[str, float]] = {}
        self._days: Dict[str, List[str]] = {}
        self._intraday: Dict[tuple, Dict[str, float]] = {}
        self._intraday_stamps: Dict[tuple, List[str]] = {}

    # ---------------- fetch / cache ----------------

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.csv"

    def download(self, ticker: str, start: str, end: str) -> int:
        """Fetch daily adjusted closes into the cache. Returns rows written."""
        params = {
            "period1": _epoch(start),
            "period2": _epoch(end) + 86400,
            "interval": "1d",
            "includeAdjustedClose": "true",
        }
        resp = requests.get(CHART_URL.format(ticker=ticker.upper()), params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        result = (resp.json().get("chart") or {}).get("result")
        if not result:
            raise RuntimeError(f"No price data returned for {ticker}")
        res = result[0]
        stamps = res.get("timestamp") or []
        quote = (res.get("indicators", {}).get("quote") or [{}])[0]
        adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
        closes = quote.get("close") or []

        rows = []
        for i, ts in enumerate(stamps):
            close = adj[i] if i < len(adj) and adj[i] is not None else (
                closes[i] if i < len(closes) else None)
            if close is None:
                continue
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            rows.append((day, float(close)))

        path = self._cache_path(ticker)
        existing: Dict[str, float] = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                existing = {r["date"]: float(r["close"]) for r in csv.DictReader(f)}
        existing.update(dict(rows))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "close"])
            for day in sorted(existing):
                w.writerow([day, existing[day]])
        self._series.pop(ticker.upper(), None)
        self._days.pop(ticker.upper(), None)
        time.sleep(self.request_pause_s)   # be polite to a free endpoint
        return len(rows)

    def ensure(self, tickers: Sequence[str], start: str, end: str, refresh: bool = False,
               strict: bool = False) -> Dict[str, int]:
        """Make sure the cache covers [start, end]. Returns rows fetched per ticker,
        or -1 for a ticker that could not be fetched (unknown symbol, endpoint down).
        A dead ticker is reported, not raised, so one bad symbol cannot abort a run."""
        out: Dict[str, int] = {}
        for ticker in tickers:
            path = self._cache_path(ticker)
            if path.exists() and not refresh:
                series = self.load(ticker)
                if series and min(series) <= start[:10] and max(series) >= end[:10]:
                    out[ticker.upper()] = 0        # cache already covers the window
                    continue
            try:
                out[ticker.upper()] = self.download(ticker, start, end)
            except Exception:
                if strict:
                    raise
                out[ticker.upper()] = -1
        return out

    def has_data(self, ticker: str) -> bool:
        return bool(self.load(ticker))

    # ---------------- lookups ----------------

    def load(self, ticker: str) -> Dict[str, float]:
        key = ticker.upper()
        if key in self._series:
            return self._series[key]
        path = self._cache_path(key)
        if not path.exists():
            self._series[key] = {}
            self._days[key] = []
            return {}
        with open(path, newline="", encoding="utf-8") as f:
            series = {r["date"]: float(r["close"]) for r in csv.DictReader(f) if r.get("close")}
        self._series[key] = series
        self._days[key] = sorted(series)
        return series

    def sessions(self, ticker: str, start: str = "", end: str = "") -> List[str]:
        """Trading days available for a ticker, inclusive of both bounds."""
        self.load(ticker)
        days = self._days.get(ticker.upper(), [])
        lo = bisect_left(days, start[:10]) if start else 0
        hi = bisect_right(days, end[:10]) if end else len(days)
        return days[lo:hi]

    def close_on(self, ticker: str, day: str) -> Optional[float]:
        """Close for `day`, or the most recent close before it (last known price)."""
        series = self.load(ticker)
        if not series:
            return None
        day = day[:10]
        if day in series:
            return series[day]
        days = self._days[ticker.upper()]
        i = bisect_right(days, day) - 1
        return series[days[i]] if i >= 0 else None

    def next_session(self, ticker: str, day: str, steps: int = 1) -> Optional[str]:
        self.load(ticker)
        days = self._days.get(ticker.upper(), [])
        i = bisect_right(days, day[:10]) - 1
        j = i + steps
        return days[j] if 0 <= j < len(days) else None

    def forward_return_bps(self, ticker: str, day: str, horizon_days: int = 1) -> Optional[float]:
        """Close-to-close return in basis points over the next `horizon_days` sessions.

        This is the realized value every forecast is scored against.
        """
        start_close = self.close_on(ticker, day)
        end_day = self.next_session(ticker, day, horizon_days)
        if start_close is None or end_day is None or start_close <= 0:
            return None
        end_close = self.load(ticker).get(end_day)
        if end_close is None:
            return None
        return (end_close / start_close - 1.0) * 10000.0

    # ---------------- intraday ----------------

    def _intraday_path(self, ticker: str, interval: str) -> Path:
        directory = self.cache_dir / "intraday"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{ticker.upper()}_{interval}.csv"

    def download_intraday(self, ticker: str, interval: str = "15m", lookback: str = "60d") -> int:
        """Fetch intraday bars. Yahoo serves 15m bars for the last ~60 days only,
        which is the hard ceiling on how far back an intraday run can reach."""
        resp = requests.get(CHART_URL.format(ticker=ticker.upper()),
                            params={"interval": interval, "range": lookback},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        result = (resp.json().get("chart") or {}).get("result")
        if not result:
            raise RuntimeError(f"No intraday data returned for {ticker}")
        res = result[0]
        stamps = res.get("timestamp") or []
        closes = (res.get("indicators", {}).get("quote") or [{}])[0].get("close") or []

        rows = {}
        for i, ts in enumerate(stamps):
            close = closes[i] if i < len(closes) else None
            if close is None:
                continue
            when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows[when] = float(close)

        path = self._intraday_path(ticker, interval)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                existing = {r["timestamp"]: float(r["close"]) for r in csv.DictReader(f)}
            existing.update(rows)
            rows = existing
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "close"])
            for when in sorted(rows):
                w.writerow([when, rows[when]])
        self._intraday.pop((ticker.upper(), interval), None)
        time.sleep(self.request_pause_s)
        return len(rows)

    def ensure_intraday(self, tickers: Sequence[str], interval: str = "15m",
                        lookback: str = "60d", refresh: bool = False) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ticker in tickers:
            if self._intraday_path(ticker, interval).exists() and not refresh:
                out[ticker.upper()] = 0
                continue
            try:
                out[ticker.upper()] = self.download_intraday(ticker, interval, lookback)
            except Exception:
                out[ticker.upper()] = -1
        return out

    def load_intraday(self, ticker: str, interval: str = "15m") -> Dict[str, float]:
        key = (ticker.upper(), interval)
        if key in self._intraday:
            return self._intraday[key]
        path = self._intraday_path(ticker, interval)
        if not path.exists():
            self._intraday[key] = {}
            self._intraday_stamps[key] = []
            return {}
        with open(path, newline="", encoding="utf-8") as f:
            series = {r["timestamp"]: float(r["close"]) for r in csv.DictReader(f) if r.get("close")}
        self._intraday[key] = series
        self._intraday_stamps[key] = sorted(series)
        return series

    def price_at(self, ticker: str, timestamp: str, interval: str = "15m") -> Optional[float]:
        """Last intraday print at or before `timestamp`."""
        series = self.load_intraday(ticker, interval)
        if not series:
            return None
        stamps = self._intraday_stamps[(ticker.upper(), interval)]
        i = bisect_right(stamps, timestamp) - 1
        return series[stamps[i]] if i >= 0 else None

    def forward_return_bps_intraday(self, ticker: str, timestamp: str, bars: int = 1,
                                    interval: str = "15m") -> Optional[float]:
        series = self.load_intraday(ticker, interval)
        if not series:
            return None
        stamps = self._intraday_stamps[(ticker.upper(), interval)]
        i = bisect_right(stamps, timestamp) - 1
        if i < 0 or i + bars >= len(stamps):
            return None
        start, end = series[stamps[i]], series[stamps[i + bars]]
        if start <= 0:
            return None
        return (end / start - 1.0) * 10000.0

    def prices_at(self, tickers: Sequence[str], timestamp: str,
                  interval: str = "15m") -> Dict[str, float]:
        out = {}
        for ticker in tickers:
            px = self.price_at(ticker, timestamp, interval)
            if px is not None:
                out[ticker.upper()] = round(px, 4)
        return out

    def prices_on(self, tickers: Sequence[str], day: str) -> Dict[str, float]:
        out = {}
        for t in tickers:
            px = self.close_on(t, day)
            if px is not None:
                out[t.upper()] = round(px, 4)
        return out


HORIZON_DAYS = {"1d": 1, "2d": 2, "5d": 5, "1w": 5, "1m": 21}
# Intraday horizons, expressed in 15-minute bars.
HORIZON_BARS = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16}


def is_intraday(horizon: str) -> bool:
    return str(horizon).lower() in HORIZON_BARS


def horizon_to_sessions(horizon: str) -> int:
    """Daily sessions a horizon spans. Intraday horizons settle inside one session."""
    return HORIZON_DAYS.get(str(horizon).lower(), 1)


def horizon_to_bars(horizon: str, bar_minutes: int = 15) -> int:
    bars = HORIZON_BARS.get(str(horizon).lower())
    if bars is None:
        return max(1, (horizon_to_sessions(horizon) * 390) // bar_minutes)
    return max(1, (bars * 15) // bar_minutes)
