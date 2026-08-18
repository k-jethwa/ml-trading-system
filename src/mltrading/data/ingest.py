"""Raw market-data ingestion from yfinance.

This module pulls daily OHLCV bars for a fixed ticker universe and writes
them to Parquet under data/raw/. It is intentionally the first piece of
the pipeline in project_spec.md section 11 ("Market-data ingestion and
validation"): every downstream feature, model, and backtest depends on
this data carrying honest timestamp semantics.

Point-in-time discipline applied here:
  - Every row is stamped with `ingested_at`, the UTC time this ingestion
    run pulled the data, plus `source` and `universe_version`. This makes
    later replay able to answer "what did we have, and when did we get
    it" (project_spec.md section 5).
  - `adj_close` is yfinance's dividend/split-adjusted close. Because
    yfinance retroactively adjusts historical adjusted-close values when
    a new split/dividend occurs, `adj_close` for a past date is NOT
    guaranteed identical across two ingestion runs on different days.
    Re-running ingestion overwrites rather than appends, and each run's
    `ingested_at` records which version of history was captured.
  - Sector is fetched once per ticker from current metadata and is NOT
    point-in-time (see universe.py docstring for the limitation).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from mltrading.data.universe import INITIAL_UNIVERSE, UNIVERSE_VERSION

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
SOURCE_NAME = "yfinance"

REQUIRED_COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
    "sector", "source", "universe_version", "ingested_at",
]


def fetch_ticker_sector(ticker: str) -> str | None:
    """Best-effort current sector lookup. Returns None on failure.

    Not point-in-time: see universe.py docstring.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector")
    except Exception:
        logger.warning("Could not fetch sector for %s", ticker, exc_info=True)
        return None


def download_daily_bars(
    tickers: list[str],
    start: str,
    end: str | None = None,
    ingested_at: datetime | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV bars for `tickers` between `start` and `end`.

    Returns a tidy long-format DataFrame with one row per (date, ticker)
    and the columns in REQUIRED_COLUMNS. Does not write to disk.
    """
    if ingested_at is None:
        ingested_at = datetime.now(UTC)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {len(tickers)} tickers from {start} to {end}")

    frames = []
    sector_cache: dict[str, str | None] = {}

    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
        except KeyError:
            logger.warning("No data returned for ticker %s; skipping", ticker)
            continue

        df = df.dropna(how="all")
        if df.empty:
            logger.warning("Empty history for ticker %s; skipping", ticker)
            continue

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume",
        })
        df = df.reset_index().rename(columns={"Date": "date"})
        df["ticker"] = ticker

        if ticker not in sector_cache:
            sector_cache[ticker] = fetch_ticker_sector(ticker)
        df["sector"] = sector_cache[ticker]

        df["source"] = SOURCE_NAME
        df["universe_version"] = UNIVERSE_VERSION
        df["ingested_at"] = ingested_at

        frames.append(df[REQUIRED_COLUMNS])

    if not frames:
        raise ValueError("No usable data for any requested ticker")

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def write_raw_parquet(df: pd.DataFrame, out_dir: Path = RAW_DATA_DIR) -> Path:
    """Write ingested bars to Parquet, partitioned by ticker.

    Each ticker gets its own file so a single bad/rate-limited ticker
    can be re-ingested without touching the rest of the universe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in df.groupby("ticker"):
        path = out_dir / f"{ticker}.parquet"
        group.to_parquet(path, index=False)
        logger.info("Wrote %d rows for %s to %s", len(group), ticker, path)
    return out_dir


def run(start: str = "2015-01-01", end: str | None = None) -> Path:
    """Ingest the full INITIAL_UNIVERSE and persist it to data/raw/."""
    logging.basicConfig(level=logging.INFO)
    df = download_daily_bars(INITIAL_UNIVERSE, start=start, end=end)
    return write_raw_parquet(df)


if __name__ == "__main__":
    run()
