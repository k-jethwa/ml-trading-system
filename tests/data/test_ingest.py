"""Synthetic known-answer tests for raw ingestion.

No network calls: yf.download and yf.Ticker are mocked with a small,
hand-built dataset so the expected output is known exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from mltrading.data.ingest import REQUIRED_COLUMNS, download_daily_bars


def _fake_yf_download(*args, **kwargs):
    """Mimic yfinance's multi-ticker, group_by='ticker' return shape."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    dates.name = "Date"
    columns = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    data = [
        [100.0, 101.0, 99.0, 100.5, 100.5, 1_000_000, 200.0, 201.0, 199.0, 200.5, 200.5, 2_000_000],
        [100.5, 102.0, 100.0, 101.5, 101.5, 1_100_000, 200.5, 203.0, 200.0, 202.5, 202.5, 2_100_000],
    ]
    return pd.DataFrame(data, index=dates, columns=columns)


class _FakeTickerInfo:
    def __init__(self, sector: str):
        self.info = {"sector": sector}


def _fake_ticker(symbol: str):
    return _FakeTickerInfo({"AAPL": "Technology", "MSFT": "Technology"}[symbol])


@patch("mltrading.data.ingest.yf.Ticker", side_effect=_fake_ticker)
@patch("mltrading.data.ingest.yf.download", side_effect=_fake_yf_download)
def test_download_daily_bars_schema_and_values(mock_download, mock_ticker):
    fixed_ingest_time = datetime(2024, 1, 4, 12, 0, tzinfo=UTC)

    df = download_daily_bars(
        tickers=["AAPL", "MSFT"],
        start="2024-01-02",
        end="2024-01-04",
        ingested_at=fixed_ingest_time,
    )

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 4  # 2 tickers x 2 dates
    assert set(df["ticker"]) == {"AAPL", "MSFT"}

    aapl_first = df[(df["ticker"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert aapl_first["open"] == 100.0
    assert aapl_first["close"] == 100.5
    assert aapl_first["volume"] == 1_000_000
    assert aapl_first["sector"] == "Technology"
    assert aapl_first["source"] == "yfinance"
    assert aapl_first["ingested_at"] == fixed_ingest_time

    # Point-in-time provenance columns must be present on every row.
    assert df["ingested_at"].nunique() == 1
    assert (df["source"] == "yfinance").all()


@patch("mltrading.data.ingest.yf.download")
def test_download_daily_bars_raises_on_empty_response(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="no data"):
        download_daily_bars(["AAPL"], start="2024-01-02")
