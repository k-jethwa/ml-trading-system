"""Synthetic known-answer tests for raw-data validation checks."""

from __future__ import annotations

import pandas as pd

from mltrading.data.validate import validate_ticker


def _bar(date, close, volume=1_000_000, open_=None, high=None, low=None):
    open_ = open_ if open_ is not None else close
    high = high if high is not None else close
    low = low if low is not None else close
    return {
        "date": pd.Timestamp(date), "open": open_, "high": high, "low": low,
        "close": close, "adj_close": close, "volume": volume,
    }


def test_clean_series_has_no_findings():
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame([
        _bar("2024-01-02", 100.0),
        _bar("2024-01-03", 101.0),
        _bar("2024-01-04", 102.0),
    ])
    report = validate_ticker(df, "TEST", calendar)
    assert report.is_clean
    assert report.missing_trading_days == []
    assert report.n_rows == 3


def test_detects_missing_trading_day():
    # Universe traded on 01-02, 01-03, 01-04; this ticker skips 01-03.
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame([
        _bar("2024-01-02", 100.0),
        _bar("2024-01-04", 102.0),
    ])
    report = validate_ticker(df, "TEST", calendar)
    assert report.missing_trading_days == [pd.Timestamp("2024-01-03")]
    assert not report.is_clean


def test_detects_large_jump():
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame([
        _bar("2024-01-02", 100.0),
        _bar("2024-01-03", 200.0),  # +100% in one day
    ])
    report = validate_ticker(df, "TEST", calendar)
    assert report.large_jump_dates == [pd.Timestamp("2024-01-03")]
    assert not report.is_clean


def test_detects_stale_run_and_duplicate_and_non_positive():
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    df = pd.DataFrame([
        _bar("2024-01-02", 100.0),
        _bar("2024-01-03", 100.0),
        _bar("2024-01-04", 100.0),
        _bar("2024-01-05", 100.0),
        _bar("2024-01-08", 100.0),  # 5 identical closes in a row -> stale
        _bar("2024-01-09", -5.0),   # non-positive price
        _bar("2024-01-09", 101.0),  # duplicate date
    ])
    report = validate_ticker(df, "TEST", calendar)
    assert report.max_stale_run == 5
    assert report.non_positive_price_rows == 1
    assert report.duplicate_dates == 1
    assert not report.is_clean
