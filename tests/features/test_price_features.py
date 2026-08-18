"""Synthetic known-answer tests for per-ticker price features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltrading.features.price_features import add_price_features


def _ohlcv(closes, volumes=None, highs=None, lows=None):
    n = len(closes)
    volumes = volumes or [1_000_000] * n
    dates = pd.bdate_range("2024-01-02", periods=n)
    closes = pd.Series(closes, dtype=float)
    highs = pd.Series(highs, dtype=float) if highs is not None else closes * 1.02
    lows = pd.Series(lows, dtype=float) if lows is not None else closes * 0.98
    return pd.DataFrame({
        "date": dates, "ticker": "TEST", "sector": "S1",
        "open": closes, "high": highs.values, "low": lows.values,
        "close": closes, "adj_close": closes, "volume": volumes,
    })


def test_ret_1d_and_ret_5d_known_values():
    # 26 rows so ret_20d has one valid value too.
    closes = [100.0 + i for i in range(26)]  # 100, 101, ..., 125
    df = add_price_features(_ohlcv(closes))

    # ret_1d at row 5 (close=105) vs row 4 (close=104): 105/104 - 1
    assert np.isclose(df["ret_1d"].iloc[5], 105 / 104 - 1)
    # ret_5d at row 10 (close=110) vs row 5 (close=105): 110/105 - 1
    assert np.isclose(df["ret_5d"].iloc[10], 110 / 105 - 1)
    # Warm-up: ret_1d undefined for the very first row.
    assert pd.isna(df["ret_1d"].iloc[0])
    # ret_20d undefined until row 20 (needs 20 prior rows).
    assert pd.isna(df["ret_20d"].iloc[19])
    assert np.isclose(df["ret_20d"].iloc[20], 120 / 100 - 1)


def test_dist_from_moving_average():
    closes = [100.0] * 19 + [200.0]  # 19 flat rows then a jump
    df = add_price_features(_ohlcv(closes))
    # 20-day MA of the last row = (19*100 + 200) / 20 = 105
    expected_ma20 = (19 * 100 + 200) / 20
    assert np.isclose(df["dist_ma_20"].iloc[19], 200 / expected_ma20 - 1)


def test_range_position_bounds():
    # Strictly increasing highs/lows/closes over 20 rows: the last row's
    # close is the highest high and highest low seen -> range position = 1.
    closes = list(range(100, 120))
    df = add_price_features(_ohlcv(closes, highs=[c * 1.0 for c in closes], lows=[c * 1.0 for c in closes]))
    assert np.isclose(df["range_position_20"].iloc[19], 1.0)


def test_volume_ratio_known_value():
    closes = [100.0] * 21
    volumes = [1_000_000] * 20 + [3_000_000]
    df = add_price_features(_ohlcv(closes, volumes=volumes))
    # The 20-day rolling average is inclusive of today (pandas rolling()
    # semantics), so at row 20 it covers rows [1..20]: 19 days of 1M plus
    # today's 3M -> (19_000_000 + 3_000_000) / 20 = 1,100,000.
    expected_avg = (19 * 1_000_000 + 3_000_000) / 20
    assert np.isclose(df["avg_volume_20"].iloc[20], expected_avg)
    assert np.isclose(df["volume_ratio_20"].iloc[20], 3_000_000 / expected_avg)
