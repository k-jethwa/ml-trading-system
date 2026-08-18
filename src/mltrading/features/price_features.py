"""Per-ticker, point-in-time price and volume features (V1 feature set).

Every feature here is computed from a single ticker's own OHLCV history
using a rolling or trailing window that ends at row t. None of them can
see a row after t, by construction — see
tests/features/test_no_lookahead.py for an automated proof of this using
a synthetic panel where future prices are perturbed and past feature
values are checked to be unaffected.

Momentum and volatility statistics use `adj_close` (dividend- and
split-adjusted) so they behave like a total-return series. The
range-based volatility and technical-state features use raw
high/low/close, since Yahoo does not provide a dividend-adjusted
intraday range; splits are already reflected in the raw series (no
false single-day jump was observed around any known historical split in
this universe — see docs/data_validation_report.md).

Every feature documents: definition, lookback, dependencies,
availability time, and rationale (project_spec.md section 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given one ticker's OHLCV history (sorted by date ascending),
    return a copy with all V1 price/volume feature columns appended.

    Availability: every column here is computable immediately after the
    close on date `date` using only that ticker's own history through
    `date`.
    """
    out = df.copy()
    px = out["adj_close"]

    # --- Momentum ---
    # ret_Nd: definition = pct change of adj_close over N trading days;
    # lookback = N days; dependencies = adj_close; rationale = short- and
    # medium-term trend persistence is the core cross-sectional momentum
    # effect this system is testing (project_spec.md section 2).
    out["ret_1d"] = px.pct_change(1)
    out["ret_5d"] = px.pct_change(5)
    out["ret_20d"] = px.pct_change(20)
    out["ret_60d"] = px.pct_change(60)

    # mom_12_1: classic Jegadeesh-Titman "relative strength" momentum —
    # price 1 month ago vs. price 12 months ago, deliberately skipping
    # the most recent month because very recent returns tend to
    # short-term reverse rather than continue. lookback = 252 trading
    # days; dependencies = adj_close.
    out["mom_12_1"] = px.shift(21) / px.shift(252) - 1

    # --- Distance from moving averages ---
    # dist_ma_N: (price / N-day simple moving average) - 1. lookback = N
    # days; rationale = a cheap, standard trend-following signal —
    # positive when price is above its recent average.
    ma20 = px.rolling(20, min_periods=20).mean()
    ma50 = px.rolling(50, min_periods=50).mean()
    out["dist_ma_20"] = px / ma20 - 1
    out["dist_ma_50"] = px / ma50 - 1

    # --- Volatility ---
    daily_ret = px.pct_change(1)
    # realized_vol_20: annualized rolling std of daily returns. lookback
    # = 20 days; rationale = simple realized-volatility estimate, used
    # both as a feature and, later, as a risk-model input.
    out["realized_vol_20"] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # ewma_vol_20: exponentially-weighted volatility (span=20), so recent
    # days count more than a flat 20-day window — reacts faster to
    # volatility regime changes.
    out["ewma_vol_20"] = daily_ret.ewm(span=20, min_periods=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # range_vol_20: rolling mean of the daily (high-low)/close range, a
    # simplified Parkinson-style range volatility estimate. Uses raw
    # high/low/close (see module docstring).
    daily_range = (out["high"] - out["low"]) / out["close"]
    out["range_vol_20"] = daily_range.rolling(20, min_periods=20).mean()

    # --- Liquidity ---
    # dollar_volume_20: rolling mean of close * volume. rationale =
    # eligibility/tradability proxy — how much capital can realistically
    # move through this name per day.
    dollar_vol = out["close"] * out["volume"]
    out["dollar_volume_20"] = dollar_vol.rolling(20, min_periods=20).mean()
    # avg_volume_20: rolling mean share volume.
    avg_vol_20 = out["volume"].rolling(20, min_periods=20).mean()
    out["avg_volume_20"] = avg_vol_20
    # volume_ratio_20: today's volume relative to its own 20-day average
    # — spikes indicate unusual interest/news flow.
    out["volume_ratio_20"] = out["volume"] / avg_vol_20

    # --- Simple technical state ---
    # range_position_20: where today's close sits within the trailing
    # 20-day high/low range, bounded [0, 1]. rationale = cheap,
    # interpretable overbought/oversold proxy.
    roll_min_20 = out["low"].rolling(20, min_periods=20).min()
    roll_max_20 = out["high"].rolling(20, min_periods=20).max()
    out["range_position_20"] = (out["close"] - roll_min_20) / (roll_max_20 - roll_min_20)

    return out
