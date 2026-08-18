"""Synthetic known-answer tests for the forward-return target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltrading.features.target import add_forward_returns, add_target


def _panel(ticker_prices: dict[str, list[float]], sector: dict[str, str]) -> pd.DataFrame:
    n = len(next(iter(ticker_prices.values())))
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = []
    for ticker, prices in ticker_prices.items():
        for date, price in zip(dates, prices):
            rows.append({"date": date, "ticker": ticker, "sector": sector[ticker], "adj_close": price})
    return pd.DataFrame(rows)


def test_forward_return_known_value():
    # 10 flat business days, single ticker, price rises from day 0.
    prices = [100.0, 101, 102, 103, 104, 110, 106, 107, 108, 109]
    panel = _panel({"A": prices}, {"A": "S1"})
    out = add_forward_returns(panel)
    # fwd_ret_5d at row 0: adj_close(row5)/adj_close(row0) - 1 = 110/100 - 1
    assert np.isclose(out["fwd_ret_5d"].iloc[0], 110 / 100 - 1)
    # Last 5 rows have no forward data -> NaN.
    assert pd.isna(out["fwd_ret_5d"].iloc[-1])


def test_target_is_forward_return_minus_leave_one_out_sector_mean():
    n = 8
    a_prices = [100.0] * n
    a_prices[5] = 110.0  # A's day-0 forward-5 return = +10%
    b_prices = [50.0] * n
    b_prices[5] = 55.0  # B's day-0 forward-5 return = +10% (same sector, same move)
    c_prices = [200.0] * n
    c_prices[5] = 220.0  # C's day-0 forward-5 return = +10% (same sector)

    panel = _panel({"A": a_prices, "B": b_prices, "C": c_prices}, {"A": "S1", "B": "S1", "C": "S1"})
    out = add_target(panel)

    a0 = out[(out["ticker"] == "A")].iloc[0]
    # A's forward return is +10%; sector leave-one-out mean (B, C) is
    # also +10% (they moved identically) -> target should be ~0.
    assert np.isclose(a0["fwd_ret_5d"], 0.10)
    assert np.isclose(a0["target_5d_sector_rel"], 0.0, atol=1e-9)


def test_target_positive_when_outperforming_sector():
    n = 8
    a_prices = [100.0] * n
    a_prices[5] = 120.0  # A: +20%
    b_prices = [50.0] * n
    b_prices[5] = 51.0  # B: +2%

    panel = _panel({"A": a_prices, "B": b_prices}, {"A": "S1", "B": "S1"})
    out = add_target(panel)

    a0 = out[(out["ticker"] == "A")].iloc[0]
    # A's sector benchmark (leave-one-out) is just B's return: +2%.
    assert np.isclose(a0["fwd_ret_5d"], 0.20)
    assert np.isclose(a0["target_5d_sector_rel"], 0.20 - 0.02)
