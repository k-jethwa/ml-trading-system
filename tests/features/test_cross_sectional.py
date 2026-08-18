"""Synthetic known-answer tests for leave-one-out cross-sectional means."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltrading.features.cross_sectional import leave_one_out_mean


def test_leave_one_out_mean_basic():
    panel = pd.DataFrame({
        "date": ["d1", "d1", "d1"],
        "ticker": ["A", "B", "C"],
        "val": [10.0, 20.0, 30.0],
    })
    loo = leave_one_out_mean(panel, "val", ["date"])
    # A excludes itself: mean(20, 30) = 25. B: mean(10, 30) = 20. C: mean(10, 20) = 15.
    assert np.isclose(loo.iloc[0], 25.0)
    assert np.isclose(loo.iloc[1], 20.0)
    assert np.isclose(loo.iloc[2], 15.0)


def test_leave_one_out_mean_ignores_nan_and_excludes_correctly():
    panel = pd.DataFrame({
        "date": ["d1", "d1", "d1"],
        "ticker": ["A", "B", "C"],
        "val": [10.0, np.nan, 30.0],
    })
    loo = leave_one_out_mean(panel, "val", ["date"])
    # A excludes itself, B is NaN (ignored): mean(30) = 30.
    assert np.isclose(loo.iloc[0], 30.0)
    # B has its own NaN; the group excluding B is (10, 30) -> mean = 20.
    assert np.isclose(loo.iloc[1], 20.0)
    # C excludes itself; B is NaN and ignored: mean(10) = 10.
    assert np.isclose(loo.iloc[2], 10.0)


def test_leave_one_out_mean_undefined_for_singleton_group():
    panel = pd.DataFrame({"date": ["d1"], "ticker": ["A"], "val": [10.0]})
    loo = leave_one_out_mean(panel, "val", ["date"])
    assert pd.isna(loo.iloc[0])


def test_leave_one_out_mean_respects_multiple_group_columns():
    panel = pd.DataFrame({
        "date": ["d1", "d1", "d1", "d1"],
        "sector": ["S1", "S1", "S2", "S2"],
        "ticker": ["A", "B", "C", "D"],
        "val": [10.0, 20.0, 100.0, 200.0],
    })
    loo = leave_one_out_mean(panel, "val", ["date", "sector"])
    # A and B only see each other (S1); C and D only see each other (S2).
    assert np.isclose(loo.iloc[0], 20.0)  # A -> B
    assert np.isclose(loo.iloc[1], 10.0)  # B -> A
    assert np.isclose(loo.iloc[2], 200.0)  # C -> D
    assert np.isclose(loo.iloc[3], 100.0)  # D -> C
