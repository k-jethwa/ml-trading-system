"""Chronological walk-forward folds (project_spec.md section 8).

    Train: all history up to the boundary  ->  Test: the next calendar year

The train window EXPANDS (never shuffled, never looks past the test
start) and the model is retrained once per test year.

Purge / embargo. The label at date t is the return from t to t+5
trading days, so a training row dated within 5 trading days of the test
start has a label window that overlaps test-period prices. Training on
it would let the model "see" returns that occur in the test window.
Each fold therefore drops the last `embargo_days` trading dates before
the test start from its training set (tests/models/test_walk_forward.py
asserts no training label window reaches the test period).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_end: pd.Timestamp   # last date whose row may be used for training
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return dates <= self.train_end

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.test_start) & (dates <= self.test_end)


def make_folds(dates: pd.Series, first_test_year: int, embargo_days: int) -> list[Fold]:
    unique = np.sort(dates.unique())
    unique = pd.DatetimeIndex(unique)
    folds: list[Fold] = []
    for year in range(first_test_year, unique.max().year + 1):
        in_year = unique[unique.year == year]
        if len(in_year) == 0:
            continue
        test_start, test_end = in_year.min(), in_year.max()
        start_idx = unique.get_loc(test_start)
        train_end_idx = start_idx - 1 - embargo_days
        if train_end_idx < 0:
            raise ValueError(f"Not enough history before {test_start.date()} for a {embargo_days}-day embargo")
        folds.append(Fold(year, unique[train_end_idx], test_start, test_end))
    return folds
