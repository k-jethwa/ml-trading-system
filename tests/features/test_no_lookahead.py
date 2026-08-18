"""The core anti-leakage check for the feature engine.

project_spec.md section 5 requires that every feature and universe
decision have explicit timestamp semantics, and section 13 names
look-ahead bias as an invalidating failure mode. This test makes that
guarantee mechanical rather than aspirational: it builds a synthetic
panel, mutates prices strictly AFTER a chosen date, rebuilds the feature
table, and asserts

  1. every FEATURE_COLUMNS value strictly before the mutation date is
     bit-for-bit identical between the two runs (no feature can see the
     future), and
  2. the target column DOES change for rows whose forward window
     reaches into the mutated region (proving the target computation
     really does use future data, as a label correctly should — this
     half of the test would also catch a bug where the target was
     accidentally computed as backward-looking).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltrading.features.build import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_table

N_DAYS = 320
MUTATE_FROM = 300
MUTATION_FACTOR = 5.0
FORWARD_HORIZON = 5


def _make_synthetic_panel(mutate_from: int | None = None, mutation_factor: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=N_DAYS)
    tickers = {"AAA": "S1", "BBB": "S1", "CCC": "S2", "DDD": "S2"}
    base_price = {"AAA": 100.0, "BBB": 150.0, "CCC": 50.0, "DDD": 80.0}
    slope = {"AAA": 0.15, "BBB": -0.05, "CCC": 0.30, "DDD": 0.02}

    rows = []
    for ticker, sector in tickers.items():
        for idx, date in enumerate(dates):
            price = base_price[ticker] + slope[ticker] * idx
            if mutate_from is not None and idx >= mutate_from:
                price = price * mutation_factor
            rows.append({
                "date": date, "ticker": ticker, "sector": sector,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "adj_close": price, "volume": 1_000_000 + idx * 100,
            })
    return pd.DataFrame(rows)


def test_features_are_unaffected_by_future_price_mutations():
    baseline = build_feature_table(_make_synthetic_panel())
    mutated = build_feature_table(_make_synthetic_panel(mutate_from=MUTATE_FROM, mutation_factor=MUTATION_FACTOR))

    for ticker in sorted(baseline["ticker"].unique()):
        b = baseline[baseline["ticker"] == ticker].reset_index(drop=True)
        m = mutated[mutated["ticker"] == ticker].reset_index(drop=True)
        pre_mutation = b.index < MUTATE_FROM

        for col in FEATURE_COLUMNS:
            pd.testing.assert_series_equal(
                b.loc[pre_mutation, col].reset_index(drop=True),
                m.loc[pre_mutation, col].reset_index(drop=True),
                check_names=False,
                obj=f"{ticker}.{col}",
            )


def test_target_is_unaffected_before_its_forward_window_and_changes_after():
    baseline = build_feature_table(_make_synthetic_panel())
    mutated = build_feature_table(_make_synthetic_panel(mutate_from=MUTATE_FROM, mutation_factor=MUTATION_FACTOR))

    for ticker in sorted(baseline["ticker"].unique()):
        b = baseline[baseline["ticker"] == ticker].reset_index(drop=True)
        m = mutated[mutated["ticker"] == ticker].reset_index(drop=True)

        # Rows whose forward window (t+1..t+5) lands entirely before the
        # mutation point must be unchanged: t + 5 < MUTATE_FROM.
        safe = b.index <= (MUTATE_FROM - 1 - FORWARD_HORIZON)
        pd.testing.assert_series_equal(
            b.loc[safe, TARGET_COLUMN].reset_index(drop=True),
            m.loc[safe, TARGET_COLUMN].reset_index(drop=True),
            check_names=False,
        )

        # A row whose forward window reaches into the mutated region
        # must actually change value -- proving the target genuinely
        # depends on future prices (it is a label, not a feature).
        touched_idx = MUTATE_FROM - 3  # touched_idx + 5 >= MUTATE_FROM
        assert not np.isclose(
            b.loc[touched_idx, TARGET_COLUMN],
            m.loc[touched_idx, TARGET_COLUMN],
        )
