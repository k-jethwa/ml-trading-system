"""Model-ready frame: cross-sectional feature ranks + the label.

Raw features are on wildly different scales (a 5-day return is ~0.02,
20-day dollar volume is ~1e9) and non-stationary (dollar volume trends
up for a decade). Each feature is therefore replaced by its
**same-date cross-sectional percentile rank, centered to [-0.5, 0.5]**.

Leakage check: a rank on date t uses only the other tickers' values on
date t, i.e. information already available at the same decision time.
Nothing is fit across time, so there is no train-vs-test statistic that
could leak (contrast with a global StandardScaler fit on the whole
sample, which is exactly the failure mode in project_spec.md section
13). The only fitted scaler in the system is the one inside the Ridge
pipeline, and it is fit on each training fold alone.
"""

from __future__ import annotations

import pandas as pd

from mltrading.features.build import FEATURE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN


def rank_features(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    ranked = df.groupby("date")[feature_cols].rank(pct=True) - 0.5
    return ranked


def build_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, ticker) with ranked features and the raw label.

    Rows with any missing feature are dropped (the earliest year of each
    ticker, which lacks 252 days of history for `mom_12_1`). Rows whose
    label is still unknown (the last 5 days of the sample) are KEPT with
    a NaN target so they can still be scored at inference time; training
    code must filter on `target.notna()`.
    """
    ranked = rank_features(features)
    frame = pd.concat(
        [features[IDENTIFIER_COLUMNS], ranked, features[[TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "target"})],
        axis=1,
    )
    frame = frame.dropna(subset=FEATURE_COLUMNS)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)
