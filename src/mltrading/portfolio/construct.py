"""Baseline portfolio: long the top decile, short the bottom decile.

Equal weight within each side; the long side sums to +1.0 and the short
side to -1.0 of NAV, so net exposure is zero (dollar-neutral) and gross
exposure is 2.0. Scores are used only through their ORDER, so the
baseline works for any model, including the non-return-unit heuristic.

Ties are broken by ticker so results are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_weights(scores: pd.Series, quantile: float = 0.10) -> pd.Series:
    scores = scores.dropna()
    n = len(scores)
    if n < 2:
        return pd.Series(0.0, index=scores.index)
    k = max(1, round(quantile * n))
    k = min(k, n // 2)
    # primary key: score; ties broken by ticker (np.lexsort's LAST key is the primary one)
    ordered = scores.index.to_numpy()[np.lexsort((scores.index.to_numpy(), scores.to_numpy()))]
    weights = pd.Series(0.0, index=scores.index)
    weights[ordered[-k:]] = 1.0 / k
    weights[ordered[:k]] = -1.0 / k
    return weights
