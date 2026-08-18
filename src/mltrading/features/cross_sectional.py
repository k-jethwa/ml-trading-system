"""Cross-sectional (same-date, across-ticker) computations.

Used for both the market-/sector-relative features (backward-looking)
and the sector-relative target (forward-looking): a leave-one-out group
mean, so a ticker is never compared against a benchmark that is partly
made of itself. This matters most for small sectors — Energy has only 4
names in the V1 universe, so including the stock itself in its own
"sector average" would materially distort the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_out_mean(panel: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.Series:
    """For each row, the mean of `value_col` across all OTHER rows
    sharing the same `group_cols` (e.g. group_cols=["date"] for a
    market-wide mean, or ["date", "sector"] for a sector mean).

    Returns a Series aligned to panel.index. A row's own NaN value does
    not count against the group; a group with fewer than 2 non-null
    members (so the leave-one-out mean is undefined) returns NaN.
    """
    values = panel[value_col]
    grp = panel.groupby(group_cols)[value_col]
    grp_sum = grp.transform("sum")
    grp_count = grp.transform("count").astype(float)

    has_own = values.notna()
    loo_sum = grp_sum - values.where(has_own, 0.0)
    loo_count = grp_count - has_own.astype(float)
    loo_count = loo_count.where(loo_count > 0, np.nan)

    return loo_sum / loo_count
