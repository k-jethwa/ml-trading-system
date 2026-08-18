"""V1 prediction target: five-trading-day forward sector-relative return.

    target(i, t) = fwd_ret_5d(i, t) - leave_one_out_mean(fwd_ret_5d, sector, t)
    fwd_ret_5d(i, t) = adj_close(i, t+5) / adj_close(i, t) - 1

"t+5" means five ROWS ahead in ticker i's own trading history, not five
calendar days. For the handful of tickers with 1-3 missing recent days
(docs/data_validation_report.md), this can shift which exact calendar
date "5 trading days later" lands on relative to a fully dense ticker —
but only within the last month of history, and never before.

This module produces LABELS ONLY. `fwd_ret_5d` and
`target_5d_sector_rel` are never valid model inputs: build.py keeps
them in clearly named, clearly documented columns and FEATURE_COLUMNS
never includes them, so a downstream `X = df[FEATURE_COLUMNS]` cannot
accidentally leak the label into training. See
tests/features/test_no_lookahead.py, which asserts these forward
columns actually change when future prices are perturbed (proving they
really do use future data, as a label should) while every feature
column stays bit-for-bit identical (proving features do not).
"""

from __future__ import annotations

import pandas as pd

from mltrading.features.cross_sectional import leave_one_out_mean

FORWARD_HORIZON_DAYS = 5


def add_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["fwd_ret_5d"] = (
        out.groupby("ticker")["adj_close"].shift(-FORWARD_HORIZON_DAYS) / out["adj_close"] - 1
    )
    return out


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel if "fwd_ret_5d" in panel.columns else add_forward_returns(panel)
    out = out.copy()
    sector_fwd_loo = leave_one_out_mean(out, "fwd_ret_5d", ["date", "sector"])
    out["target_5d_sector_rel"] = out["fwd_ret_5d"] - sector_fwd_loo
    return out
