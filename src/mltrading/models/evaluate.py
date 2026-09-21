"""ML evaluation metrics (project_spec.md section 8).

  IC        per-date Pearson correlation between prediction and realized
            target, averaged over dates.
  Rank IC   the same with Spearman (rank) correlation — the headline
            metric, because the portfolio only uses the RANKING.
  MAE / MSE pooled over all rows; `oos_r2` is 1 - MSE/MSE(mean baseline),
            so it is negative when the model is worse than a constant.
  Decile spread  mean target of the top predicted decile minus the
            bottom, per date — closer to what a long/short book earns.

Overlapping labels. Each 5-day label overlaps its four neighbours, so
consecutive daily ICs are strongly autocorrelated and the naive
t-statistic overstates significance by roughly sqrt(5). `ic_tstat`
therefore uses a non-overlapping subsample (every 5th date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES_PER_DATE = 10
NON_OVERLAP_STEP = 5


def _per_date_corr(df: pd.DataFrame, method: str) -> pd.Series:
    def corr(group: pd.DataFrame) -> float:
        if len(group) < MIN_NAMES_PER_DATE or group["pred"].nunique() < 2:
            return np.nan
        return group["pred"].corr(group["target"], method=method)

    return df.groupby("date")[["pred", "target"]].apply(corr).dropna()


def per_date_ic(df: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """`df` needs columns date, pred, target."""
    return _per_date_corr(df.dropna(subset=["pred", "target"]), method)


def decile_spread(df: pd.DataFrame, q: float = 0.10) -> pd.Series:
    def spread(group: pd.DataFrame) -> float:
        n = len(group)
        if n < MIN_NAMES_PER_DATE or group["pred"].nunique() < 2:
            return np.nan
        k = max(1, round(q * n))
        ordered = group.sort_values("pred", kind="stable")
        return ordered["target"].iloc[-k:].mean() - ordered["target"].iloc[:k].mean()

    return df.dropna(subset=["pred", "target"]).groupby("date")[["pred", "target"]].apply(spread).dropna()


def _tstat(series: pd.Series) -> float:
    sub = series.iloc[::NON_OVERLAP_STEP]
    if len(sub) < 3 or sub.std(ddof=1) == 0:
        return np.nan
    return float(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))))


def summarize(df: pd.DataFrame, mean_mse: float | None = None) -> dict[str, float]:
    """Summary for one model over one evaluation window."""
    d = df.dropna(subset=["pred", "target"])
    ic = per_date_ic(d, "pearson")
    ric = per_date_ic(d, "spearman")
    spread = decile_spread(d)
    err = d["pred"] - d["target"]
    mse = float((err**2).mean())
    return {
        "n_rows": len(d),
        "n_dates": int(d["date"].nunique()),
        "mae": float(err.abs().mean()),
        "mse": mse,
        "oos_r2": float(1 - mse / mean_mse) if mean_mse else np.nan,
        "ic_mean": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic_mean": float(ric.mean()) if len(ric) else np.nan,
        "rank_ic_std": float(ric.std(ddof=1)) if len(ric) > 1 else np.nan,
        "rank_ic_ir": float(ric.mean() / ric.std(ddof=1)) if len(ric) > 1 and ric.std(ddof=1) > 0 else np.nan,
        "rank_ic_tstat_nonoverlap": _tstat(ric),
        "rank_ic_hit_rate": float((ric > 0).mean()) if len(ric) else np.nan,
        "decile_spread_5d": float(spread.mean()) if len(spread) else np.nan,
    }
