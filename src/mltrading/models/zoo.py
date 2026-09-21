"""The model progression from project_spec.md section 7.

  1. mean          constant predictor (the minimum benchmark)
  2. ols           linear regression
  3. ridge         L2-regularized linear regression + scaler
  4. hgb           gradient-boosted trees (HistGradientBoostingRegressor)

plus one non-ML reference, `reversal_5d` (short-term reversal: rank
stocks by minus their 5-day sector-relative return), so that any ML
model has to beat a one-line heuristic, not just a constant.

Deviation from the spec, disclosed: section 7 names XGBoost or LightGBM
for step 4. Both need the system OpenMP runtime (`brew install libomp`
on macOS), which this environment does not have. scikit-learn's
HistGradientBoostingRegressor is the same algorithm family (histogram-
binned, leaf-wise gradient-boosted trees, the design LightGBM
popularized) and ships its own OpenMP, so it is used for the nonlinear
benchmark. The factory below is the only place that would change.

All hyperparameters come from configs/v1.toml and were fixed a priori;
none were tuned on out-of-sample results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mltrading.config import ModelConfig

RETURN_UNIT_MODELS = ("mean", "ols", "ridge", "hgb")  # predictions are return forecasts (used by the optimizer)


class ReversalHeuristic(BaseEstimator, RegressorMixin):
    """Score = -(ranked 5-day sector-relative return). No fitting."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -X["sector_rel_ret_5d"].to_numpy(dtype=float)


def build_model(name: str, cfg: ModelConfig | None = None):
    cfg = cfg or ModelConfig()
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "reversal_5d":
        return ReversalHeuristic()
    if name == "ols":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=cfg.hgb_max_depth,
            learning_rate=cfg.hgb_learning_rate,
            max_iter=cfg.hgb_max_iter,
            min_samples_leaf=cfg.hgb_min_samples_leaf,
            l2_regularization=cfg.hgb_l2_regularization,
            early_stopping=False,
            random_state=cfg.seed,
        )
    raise ValueError(f"Unknown model: {name!r}")
