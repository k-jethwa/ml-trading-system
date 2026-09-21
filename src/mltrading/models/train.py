"""Walk-forward training and out-of-sample prediction.

For each test year (models/walk_forward.py) and each model in the
config: fit on the purged, expanding training window, predict the test
year, and append the predictions. Nothing about a test row is visible
to a fit that scores it. Also trains one "production" model per name
on all labelled history, saved to the model registry for batch
inference (models/infer.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mltrading.config import (
    DEFAULT_CONFIG_PATH,
    ExperimentConfig,
    config_hash,
    load_config,
)
from mltrading.features.build import FEATURE_COLUMNS, FEATURE_TABLE_PATH
from mltrading.models.preprocess import build_model_frame
from mltrading.models.registry import SavedModel, file_fingerprint, save_model
from mltrading.models.walk_forward import Fold, make_folds
from mltrading.models.zoo import build_model
from mltrading.observability import RunMetrics, log_event

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = ["date", "ticker", "model", "pred", "target", "fold_year", "period"]


def fit_predict_fold(frame: pd.DataFrame, fold: Fold, model_name: str, cfg: ExperimentConfig) -> pd.DataFrame:
    train = frame[fold.train_mask(frame["date"]) & frame["target"].notna()]
    test = frame[fold.test_mask(frame["date"])]
    model = build_model(model_name, cfg.models)
    model.fit(train[FEATURE_COLUMNS], train["target"])
    out = test[["date", "ticker", "target"]].copy()
    out["model"] = model_name
    out["pred"] = model.predict(test[FEATURE_COLUMNS])
    out["fold_year"] = fold.test_year
    return out


def run_walk_forward(frame: pd.DataFrame, cfg: ExperimentConfig, metrics: RunMetrics | None = None) -> pd.DataFrame:
    metrics = metrics or RunMetrics()
    folds = make_folds(frame["date"], cfg.walk_forward.first_test_year, cfg.walk_forward.embargo_days)
    holdout_start = pd.Timestamp(cfg.walk_forward.holdout_start)
    parts = []
    for fold in folds:
        for name in cfg.models.names:
            with metrics.timer(f"fit_predict.{name}"):
                parts.append(fit_predict_fold(frame, fold, name, cfg))
        log_event(logger, "fold complete", test_year=fold.test_year, train_end=str(fold.train_end.date()))
    preds = pd.concat(parts, ignore_index=True)
    preds["period"] = preds["date"].map(lambda d: "holdout" if d >= holdout_start else "development")
    return preds[PREDICTION_COLUMNS].sort_values(["model", "date", "ticker"]).reset_index(drop=True)


def train_production_models(
    frame: pd.DataFrame, cfg: ExperimentConfig, data_fingerprint: str, artifact_dir: Path | None = None
) -> dict[str, SavedModel]:
    """One model per name fit on ALL labelled history (for inference)."""
    labelled = frame[frame["target"].notna()]
    saved: dict[str, SavedModel] = {}
    kwargs = {"artifact_dir": artifact_dir} if artifact_dir is not None else {}
    for name in cfg.models.names:
        model = build_model(name, cfg.models)
        model.fit(labelled[FEATURE_COLUMNS], labelled["target"])
        saved[name] = save_model(
            model, name, labelled["date"], len(labelled), config_hash(cfg), data_fingerprint,
            params=model.get_params(deep=False) if hasattr(model, "get_params") else {}, **kwargs,
        )
    return saved


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    from mltrading.observability import configure_logging

    configure_logging()
    cfg = load_config(config_path)
    frame = build_model_frame(pd.read_parquet(FEATURE_TABLE_PATH))
    saved = train_production_models(frame, cfg, file_fingerprint(FEATURE_TABLE_PATH))
    for s in saved.values():
        log_event(logger, "saved model", model_id=s.model_id)


if __name__ == "__main__":
    main()
