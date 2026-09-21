"""Batch inference: score the latest decision date with a registered model.

    python -m mltrading.models.infer --model ridge

Failure handling (project_spec.md section 15, systems checkpoints):

  * stale data      the latest bar is older than `max_age_days` -> StaleDataError
                    (refuse to emit "fresh-looking" predictions from old data;
                    `allow_stale=True` overrides for replay/debugging and the
                    result is still marked stale)
  * feature drift   the model was trained on a different feature list/version
                    than the code now produces -> FeatureVersionMismatch
  * missing names   tickers absent from the latest date or with incomplete
                    features are excluded, counted and listed in `health`,
                    never silently filled

Every prediction row carries the model_id, so "which exact model made
this prediction?" is answerable from the output alone.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mltrading.features.build import FEATURE_COLUMNS, FEATURE_TABLE_PATH
from mltrading.models.preprocess import build_model_frame
from mltrading.models.registry import (
    ARTIFACT_DIR,
    feature_version,
    load_latest,
    load_model,
)
from mltrading.observability import configure_logging, log_event

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_DAYS = 7


class StaleDataError(RuntimeError):
    pass


class FeatureVersionMismatch(RuntimeError):
    pass


@dataclass
class InferenceResult:
    predictions: pd.DataFrame
    health: dict


def score_latest(
    features: pd.DataFrame,
    model_name: str,
    artifact_dir: Path = ARTIFACT_DIR,
    now: datetime | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    allow_stale: bool = False,
    model_cache: dict | None = None,
) -> InferenceResult:
    started = time.perf_counter()
    now = now or datetime.now(UTC)
    saved = load_latest(model_name, artifact_dir)
    if saved.metadata["feature_version"] != feature_version() or saved.metadata["feature_columns"] != FEATURE_COLUMNS:
        raise FeatureVersionMismatch(
            f"model {saved.model_id} was trained with feature_version {saved.metadata['feature_version']}, "
            f"current code produces {feature_version()}; retrain (python -m mltrading.models.train)"
        )

    last_date = features["date"].max()
    age_days = (now.date() - last_date.date()).days
    stale = age_days > max_age_days
    if stale and not allow_stale:
        raise StaleDataError(f"latest bar is {last_date.date()} ({age_days} days old, limit {max_age_days})")

    latest = features[features["date"] == last_date]
    expected = sorted(features["ticker"].unique())
    frame = build_model_frame(latest)  # same-date ranks; drops names with incomplete features
    scored = sorted(frame["ticker"])
    missing = sorted(set(expected) - set(scored))

    cache = model_cache if model_cache is not None else {}
    if saved.model_id not in cache:
        cache[saved.model_id] = load_model(saved)
    model = cache[saved.model_id]
    preds = frame[["date", "ticker", "sector"]].copy()
    preds["pred"] = model.predict(frame[FEATURE_COLUMNS])
    preds["model_id"] = saved.model_id
    preds = preds.sort_values("pred", ascending=False).reset_index(drop=True)

    health = {
        "status": "stale" if stale else "ok",
        "model_id": saved.model_id,
        "feature_version": saved.metadata["feature_version"],
        "trained_through": saved.metadata["train_end"],
        "last_bar_date": str(last_date.date()),
        "age_days": age_days,
        "n_scored": len(scored),
        "n_expected": len(expected),
        "missing_tickers": missing,
        "pred_mean": float(preds["pred"].mean()),
        "pred_std": float(preds["pred"].std()),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if missing:
        log_event(logger, "tickers excluded from scoring", level=logging.WARNING, missing=missing, date=health["last_bar_date"])
    log_event(logger, "batch inference", **{k: v for k, v in health.items() if k != "missing_tickers"})
    return InferenceResult(preds, health)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE_PATH)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV path for the predictions")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()
    result = score_latest(pd.read_parquet(args.features), args.model, max_age_days=args.max_age_days, allow_stale=args.allow_stale)
    if args.out:
        result.predictions.to_csv(args.out, index=False)
    print(result.predictions.head(10).to_string(index=False))
    print(result.health)


if __name__ == "__main__":
    main()
