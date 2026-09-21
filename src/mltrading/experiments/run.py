"""One-command reproducible experiment (project_spec.md "Final evidence").

    python -m mltrading.experiments.run --config configs/v1.toml --note "why this run"

Pipeline: load the feature table -> walk-forward predictions for every
model -> ML metrics -> baseline and risk-aware backtests -> cost
sensitivity -> robustness checks (label shuffle, random-score
portfolios, parameter sensitivity, per-year stability) -> monitoring
metrics -> outputs under results/<config name>/.

Every full run appends one line to docs/attempt_log.jsonl (config hash,
git commit, note, headline numbers). That file is the count of
"attempts" that project_spec.md section 13 asks results to disclose.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mltrading.backtest.engine import MarketData
from mltrading.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ExperimentConfig,
    config_hash,
    load_config,
)
from mltrading.data.universe import UNIVERSE_VERSION
from mltrading.experiments import analysis as an
from mltrading.features.build import FEATURE_COLUMNS, FEATURE_TABLE_PATH
from mltrading.models.evaluate import per_date_ic, summarize
from mltrading.models.preprocess import build_model_frame
from mltrading.models.registry import feature_version, file_fingerprint, git_state
from mltrading.models.train import run_walk_forward, train_production_models
from mltrading.models.zoo import RETURN_UNIT_MODELS
from mltrading.observability import RunMetrics, configure_logging, log_event

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
ATTEMPT_LOG = PROJECT_ROOT / "docs" / "attempt_log.jsonl"
COST_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
BASELINE_MODELS = ("reversal_5d", "ols", "ridge", "hgb")
RISK_AWARE_MODELS = ("ols", "ridge", "hgb")
SENSITIVITY_MODELS = ("ridge", "hgb")


def data_freshness(features: pd.DataFrame, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    last = features["date"].max()
    per_ticker_last = features.groupby("ticker")["date"].max()
    return {
        "last_bar_date": str(last.date()),
        "age_days": (now.date() - last.date()).days,
        "n_tickers": int(features["ticker"].nunique()),
        "n_tickers_missing_last_bar": int((per_ticker_last < last).sum()),
    }


def ml_metrics_table(preds: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    bounds = {"all": preds["date"].between("1900-01-01", "2100-01-01"),
              "development": preds["period"] == "development", "holdout": preds["period"] == "holdout"}
    mean_rows = preds[preds["model"] == "mean"]
    for period, mask in bounds.items():
        sub = preds[mask]
        base = mean_rows[mask.loc[mean_rows.index]].dropna(subset=["target"])
        mean_mse = float(((base["pred"] - base["target"]) ** 2).mean())
        for model, g in sub.groupby("model"):
            s = summarize(g, mean_mse)
            if model not in RETURN_UNIT_MODELS:
                s["mae"] = s["mse"] = s["oos_r2"] = float("nan")   # scores are not return forecasts
            rows.append({"model": model, "period": period, **s})
    return pd.DataFrame(rows)


def ic_by_year(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, year), g in preds.assign(year=preds["date"].dt.year).groupby(["model", "year"]):
        ric = per_date_ic(g.dropna(subset=["target"]), "spearman")
        if len(ric):
            rows.append({"model": model, "year": year, "rank_ic_mean": ric.mean(), "n_dates": len(ric)})
    return pd.DataFrame(rows)


def _flat(name: str, kind: str, mult: float, per_period: dict) -> list[dict]:
    return [{"model": name, "portfolio": kind, "cost_multiplier": mult, "period": p, **m} for p, m in per_period.items()]


def run(config_path: Path = DEFAULT_CONFIG_PATH, note: str = "", n_random_seeds: int = 20,
        out_root: Path = RESULTS_DIR, log_attempt: bool = True) -> Path:
    configure_logging()
    cfg = load_config(config_path)
    metrics = RunMetrics()
    chash = config_hash(cfg)
    out = Path(out_root) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log_event(logger, "run start", config=cfg.name, config_hash=chash, note=note)

    with metrics.timer("load_features"):
        features = pd.read_parquet(FEATURE_TABLE_PATH)
        data_fp = file_fingerprint(FEATURE_TABLE_PATH)
    fresh = data_freshness(features)
    metrics.set("data_freshness", fresh)
    if fresh["age_days"] > 7:
        log_event(logger, "feature data is stale", level=logging.WARNING, **fresh)

    with metrics.timer("build_model_frame"):
        frame = build_model_frame(features)
    with metrics.timer("walk_forward"):
        preds = run_walk_forward(frame, cfg, metrics)
    preds.to_parquet(out / "predictions.parquet", index=False)

    ml = ml_metrics_table(preds, cfg)
    ml.to_csv(out / "ml_metrics.csv", index=False)
    ic_by_year(preds).to_csv(out / "ic_by_year.csv", index=False)
    an.prediction_distribution(preds).to_csv(out / "prediction_distribution.csv", index=False)

    with metrics.timer("production_models"):
        saved = train_production_models(frame, cfg, data_fp)
    metrics.set("production_model_ids", {k: v.model_id for k, v in saved.items()})

    data = MarketData.from_features(features)
    trading_rows, navs, yearly, risk_logs = [], {}, [], {}
    cache: dict[tuple, dict] = {}

    def do_backtest(model: str, kind: str, mult: float, keep: bool = False, **kw):
        costs = cfg.costs.scaled(mult)
        scores = preds[preds["model"] == model][["date", "ticker", "pred"]]
        with metrics.timer(f"backtest.{kind}"):
            result, strat = an.backtest(scores, data, cfg, kind, costs, **kw)
        per_period = an.period_metrics(result, data, cfg)
        if keep:
            navs[f"{model}|{kind}"] = result.daily["nav"]
            yearly.append(an.yearly_metrics(result, data).assign(model=model, portfolio=kind))
            if kind == "risk_aware":
                risk_logs[model] = pd.DataFrame(strat.log)
        cache[(model, kind, mult)] = per_period
        log_event(logger, "backtest", model=model, kind=kind, cost_mult=mult,
                  sharpe_net_all=round(per_period["all"]["sharpe_net"], 3))
        return per_period

    for model in BASELINE_MODELS:
        trading_rows += _flat(model, "baseline", 1.0, do_backtest(model, "baseline", 1.0, keep=True))
    for model in RISK_AWARE_MODELS:
        trading_rows += _flat(model, "risk_aware", 1.0, do_backtest(model, "risk_aware", 1.0, keep=True))

    first = min(nv.index[0] for nv in navs.values())
    for period, m in an.benchmark_metrics(data, cfg, first).items():
        trading_rows.append({"model": "equal_weight_universe", "portfolio": "buy_and_hold_long_only",
                             "cost_multiplier": 0.0, "period": period, **m})
    pd.DataFrame(trading_rows).to_csv(out / "trading_metrics.csv", index=False)
    pd.DataFrame(navs).to_parquet(out / "nav.parquet")
    pd.concat(yearly).to_csv(out / "yearly_metrics.csv", index=False)
    for model, lg in risk_logs.items():
        lg.to_csv(out / f"optimizer_log_{model}.csv", index=False)

    # ---- cost sensitivity (spec section 10: 0.5x, 1x, 2x, 4x) ----
    sens_rows = []
    for model in SENSITIVITY_MODELS:
        for kind in ("baseline", "risk_aware"):
            for mult in COST_MULTIPLIERS:
                per_period = cache.get((model, kind, mult)) or do_backtest(model, kind, mult)
                sens_rows += _flat(model, kind, mult, per_period)
    pd.DataFrame(sens_rows).to_csv(out / "cost_sensitivity.csv", index=False)

    # ---- robustness ----
    with metrics.timer("robustness.label_shuffle"):
        an.label_shuffle_check(frame, cfg).to_csv(out / "label_shuffle_check.csv", index=False)
    template = preds[preds["model"] == "ridge"][["date", "ticker"]]
    with metrics.timer("robustness.random_scores"):
        an.random_score_distribution(template, data, cfg, n_random_seeds).to_csv(out / "random_score_baseline.csv", index=False)
    param_rows = []
    ridge_scores = preds[preds["model"] == "ridge"][["date", "ticker", "pred"]]
    with metrics.timer("robustness.parameters"):
        for q in (0.05, 0.10, 0.20):
            r, _ = an.backtest(ridge_scores, data, cfg, "baseline", quantile=q)
            param_rows += _flat(f"ridge q={q}", "baseline", 1.0, an.period_metrics(r, data, cfg))
        for every in (5, 10, 21):
            r, _ = an.backtest(ridge_scores, data, cfg, "baseline", rebalance_every=every)
            param_rows += _flat(f"ridge rebalance_every={every}", "baseline", 1.0, an.period_metrics(r, data, cfg))
    pd.DataFrame(param_rows).to_csv(out / "parameter_sensitivity.csv", index=False)

    # ---- manifest / monitoring ----
    manifest = {
        "run_id": f"{cfg.name}-{chash}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "note": note, "config": cfg.to_dict(), "config_hash": chash, "git": git_state(),
        "feature_version": feature_version(), "n_features": len(FEATURE_COLUMNS),
        "universe_version": UNIVERSE_VERSION, "data_fingerprint": data_fp,
        "n_prediction_rows": len(preds), "n_random_seeds": n_random_seeds,
        "metrics": metrics.to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if log_attempt:
        tm = pd.DataFrame(trading_rows)
        head = {}
        for model, kind in (("ridge", "baseline"), ("hgb", "baseline"), ("ridge", "risk_aware"), ("hgb", "risk_aware")):
            for period in ("development", "holdout"):
                row = tm[(tm.model == model) & (tm.portfolio == kind) & (tm.period == period)].iloc[0]
                head[f"{model}_{kind}_{period}_sharpe_net"] = round(float(row["sharpe_net"]), 3)
        ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTEMPT_LOG, "a") as fh:
            fh.write(json.dumps({"run_id": manifest["run_id"], "config_hash": chash, "git": manifest["git"],
                                 "note": note, "headline": head}) + "\n")
    log_event(logger, "run complete", out_dir=str(out), seconds=round(sum(metrics.timings.values()), 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--note", default="", help="Why this run is being made (goes in the attempt log)")
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--no-attempt-log", action="store_true", help="Do not append to docs/attempt_log.jsonl (tests only)")
    args = parser.parse_args()
    run(args.config, args.note, args.random_seeds, log_attempt=not args.no_attempt_log)


if __name__ == "__main__":
    main()
