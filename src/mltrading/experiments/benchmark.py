"""Stage-level benchmark + profile (project_spec.md section 12).

    python -m mltrading.experiments.benchmark --label before
    python -m mltrading.experiments.benchmark --label after

Times each pipeline stage the spec names — data loading, feature
computation, inference, portfolio construction, backtesting — using the
walk-forward predictions in results/<config>/predictions.parquet, and
writes results/benchmark_<label>.json. `--profile` additionally prints
the cProfile top entries for the risk-aware backtest, which is how the
bottleneck was identified before optimizing it (docs/benchmarks.md).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from pathlib import Path

import pandas as pd

from mltrading.backtest.engine import MarketData
from mltrading.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from mltrading.experiments import analysis as an
from mltrading.features.build import FEATURE_COLUMNS, FEATURE_TABLE_PATH, load_and_build
from mltrading.models.preprocess import build_model_frame
from mltrading.models.registry import load_latest, load_model
from mltrading.portfolio.optimize import optimize_weights
from mltrading.portfolio.risk import estimate_risk

RESULTS_DIR = PROJECT_ROOT / "results"


def _time(fn, repeats: int) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return {"median_s": round(statistics.median(samples), 4), "min_s": round(min(samples), 4), "repeats": repeats}


def run_benchmark(config_path: Path = DEFAULT_CONFIG_PATH, profile: bool = False) -> dict:
    cfg = load_config(config_path)
    preds = pd.read_parquet(RESULTS_DIR / cfg.name / "predictions.parquet")
    out: dict[str, dict] = {}

    out["data_loading (read features.parquet)"] = _time(lambda: pd.read_parquet(FEATURE_TABLE_PATH), 5)
    features = pd.read_parquet(FEATURE_TABLE_PATH)
    out["feature_computation (raw panel -> feature table)"] = _time(load_and_build, 3)
    out["model_frame (cross-sectional ranks)"] = _time(lambda: build_model_frame(features), 3)
    frame = build_model_frame(features)

    ridge = load_model(load_latest("ridge"))
    hgb = load_model(load_latest("hgb"))
    last = frame[frame["date"] == frame["date"].max()]
    out["inference latency: hgb, 1 date x 55 names"] = _time(lambda: hgb.predict(last[FEATURE_COLUMNS]), 20)
    out[f"inference throughput: hgb, whole panel ({len(frame)} rows)"] = _time(lambda: hgb.predict(frame[FEATURE_COLUMNS]), 3)
    out["inference throughput: ridge, whole panel"] = _time(lambda: ridge.predict(frame[FEATURE_COLUMNS]), 3)

    data = MarketData.from_features(features)
    scores = preds[preds["model"] == "ridge"][["date", "ticker", "pred"]]
    date = scores["date"].max() - pd.Timedelta(days=400)
    date = data.returns.index[data.returns.index.get_indexer([date], method="nearest")[0]]
    window = data.returns.loc[:date].tail(cfg.risk.cov_lookback).dropna(axis=1)
    day_scores = scores[scores["date"] == date].set_index("ticker")["pred"]
    alpha = day_scores.reindex(window.columns).dropna()
    risk = estimate_risk(window)
    cost_frac = (cfg.costs.commission_bps + cfg.costs.half_spread_bps + cfg.costs.slippage_bps) * 1e-4
    prev = pd.Series(0.0, index=alpha.index)
    out["portfolio construction: one risk-aware optimization"] = _time(
        lambda: optimize_weights(alpha, prev, risk, data.sectors, cfg.risk, cost_frac, cfg.portfolio.rebalance_every), 20)

    out["backtest: baseline, ridge, full period"] = _time(lambda: an.backtest(scores, data, cfg, "baseline"), 3)
    out["backtest: risk-aware, ridge, full period"] = _time(lambda: an.backtest(scores, data, cfg, "risk_aware"), 1)

    if profile:
        prof = cProfile.Profile()
        prof.enable()
        an.backtest(scores, data, cfg, "risk_aware")
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        out["_profile_risk_aware_backtest"] = {"top_cumulative": buf.getvalue()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(args.config, args.profile)
    path = RESULTS_DIR / f"benchmark_{args.label}.json"
    path.write_text(json.dumps(result, indent=2))
    for stage, v in result.items():
        if not stage.startswith("_"):
            print(f"{stage:<62} median {v['median_s']:>9.4f}s   (min {v['min_s']:.4f}s, n={v['repeats']})")
    if args.profile:
        print(result["_profile_risk_aware_backtest"]["top_cumulative"])


if __name__ == "__main__":
    main()
