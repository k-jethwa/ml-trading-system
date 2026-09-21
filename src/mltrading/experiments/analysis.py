"""Backtest orchestration, cost sensitivity and robustness checks.

Everything here consumes the walk-forward predictions produced by
models/train.py; nothing refits a model on out-of-sample data except
`label_shuffle_check`, which deliberately trains on destroyed labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltrading.backtest.engine import BacktestResult, MarketData, run_backtest
from mltrading.backtest.metrics import compute_metrics
from mltrading.backtest.strategies import BaselineStrategy, RiskAwareStrategy
from mltrading.config import CostConfig, ExperimentConfig
from mltrading.features.build import FEATURE_COLUMNS
from mltrading.models.evaluate import per_date_ic
from mltrading.models.walk_forward import make_folds
from mltrading.models.zoo import build_model

PERIODS = ("all", "development", "holdout")


def period_bounds(cfg: ExperimentConfig, first_date: pd.Timestamp, last_date: pd.Timestamp) -> dict[str, tuple]:
    holdout_start = pd.Timestamp(cfg.walk_forward.holdout_start)
    return {
        "all": (first_date, last_date),
        "development": (first_date, holdout_start - pd.Timedelta(days=1)),
        "holdout": (holdout_start, last_date),
    }


def make_strategy(kind: str, cfg: ExperimentConfig, costs: CostConfig):
    if kind == "baseline":
        return BaselineStrategy(cfg.portfolio.quantile)
    if kind == "risk_aware":
        return RiskAwareStrategy(cfg.risk, costs, cfg.portfolio.rebalance_every)
    raise ValueError(kind)


def backtest(scores: pd.DataFrame, data: MarketData, cfg: ExperimentConfig, kind: str,
             costs: CostConfig | None = None, rebalance_every: int | None = None, quantile: float | None = None):
    costs = costs or cfg.costs
    strategy = make_strategy(kind, cfg, costs)
    if quantile is not None and kind == "baseline":
        strategy = BaselineStrategy(quantile)
    result = run_backtest(scores, data, strategy, cfg.portfolio.initial_capital,
                          rebalance_every or cfg.portfolio.rebalance_every, costs)
    return result, strategy


def period_metrics(result: BacktestResult, data: MarketData, cfg: ExperimentConfig) -> dict[str, dict]:
    market = data.returns.mean(axis=1)
    bounds = period_bounds(cfg, result.daily.index[0], result.daily.index[-1])
    return {
        name: compute_metrics(result.daily, result.positions, data.sectors, market, start, end)
        for name, (start, end) in bounds.items()
    }


def yearly_metrics(result: BacktestResult, data: MarketData) -> pd.DataFrame:
    market = data.returns.mean(axis=1)
    rows = []
    for year in sorted(result.daily.index.year.unique()):
        start, end = f"{year}-01-01", f"{year}-12-31"
        if len(result.daily.loc[start:end]) < 20:
            continue
        m = compute_metrics(result.daily, result.positions, data.sectors, market, start, end)
        rows.append({"year": year, **{k: m[k] for k in ("total_return_net", "sharpe_net", "sharpe_gross", "max_drawdown", "beta_to_universe")}})
    return pd.DataFrame(rows)


def benchmark_metrics(data: MarketData, cfg: ExperimentConfig, first_date: pd.Timestamp) -> dict[str, dict]:
    """Buy-and-hold equal-weight universe (long-only context for the market-neutral books)."""
    r = data.returns.mean(axis=1).loc[first_date:].dropna()
    nav = (1 + r).cumprod() * cfg.portfolio.initial_capital
    out = {}
    for name, (start, end) in period_bounds(cfg, r.index[0], r.index[-1]).items():
        sub = r.loc[start:end]
        n = (1 + sub).cumprod()
        out[name] = {
            "ann_return_net": float(n.iloc[-1] ** (252 / len(sub)) - 1),
            "ann_vol": float(sub.std(ddof=1) * np.sqrt(252)),
            "sharpe_net": float(sub.mean() / sub.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float((n / n.cummax() - 1).min()),
        }
    del nav
    return out


def random_score_distribution(template: pd.DataFrame, data: MarketData, cfg: ExperimentConfig, n_seeds: int) -> pd.DataFrame:
    """Baseline decile portfolio on pure-noise scores: what a signal-free strategy earns after costs."""
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        scores = template[["date", "ticker"]].copy()
        scores["pred"] = rng.normal(size=len(scores))
        result, _ = backtest(scores, data, cfg, "baseline")
        m = period_metrics(result, data, cfg)
        rows.append({"seed": seed, **{f"{p}_{k}": m[p][k] for p in PERIODS for k in ("sharpe_net", "sharpe_gross", "ann_return_net")}})
    return pd.DataFrame(rows)


def label_shuffle_check(frame: pd.DataFrame, cfg: ExperimentConfig, model_names=("ridge", "hgb"), seed: int = 0) -> pd.DataFrame:
    """Leakage / pipeline sanity check: train on PERMUTED labels. Any out-of-sample rank IC that
    is meaningfully above zero would mean the evaluation pipeline manufactures signal."""
    rng = np.random.default_rng(seed)
    folds = make_folds(frame["date"], cfg.walk_forward.first_test_year, cfg.walk_forward.embargo_days)
    rows = []
    for name in model_names:
        parts = []
        for fold in folds:
            train = frame[fold.train_mask(frame["date"]) & frame["target"].notna()]
            test = frame[fold.test_mask(frame["date"]) & frame["target"].notna()]
            model = build_model(name, cfg.models)
            model.fit(train[FEATURE_COLUMNS], rng.permutation(train["target"].to_numpy()))
            part = test[["date", "target"]].copy()
            part["pred"] = model.predict(test[FEATURE_COLUMNS])
            parts.append(part)
        ic = per_date_ic(pd.concat(parts), "spearman")
        rows.append({"model": name, "shuffled_rank_ic_mean": float(ic.mean()),
                     "shuffled_rank_ic_tstat": float(ic.iloc[::5].mean() / (ic.iloc[::5].std(ddof=1) / np.sqrt(len(ic.iloc[::5]))))})
    return pd.DataFrame(rows)


def prediction_distribution(preds: pd.DataFrame) -> pd.DataFrame:
    """Monitoring: per model and year, the shape of the scores (drift / degenerate-output detector)."""
    g = preds.assign(year=preds["date"].dt.year).groupby(["model", "year"])["pred"]
    return g.agg(mean="mean", std="std", p01=lambda s: s.quantile(0.01), p99=lambda s: s.quantile(0.99),
                 frac_nan=lambda s: s.isna().mean()).reset_index()
