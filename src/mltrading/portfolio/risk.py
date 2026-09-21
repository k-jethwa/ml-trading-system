"""Risk model inputs and post-trade risk checks for portfolio construction.

Covariance: Ledoit-Wolf shrinkage of the trailing daily-return matrix.
With ~55 names and a 60-day window the raw sample covariance is
noisy and close to singular; shrinking toward a scaled identity makes
it well-conditioned at the price of some bias, the standard trade-off.

Beta: each stock's regression beta to an equal-weight average of the
universe (the dataset has no index series, so the "market" is
the universe itself; documented limitation).

Every input at decision date t is computed from returns dated <= t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from mltrading.config import RiskConfig

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RiskModel:
    cov_daily: pd.DataFrame
    betas: pd.Series


def estimate_risk(window: pd.DataFrame) -> RiskModel:
    """`window`: rows = trailing dates, columns = tickers, values = daily returns (no NaN)."""
    cov = LedoitWolf().fit(window.to_numpy()).covariance_
    market = window.mean(axis=1)
    # beta_i = cov(r_i, m) / var(m), vectorized across all columns at once (the per-column
    # `apply` version was a measured hotspot: 2.7s across 384 rebalances)
    market_c = market - market.mean()
    cov_im = (window - window.mean()).mul(market_c, axis=0).sum() / (len(window) - 1)
    betas = cov_im / market.var(ddof=1)
    return RiskModel(pd.DataFrame(cov, index=window.columns, columns=window.columns), betas)


def ex_ante_vol_annual(weights: pd.Series, cov_daily: pd.DataFrame) -> float:
    w = weights.reindex(cov_daily.index).fillna(0.0).to_numpy()
    return float(np.sqrt(w @ cov_daily.to_numpy() @ w * TRADING_DAYS_PER_YEAR))


def exposure_summary(weights: pd.Series, sectors: pd.Series, betas: pd.Series | None = None) -> dict[str, float]:
    sec_net = weights.groupby(sectors.reindex(weights.index)).sum()
    out = {
        "gross": float(weights.abs().sum()),
        "net": float(weights.sum()),
        "max_abs_weight": float(weights.abs().max()) if len(weights) else 0.0,
        "max_abs_sector_net": float(sec_net.abs().max()) if len(sec_net) else 0.0,
    }
    if betas is not None:
        out["beta"] = float((weights * betas.reindex(weights.index).fillna(0.0)).sum())
    return out


def check_constraints(
    weights: pd.Series,
    prev_weights: pd.Series | None,
    sectors: pd.Series,
    risk: RiskModel,
    cfg: RiskConfig,
    tol: float = 1e-6,
    enforce_turnover: bool = True,
) -> list[str]:
    """Return a list of violated constraints (empty == all satisfied)."""
    bad: list[str] = []
    ex = exposure_summary(weights, sectors, risk.betas)
    if ex["gross"] > cfg.max_gross + tol:
        bad.append(f"gross {ex['gross']:.4f} > {cfg.max_gross}")
    if abs(ex["net"]) > cfg.net_tolerance + tol:
        bad.append(f"|net| {abs(ex['net']):.4f} > {cfg.net_tolerance}")
    if ex["max_abs_weight"] > cfg.max_abs_weight + tol:
        bad.append(f"max |w| {ex['max_abs_weight']:.4f} > {cfg.max_abs_weight}")
    if ex["max_abs_sector_net"] > cfg.sector_net_limit + tol:
        bad.append(f"sector net {ex['max_abs_sector_net']:.4f} > {cfg.sector_net_limit}")
    if abs(ex["beta"]) > cfg.beta_limit + tol:
        bad.append(f"|beta| {abs(ex['beta']):.4f} > {cfg.beta_limit}")
    vol = ex_ante_vol_annual(weights, risk.cov_daily)
    if vol > cfg.target_vol_annual + tol:
        bad.append(f"vol {vol:.4f} > {cfg.target_vol_annual}")
    if enforce_turnover and prev_weights is not None:
        turnover = float((weights - prev_weights.reindex(weights.index).fillna(0.0)).abs().sum())
        if turnover > cfg.turnover_limit + tol:
            bad.append(f"turnover {turnover:.4f} > {cfg.turnover_limit}")
    return bad
