"""Reference (uncached) formulation of the risk-aware optimizer.

This is the original straightforward implementation, rebuilt from
scratch on every call. It was replaced in src/ by a compile-once
parametrized problem (see docs/benchmarks.md); it is kept here ONLY so
tests can assert the fast version returns the same weights.
"""

import cvxpy as cp
import numpy as np

from mltrading.portfolio.risk import TRADING_DAYS_PER_YEAR


def _psd_factor(cov: np.ndarray) -> np.ndarray:
    """L with L @ L.T == cov (eigen-decomposition; robust to tiny negative eigenvalues)."""
    vals, vecs = np.linalg.eigh(cov)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def solve_reference(alpha, prev, factor, betas, sector_ids, n_sectors, cfg, cost_frac, holding_days, use_turnover):
    n = len(alpha)
    w = cp.Variable(n)
    risk_term = cp.sum_squares(factor.T @ w)  # w' Sigma_d w
    turnover = cp.norm1(w - prev)
    objective = (
        alpha @ w
        - 0.5 * cfg.risk_aversion * holding_days * risk_term
        - cost_frac * turnover
    )
    constraints = [
        cp.norm1(w) <= cfg.max_gross,
        cp.abs(cp.sum(w)) <= cfg.net_tolerance,
        cp.abs(w) <= cfg.max_abs_weight,
        cp.abs(betas @ w) <= cfg.beta_limit,
        TRADING_DAYS_PER_YEAR * risk_term <= cfg.target_vol_annual**2,
    ]
    for s in range(n_sectors):
        constraints.append(cp.abs(cp.sum(w[sector_ids == s])) <= cfg.sector_net_limit)
    if use_turnover:
        constraints.append(turnover <= cfg.turnover_limit)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status == "optimal_inaccurate":
        # first pass hit the solver's default tolerance limits; retry tighter before accepting
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500)
    return w, problem


