"""Risk-aware portfolio: constrained mean-variance optimization with costs.

For decision date t the optimizer solves

    maximize    alpha' w  -  (gamma/2) * w' (H * Sigma_d) w  -  c * ||w - w_prev||_1
    subject to  ||w||_1                     <= max_gross
                |1' w|                      <= net_tolerance
                |w_i|                       <= max_abs_weight        (0 if ineligible)
                |sum_{i in sector s} w_i|   <= sector_net_limit      for each sector s
                |beta' w|                   <= beta_limit
                sqrt(252 * w' Sigma_d w)    <= target_vol_annual
                ||w - w_prev||_1            <= turnover_limit        (skipped at first rebalance)

  alpha   the model's predicted 5-day sector-relative return per stock
          (return units, so it is directly comparable to `c`)
  Sigma_d Ledoit-Wolf daily covariance; H = holding period in days, so the
          risk term is the variance of the return earned until the next
          rebalance
  c       one-way transaction cost as a fraction of traded notional
          (commission + half-spread + slippage); trades only happen when
          the expected alpha gain exceeds the cost
  gamma   risk aversion (RiskConfig.risk_aversion)

Each term is explainable: alpha is what we want, the variance term
penalizes risk, the L1 term is the cost of changing the book, and the
constraints are hard limits (exposure, concentration, sector, beta,
volatility, turnover) rather than penalties.

If the solver fails numerically the caller receives status "failed" and
the previous weights, never a silently bad book. If the problem is
infeasible with the turnover limit (drifted prices can make the old book
violate the new risk limits) it is retried without that one limit and
the result is flagged `turnover_constraint_dropped`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from mltrading.config import RiskConfig
from mltrading.portfolio.risk import TRADING_DAYS_PER_YEAR, RiskModel


@dataclass
class OptimizationResult:
    weights: pd.Series
    status: str
    turnover_constraint_dropped: bool = False
    info: dict = field(default_factory=dict)


def _psd_factor(cov: np.ndarray) -> np.ndarray:
    """L with L @ L.T == cov (eigen-decomposition; robust to tiny negative eigenvalues)."""
    vals, vecs = np.linalg.eigh(cov)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


class _CompiledProblem:
    """The optimization above, expressed once over a FIXED universe with cvxpy Parameters.

    cvxpy spends most of a solve canonicalizing the problem (measured: 11.1s of
    15.0s across 384 rebalances). The structure never changes between rebalances,
    only the numbers do, so the problem is built once as a DPP (disciplined
    parametrized programming) problem and re-solved by updating Parameter values.

    Fixed size is handled with an eligibility mask: an ineligible ticker gets a
    zero position bound, zero alpha, zero beta and zero rows in the covariance
    factor, which is exactly equivalent to leaving it out of the problem.
    """

    def __init__(self, sector_codes: np.ndarray, n_sectors: int, cfg: RiskConfig, cost_frac: float, holding_days: int):
        n = len(sector_codes)
        self.alpha = cp.Parameter(n)
        self.prev = cp.Parameter(n)
        self.factor = cp.Parameter((n, n))
        self.betas = cp.Parameter(n)
        self.elig = cp.Parameter(n, nonneg=True)
        self.turnover_cap = cp.Parameter(nonneg=True)
        self.w = cp.Variable(n)
        w = self.w
        risk_term = cp.sum_squares(self.factor.T @ w)  # w' Sigma_d w
        turnover = cp.norm1(w - self.prev)
        objective = self.alpha @ w - 0.5 * cfg.risk_aversion * holding_days * risk_term - cost_frac * turnover
        constraints = [
            cp.norm1(w) <= cfg.max_gross,
            cp.abs(cp.sum(w)) <= cfg.net_tolerance,
            cp.abs(w) <= cfg.max_abs_weight * self.elig,
            cp.abs(self.betas @ w) <= cfg.beta_limit,
            # target vol as a direct second-order-cone constraint. The algebraically identical
            # `252 * w'Sigma w <= sigma^2` form is poorly scaled for the interior-point solver: on 172
            # captured instances it produced 1 SolverError + 8 inaccurate solves, the SOC form 0.
            cp.norm2(self.factor.T @ w) <= cfg.target_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR),
            turnover <= self.turnover_cap,
        ]
        for s in range(n_sectors):
            constraints.append(cp.abs(cp.sum(w[sector_codes == s])) <= cfg.sector_net_limit)
        self.problem = cp.Problem(cp.Maximize(objective), constraints)
        if not self.problem.is_dpp():
            raise RuntimeError("optimizer problem is not DPP; it would be re-canonicalized on every solve")

    def solve(self, alpha, prev, factor, betas, elig, turnover_cap):
        self.alpha.value, self.prev.value, self.factor.value = alpha, prev, factor
        self.betas.value, self.elig.value, self.turnover_cap.value = betas, elig, turnover_cap
        self.problem.solve(solver=cp.CLARABEL)
        return self.w.value, self.problem


_COMPILED: dict[tuple, _CompiledProblem] = {}


def _compiled_problem(universe: tuple[str, ...], sector_codes, n_sectors, cfg, cost_frac, holding_days) -> _CompiledProblem:
    key = (universe, tuple(sector_codes), cfg, cost_frac, holding_days)
    if key not in _COMPILED:
        _COMPILED[key] = _CompiledProblem(np.asarray(sector_codes), n_sectors, cfg, cost_frac, holding_days)
    return _COMPILED[key]


def optimize_weights(
    alpha: pd.Series,
    prev_weights: pd.Series | None,
    risk: RiskModel,
    sectors: pd.Series,
    cfg: RiskConfig,
    cost_frac: float,
    holding_days: int,
) -> OptimizationResult:
    """`sectors` indexes the FULL universe (fixes the problem's dimension); `alpha` and `risk`
    cover the eligible subset."""
    universe = tuple(sectors.index)
    eligible = [t for t in alpha.dropna().index if t in risk.cov_daily.index]
    pos = pd.Index(universe).get_indexer(eligible)
    n = len(universe)

    def embed(values: np.ndarray) -> np.ndarray:
        full = np.zeros(n)
        full[pos] = values
        return full

    factor = np.zeros((n, n))
    factor[np.ix_(pos, pos)] = _psd_factor(risk.cov_daily.loc[eligible, eligible].to_numpy())
    a = embed(alpha.reindex(eligible).to_numpy(dtype=float))
    betas = embed(risk.betas.reindex(eligible).to_numpy(dtype=float))
    elig = embed(np.ones(len(eligible)))
    have_prev = prev_weights is not None
    prev = embed(prev_weights.reindex(eligible).fillna(0.0).to_numpy()) if have_prev else np.zeros(n)
    sector_codes, uniques = pd.factorize(sectors)
    compiled = _compiled_problem(universe, sector_codes, len(uniques), cfg, cost_frac, holding_days)

    # holdings dropped from the eligible set are left out of the optimization (weight 0), so
    # turnover is under-counted by their size; the engine liquidates them regardless.
    #
    # Fallback policy: only genuine INFEASIBILITY relaxes the turnover limit (drifted prices can make
    # the old book violate the new risk limits; the relaxation is flagged in the result). A numerical
    # SolverError never loosens a limit: it returns status "failed" and the caller holds its book.
    for use_turnover in ([True, False] if have_prev else [False]):
        # "no turnover limit" = the tightest cap that can never bind, ||w-prev||_1 <= max_gross + ||prev||_1.
        # (A huge sentinel such as 1e9 ruins the solver's conditioning.)
        cap = cfg.turnover_limit if use_turnover else cfg.max_gross + float(np.abs(prev).sum()) + 1.0
        try:
            value, problem = compiled.solve(a, prev, factor, betas, elig, cap)
        except cp.error.SolverError:
            break
        if problem.status in ("optimal", "optimal_inaccurate") and value is not None:
            weights = pd.Series(np.asarray(value).ravel()[pos], index=eligible)
            weights[weights.abs() < 1e-8] = 0.0
            return OptimizationResult(
                weights, problem.status, turnover_constraint_dropped=have_prev and not use_turnover,
                info={"objective": float(problem.value)},
            )
    fallback = prev_weights.copy() if have_prev else pd.Series(0.0, index=eligible)
    return OptimizationResult(fallback, "failed", info={})
