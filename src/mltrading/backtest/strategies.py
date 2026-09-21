"""Strategies plugged into the backtest engine.

Both see only (date, that date's scores, current weights, data dated
<= date); the engine never gives them a future row.
"""

from __future__ import annotations

import logging

from mltrading.config import CostConfig, RiskConfig
from mltrading.portfolio.construct import baseline_weights
from mltrading.portfolio.optimize import optimize_weights
from mltrading.portfolio.risk import check_constraints, estimate_risk, exposure_summary

logger = logging.getLogger(__name__)


class BaselineStrategy:
    """Long top decile / short bottom decile, equal weight (spec section 9, baseline)."""

    def __init__(self, quantile: float = 0.10):
        self.quantile = quantile

    def target_weights(self, date, scores, current_weights, data):
        return baseline_weights(scores, self.quantile)


class RiskAwareStrategy:
    """Constrained optimizer over predicted return, risk and costs (spec section 9, risk-aware)."""

    def __init__(self, risk: RiskConfig, costs: CostConfig, holding_days: int):
        self.risk = risk
        self.cost_frac = (costs.commission_bps + costs.half_spread_bps + costs.slippage_bps) * 1e-4
        self.holding_days = holding_days
        self.log: list[dict] = []

    def target_weights(self, date, scores, current_weights, data):
        window = data.returns.loc[:date].tail(self.risk.cov_lookback)
        complete = window.columns[window.notna().all()]
        eligible = [t for t in scores.dropna().index if t in complete]
        if len(window) < self.risk.cov_lookback or len(eligible) < 10:
            self.log.append({"date": date, "status": "skipped_insufficient_data", "violations": 0})
            return current_weights
        risk = estimate_risk(window[eligible])
        result = optimize_weights(
            scores.reindex(eligible), current_weights.reindex(eligible).fillna(0.0), risk,
            data.sectors, self.risk, self.cost_frac, self.holding_days,
        )
        if result.status == "failed":
            logger.warning("optimizer failed on %s; holding current weights", date.date())
            self.log.append({"date": date, "status": "failed", "violations": 0})
            return current_weights
        violations = check_constraints(
            result.weights, current_weights.reindex(eligible).fillna(0.0), data.sectors, risk, self.risk,
            tol=1e-4, enforce_turnover=not result.turnover_constraint_dropped,
        )
        self.log.append({
            "date": date, "status": result.status if not violations else "rejected_violations",
            "turnover_dropped": result.turnover_constraint_dropped,
            "violations": len(violations), **exposure_summary(result.weights, data.sectors, risk.betas),
        })
        if violations:
            # never trade a book that fails its own risk checks: hold what we have instead
            logger.warning("risk check failed on %s (%s); holding current weights", date.date(), "; ".join(violations))
            return current_weights
        return result.weights
