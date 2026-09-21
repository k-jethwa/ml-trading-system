"""Known-answer tests for the baseline and risk-aware portfolios."""

import numpy as np
import pandas as pd
import pytest

from mltrading.config import RiskConfig
from mltrading.portfolio.construct import baseline_weights
from mltrading.portfolio.optimize import optimize_weights
from mltrading.portfolio.risk import (
    check_constraints,
    estimate_risk,
    ex_ante_vol_annual,
    exposure_summary,
)

TICKERS = [f"T{i:02d}" for i in range(20)]
SECTORS = pd.Series(["A"] * 10 + ["B"] * 10, index=TICKERS)


def _returns(seed=0, n_days=120):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.008, size=(n_days, 1))
    idio = rng.normal(0, 0.012, size=(n_days, len(TICKERS)))
    return pd.DataFrame(0.9 * market + idio, columns=TICKERS)


# ---------- baseline ----------

def test_baseline_is_long_top_decile_short_bottom_dollar_neutral():
    scores = pd.Series(np.arange(20.0), index=TICKERS)
    w = baseline_weights(scores, 0.10)  # k = 2
    assert w[["T18", "T19"]].tolist() == [0.5, 0.5]
    assert w[["T00", "T01"]].tolist() == [-0.5, -0.5]
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(2.0)
    assert (w != 0).sum() == 4


def test_baseline_uses_only_score_order_not_scale():
    s = pd.Series(np.arange(20.0), index=TICKERS)
    assert baseline_weights(s).equals(baseline_weights(s * 1e-6 + 100))


def test_baseline_tie_break_is_deterministic():
    s = pd.Series(1.0, index=TICKERS)
    assert baseline_weights(s).equals(baseline_weights(s.sample(frac=1, random_state=3)).reindex(TICKERS))


def test_baseline_ignores_nan_scores_and_handles_tiny_universe():
    s = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
    w = baseline_weights(s)
    assert list(w.index) == ["a", "c"] and w["c"] == 1.0 and w["a"] == -1.0
    assert baseline_weights(pd.Series([1.0], index=["a"])).tolist() == [0.0]


# ---------- risk model ----------

def test_estimated_covariance_is_positive_definite_and_beta_averages_to_one():
    risk = estimate_risk(_returns())
    assert np.linalg.eigvalsh(risk.cov_daily.to_numpy()).min() > 0
    assert risk.betas.mean() == pytest.approx(1.0, abs=1e-9)  # EW-market betas average to exactly 1


def test_ex_ante_vol_known_answer():
    cov = pd.DataFrame(np.diag([0.0001, 0.0004]), index=["a", "b"], columns=["a", "b"])
    w = pd.Series({"a": 1.0, "b": -1.0})
    assert ex_ante_vol_annual(w, cov) == pytest.approx(np.sqrt(0.0005 * 252))


# ---------- risk-aware optimizer ----------

def _opt(alpha, prev=None, cfg=None, cost=0.0005, seed=0):
    cfg = cfg or RiskConfig()
    risk = estimate_risk(_returns(seed))
    return optimize_weights(alpha, prev, risk, SECTORS, cfg, cost, 5), risk


def test_optimizer_satisfies_every_constraint():
    rng = np.random.default_rng(1)
    alpha = pd.Series(rng.normal(0, 0.004, len(TICKERS)), index=TICKERS)
    res, risk = _opt(alpha)
    assert res.status in ("optimal", "optimal_inaccurate")
    assert check_constraints(res.weights, None, SECTORS, risk, RiskConfig(), tol=1e-5, enforce_turnover=False) == []


def test_optimizer_goes_long_high_alpha_short_low_alpha():
    alpha = pd.Series(np.linspace(-0.01, 0.01, 20), index=TICKERS)
    res, _ = _opt(alpha)
    w = res.weights
    assert w.iloc[-5:].sum() > 0 > w.iloc[:5].sum()
    assert np.corrcoef(w.to_numpy(), alpha.to_numpy())[0, 1] > 0.5


def test_zero_alpha_gives_no_position():
    res, _ = _opt(pd.Series(0.0, index=TICKERS))
    assert res.weights.abs().sum() < 1e-6


def test_high_costs_suppress_trading_when_alpha_is_small():
    alpha = pd.Series(np.linspace(-0.0004, 0.0004, 20), index=TICKERS)
    cheap, _ = _opt(alpha, cost=0.0)
    costly, _ = _opt(alpha, cost=0.01)
    assert costly.weights.abs().sum() < cheap.weights.abs().sum()
    assert costly.weights.abs().sum() < 1e-6


def test_turnover_limit_is_respected_and_holdings_are_sticky():
    alpha = pd.Series(np.linspace(-0.01, 0.01, 20), index=TICKERS)
    prev = pd.Series(0.0, index=TICKERS)
    cfg = RiskConfig(turnover_limit=0.2)
    res, _ = _opt(alpha, prev=prev, cfg=cfg)
    assert (res.weights - prev).abs().sum() <= 0.2 + 1e-6
    assert not res.turnover_constraint_dropped


def test_sector_neutrality_binds_when_alpha_favors_one_sector():
    alpha = pd.Series([0.01] * 10 + [-0.01] * 10, index=TICKERS)
    res, _ = _opt(alpha, cfg=RiskConfig(sector_net_limit=0.05))
    net = res.weights.groupby(SECTORS).sum()
    assert net.abs().max() <= 0.05 + 1e-6


def test_exposure_summary_reports_gross_net_and_sector():
    w = pd.Series({"T00": 0.3, "T01": -0.1, "T10": -0.2})
    ex = exposure_summary(w, SECTORS)
    assert ex["gross"] == pytest.approx(0.6) and ex["net"] == pytest.approx(0.0)
    assert ex["max_abs_sector_net"] == pytest.approx(0.2)


# ---------- compile-once optimizer == original formulation ----------

def _reference_weights(alpha, prev, risk, cfg, cost, use_turnover):
    from tests.portfolio.reference_optimizer import _psd_factor, solve_reference

    tickers = list(alpha.index)
    codes, uniques = pd.factorize(SECTORS.reindex(tickers))
    w, problem = solve_reference(
        alpha.to_numpy(), prev.reindex(tickers).fillna(0.0).to_numpy(), _psd_factor(risk.cov_daily.loc[tickers, tickers].to_numpy()),
        risk.betas.reindex(tickers).to_numpy(), codes, len(uniques), cfg, cost, 5, use_turnover,
    )
    return pd.Series(np.asarray(w.value).ravel(), index=tickers), problem.value


@pytest.mark.parametrize("seed", range(6))
def test_fast_optimizer_matches_reference_formulation(seed):
    rng = np.random.default_rng(seed)
    n_elig = [20, 20, 17, 14, 20, 12][seed]
    tickers = list(rng.permutation(TICKERS)[:n_elig])
    alpha = pd.Series(rng.normal(0, 0.004, n_elig), index=tickers)
    risk = estimate_risk(_returns(seed)[tickers])
    prev = pd.Series(rng.normal(0, 0.03, n_elig), index=tickers) if seed % 2 else None
    cfg = RiskConfig()
    res = optimize_weights(alpha, prev, risk, SECTORS, cfg, 0.0005, 5)
    ref_w, ref_obj = _reference_weights(alpha, prev if prev is not None else pd.Series(0.0, index=tickers), risk, cfg, 0.0005, prev is not None)
    assert res.status in ("optimal", "optimal_inaccurate")
    assert set(res.weights.index) == set(tickers)
    # Same optimum: objectives agree to ~1e-9. Weights can differ slightly (observed up to ~2e-4, i.e. 0.02%
    # of NAV) in near-flat directions of the objective, where two solver runs stop at different but equally
    # optimal points, so the weight tolerance reflects that degeneracy rather than solver precision.
    assert res.info["objective"] == pytest.approx(ref_obj, abs=1e-8)
    assert np.abs(res.weights.reindex(tickers) - ref_w).max() < 1e-3


def test_problem_is_compiled_once_and_reused():
    from mltrading.portfolio import optimize

    optimize._COMPILED.clear()
    alpha = pd.Series(np.linspace(-0.005, 0.005, 20), index=TICKERS)
    risk = estimate_risk(_returns())
    optimize_weights(alpha, None, risk, SECTORS, RiskConfig(), 0.0005, 5)
    optimize_weights(alpha * 2, None, estimate_risk(_returns(1)), SECTORS, RiskConfig(), 0.0005, 5)
    assert len(optimize._COMPILED) == 1
    (compiled,) = optimize._COMPILED.values()
    assert compiled.problem.is_dpp()


def test_vectorized_betas_match_pairwise_covariance_definition():
    window = _returns()
    risk = estimate_risk(window)
    market = window.mean(axis=1)
    expected = pd.Series({t: window[t].cov(market) / market.var(ddof=1) for t in window.columns})
    assert np.allclose(risk.betas.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)


def test_numerical_solver_failure_holds_book_and_never_relaxes_turnover(monkeypatch):
    """A SolverError must yield status 'failed' + the previous weights, not a silently looser retry."""
    import cvxpy as cp

    from mltrading.portfolio import optimize

    calls = []

    def boom(self, *args):
        calls.append(args[-1])  # turnover cap used
        raise cp.error.SolverError("numerical trouble")

    monkeypatch.setattr(optimize._CompiledProblem, "solve", boom)
    alpha = pd.Series(np.linspace(-0.005, 0.005, 20), index=TICKERS)
    prev = pd.Series(0.02, index=TICKERS)
    res = optimize_weights(alpha, prev, estimate_risk(_returns()), SECTORS, RiskConfig(), 0.0005, 5)
    assert res.status == "failed" and res.weights.equals(prev)
    assert len(calls) == 1 and not res.turnover_constraint_dropped
