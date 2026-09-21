"""The risk-aware strategy may only use data dated <= the decision date."""

import numpy as np
import pandas as pd

from mltrading.backtest.engine import MarketData
from mltrading.backtest.strategies import BaselineStrategy, RiskAwareStrategy
from mltrading.config import CostConfig, RiskConfig

TICKERS = [f"T{i:02d}" for i in range(20)]


def _data(seed=0, n_days=150, scramble_after=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rets = pd.DataFrame(rng.normal(0, 0.012, size=(n_days, 20)), index=dates, columns=TICKERS)
    if scramble_after is not None:
        future = rets.index > scramble_after
        rets.loc[future] = np.random.default_rng(seed + 99).normal(0, 0.05, size=(future.sum(), 20))
    close = 100 * (1 + rets).cumprod()
    sectors = pd.Series(["A"] * 10 + ["B"] * 10, index=TICKERS)
    dv = pd.DataFrame(1e9, index=dates, columns=TICKERS)
    return MarketData(close.copy(), close, dv, rets, sectors), dates


def _weights(data, date):
    strat = RiskAwareStrategy(RiskConfig(), CostConfig(), holding_days=5)
    scores = pd.Series(np.random.default_rng(5).normal(0, 0.003, 20), index=TICKERS)
    return strat.target_weights(date, scores, pd.Series(0.0, index=TICKERS), data), strat


def test_risk_aware_weights_ignore_future_data():
    base, dates = _data()
    date = dates[100]
    w1, _ = _weights(base, date)
    scrambled, _ = _data(scramble_after=date)
    w2, _ = _weights(scrambled, date)
    pd.testing.assert_series_equal(w1, w2)


def test_risk_aware_logs_constraint_check_and_no_violations():
    data, dates = _data()
    _, strat = _weights(data, dates[100])
    row = strat.log[-1]
    assert row["status"] in ("optimal", "optimal_inaccurate") and row["violations"] == 0
    assert row["gross"] <= RiskConfig().max_gross + 1e-4


def test_risk_aware_holds_when_history_is_too_short():
    data, dates = _data()
    w, strat = _weights(data, dates[10])
    assert strat.log[-1]["status"] == "skipped_insufficient_data" and (w == 0).all()


def test_baseline_strategy_is_scale_free():
    data, _ = _data()
    s = pd.Series(np.arange(20.0), index=TICKERS)
    assert BaselineStrategy().target_weights(None, s, None, data).equals(BaselineStrategy().target_weights(None, s * 7, None, data))


def test_strategy_holds_current_book_when_post_solve_check_fails(monkeypatch):
    import mltrading.backtest.strategies as strategies_mod

    data, dates = _data()
    monkeypatch.setattr(strategies_mod, "check_constraints", lambda *a, **k: ["forced violation"])
    strat = RiskAwareStrategy(RiskConfig(), CostConfig(), holding_days=5)
    scores = pd.Series(np.random.default_rng(5).normal(0, 0.003, 20), index=TICKERS)
    current = pd.Series(0.01, index=TICKERS)
    w = strat.target_weights(dates[100], scores, current, data)
    assert w.equals(current) and strat.log[-1]["status"] == "rejected_violations"
