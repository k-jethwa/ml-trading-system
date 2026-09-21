"""Known-answer tests for backtest timing, accounting, costs and determinism."""

import numpy as np
import pandas as pd
import pytest

from mltrading.backtest.engine import MarketData, run_backtest
from mltrading.backtest.metrics import compute_metrics
from mltrading.config import CostConfig

ZERO = CostConfig(commission_bps=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0, adv_participation_cap=1.0)
DATES = pd.bdate_range("2024-01-02", periods=8)


def _wide(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES)


def _data(open_, close, dv=1e12, sectors=None) -> MarketData:
    tickers = list(close.columns)
    return MarketData(
        open=open_, close=close,
        dollar_volume=pd.DataFrame(dv, index=DATES, columns=tickers),
        returns=close.pct_change(),
        sectors=pd.Series(sectors or {t: "S" for t in tickers}),
    )


class FixedWeights:
    def __init__(self, w): self.w = pd.Series(w, dtype=float)
    def target_weights(self, date, scores, current_weights, data): return self.w


def _scores(dates, tickers):
    return pd.DataFrame([(d, t, 0.0) for d in dates for t in tickers], columns=["date", "ticker", "pred"])


def _run(data, weights, costs=ZERO, dates=None, every=100, capital=1_000_000.0):
    dates = DATES[:1] if dates is None else dates
    return run_backtest(_scores(dates, data.close.columns), data, FixedWeights(weights), capital, every, costs)


def test_fill_happens_at_next_open_never_the_decision_close():
    close = _wide({"A": [100] * 8})
    open_ = _wide({"A": [100, 110, 110, 110, 110, 110, 110, 110]})  # overnight gap up on day 2
    res = _run(_data(open_, close), {"A": 1.0})
    fill = res.fills.iloc[0]
    assert fill["decision_date"] == DATES[0] and fill["date"] == DATES[1]
    assert fill["mid_price"] == 110.0


def test_overnight_gap_gives_no_free_return():
    """Decide at close 100, tomorrow opens 110: we pay 110, so NAV cannot rise on the gap."""
    close = _wide({"A": [100, 110, 110, 110, 110, 110, 110, 110]})
    open_ = _wide({"A": [100, 110, 110, 110, 110, 110, 110, 110]})
    res = _run(_data(open_, close), {"A": 1.0})
    assert res.daily["nav"].iloc[1] == pytest.approx(1_000_000.0)


def test_pnl_known_answer_long_short():
    close = _wide({"A": [100, 100, 110, 110, 110, 110, 110, 110], "B": [100] * 8})
    open_ = close.copy()
    res = _run(_data(open_, close), {"A": 0.5, "B": -0.5})
    # 5000 sh long A, 5000 sh short B; A +10% => +50,000
    assert res.daily["nav"].iloc[-1] == pytest.approx(1_050_000.0)
    assert res.daily["nav"].iloc[0] == pytest.approx(1_000_000.0)  # nothing held on the decision day


def test_costs_are_charged_and_itemized():
    close = _wide({"A": [100] * 8, "B": [100] * 8})
    costs = CostConfig(commission_bps=1, half_spread_bps=2, slippage_bps=3, borrow_bps_annual=0, adv_participation_cap=1.0)
    res = _run(_data(close.copy(), close), {"A": 0.5, "B": -0.5}, costs)
    traded = 5000 * 100 * 2
    last = res.daily.iloc[-1]
    assert last["commission"] == pytest.approx(traded * 1e-4)
    assert last["spread"] == pytest.approx(traded * 2e-4)
    assert last["slippage"] == pytest.approx(traded * 3e-4)
    # flat prices => NAV loss equals total costs exactly
    assert last["nav"] == pytest.approx(1_000_000.0 - traded * 6e-4)


def test_buys_pay_up_and_sells_receive_less():
    close = _wide({"A": [100] * 8, "B": [100] * 8})
    costs = CostConfig(commission_bps=0, half_spread_bps=10, slippage_bps=0, borrow_bps_annual=0, adv_participation_cap=1.0)
    res = _run(_data(close.copy(), close), {"A": 0.5, "B": -0.5}, costs)
    f = res.fills.set_index("ticker")
    assert f.loc["A", "fill_price"] == pytest.approx(100.1)
    assert f.loc["B", "fill_price"] == pytest.approx(99.9)


def test_borrow_cost_accrues_on_short_market_value():
    close = _wide({"A": [100] * 8, "B": [100] * 8})
    costs = CostConfig(commission_bps=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=252.0, adv_participation_cap=1.0)
    res = _run(_data(close.copy(), close), {"A": 0.5, "B": -0.5}, costs)
    # short MV 500,000 at 252 bps/yr => 0.0252/252 = 1e-4/day => 50 per day; short exists from day-2 close on,
    # so borrow accrues on days 3..8 (accrual uses the PRIOR close's short MV)
    assert res.daily["borrow"].iloc[-1] == pytest.approx(50.0 * 6)


def test_liquidity_cap_truncates_large_orders():
    close = _wide({"A": [100] * 8})
    costs = CostConfig(commission_bps=0, half_spread_bps=0, slippage_bps=0, borrow_bps_annual=0, adv_participation_cap=0.05)
    data = _data(close.copy(), close, dv=2_000_000.0)  # cap = 5% * 2M = $100k => 1000 shares
    res = _run(data, {"A": 0.5}, costs)
    assert res.fills["shares"].iloc[0] == 1000
    assert res.counters["partial_fills"] == 1


def test_order_lapses_when_ticker_has_no_bar_on_fill_date():
    close = _wide({"A": [100] * 8})
    open_ = close.copy()
    open_.iloc[1, 0] = np.nan
    res = _run(_data(open_, close), {"A": 0.5})
    assert res.counters["lapsed_orders"] == 1 and len(res.fills) == 0


def test_rebalance_cadence_and_liquidation_of_dropped_names():
    close = _wide({"A": [100] * 8})

    class Flip:
        def __init__(self): self.calls = 0
        def target_weights(self, date, scores, cw, data):
            self.calls += 1
            return pd.Series({"A": 0.5 if self.calls == 1 else 0.0})

    res = run_backtest(_scores(DATES, ["A"]), _data(close.copy(), close), Flip(), 1_000_000.0, 3, ZERO)
    assert list(res.rebalances["date"]) == [DATES[0], DATES[3], DATES[6]]
    assert res.positions["A"].iloc[2] > 0 and res.positions["A"].iloc[5] == 0  # bought after d0, sold after d3


def test_deterministic_replay():
    rng = np.random.default_rng(0)
    close = _wide({"A": 100 + rng.normal(size=8).cumsum(), "B": 50 + rng.normal(size=8).cumsum()})
    data = _data(close.copy(), close)
    a = _run(data, {"A": 0.4, "B": -0.4}, CostConfig(), dates=DATES, every=2)
    b = _run(data, {"A": 0.4, "B": -0.4}, CostConfig(), dates=DATES, every=2)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.fills, b.fills)


# ---------- metrics ----------

def _result_from_nav(nav):
    idx = pd.bdate_range("2020-01-01", periods=len(nav))
    daily = pd.DataFrame({"nav": nav, "commission": 0.0, "spread": 0.0, "slippage": 0.0, "borrow": 0.0, "traded_notional": 0.0}, index=idx)
    positions = pd.DataFrame({"A": 0.0}, index=idx)
    return daily, positions, pd.Series({"A": "S"}), pd.Series(np.linspace(-0.001, 0.001, len(nav)), index=idx)


def test_max_drawdown_known_answer():
    daily, pos, sec, mkt = _result_from_nav([100.0, 120.0, 90.0, 95.0, 130.0])
    m = compute_metrics(daily, pos, sec, mkt)
    assert m["max_drawdown"] == pytest.approx(90 / 120 - 1)


def test_sharpe_of_constant_growth_is_undefined_and_of_known_series_matches_formula():
    daily, pos, sec, mkt = _result_from_nav(list(100 * np.cumprod([1.0] + [1.01, 0.99] * 10)))
    m = compute_metrics(daily, pos, sec, mkt)
    r = daily["nav"].pct_change().dropna()
    assert m["sharpe_net"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert m["hit_rate_daily"] == pytest.approx(0.5)
