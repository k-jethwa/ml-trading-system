"""Trading metrics (project_spec.md section 8).

Returns are daily NAV changes. "Gross" adds cumulative costs back into
NAV (what the strategy would have earned at zero cost); "net" is what
was actually earned after commissions, spread, slippage and borrow.

Sharpe/Sortino use zero as the risk-free rate: the book is
dollar-neutral, so the return is (approximately) an excess return; this
flatters nothing but ignores the interest on cash collateral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan


def _sortino(r: pd.Series) -> float:
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def compute_metrics(
    daily: pd.DataFrame,
    positions: pd.DataFrame,
    sectors: pd.Series,
    market_returns: pd.Series,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Metrics over [start, end]. `market_returns`: equal-weight universe daily return by date."""
    d = daily.loc[start:end]
    pos = positions.loc[d.index]
    nav = d["nav"]
    costs = d[["commission", "spread", "slippage", "borrow"]].sum(axis=1)
    net_r = nav.pct_change().dropna()
    prev_nav = nav.shift(1)
    # gross return: the same P&L with the period's incremental costs added back
    gross_r = ((nav.diff() + costs.diff()) / prev_nav).dropna()

    years = len(d) / TRADING_DAYS
    exposure_gross = (pos.abs().sum(axis=1) / nav)
    exposure_net = (pos.sum(axis=1) / nav)
    sector_net = pos.T.groupby(sectors.reindex(pos.columns)).sum().T.div(nav, axis=0)

    mkt = market_returns.reindex(net_r.index)
    beta, alpha_ann, alpha_t = np.nan, np.nan, np.nan
    if mkt.notna().all() and mkt.var(ddof=1) > 0 and len(net_r) > 10:
        # OLS of strategy daily return on the equal-weight universe return: r = a + b*m + e
        beta = float(net_r.cov(mkt) / mkt.var(ddof=1))
        resid = net_r - beta * mkt
        alpha_daily = float(resid.mean())
        alpha_ann = alpha_daily * TRADING_DAYS
        se = float(resid.std(ddof=2) / np.sqrt(len(resid)))
        alpha_t = alpha_daily / se if se > 0 else np.nan

    period_rets = nav.iloc[::5].pct_change().dropna()
    n = len(net_r)
    return {
        "start": str(d.index[0].date()),
        "end": str(d.index[-1].date()),
        "n_days": len(d),
        "total_return_net": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "ann_return_net": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else np.nan,
        "ann_return_gross": float(gross_r.mean() * TRADING_DAYS),
        "ann_vol": float(net_r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe_net": _sharpe(net_r),
        "sharpe_gross": _sharpe(gross_r),
        "sortino_net": _sortino(net_r),
        "max_drawdown": _max_drawdown(nav),
        "annual_turnover": float((d["traded_notional"].iloc[-1] - d["traded_notional"].iloc[0]) / nav.mean() / years) if years > 0 else np.nan,
        "annual_cost_drag": float((costs.iloc[-1] - costs.iloc[0]) / nav.mean() / years) if years > 0 else np.nan,
        "hit_rate_daily": float((net_r > 0).mean()),
        "hit_rate_5d": float((period_rets > 0).mean()) if len(period_rets) else np.nan,
        "avg_gross_exposure": float(exposure_gross.mean()),
        "avg_net_exposure": float(exposure_net.mean()),
        "avg_max_abs_sector_net": float(sector_net.abs().max(axis=1).mean()),
        "beta_to_universe": beta,
        "alpha_ann_vs_universe": alpha_ann,
        "alpha_tstat": alpha_t,
        "n_obs": int(n),
    }
