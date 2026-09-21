"""Portfolio accounting: cash, positions, costs, NAV.

NAV = cash + sum(shares * price). Short sales add their proceeds to
cash and carry a negative position value, so NAV moves only with price
changes and costs. Prices are dividend-adjusted (total-return) so
dividends are implicitly reinvested; see docs/limitations.md.
"""

from __future__ import annotations

import pandas as pd

from mltrading.backtest.events import FillEvent


class Book:
    def __init__(self, initial_capital: float, tickers: list[str]):
        self.cash = float(initial_capital)
        self.shares = pd.Series(0, index=tickers, dtype="int64")
        self.commission = 0.0
        self.spread = 0.0
        self.slippage = 0.0
        self.borrow = 0.0
        self.traded_notional = 0.0

    def apply_fill(self, fill: FillEvent) -> None:
        self.cash -= fill.shares * fill.fill_price + fill.commission
        self.shares[fill.ticker] += fill.shares
        self.commission += fill.commission
        self.spread += fill.spread_cost
        self.slippage += fill.slippage_cost
        self.traded_notional += abs(fill.shares) * fill.mid_price

    def accrue_borrow(self, prev_close: pd.Series, annual_bps: float) -> None:
        short_mv = float((-self.shares.clip(upper=0) * prev_close.reindex(self.shares.index).fillna(0.0)).sum())
        cost = short_mv * annual_bps * 1e-4 / 252
        self.cash -= cost
        self.borrow += cost

    def position_values(self, close: pd.Series) -> pd.Series:
        return self.shares * close.reindex(self.shares.index).fillna(0.0)

    def nav(self, close: pd.Series) -> float:
        return float(self.cash + self.position_values(close).sum())

    @property
    def total_costs(self) -> float:
        return self.commission + self.spread + self.slippage + self.borrow
