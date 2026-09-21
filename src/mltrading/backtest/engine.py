"""Event-driven backtest engine (project_spec.md sections 10-11).

Per trading day d (in calendar order) the event queue runs

    MarketEvent(d)  -> accrue borrow on the book held overnight into d
                    -> execute yesterday's pending orders at d's OPEN  (FillEvents)
                    -> MarkEvent(d): mark to d's CLOSE, record NAV
                    -> SignalEvent(d)   if d is a rebalance date: the strategy
                                        sees scores dated d and returns target weights
    SignalEvent(d)  -> OrderEvents sized off d's close, queued for d+1's open

So a decision made with information through the close of d can never
trade at a price from d or earlier: the earliest fill is the next open.
`Strategy.target_weights` receives only data dated <= d.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from mltrading.backtest.events import (
    FillEvent,
    MarketEvent,
    MarkEvent,
    OrderEvent,
    SignalEvent,
)
from mltrading.backtest.execution import simulate_fill
from mltrading.backtest.state import Book
from mltrading.config import CostConfig

MIN_ORDER_NOTIONAL = 1_000.0


@dataclass(frozen=True)
class MarketData:
    """Wide (date x ticker) adjusted prices + liquidity, plus static sector map."""

    open: pd.DataFrame
    close: pd.DataFrame
    dollar_volume: pd.DataFrame
    returns: pd.DataFrame
    sectors: pd.Series

    @classmethod
    def from_features(cls, features: pd.DataFrame) -> MarketData:
        df = features.copy()
        # open is split-adjusted only; scale it onto the dividend-adjusted close so the
        # open->close move and the overnight gap are on the same (total-return) basis
        df["adj_open"] = df["open"] * df["adj_close"] / df["close"]
        wide = lambda col: df.pivot(index="date", columns="ticker", values=col).sort_index()
        sectors = df.drop_duplicates("ticker").set_index("ticker")["sector"].sort_index()
        return cls(wide("adj_open"), wide("adj_close"), wide("dollar_volume_20"), wide("ret_1d"), sectors)


class Strategy(Protocol):
    def target_weights(self, date: pd.Timestamp, scores: pd.Series, current_weights: pd.Series, data: MarketData) -> pd.Series:
        """Target weights (fraction of NAV) by ticker; tickers absent or NaN are liquidated."""


@dataclass
class BacktestResult:
    daily: pd.DataFrame                       # one row per date
    positions: pd.DataFrame                   # date x ticker position market value at the close
    fills: pd.DataFrame
    rebalances: pd.DataFrame                  # one row per rebalance decision
    counters: dict = field(default_factory=dict)


def run_backtest(
    scores: pd.DataFrame,
    data: MarketData,
    strategy: Strategy,
    initial_capital: float,
    rebalance_every: int,
    costs: CostConfig,
) -> BacktestResult:
    """`scores`: long frame with columns date, ticker, pred (one model's predictions)."""
    score_by_date = {d: g.set_index("ticker")["pred"].sort_index() for d, g in scores.groupby("date")}
    pred_dates = sorted(score_by_date)
    if not pred_dates:
        raise ValueError("No predictions to backtest")
    rebalance_dates = set(pred_dates[::rebalance_every])
    calendar = [d for d in data.close.index if d >= pred_dates[0]]
    close_ffill = data.close.ffill()
    tickers = list(data.close.columns)

    book = Book(initial_capital, tickers)
    queue: deque = deque()
    pending: list[OrderEvent] = []
    daily_rows, position_rows, fill_rows, rebalance_rows = [], {}, [], []
    counters = {"orders": 0, "fills": 0, "lapsed_orders": 0, "partial_fills": 0}
    prev_date = None

    for date in calendar:
        queue.append(MarketEvent(date))
        while queue:
            ev = queue.popleft()

            if isinstance(ev, MarketEvent):
                # 0. borrow for the night just ended, on the book as it stood at the prior
                #    close (before today's fills change it)
                if prev_date is not None:
                    book.accrue_borrow(close_ffill.loc[prev_date], costs.borrow_bps_annual)
                # 1. execute yesterday's decisions at today's open
                orders, pending = pending, []
                for order in orders:
                    fill = simulate_fill(order, ev.date, data.open.at[ev.date, order.ticker], costs)
                    if fill is None:
                        counters["lapsed_orders"] += 1
                        continue
                    if abs(fill.shares) < abs(order.shares):
                        counters["partial_fills"] += 1
                    queue.append(fill)
                queue.append(MarkEvent(ev.date))
                if ev.date in rebalance_dates:
                    queue.append(SignalEvent(ev.date, score_by_date[ev.date]))

            elif isinstance(ev, FillEvent):
                book.apply_fill(ev)
                counters["fills"] += 1
                fill_rows.append(ev.__dict__)

            elif isinstance(ev, MarkEvent):
                prices = close_ffill.loc[ev.date]
                pos = book.position_values(prices)
                nav = book.nav(prices)
                daily_rows.append({
                    "date": ev.date, "nav": nav, "cash": book.cash,
                    "long_mv": float(pos.clip(lower=0).sum()), "short_mv": float(-pos.clip(upper=0).sum()),
                    "commission": book.commission, "spread": book.spread, "slippage": book.slippage,
                    "borrow": book.borrow, "traded_notional": book.traded_notional,
                })
                position_rows[ev.date] = pos
                prev_date = ev.date

            elif isinstance(ev, SignalEvent):
                prices = close_ffill.loc[ev.date]
                nav = book.nav(prices)
                current_w = book.position_values(prices) / nav
                target = strategy.target_weights(ev.date, ev.scores, current_w, data).reindex(tickers).fillna(0.0)
                target_shares = (target * nav / prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                delta = (target_shares - book.shares).apply(np.trunc).astype("int64")
                n_orders = 0
                for ticker in tickers:
                    qty = int(delta[ticker])
                    if qty == 0 or abs(qty) * prices[ticker] < MIN_ORDER_NOTIONAL:
                        continue
                    queue.append(OrderEvent(ev.date, ticker, qty, float(data.dollar_volume.at[ev.date, ticker])))
                    n_orders += 1
                rebalance_rows.append({
                    "date": ev.date, "n_orders": n_orders, "nav": nav,
                    "target_gross": float(target.abs().sum()), "target_net": float(target.sum()),
                })

            elif isinstance(ev, OrderEvent):
                pending.append(ev)
                counters["orders"] += 1

    daily = pd.DataFrame(daily_rows).set_index("date")
    return BacktestResult(
        daily=daily,
        positions=pd.DataFrame(position_rows).T.sort_index(),
        fills=pd.DataFrame(fill_rows),
        rebalances=pd.DataFrame(rebalance_rows),
        counters=counters,
    )
