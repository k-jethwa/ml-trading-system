"""Explicit state-transition events (project_spec.md section 11).

    MarketEvent  a new trading day's bars are available
    FillEvent    an order from an earlier decision executed at today's open
    MarkEvent    end of day: mark the book to the close and record NAV
    SignalEvent  after the close: the strategy is asked for target weights
    OrderEvent   the strategy's decision, queued to execute at the NEXT open

Given the same event sequence and configuration the engine produces
identical state (tests/backtest/test_engine.py::test_deterministic_replay).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    date: pd.Timestamp


@dataclass(frozen=True)
class OrderEvent:
    decision_date: pd.Timestamp     # close on which the decision was made
    ticker: str
    shares: int                     # signed: + buy, - sell/short
    ref_dollar_volume: float        # trailing 20d average dollar volume known at decision time


@dataclass(frozen=True)
class FillEvent:
    date: pd.Timestamp              # execution date (strictly after decision_date)
    decision_date: pd.Timestamp
    ticker: str
    shares: int                     # filled quantity (may be smaller than ordered: liquidity cap)
    mid_price: float                # the open used as the reference price
    fill_price: float               # mid adjusted for half-spread and slippage
    commission: float
    spread_cost: float
    slippage_cost: float


@dataclass(frozen=True)
class MarkEvent:
    date: pd.Timestamp


@dataclass(frozen=True)
class SignalEvent:
    date: pd.Timestamp              # decision time: after the close of this date
    scores: pd.Series               # model predictions by ticker
