"""Execution simulator: turns an order into a fill with explicit costs.

Assumptions (all stated, none free):

  * Timing: an order decided after the close of day t fills at the OPEN
    of the next trading day for that ticker. If the ticker has no bar
    that day the order lapses (it is not retried); the next rebalance
    re-decides.
  * Price: the open is treated as the mid. A buy pays
    mid * (1 + (half_spread + slippage)), a sell receives
    mid * (1 - (half_spread + slippage)).
  * Commission: bps of traded notional at the mid.
  * Liquidity: a single order may not exceed `adv_participation_cap` of
    the ticker's trailing 20-day average DOLLAR volume known at decision
    time; anything larger is truncated (partial fill).
  * NOT modeled: market impact that grows with order size beyond the
    flat slippage, intraday price path, opening-auction dynamics, short
    availability/recall, financing on the long side, taxes.
"""

from __future__ import annotations

import numpy as np

from mltrading.backtest.events import FillEvent, OrderEvent
from mltrading.config import CostConfig

BPS = 1e-4


def simulate_fill(order: OrderEvent, date, open_price: float, cfg: CostConfig) -> FillEvent | None:
    if not np.isfinite(open_price) or open_price <= 0 or order.shares == 0:
        return None
    shares = order.shares
    if np.isfinite(order.ref_dollar_volume) and order.ref_dollar_volume > 0:
        max_shares = int(cfg.adv_participation_cap * order.ref_dollar_volume // open_price)
        shares = int(np.sign(shares)) * min(abs(shares), max_shares)
    if shares == 0:
        return None
    sign = np.sign(shares)
    notional = abs(shares) * open_price
    spread_cost = notional * cfg.half_spread_bps * BPS
    slippage_cost = notional * cfg.slippage_bps * BPS
    fill_price = open_price * (1 + sign * (cfg.half_spread_bps + cfg.slippage_bps) * BPS)
    return FillEvent(
        date=date, decision_date=order.decision_date, ticker=order.ticker, shares=int(shares),
        mid_price=float(open_price), fill_price=float(fill_price),
        commission=float(notional * cfg.commission_bps * BPS),
        spread_cost=float(spread_cost), slippage_cost=float(slippage_cost),
    )
