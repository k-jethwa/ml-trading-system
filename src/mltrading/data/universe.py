"""Tradable universe definition for V1.

LIMITATION (documented per project_spec.md section 3 and 13):
This universe is a static, present-day list of liquid large/mid-cap US
equities. It is NOT reconstructed from historical index membership, so
using it to backtest earlier periods introduces **survivorship bias** —
companies that were delisted, acquired, or dropped from an index before
today are absent by construction. yfinance does not expose point-in-time
historical constituents, so this limitation cannot be fully removed
without a paid point-in-time dataset (e.g. Norgate, Sharadar). V1 accepts
this limitation explicitly rather than hiding it; see docs/limitations.md.

Sector labels come from yfinance's `Ticker.info["sector"]`, which reflects
the CURRENT GICS sector classification, not the sector a company was
classified under at each historical date. This is a second, smaller
survivorship/point-in-time gap in the same spirit.
"""

from __future__ import annotations

# A fixed starting universe: ~50 liquid, well-known US equities spanning
# multiple sectors, chosen for name recognition and data availability
# rather than any historical index snapshot. Expand toward the spec's
# target of 100-500 names once the ingestion pipeline is validated.
INITIAL_UNIVERSE: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO", "ADBE", "CRM",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE", "LMT", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST",
    # Communication Services
    "DIS", "NFLX", "CMCSA", "T", "VZ",
]

UNIVERSE_VERSION = "v1-static-2026-08"
