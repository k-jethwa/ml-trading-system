# V1 Known Limitations — Data Source

Per `project_spec.md` section 3 ("The tradable universe must be defined
historically... must not use present-day index membership as a proxy for
past eligibility without documenting the resulting survivorship
limitation") and section 13 (survivorship bias / data leakage as
invalidating failure modes), this file records the limitations introduced
by choosing yfinance as the sole V1 data source (a deliberate $0-cost
decision).

## 1. Survivorship bias in the universe

`mltrading.data.universe.INITIAL_UNIVERSE` is a static, hand-picked list
of currently-liquid large/mid-cap US equities. It is not derived from any
historical index snapshot. Consequences:

- Companies that were delisted, went bankrupt, or were acquired before
  today are absent, even during periods when they were liquid and
  tradable. Backtests over those periods will look better than a
  contemporaneous strategy could actually have performed, because the
  losers have been filtered out with hindsight.
- yfinance does not expose historical point-in-time index constituents
  (e.g. "who was in the S&P 500 on 2016-03-01"), so this cannot be fixed
  without a different data source.

**Mitigation in V1:** none beyond disclosure. Any reported backtest
result must state that it is subject to survivorship bias from universe
construction, and results should be treated as an upper bound on
realistic performance, not a realistic estimate.

**Phase 2 fix:** switch to a point-in-time universe dataset (e.g.
Norgate Data, Sharadar/Tiingo fundamentals with historical index
membership) once budget allows.

## 2. Non-point-in-time sector classification

`mltrading.data.ingest.fetch_ticker_sector` reads `Ticker.info["sector"]`,
which reflects each company's *current* GICS sector. A company that
changed sector classification historically (reclassifications happen,
e.g. GICS's 2018 Communication Services carve-out) will be labeled with
its present-day sector for all historical dates.

**Impact:** the target definition (`project_spec.md` section 4) relies on
sector-relative returns, so a sector mislabel in old data slightly
corrupts the target for affected tickers/periods. Expected to be a small
effect for the initial universe, which was chosen to avoid tickers with
well-known major reclassifications, but it is not verified name-by-name.

## 3. Adjusted-close is retroactively revised

yfinance's `adj_close` is recomputed for the full history whenever a new
split or dividend occurs. Two ingestion runs on different calendar days
can therefore return slightly different `adj_close` values for the same
historical date. Every ingested row carries `ingested_at` and
`universe_version` specifically so a given experiment's inputs can be
tied to the exact ingestion run that produced them (`project_spec.md`
section 5's "what did the system know at this decision time" replay
requirement) — but note this is about *data provenance*, not about
removing the revision behavior itself.

## 4. No delisted-name coverage, no corporate-action audit trail

yfinance returns no data at all for tickers it no longer recognizes, and
does not provide a structured corporate-actions history separate from
its own price adjustments. Compare against `project_spec.md` section 5's
requirement that "corporate actions... must be documented with their
availability assumptions" — V1 currently relies entirely on yfinance's
opaque adjustment logic and does not independently verify it.

## Summary

These are accepted, documented V1 limitations in exchange for a $0
data budget. They must be restated alongside any backtest result
produced from this data (per section 13: reported performance built on
an undisclosed survivorship-biased universe is exactly the kind of
result the spec says must be treated as invalidated).
