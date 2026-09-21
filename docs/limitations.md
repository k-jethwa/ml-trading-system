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

---

# Limitations added by the modeling, portfolio and backtest stages

## 5. Model and evaluation

* **Gradient boosting deviates from the spec.** §7 names XGBoost or LightGBM; scikit-learn's `HistGradientBoostingRegressor` is used because both need
  the system OpenMP runtime (`brew install libomp`), which was not installed. Same algorithm family, but not the library the spec names, and results with the named libraries could differ.
* **No hyperparameter tuning, by design.** One setting per model, fixed in advance (`configs/v1.toml`). This avoids overfitting the backtest but means the models are not
  optimized; better-tuned models might do better or worse.
* **The holdout is short and partly seen.** 2025-01-01 onward is 401 trading days (~1.6 years, Sharpe standard error ≈ ±0.8–1.0). Smoke tests printed holdout ML metrics before the backtest was built
  (see `docs/attempt_log.jsonl`); no configuration value was changed as a result, but the holdout is not pristine.
* **Eight test years, one market regime.** 2019–2026 is largely a bull market; results say little about crashes or a rising-rate regime.
* **No multiple-comparison correction** across the ~8 model/portfolio variants; the reported t-statistics are per-strategy.
* **Overlapping labels** make daily rank ICs autocorrelated; t-statistics use every 5th date, which is conservative but discards data.

## 6. Portfolio and risk model

* **Beta is measured against an equal-weight average of the 55-name universe**, not an index (the dataset has no index series). It is a proxy for market exposure, and it includes each stock in its own benchmark.
* **Ex-ante versus realized risk.** Constraints control *estimated* beta/volatility from a 60-day Ledoit-Wolf covariance; realized beta was 0.04–0.10 against a 0.05 limit and gross exposure drifts around 2.0 between rebalances as prices move.
* **Predictions are treated as expected returns** in the optimizer. HGB's predictions are poorly calibrated in magnitude (negative OOS R²), which the optimizer cannot know.
* **Positions dropped from the eligible set are left out of the optimization** (weight 0) and liquidated by the engine, so the turnover constraint under-counts those trades. Rare in this universe.

## 7. Execution and costs

* **Simplified fills.** The open is treated as the mid; costs are flat basis points (1 commission + 2 half-spread + 2 slippage per side). No size-dependent market impact, no opening-auction dynamics, no intraday path.
  The 4× cost sweep is a robustness check, not a claim about real fills.
* **Short selling.** A flat 50 bp/yr borrow rate; no locate/availability, recalls, hard-to-borrow fees, or margin financing. Interest on cash and short proceeds is ignored.
* **Total-return prices.** Backtests use dividend-adjusted prices (dividends implicitly reinvested); open prices are scaled onto the adjusted-close basis. Not tax-aware.
* **Whole shares, $10M NAV.** At this size the 5%-of-ADV cap essentially never binds in mega-cap names; it would matter at larger size.

## 8. Data freshness

* The feature snapshot ends 2026-08-17. `adj_close` is retroactively revised by yfinance, so a re-ingest changes historical values slightly; a re-run on new data is a new experiment, not a reproduction.
* Inference refuses data older than 7 days unless `--allow-stale` is passed, so the batch/endpoint paths will report `stale` on this snapshot.

## 9. Interpretation

Every result inherits the survivorship bias in §1: the universe is today's liquid stocks, so backtests over past dates omit companies that later failed or were removed. Reported performance should be read as an
upper bound. `docs/results.md` concludes that the hypothesis is not demonstrated even under these favorable conditions.
