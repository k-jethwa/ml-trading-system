# Architecture

A concise map of the system. For the reasoning and math behind each piece see
[how_it_works.md](how_it_works.md); for results see [results.md](results.md).

```text
                    configs/v1.toml  ──►  mltrading.config (frozen dataclasses, config hash)
                          │
 yfinance ─► data/ingest ─► data/validate ─► features/{price_features, cross_sectional, target, build}
   (raw parquet per ticker)   (5 checks)       └─► data/processed/features.parquet  (160,702 × 31)
                                                          │
                                        models/preprocess (same-date cross-sectional ranks)
                                                          │
                            models/walk_forward (expanding window, 5-day purge)
                                                          │
        models/train ─► models/zoo {mean, reversal_5d, OLS, Ridge, HGB} ─► predictions (out-of-sample)
              │                                                          │
              └─► models/registry  artifacts/<name>/<model_id>/{model.joblib, metadata.json}
                          │                                              │
        models/infer  (batch: latest date; stale/version/missing-name checks)     models/evaluate (IC, rank IC, MAE, MSE, decile spread)
                          │
        models/serve  (GET /health, /predict, /metrics)
                                                          │
                     backtest/engine  ◄── portfolio/construct (baseline decile)
                     event queue:         portfolio/optimize  (cvxpy, compile-once)
                     Market → Fill →      portfolio/risk      (Ledoit-Wolf cov, betas, checks)
                     Mark → Signal →      backtest/execution  (next-open fills, costs, ADV cap)
                     Order                backtest/state      (cash, positions, cost accounting)
                          │                backtest/metrics    (Sharpe, DD, turnover, alpha/beta, exposures)
                          ▼
     experiments/run  (walk-forward → backtests → cost sensitivity → robustness → results/<name>/)
     experiments/{analysis, report, plots, benchmark}   observability (JSON logs, timers, run manifest)
```

## Package boundaries

| Package | Owns | Depends on | Does not know about |
|---|---|---|---|
| `data` | ingestion, validation, universe | yfinance, pandas | features, models |
| `features` | point-in-time features + label | `data` (raw panel) | models, portfolio |
| `models` | ranking, walk-forward, metrics, registry, batch inference, serving | `features` | portfolio, backtest |
| `portfolio` | scores → weights, risk model, risk checks | `config` | models, backtest, prices |
| `backtest` | events, execution + costs, accounting, metrics, strategies | `portfolio`, `config` | model internals (sees only `date, ticker, pred`) |
| `experiments` | orchestration, sensitivity, reports, benchmarks | everything above | – |

The seam between `models` and `backtest` is a plain table of `(date, ticker, pred)`. Any scorer that
produces that table can be backtested, which is how the non-ML `reversal_5d` heuristic and 20 random-score
portfolios run through exactly the same engine.

## Time and information discipline

| Moment | What is known | What may happen |
|---|---|---|
| Close of day *t* | all bars through *t* | features computed, model scores, strategy returns target weights (`SignalEvent`) |
| Open of day *t+1* | the above + open *t+1* | orders fill at the open with spread + slippage + commission (`FillEvent`) |
| Close of day *t+1* | bars through *t+1* | mark to market, accrue costs (`MarkEvent`) |
| Label for *t* | close of *t+5* | used **only** to train models trained before the test window, and to score IC; purged 5 days before each test year |

## Reproducibility and provenance

* One command, one TOML config; `config_hash` in the manifest, every artifact's metadata, and every run's attempt-log line.
* Each model artifact records feature list + version hash, training window, row count, data fingerprint, git commit and library versions.
* The backtest is deterministic: identical inputs give identical NAV, fills and metrics (tested), and the walk-forward output was bit-identical across two runs of the same config.
* Structured (JSON) logs with stage timings, data freshness, and prediction-distribution statistics are written per run.

## Failure handling

| Condition | Behavior |
|---|---|
| Latest bar older than 7 days | batch inference raises `StaleDataError`; the endpoint returns 503; experiment runs log a warning |
| Feature list changed since the model was trained | `FeatureVersionMismatch` (endpoint 409) |
| Ticker missing or has incomplete features on the latest date | excluded from scoring, counted and listed in the health report; never filled |
| Order for a ticker with no bar on the fill date | order lapses (counted); the next rebalance re-decides |
| Order larger than 5% of trailing average dollar volume | partial fill (counted) |
| Optimizer numerical failure | strategy holds its current book; no limit is relaxed |
| Optimizer infeasible with the turnover limit | retried without that one limit, result flagged and logged |
| Optimizer output fails the post-solve risk check | book is rejected, current book held |
