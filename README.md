# ml-trading-system

An end-to-end machine learning system for cross-sectional equity trading. It pulls daily market data, engineers point-in-time features, trains and validates ranking models, turns their predictions into a risk-constrained long/short portfolio, and evaluates it in an event-driven backtest with realistic trading costs.

The target is deliberately not "where will the price go?" but **"which of these 55 large US stocks will beat their own sector over the next five trading days?"** Ranking stocks against each other is a more stable problem than forecasting the market, and it lets the portfolio go long the likely winners and short the likely losers so overall market moves largely cancel out.

## What it does

```text
yfinance ─► validation ─► features + label ─► walk-forward training ─► portfolio ─► backtest ─► metrics
 (OHLCV)   (gaps, jumps,   (20 point-in-time   (5 models, purged,      (decile or    (next-open     (Sharpe, alpha/beta,
            stale data)     signals, 5-day       out-of-sample          cvxpy         fills, costs,   drawdown, turnover,
                            sector-relative)     predictions)           optimizer)    borrow)         cost sensitivity)
```

1. **Ingest and validate.** Daily bars for 55 stocks across 8 sectors, stored as Parquet, with checks for missing days, bad prices, suspicious jumps and frozen feeds.
2. **Build features and the label.** Momentum, volatility, liquidity, and market- and sector-relative signals, each computed only from information available at that day's close. The label is the next 5 days' return minus the sector's.
3. **Train and validate models.** A ladder of increasing complexity (constant predictor, a one-line reversal rule, OLS, Ridge, gradient-boosted trees), evaluated with walk-forward validation on data the model never saw.
4. **Construct portfolios.** Either a simple top-decile long / bottom-decile short book, or a convex optimizer that maximizes predicted return net of risk and trading cost under exposure, sector, beta, volatility and turnover limits.
5. **Backtest.** An event-driven simulator: a decision made after the close fills at the next open, paying commission, spread, slippage and short-borrow fees.
6. **Stress-test.** Cost sweeps, random-signal baselines, a label-shuffle leakage check, parameter sensitivity and per-year stability, all from one command.
7. **Serve.** Versioned model artifacts, batch inference that refuses stale data, and a small HTTP endpoint for predictions and health.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | pandas, NumPy, PyArrow (Parquet), yfinance |
| Modeling | scikit-learn (OLS, Ridge, `HistGradientBoostingRegressor`, Ledoit-Wolf covariance), joblib |
| Optimization | cvxpy with the Clarabel solver, SciPy |
| Backtesting | custom event-driven engine (Market / Signal / Order / Fill events) |
| Serving | stdlib `http.server` JSON endpoint |
| Config & tooling | TOML configs, pytest (85 tests), ruff, matplotlib for figures |

## ML concepts used

- **Cross-sectional prediction:** a relative-return target instead of an absolute-price forecast.
- **Point-in-time feature engineering:** momentum (including 12-1 month), realized/EWMA/range volatility, dollar-volume liquidity, distance from moving averages, and leave-one-out market- and sector-relative returns.
- **Cross-sectional rank normalization:** features are replaced by same-date percentile ranks, which removes scale and non-stationarity without fitting anything across time.
- **Walk-forward validation with purging:** expanding training window, yearly retraining, and a 5-day embargo so overlapping labels can't leak into the test period.
- **Leakage prevention:** an automated look-ahead test (perturb the future, assert past features don't move), scalers fit on training folds only, and a label-shuffle sanity check.
- **Regularization and model progression:** simple baselines first, Ridge shrinkage, shallow heavily-regularized boosted trees, and fixed hyperparameters to avoid backtest overfitting.
- **Ranking metrics vs. error metrics:** IC, rank IC (Spearman), decile spread, MAE, MSE and out-of-sample R², plus non-overlapping t-statistics for 5-day labels.
- **Risk modeling:** Ledoit-Wolf covariance shrinkage, betas to the universe, ex-ante volatility, sector and gross/net exposure control.
- **Constrained optimization:** mean-variance with an L1 transaction-cost term, expressed as a compile-once parametrized (DPP) convex program with a direct second-order-cone volatility constraint.
- **Execution modeling:** next-open fills, half-spread, slippage, commissions, borrow costs, participation caps, and turnover-based cost accounting.
- **Performance attribution:** gross vs. net returns, Sharpe/Sortino, drawdown, and an alpha/beta regression against the universe so market exposure isn't mistaken for skill.

## Engineering practices

- Everything is driven by one config file ([configs/v1.toml](configs/v1.toml)); every run records a config hash, data fingerprint and git commit.
- Model artifacts carry metadata (feature version, training window, library versions), and every prediction is tagged with the `model_id` that produced it.
- The backtest is deterministic, and the portfolio code is tested against an independent reference formulation.
- Structured JSON logs, stage timings, a data-freshness check and prediction-distribution monitoring.
- Profiled and optimized with before/after benchmarks: caching the compiled optimizer cut portfolio-construction time by 2.7×.
- 85 tests, mostly small hand-computed examples (fill timing, costs, accounting, metrics); no network needed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,viz]'

python -m mltrading.data.ingest        # download raw data
python -m mltrading.data.validate      # data-quality checks
python -m mltrading.features.build     # features + label

# train, backtest and stress-test everything (~2.5 min)
python -m mltrading.experiments.run --config configs/v1.toml --note "what this run is for"

python -m mltrading.experiments.report   # tables
python -m mltrading.experiments.plots    # figures
python -m pytest -q                      # tests
```

Batch scoring: `python -m mltrading.models.infer --model ridge`. Endpoint: `python -m mltrading.models.serve`. Both refuse data older than 7 days unless told otherwise.

## Project layout

| Path | Contents |
|---|---|
| `src/mltrading/data`, `features` | ingestion, validation, features, label |
| `src/mltrading/models` | model zoo, walk-forward, metrics, registry, inference, serving |
| `src/mltrading/portfolio` | decile and optimizer portfolios, risk model |
| `src/mltrading/backtest` | event engine, execution and costs, accounting, metrics |
| `src/mltrading/experiments` | one-command experiment, robustness checks, reports, benchmarks |
| `configs/`, `tests/` | experiment config, test suite |
| `docs/` | specification, architecture, results, benchmarks, limitations |

## Docs

- [docs/how_it_works.md](docs/how_it_works.md): a learning log explaining each concept, its math and its tests
- [docs/architecture.md](docs/architecture.md): how the pieces fit together
- [docs/results.md](docs/results.md): experiment setup, validation protocol and write-up
- [docs/benchmarks.md](docs/benchmarks.md): profiling and the optimization
- [docs/limitations.md](docs/limitations.md): known limitations and spec deviations
- [docs/project_spec.md](docs/project_spec.md): the original specification

Not investment advice; a research and engineering project.
