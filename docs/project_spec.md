# ML Cross-Sectional Trading System — Project Specification

## 1. Objective

Build an end-to-end machine-learning trading system that uses point-in-time daily US equity data to predict **five-trading-day sector-relative returns**. Convert these predictions into a risk-controlled long/short portfolio, then evaluate it through a realistic, strictly out-of-sample backtest.

The project is primarily a production-style ML and systems project. The trading strategy is the initial workload used to demonstrate data pipelines, feature computation, model lifecycle management, evaluation, decision optimization, simulation, and observability.

## 2. Research Hypothesis

At the close of trading day *t*, a model using only information available by that time can rank liquid US equities by their expected five-trading-day sector-relative return. A portfolio that is long the highest-ranked securities and short the lowest-ranked securities may retain useful risk-adjusted performance after reasonable transaction costs and risk constraints.

This is a falsifiable hypothesis. The goal is not to maximize a backtest statistic; it is to determine whether the result remains credible under out-of-sample testing and sensitivity analysis.

## 3. Initial Universe and Data

- **Asset class:** US equities.
- **Frequency:** Daily bars.
- **Initial scale:** Approximately 100–500 liquid, eligible securities.
- **Core fields:** timestamp, ticker, open, high, low, close, adjusted close when available, volume, and sector classification.
- **Storage:** Reproducible raw and processed datasets, initially using Parquet and a local analytical query layer.

The tradable universe must be defined historically. The system must not use present-day index membership as a proxy for past eligibility without documenting the resulting survivorship limitation.

## 4. Prediction Target

For stock *i* on date *t*, the initial target is:

```text
target(i, t) = five-day forward stock return(i, t)
             − five-day forward sector return(sector(i), t)
```

The target asks whether a security outperforms or underperforms its sector over the next five trading days. The primary use of predictions is **cross-sectional ranking**, rather than attempting to forecast an exact price.

Initial operational timing:

1. Compute features after the close on day *t*.
2. Generate predictions using only data available at that time.
3. Create orders for execution no earlier than the next tradable opportunity.

## 5. Allowed Information and Point-in-Time Rules

Every feature and universe decision must have explicit timestamp semantics.

- A feature for date *t* may use observations available on or before the defined decision time on *t*.
- The target may use future returns only as a training/evaluation label, never as a model input.
- Trades cannot execute at a price that was unavailable when the decision was made.
- Data revisions, corporate actions, and classification changes must be documented with their availability assumptions.
- Validation and replay must make it possible to answer: **“What did the system know at this decision time?”**

## 6. Initial Feature Families

V1 features will be price- and volume-derived, with a small, interpretable feature set.

- **Momentum:** 1-, 5-, 20-, and 60-trading-day returns; relative strength; distance from moving averages.
- **Volatility:** rolling realized volatility, EWMA volatility, and simple range-based measures.
- **Liquidity:** dollar volume, rolling average volume, and volume relative to recent history.
- **Market-relative:** stock returns less broad-market returns over selected lookbacks.
- **Sector-relative:** stock returns less sector returns over selected lookbacks.
- **Simple technical state:** rolling high/low position and related bounded indicators where useful.

Each feature must document its definition, lookback, dependencies, availability time, and economic rationale. V1 explicitly avoids a large, unexamined “feature zoo.”

## 7. Model Progression

Models will be introduced only after simpler baselines are recorded.

1. **Mean / constant predictor** — establishes a minimum benchmark.
2. **Linear regression** — tests a simple additive relationship.
3. **Ridge regression** — adds regularization and feature scaling discipline.
4. **XGBoost or LightGBM** — the initial nonlinear benchmark.

The system will favor the simplest model that provides a robust improvement. Neural networks, transformers, and reinforcement learning are not V1 requirements.

## 8. Validation and Evaluation

### Walk-forward evaluation

Evaluation must respect chronological order. Training windows precede test windows; observations must never be randomly shuffled across time.

Example:

```text
Train: 2015–2018  →  Test: 2019
Train: 2015–2019  →  Test: 2020
Train: 2015–2020  →  Test: 2021
```

Retraining cadence, feature version, model parameters, and universe definition must be recorded for every experiment.

### ML metrics

- Mean absolute error (MAE)
- Mean squared error (MSE)
- Spearman rank correlation
- Information coefficient (IC)

Rank-based metrics are central because the model drives a ranked portfolio rather than a single-security price forecast.

### Trading metrics

- Gross and net PnL
- Annualized return and volatility
- Sharpe ratio and, where useful, Sortino ratio
- Maximum drawdown
- Turnover
- Hit rate
- Gross and net exposure
- Sector and beta exposure
- Performance before and after estimated costs

## 9. Portfolio Construction

The model produces expected relative-return scores; portfolio construction converts those scores into tradable weights.

### Baseline portfolio

- Rank eligible securities by predicted return.
- Long the top decile and short the bottom decile.
- Use equal weights within each side.
- Target approximately market-neutral net exposure.

### Risk-aware portfolio

Subsequent versions will incorporate constraints rather than directly trading raw scores:

- maximum absolute position weight
- gross and net exposure limits
- sector exposure limits
- portfolio beta constraint
- target volatility or volatility scaling
- turnover limits

The optimizer will balance expected return, estimated risk, and trading costs. Any solver may be used initially, but the objective, constraints, and resulting trade-offs must be explainable.

## 10. Execution and Transaction Costs

The backtest must not assume costless, instantaneous fills at the midpoint.

V1 execution modeling includes:

- commissions or fee assumptions
- bid/ask-spread component
- slippage component
- turnover-based cost accounting
- execution no earlier than the permitted time after a decision

Cost assumptions will be explicit and tested through sensitivity analysis (for example, 0.5×, 1×, 2×, and 4× baseline costs). V1 may use simplified fills, but the limitations must be stated.

## 11. System Components

```text
Market-data ingestion and validation
        ↓
Point-in-time data store and universe selection
        ↓
Feature engine and persisted feature tables
        ↓
Walk-forward training and experiment tracking
        ↓
Versioned model artifact / registry metadata
        ↓
Batch inference
        ↓
Portfolio construction and risk checks
        ↓
Event-driven backtest and execution simulation
        ↓
Portfolio accounting, metrics, monitoring, and replay
```

The backtester will model explicit state transitions such as `MarketEvent`, `SignalEvent`, `OrderEvent`, and `FillEvent`. Given the same event sequence and configuration, the system should produce deterministic portfolio state and results.

## 12. Engineering Goals

V1 should demonstrate:

- modular, testable software boundaries between data, features, models, portfolio logic, execution, and backtesting
- reproducible ingestion, training, and backtest runs
- configuration-driven experiments
- unit and integration tests, including synthetic known-answer cases
- model artifacts and metadata that identify the exact model behind a prediction
- structured logs and basic metrics for latency, failures, data freshness, model version, and prediction distributions
- profiling of data loading, feature computation, inference, portfolio construction, and backtesting
- one measured optimization of a real bottleneck, with before/after benchmarks and an explanation of the result

An optional V1 endpoint may expose model prediction and health metadata. It exists to demonstrate model serving, not to create a frontend-heavy application.

## 13. Invalidating Failure Modes

The following failures can invalidate reported performance and must be actively guarded against.

### Look-ahead bias

The model or trading logic uses information unavailable at the decision time—for example, deciding after a close while assuming execution at that same observed close.

### Survivorship bias

The historical universe includes only companies that are known to have survived to the present, such as backtesting old dates using today’s S&P 500 constituents.

### Data leakage

Future information enters a feature, preprocessing step, target transform, or normalization calculation. For example, fitting a scaler on the complete dataset before splitting time periods.

### Unrealistic execution assumptions

The simulation ignores spread, slippage, commissions, liquidity constraints, event timing, or the inability to fill every desired order at a favorable price.

### Overfitting and excessive backtest tuning

Repeatedly altering features, parameters, constraints, or date ranges to optimize a single historical result without preserving a clean holdout period or reporting the number of attempts.

Any discovery of these issues requires the affected experiment to be re-run and documented as corrected.

## 14. V1 Non-Goals

V1 will not attempt to build:

- live trading or execution with real capital
- high-frequency or order-book trading
- real-time streaming infrastructure
- distributed deployment, Kubernetes, Kafka, or cloud scale-out for appearance alone
- complex deep-learning architectures, LLM components, or reinforcement learning
- a polished frontend/dashboard as the project centerpiece
- a claim of deployable investment performance

These may become Phase 2 extensions only after the core pipeline is correct, tested, benchmarked, and well documented.

## 15. Learning and Checkpoints

The project is complete only when its design choices can be defended. For each module, maintain a brief learning log with:

1. A plain-language definition of the concept.
2. The core mathematical or operational logic.
3. The simplest implementation.
4. At least one failure mode or test.
5. A baseline comparison or experiment result.
6. A short conclusion, including limitations.

### Finance checkpoints

Explain and calculate: return, sector-relative return, volatility, correlation, beta, market neutrality, gross/net exposure, turnover, slippage, drawdown, and Sharpe ratio.

### ML checkpoints

Explain: target construction, feature leakage, regularization, ranking versus regression error, walk-forward validation, overfitting, and why predictive accuracy may not translate into PnL.

### Systems checkpoints

Explain: batch versus online inference, latency versus throughput, profiling, caching, deterministic replay, state persistence, and failure handling for missing or stale data.

### Final evidence

The final repository should contain a reproducible experiment command, a concise architecture description, benchmark results, a results report, robustness and cost-sensitivity experiments, and an honest account of failure modes and limitations.
