# ml-trading-system

End-to-end machine learning trading system with point-in-time data processing, feature engineering, model training and inference, portfolio construction, risk management, and event-driven backtesting.

**Status: V1 complete.** The pipeline runs from raw data to a costed, risk-constrained, out-of-sample backtest. **The trading hypothesis is not demonstrated:** the models find a small, unstable ranking signal
(rank IC ≈ 0.01) that does not survive costs and beta-adjustment as statistically significant alpha, and no strategy beat buying and holding the same 55 stocks. See [docs/results.md](docs/results.md).
This is a research/engineering project, not investment advice, and nothing in it is a claim of deployable performance.

## Layout

| Path | What |
|---|---|
| `src/mltrading/data`, `features` | ingestion, validation, point-in-time features, 5-day sector-relative label |
| `src/mltrading/models` | model zoo, walk-forward validation, metrics, registry, batch inference, serving endpoint |
| `src/mltrading/portfolio` | baseline decile and constrained (cvxpy) portfolios, risk model |
| `src/mltrading/backtest` | event-driven engine, execution/cost model, accounting, metrics |
| `src/mltrading/experiments` | one-command experiment, sensitivity/robustness, report, plots, benchmark |
| `configs/v1.toml` | the experiment configuration (fixed before results were seen) |
| `docs/` | [spec](docs/project_spec.md), [architecture](docs/architecture.md), [learning log](docs/how_it_works.md), [results](docs/results.md), [generated tables](docs/results_tables.md), [benchmarks](docs/benchmarks.md), [limitations](docs/limitations.md), [attempt log](docs/attempt_log.jsonl) |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,viz]'

python -m mltrading.data.ingest && python -m mltrading.data.validate && python -m mltrading.features.build   # data + features
python -m mltrading.experiments.run --config configs/v1.toml --note "why this run"     # train, backtest, robustness (~2.5 min)
python -m mltrading.experiments.report && python -m mltrading.experiments.plots        # docs/results_tables.md, figures
python -m pytest -q                                                                     # 85 tests, no network
```

Batch inference and the optional endpoint: `python -m mltrading.models.infer --model ridge` and `python -m mltrading.models.serve`
(both refuse data older than 7 days unless told otherwise).

Known deviations from the spec are listed in [docs/limitations.md](docs/limitations.md): survivorship-biased static universe, scikit-learn histogram gradient boosting instead of XGBoost/LightGBM, simplified fills.
