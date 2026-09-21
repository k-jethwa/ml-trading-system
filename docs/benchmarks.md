# Profiling and one measured optimization

project_spec.md §12 asks for profiling of data loading, feature computation, inference,
portfolio construction and backtesting, and for **one measured optimization of a real bottleneck
with before/after benchmarks and an explanation**. Reproduce with:

```bash
python -m mltrading.experiments.benchmark --label before --profile   # (on the pre-optimization commit)
python -m mltrading.experiments.benchmark --label after  --profile
```

Machine: Apple-silicon Mac, Python 3.13, single process. Fast stages are medians of 3–20 runs. The risk-aware
backtest is a single timed run per benchmark (a multi-second call); its "after" value was reproduced at
6.8 s (Ridge, OLS) and 7.5 s (HGB) in separate re-runs, and the sum over all 9 risk-aware backtests inside a full
experiment run gives an independent check (Section 4). "Before" was measured once.

## 1. Where the time goes (before)

| Stage | Median | Note |
|---|---|---|
| Data loading (read `features.parquet`) | 0.015 s | 146,842 rows × 31 cols |
| Feature computation (raw panel → feature table) | 0.26 s | 55 tickers × ~2,900 days |
| Model frame (cross-sectional ranks) | 0.50 s | |
| Inference latency, HGB, one date × 55 names | 8.6 ms | batch-of-55 latency |
| Inference throughput, HGB, whole panel | 0.18 s | ≈ 800k rows/s |
| Inference throughput, Ridge, whole panel | 0.007 s | |
| One risk-aware optimization | 22.1 ms | |
| Backtest, baseline, full period (1,911 days) | 2.1 s | |
| **Backtest, risk-aware, full period** | **12.9 s** | 384 optimizations |

Across a full experiment (`run.py`) the timings manifest showed the risk-aware backtests were
**119 s of 213 s (56%)**, i.e. the clear bottleneck; everything in the ML path (fit, predict) is
under 5 s in total. Batch versus online: scoring the whole 10-year panel takes 0.18 s but a single date
takes ~5–9 ms, dominated by per-call overhead rather than compute, which is why the endpoint caches the loaded model.

## 2. Profile of the risk-aware backtest (before)

`cProfile`, 384 optimizer calls (times inflated ~2× by the profiler; the proportions are what matter):

| Cumulative | What |
|---|---|
| 26.9 s | whole backtest |
| 21.3 s | `RiskAwareStrategy.target_weights` |
| 15.0 s | `cvxpy Problem.solve` |
| **11.1 s** | ↳ **`get_problem_data`: cvxpy re-canonicalizing the problem from scratch every call** |
| 3.3 s | ↳ the actual Clarabel interior-point solve |
| 3.1 s | `estimate_risk`, of which 2.7 s a per-column `pandas.apply` computing betas |

Each optimizer call has the *same structure* (same variables, same constraints); only the numbers
(alphas, covariance, previous weights) change. cvxpy was rebuilding and re-reducing the whole problem
each time, which cost 3.4× more than solving it.

## 3. The optimization

1. **Compile once.** The problem is expressed as a DPP (disciplined parametrized programming)
   problem over cvxpy `Parameter`s and cached per (universe, config). A fixed-size universe plus an
   eligibility mask handles the number of eligible names changing between rebalances (an ineligible
   name has a zero position bound, zero alpha/beta and a zero covariance-factor row, which is
   equivalent to removing it).
2. **Vectorize beta estimation** (one matrix expression instead of a Python-level `apply` over 55 columns).

Correctness was not assumed: the original formulation is kept in `tests/portfolio/reference_optimizer.py` and
`test_fast_optimizer_matches_reference_formulation` asserts the same optimum (objective within 1e-8; weights
within 1e-3, since near-flat directions of the objective let two equally optimal solves differ by up to
~2e-4) on random instances including partial eligibility and a non-zero previous book.
A full re-run reproduced the walk-forward predictions bit-for-bit and every Sharpe ratio to within 0.0005.

## 4. Result

| Stage | Before | After | Speedup |
|---|---|---|---|
| One risk-aware optimization | 22.1 ms | 8.3 ms | **2.7×** |
| Backtest, risk-aware, full period | 12.95 s | 6.82 s | **1.9×** |
| All risk-aware backtests inside one experiment run (`run.py` timing manifest) | 119.4 s | 64.5 s | 1.9× |
| Full experiment run (`run.py`) | 213 s | 155 s | 1.4× |

The unrelated stages (data load, features, model frame, baseline backtest, Ridge inference) moved by
≤ 4%, which is the run-to-run noise floor. HGB inference also appears to move (single-date latency 8.6 → 5.1 ms; whole-panel 0.18 → 0.165 s), but no code
on that path changed, so that is noise (a median of a ~5 ms call; a 3-run median of a ~0.17 s call), not a result.

After the change the profile is flat: 2.7 s of the 12.7 s profiled backtest is the solver itself, and most
of the rest is small pandas operations in the engine loop (`clip`, `where`, indexing in the borrow accrual and
mark-to-market). Removing those would be the next lever, but at 6.8 s per backtest the remaining gain is not worth the added complexity.

## 5. A correctness bug found while optimizing (why the tests matter)

The first version of the fast optimizer was **not** a pure speed-up. Re-running the backtest showed 141 of 384
HGB solves returning `optimal_inaccurate` (the original had 4), which the unit tests had accepted as a pass. Investigating:

* the numbers were not the problem; the same inputs were replayed under variants of the formulation;
* writing the volatility limit as `252·wᵀΣw ≤ σ²` is poorly scaled for the interior-point solver (1 hard `SolverError` and
  8 inaccurate solves on 172 captured instances); writing it as the direct second-order cone `‖Fᵀw‖₂ ≤ σ/√252` gave 172/172 optimal;
* my first attempt to "fix" it (retrying with tighter tolerances) made it worse (54 of 60 inaccurate);
* the same investigation exposed a design flaw in the original fallback: a numerical `SolverError` fell through to a retry
  *with the turnover limit silently dropped*. It now returns "failed" and the strategy holds its current book; only
  genuine infeasibility relaxes turnover, and that is flagged. A test covers this.

Because the formulation changed, the full experiment was re-run (Run 2 in the attempt log), so that the reported results come from the committed code.
