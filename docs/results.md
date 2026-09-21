# V1 Results Report

**Question (project_spec.md §2):** at the close of day *t*, can a model that
uses only information available then rank 55 liquid US stocks by their
5-day sector-relative return well enough that a long/short book keeps a
useful risk-adjusted return after costs and risk limits?

**Answer: not demonstrated.** The models find a small amount of real
ranking signal, but it is too weak and too unstable to distinguish from zero
once costs and market exposure are accounted for. Nothing here is a claim
of deployable performance.

All numbers below come from one reproducible command
(`python -m mltrading.experiments.run`, config `configs/v1.toml`) and are
laid out in full in [results_tables.md](results_tables.md), which is generated
from the run outputs rather than typed. Figures: [equity_curves.png](equity_curves.png),
[cost_sensitivity.png](cost_sensitivity.png).

---

## 1. Headline findings

| | Finding | Evidence |
|---|---|---|
| 1 | Ranking signal is real but tiny: rank IC ≈ 0.009–0.014 over 2019–2026 | t = 1.7–2.2 on non-overlapping dates (tables §1) |
| 2 | The best linear models mostly rediscover short-term reversal | 2025-26 holdout rank IC: Ridge 0.031, one-line `reversal_5d` heuristic 0.034 |
| 3 | The signal is unstable across years | Ridge rank IC by year ranges from −0.013 (2019) to +0.038 (2025); negative in 2019, 2021, 2023 |
| 4 | Higher rank IC did not mean better PnL, and the best PnL model was not the best MSE model | HGB: best full-period rank IC (0.014) but OOS R² −0.24% (worse than a constant); Ridge: best baseline net Sharpe (0.52) |
| 5 | The baseline decile books are mostly market exposure, not alpha | beta 0.28–0.60; alpha vs the universe ≈ 0 (|t| < 0.2 for OLS/Ridge/HGB) |
| 6 | Costs dominate: ~117× annual turnover costs the baseline ≈ 6.3% a year | signal-free random-score portfolios lose −11% a year net (Sharpe −0.71 ± 0.49) |
| 7 | The risk-aware book is far cheaper and more robust to costs, but its alpha is not significant | cost drag ≈ 2%/yr; net Sharpe 0.49–0.62; alpha 2.8–5.4%/yr, t = 0.8–1.5; still positive at 4× costs (Sharpe 0.15) |
| 8 | Every strategy underperformed buying and holding the same 55 stocks | buy & hold Sharpe 1.07 (2019–2026 was a strong bull market); strategies 0.09–0.62 |
| 9 | Nothing suggests the pipeline manufactures signal | models trained on shuffled labels: out-of-sample rank IC −0.002 and −0.007 (t ≈ −0.5, −0.7) |

### Net performance, 1× costs, full test period 2019-01 → 2026-08

| Strategy | Net Sharpe | Net ann. return | Ann. vol | Max DD | Beta | Alpha (ann., t) |
|---|---|---|---|---|---|---|
| Ridge · baseline decile | 0.52 | 10.5% | 24.9% | −30% | 0.60 | 0.9% (0.1) |
| HGB · baseline decile | 0.27 | 3.4% | 20.2% | −40% | 0.28 | −0.2% (−0.0) |
| Ridge · risk-aware | 0.51 | 4.5% | 9.5% | −16% | 0.10 | 2.9% (0.9) |
| HGB · risk-aware | 0.62 | 5.9% | 9.9% | −25% | 0.04 | 5.4% (1.5) |
| Buy & hold equal-weight (long-only) | 1.07 | 20.2% | 19.0% | −35% | – | – |

For a sense of scale: the 20 pure-noise portfolios have a net-Sharpe standard
deviation of 0.49, so gaps of ~0.5 between these rows are about one standard
deviation of luck. See `docs/results_tables.md` for OLS, the heuristic and the
development/holdout splits.

---

## 2. What was done, and how far to trust it

**Data.** 55 large-cap US stocks, daily bars 2015-01-02 → 2026-08-17 from yfinance, 8 sectors.
Features start 2016 (the momentum feature needs a year of history). Feature and label
definitions are in `docs/how_it_works.md` §5–§6; the label is the 5-day forward return minus the
leave-one-out sector mean.

**Validation.** Expanding-window walk-forward, retrained once a year, first test year 2019,
with a 5-trading-day purge before each test start so no training label overlaps the test period
(tested in `tests/models/test_models.py`). Test years 2019–2024 are the *development* period; 2025-01-01
onward (401 trading days) is the *holdout*. Nothing was fit on a test period. Cross-sectional
feature ranks use only same-date information.

**Models.** Constant mean, `reversal_5d` (a one-line non-ML heuristic), OLS, Ridge (α = 1000), and
gradient-boosted trees (depth 3, 200 iterations, 500-sample leaves). Hyperparameters were set
before any out-of-sample result was seen and **were never tuned**: one configuration per model.

**Backtest.** Event-driven (`MarketEvent → FillEvent → MarkEvent → SignalEvent → OrderEvent`).
Decisions are made at the close of day *t* and fill at the **next day's open**. Costs: 1 bp
commission + 2 bp half-spread + 2 bp slippage per side, 50 bp/yr borrow on shorts, a 5%-of-ADV
participation cap, whole-share orders. $10M starting capital, rebalance every 5 trading days.
Two portfolios: an equal-weight top/bottom decile (dollar-neutral, gross 2.0) and a constrained optimizer
(max 10% per name, gross ≤ 2, |net| ≤ 2%, sector net ≤ 10%, |beta| ≤ 0.05, ex-ante vol ≤ 10%,
turnover ≤ 0.6 of NAV per rebalance).

### Attempt log (project_spec.md §13: "report the number of attempts")

The full log is `docs/attempt_log.jsonl`. In plain terms:

* **Configurations evaluated:** 5 model variants × 1 hyperparameter setting each; 1 feature set;
  2 portfolio constructions; cost multipliers 0.5/1/2/4×; 6 one-at-a-time parameter variants
  (decile 5/10/20%, rebalance 5/10/21 days) on Ridge; 20 random-score portfolios; 1 label-shuffle check.
* **Config changes after seeing any out-of-sample result: none.** `configs/v1.toml` is unchanged
  since before the first run.
* **Full runs of the final config: 2.** Run 1 exposed a numerical-quality problem in the optimizer while
  I was speeding it up (below); Run 2 is the same config on the corrected code, and it is the run reported here.
  Run 2 differs from Run 1 by ≤ 0.0005 in every Sharpe ratio, and its predictions and ML metrics are bit-identical.
* **Exploratory runs before Run 1 (disclosed because they touched out-of-sample data):** a smoke
  test of the walk-forward step printed ML metrics for *both* development and holdout periods;
  a smoke test of the backtest printed development-period results only; a smoke test of the experiment runner on 2024+
  test years printed net Sharpe ratios that included part of the holdout. No configuration value
  was changed as a result of any of them.

**Consequences for trust.** The holdout is *not* pristine: I saw its ML metrics before the backtest
was built. It is also short (401 days, ~1.6 years), so a Sharpe measured there has a standard error
of roughly ±0.8–1.0. Strong holdout numbers for Ridge/OLS (net Sharpe 1.2–1.5, t ≈ 3 rank IC) should not
be read as confirmation: they coincide with the reversal heuristic doing equally well, and the same models
were slightly negative in 2019 and 2023.

---

## 3. Findings in detail

### 3.1 Prediction quality (tables §1, §6, §7)

* **Development (2019–2024):** rank IC 0.004 (OLS/Ridge, t ≈ 0.4) and 0.011 (HGB, t = 1.7). None is
  significant. **Holdout (2025–26):** 0.031 (OLS/Ridge, t ≈ 3.0), 0.025 (HGB, t = 1.4), 0.034 (heuristic, t = 2.4).
* **Ranking vs. regression error.** Ridge and OLS are the only models with positive out-of-sample R²
  (+0.01–0.02%, i.e. essentially zero); HGB is worse than predicting the mean (−0.24%) yet has the highest
  full-period rank IC. HGB is finding a little ordering information while being badly calibrated in
  magnitude, which is exactly the "MSE vs. rank" distinction the ML checkpoint asks about. The portfolio
  only uses order, so MSE alone would have ranked these models wrongly.
* **Instability.** Rank IC by year for Ridge: −0.013, +0.017, −0.002, +0.018, −0.010, +0.012, +0.038, +0.019
  (2019–2026). A 5-day reversal effect that was strongly negative for the heuristic in 2023 (−0.041) is the kind
  of regime dependence a 55-name universe cannot average out.
* **Leakage check.** Retraining on permuted labels gives out-of-sample rank IC ≈ 0 (table §7). Together with the
  automated look-ahead test (`tests/features/test_no_lookahead.py`) and the purge test, this is the evidence that
  the small positive IC is not an evaluation artifact.

### 3.2 Baseline decile portfolio (tables §2, §4)

Ridge's net Sharpe of 0.52 sounds respectable until decomposed: beta to the equal-weight universe is 0.60
(the book is long high-beta names in a bull market) and the intercept alpha is 0.9% a year (t = 0.1). HGB's baseline has
lower beta (0.28) and zero alpha. Only ~6 names per side are held, so the book is concentrated (ann. vol 20–25%,
max drawdown 30–40%).

Costs are the other half of the story. The book trades ~117× its NAV per year and pays ≈ 6.3% a year in costs (gross
Sharpe 0.78 → net 0.52 for Ridge). The 20 signal-free portfolios average a net Sharpe of −0.71 (gross −0.10): with
no information, the same machinery loses roughly 11% a year. Ridge's and HGB's gross Sharpe (0.78, 0.59) each beat 19 of
20 noise portfolios, which is suggestive of some information but is a 20-sample comparison, not a significance test.

### 3.3 Risk-aware portfolio (tables §2, §3, §8)

The optimizer trades far less (28–30× turnover, cost drag ≈ 2%) and holds volatility near the 10% cap (realized 9.5–9.9%),
with realized beta 0.04–0.10 against a 0.05 ex-ante limit (the gap is estimation error in a 60-day beta). All 384 rebalances
per model solved to optimality, with zero post-solve risk-check violations and no turnover limit ever relaxed.

It does not produce a significant edge: alpha 2.8% (Ridge, t = 0.9) and 5.4% (HGB, t = 1.5). HGB risk-aware is the most
consistent result in the study (net Sharpe 0.63 development, 0.61 holdout), but that consistency is one model on one
55-name universe, and it is the same model whose baseline portfolio had zero alpha.

### 3.4 Cost sensitivity (table §3, figure)

Scaling all costs: baseline books go negative at 2× (HGB, −0.05) or 4× (Ridge, −0.24; HGB −0.68). Risk-aware books stay positive
at 4× (0.15 for both) but only just. So the result is robust to costs being *somewhat* worse than assumed for the risk-aware
book, and not robust for the baseline. The cost model is deliberately simple (see limitations); real slippage for a
$10M book in these names is probably within the assumed range, but that is an assumption, not a measurement.

### 3.5 Parameter sensitivity (table §5)

For Ridge baseline, net Sharpe is 0.30 / 0.52 / 0.67 for 5% / 10% / 20% quantiles, and 0.52 / 0.13 / 0.62 for
rebalancing every 5 / 10 / 21 days. The non-monotonic response to rebalance frequency (10 days is far worse than both 5
and 21) is what noise-dominated results look like; it is a reason not to trust any single point estimate here. These
variants are reported for transparency and were **not** used to change the configured values.

---

## 4. Invalidating failure modes (project_spec.md §13)

| Failure mode | What guards against it | Residual risk |
|---|---|---|
| Look-ahead bias | Features use trailing windows only; automated test perturbs future prices and asserts past features are bit-identical; decisions at close *t*, fills at open *t+1* (engine tests include an overnight-gap case that would show free profit under same-close fills); risk-aware strategy tested to ignore future rows | `adj_close` is revised retroactively by the data vendor (documented) |
| Survivorship bias | Disclosed (`docs/limitations.md`) | **Not fixed.** Universe is today's liquid names; all results are upper-biased |
| Data leakage | Same-date rank features (nothing fit across time); scaler fit on train folds only (tested); 5-day purge between train and test (tested); label-shuffle check | Holdout is short and was partly seen (§2) |
| Unrealistic execution | Next-open fills, spread + slippage + commission + borrow, ADV cap, whole shares, lapsed orders when no bar; cost sensitivity to 4× | No market-impact curve, no short-availability model, open treated as mid, no intraday path |
| Overfitting / tuning | One config per model, fixed in advance; attempt log; parameter sweeps reported not selected | Only 8 test years; multiple comparisons across ~8 strategy variants are not corrected for |

---

## 5. Conclusion and limitations

The hypothesis is **not supported at a conventional significance level**. The results are equally compatible with a small
real edge (IC ≈ 0.01–0.03) and with none, and they are far from the 1.07 Sharpe of simply holding the stocks. The most defensible
takeaways are about method rather than returns:

* the pipeline is leak-checked, deterministic, tested (85 tests), and its accounting is verified against hand-computed cases;
* transaction costs, not model quality, decide whether a 5-day-turnover strategy on this universe survives;
* a beta/alpha decomposition, a noise-portfolio baseline and a cost sweep changed the interpretation of every headline
  Sharpe ratio and should accompany any such result.

Limitations that bound every claim above (details in `docs/limitations.md`): survivorship-biased 55-name static universe;
sector labels are current, not historical; a single short holdout that was partly seen; a no-index "market" proxy for beta;
simplified costs and fills; total-return adjusted prices; gradient boosting via scikit-learn's histogram implementation rather than
XGBoost/LightGBM (spec deviation, see `docs/limitations.md`).

## 6. Reproduce

```bash
source .venv/bin/activate
python -m mltrading.experiments.run --config configs/v1.toml --note "why"   # ~2.5 min; appends docs/attempt_log.jsonl
python -m mltrading.experiments.report                                      # docs/results_tables.md
python -m mltrading.experiments.plots                                       # docs/*.png  (needs `pip install -e .[viz]`)
python -m mltrading.experiments.benchmark --label after --profile          # docs/benchmarks.md numbers
python -m pytest -q
```
