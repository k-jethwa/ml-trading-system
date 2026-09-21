# How This System Was Built — Architecture, Code, and the Finance Behind It

This is the project's learning log, in the format `project_spec.md`
section 15 asks for: for each piece of the system, a plain-language
definition of the concept, the math/logic behind it, the simplest
implementation, at least one failure mode or test, and an honest note on
limitations. It covers the whole system, in the order the pipeline actually
runs: data ingestion, validation, feature engineering, target construction
(§3–§7), then models, walk-forward validation, evaluation metrics, portfolio
construction, the event-driven backtester with costs, robustness experiments,
and the systems pieces (§11–§17). Results are summarized in §18 and in full in
`docs/results.md`; the architecture map is `docs/architecture.md`.

If you're reading this to relearn the material later, each section
stands mostly on its own; you don't need the earlier sections fresh in
mind to follow a later one.

---

## 1. The problem, in plain language

The system does **not** try to predict what a stock's price will be next
week. It tries to answer a narrower, more tractable question: *of these
~55 stocks, which ones will outperform their own sector over the next
five trading days, and which will underperform?*

That distinction — ranking stocks against each other rather than
forecasting an absolute price — is called **cross-sectional prediction**,
and it's the foundation everything else is built on. It matters for a
concrete reason: absolute price forecasting bakes in exposure to overall
market direction (if the market surges, "the model called it" even if it
just called the market), which is a much harder and noisier thing to
predict than *relative* outperformance. A cross-sectional model only has
to be right about *ranking*, not about magnitude or market direction,
which is a meaningfully easier and more stable target — the ranking can
stay useful even in a flat or falling market, since the trade is long
the winners and short the losers, not long the market.

---

## 2. Pipeline (data → features; the later stages are in §11–§17)

```text
yfinance (free, live)
        │
        ▼
┌─────────────────────┐
│  Ingestion           │  src/mltrading/data/ingest.py
│  raw OHLCV → Parquet │  universe.py defines WHICH 55 tickers
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Validation           │  src/mltrading/data/validate.py
│  gaps / stale / jumps │  → docs/data_validation_report.md
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Feature engine        │  src/mltrading/features/
│  price_features.py      →  20 point-in-time features
│  cross_sectional.py      →  leave-one-out group means
│  target.py               →  5-day sector-relative label
│  build.py                →  orchestrates + writes Parquet
└─────────────────────┘
        │
        ▼
   data/processed/features.parquet
   (160,702 rows × 31 columns, ready for model training)
```

Everything downstream of `features.parquet` (models, walk-forward, portfolio,
backtest, robustness, serving) is described in §11–§17 and mapped in
`docs/architecture.md`.

---

## 3. Stage 1 — Data ingestion (`src/mltrading/data/`)

### 3.1 The concept: point-in-time data

**Definition.** A dataset is "point-in-time" if, for any past date, you
can reconstruct *exactly* what was knowable on that date — no
information that only became available later is allowed to leak in.

**Why it matters.** A backtest that accidentally uses information from
the future will look far more profitable than any real trading strategy
could have been, because it's effectively being told the answer before
it has to guess. This is the single most common way trading-strategy
backtests lie to their authors, and it's rarely intentional — it usually
sneaks in through something as innocuous as computing a "20-day average"
using a rolling window that isn't carefully bounded, or fitting a
normalization step on the full dataset before splitting into train/test.

**How it's enforced here.** Every ingested row carries three provenance
columns that have nothing to do with price or volume:

```python
df["source"] = SOURCE_NAME              # "yfinance"
df["universe_version"] = UNIVERSE_VERSION
df["ingested_at"] = ingested_at          # UTC time this pull happened
```
(`src/mltrading/data/ingest.py`)

This doesn't *by itself* prevent leakage — that's the feature engine's
job (§6) — but it means every downstream number can be traced back to
"what we had, and when we got it," which is what lets you audit a
suspicious result later instead of just trusting it.

### 3.2 How it's built

`universe.py` defines a fixed, hand-picked list of 55 liquid large/mid-
cap US stocks across 8 sectors (Technology, Consumer Discretionary,
Financials, Healthcare, Industrials, Energy, Consumer Staples,
Communication Services). `ingest.py` downloads daily OHLCV bars for each
via `yfinance.download()`, reshapes the result into one tidy row per
(date, ticker), and writes one Parquet file per ticker to `data/raw/`.

```python
def download_daily_bars(tickers, start, end=None, ingested_at=None):
    raw = yf.download(tickers=tickers, start=start, end=end,
                       auto_adjust=False, group_by="ticker", threads=True)
    ...
```

Splitting output into per-ticker Parquet files (rather than one giant
file) is a small but deliberate choice: if one ticker's data is bad or
the feed rate-limits partway through, you can re-ingest just that ticker
without redoing all 55.

### 3.3 Known limitations (this is the $0-budget trade-off)

Choosing yfinance for zero cost means accepting two specific gaps,
written up in full in `docs/limitations.md`:

1. **Survivorship bias.** The universe is *today's* list of liquid
   names, not a reconstruction of who was actually tradable on each
   historical date. Companies that were delisted, acquired, or went
   bankrupt before today are simply absent — which means any backtest
   over this data is missing its historical losers by construction, and
   will look better than a strategy run in real time could actually have
   performed. There's no way to fully fix this without a paid
   point-in-time dataset (Norgate, Sharadar, etc.); V1 accepts and
   discloses it rather than hiding it.
2. **Sector labels aren't point-in-time either** — they reflect each
   company's *current* GICS sector, not necessarily its historical one.

**Failure mode this guards against:** reporting a backtest Sharpe ratio
without disclosing that the universe was chosen with hindsight. Section
13 of `project_spec.md` calls this out explicitly as an invalidating
failure mode, so every future results write-up needs to restate this
caveat, not just this doc.

---

## 4. Stage 2 — Data validation (`src/mltrading/data/validate.py`)

**Concept.** Before any modeling, you check whether the raw data itself
is trustworthy — the classic "garbage in, garbage out" problem, but made
concrete for price data: missing trading days, prices that are zero or
negative (a data-feed error, since prices can't actually go negative),
volume that's zero or negative, one-day price jumps large enough to
suggest an unadjusted stock split or a bad tick, and prices that freeze
(a stale/disconnected feed).

**Implementation.** Five checks, run per ticker against a reference
calendar built from the observed union of all trading days across the
universe (so a ticker is only flagged "missing" a day that some *other*
ticker in the universe actually traded on):

```python
JUMP_THRESHOLD = 0.50          # flag |single-day return| > 50%
STALE_DAYS_THRESHOLD = 5       # flag 5+ identical consecutive closes
```

**Result on real data:** 50 of 55 tickers passed every check cleanly.
The other 5 (BLK, TJX, AXP, MS, SCHW) were each missing 1–3 days, all
clustered in the most recent month before ingestion — a yfinance
backfill lag, not a real historical gap (five unrelated financial/retail
names don't halt trading together on the same two days). Full detail in
`docs/data_validation_report.md`.

**A bug this caught during development:** the first version of the
stale-run check used `close.diff().eq(0)` to count frozen days, which
undercounts a run by one — five identical consecutive prices produce
only four zero-diffs, since `diff()` has no predecessor for the first
row of the run. The synthetic test `test_detects_stale_run_and_duplicate_and_non_positive`
caught this immediately (it asserted `max_stale_run == 5` and got `4`),
and the fix (`_longest_flat_run`, which groups by "value changed from
previous row" instead of diffing) is now what runs in production. This
is a small but real example of why every check has a synthetic
known-answer test: a check that's wrong is worse than no check, because
it creates false confidence.

---

## 5. Stage 3 — Feature engineering (`src/mltrading/features/`)

Every feature below is computed from a single ticker's *own* trailing
history — a rolling or shifted window that ends at row *t* and never
reads a row after *t*. That's what makes them safe to use as model
inputs. (The mechanical proof of this is in §7.)

Five feature families, chosen deliberately small and interpretable per
`project_spec.md` section 6 ("V1 explicitly avoids a large, unexamined
feature zoo") — 20 columns total, not 200.

### 5.1 Momentum

**Concept.** The momentum anomaly is one of the most robustly documented
patterns in empirical finance (Jegadeesh & Titman, 1993): stocks that
have gone up over the past several months tend to keep going up over the
next few months, more than a random-walk model of prices would predict.
Nobody fully agrees on *why* — under-reaction to news, herding, and
slow information diffusion are all candidate explanations — but the
empirical pattern itself has held up across markets and decades. It's
the single most natural first hypothesis to test for a cross-sectional
equity model, which is why it's the largest feature family here.

**Implementation** (`price_features.py`):

```python
out["ret_1d"]  = px.pct_change(1)
out["ret_5d"]  = px.pct_change(5)
out["ret_20d"] = px.pct_change(20)
out["ret_60d"] = px.pct_change(60)
out["mom_12_1"] = px.shift(21) / px.shift(252) - 1
```

`ret_1d`/`5d`/`20d`/`60d` are simple trailing returns over increasingly
long windows — short-, medium-, and longer-term momentum are genuinely
different signals with different half-lives, so all four are kept rather
than picking just one.

`mom_12_1` is the classic academic "12-month minus 1-month" momentum
signal: the return from 12 months ago to 1 month ago, deliberately
*excluding* the most recent month. That exclusion isn't an accident —
the most recent month's return tends to partially mean-revert (a
short-term reversal effect that's almost the opposite of momentum), so
including it would mix two offsetting signals into one noisy number.

**Distance from moving average** (`dist_ma_20`, `dist_ma_50`): how far
today's price sits above or below its own 20- and 50-day average,
`price / MA - 1`. This is a cheap, standard trend-following signal —
conceptually related to momentum but capturing "am I currently above my
recent trend line" rather than "how much have I moved."

### 5.2 Volatility

**Concept.** Volatility isn't just risk to control later — it's
predictive information in its own right, because of a well-documented
phenomenon called **volatility clustering**: periods of high volatility
tend to be followed by more high volatility, and calm periods tend to
stay calm. A stock's recent volatility also matters for interpreting its
recent return — a 5% move for a stock that normally moves 1%/day is a
very different signal than the same move for a stock that normally moves
5%/day.

**Implementation:**

```python
out["realized_vol_20"] = daily_ret.rolling(20).std() * sqrt(252)
out["ewma_vol_20"]     = daily_ret.ewm(span=20).std() * sqrt(252)
out["range_vol_20"]    = ((high - low) / close).rolling(20).mean()
```

`realized_vol_20` is the standard textbook estimator: the trailing
20-day standard deviation of daily returns, annualized by multiplying by
√252 (there are ~252 trading days in a year, and variance scales
linearly with time, so standard deviation scales with its square root).

`ewma_vol_20` uses **exponentially weighted** volatility instead of a
flat rolling window — each day's contribution decays geometrically
(`span=20` corresponds to a decay factor of `2/(20+1)`), so a volatility
spike from three weeks ago fades out faster than in the flat 20-day
window. This makes it react faster to genuine regime changes.

`range_vol_20` is a simplified Parkinson-style range estimator: instead
of using only the closing price, it uses the full high-low range each
day, which captures intraday volatility that a close-to-close return can
miss entirely (a stock that swings ±5% intraday but closes flat looks
perfectly calm to `realized_vol_20`, but not to this one).

### 5.3 Liquidity

**Concept.** A stock can have a great predicted return and still be a
bad trade if you can't actually buy or sell it in size without moving
the price against yourself (market impact). Liquidity features exist to
let later stages of the system — universe eligibility, position sizing,
transaction cost modeling — filter out or down-weight names that look
attractive on paper but aren't realistically tradable at the position
sizes this strategy would need.

**Implementation:**

```python
out["dollar_volume_20"] = (close * volume).rolling(20).mean()
out["avg_volume_20"]    = volume.rolling(20).mean()
out["volume_ratio_20"]  = volume / avg_volume_20
```

`dollar_volume_20` (price × shares traded, averaged over 20 days) is the
standard "how much capital moves through this name per day" measure —
more directly useful than raw share volume, since a $5 stock trading 10
million shares and a $500 stock trading 100,000 shares carry the same
dollar liquidity despite very different share counts. `volume_ratio_20`
flags unusual volume spikes (today vs. the recent baseline), which
typically coincide with news or earnings and can mean either extra
tradability or extra risk depending on context.

### 5.4 Market-relative and sector-relative returns

**Concept.** A stock's raw 5-day return mixes together at least two
different things: how the whole market moved, how its sector moved, and
what the stock did on its own. If the market rallied 3% and a stock rose
2%, that stock actually *underperformed* even though its return was
positive — the raw number alone doesn't tell you that. Subtracting out
the market or sector return isolates the stock-specific component, which
is closer to what a genuinely predictive signal should be measuring
(and is directly aligned with the sector-relative *target* — see §6).

**A subtlety: leave-one-out.** When computing "the sector's average
return" to subtract from a stock's own return, it would be circular to
include the stock's own return in that average — for a big sector that
barely matters, but the V1 universe's Energy sector has only 4 names
(XOM, CVX, COP, SLB), so including a stock in its own benchmark would
distort the comparison by 25%. `cross_sectional.py` implements a
**leave-one-out mean**: for each stock, the average of every *other*
member of its group (sector, or the whole market) on the same date.

```python
loo_sum = grp_sum - values.where(has_own, 0.0)
loo_count = grp_count - has_own.astype(float)
return loo_sum / loo_count
```

This same function drives both the backward-looking features here and
the forward-looking target in §6 — one implementation, two uses, so
there's no chance of the feature and the target disagreeing about what
"sector-relative" means.

```python
panel[f"mkt_rel_{h}"]    = panel[h] - leave_one_out_mean(panel, h, ["date"])
panel[f"sector_rel_{h}"] = panel[h] - leave_one_out_mean(panel, h, ["date", "sector"])
```
for `h` in `{ret_1d, ret_5d, ret_20d}` — six columns total.

### 5.5 Technical state

**Concept.** "Overbought" and "oversold" are informal trading terms for
whether a stock is trading near the top or bottom of its recent range —
the intuition some traders use for short-term mean reversion, which sits
in tension with the momentum features above (momentum says "keep
going," a range-position extreme sometimes precedes "snap back"). Having
both in the feature set lets the model — not a hand-picked rule — decide
which effect dominates and when.

**Implementation:**

```python
roll_min_20 = low.rolling(20).min()
roll_max_20 = high.rolling(20).max()
out["range_position_20"] = (close - roll_min_20) / (roll_max_20 - roll_min_20)
```

Bounded between 0 (today's close is the lowest point of the past 20
days) and 1 (today's close is the highest).

---

## 6. Stage 4 — Target construction (`src/mltrading/features/target.py`)

**Concept.** The label the model is trained to predict is *not* "will
this stock go up." It's:

```text
target(i, t) = 5-day forward return(i, t) − 5-day forward sector return(sector(i), t)
```

i.e. the same sector-relative idea as §5.4, but applied to the *future*
return instead of a past one. This is the direct, testable version of
the research hypothesis in `project_spec.md` section 2: does the model
know something about which stocks will beat their own sector over the
next week, not about where the whole market is headed?

**Why sector-relative and not just raw forward return?** Two reasons.
First, it matches how the resulting portfolio actually trades (§9,
future work): long the top-ranked names, short the bottom-ranked ones,
targeting market neutrality — a sector-relative target trains the model
on exactly the kind of relative outperformance that portfolio will try
to capture. Second, it removes a huge source of noise: overall
market/sector moves are large relative to any single stock's
idiosyncratic move, so a model trained on raw returns would spend most
of its capacity (and most of its apparent "accuracy") just learning to
track the market, not learning anything stock-specific.

**Implementation:**

```python
out["fwd_ret_5d"] = (
    out.groupby("ticker")["adj_close"].shift(-5) / out["adj_close"] - 1
)
sector_fwd_loo = leave_one_out_mean(out, "fwd_ret_5d", ["date", "sector"])
out["target_5d_sector_rel"] = out["fwd_ret_5d"] - sector_fwd_loo
```

`shift(-5)` looks *forward* five rows — this is the one place in the
whole feature engine where looking into the future is not a bug, because
this becomes the training label, never a model input. To make that
separation impossible to get wrong by accident, `build.py`'s
`FEATURE_COLUMNS` list — the columns any training code should read as
`X` — never includes `fwd_ret_5d` or `target_5d_sector_rel`; anyone
writing `X = df[FEATURE_COLUMNS]` structurally cannot leak the label into
the inputs.

**On real data:** target mean across all 160,702 rows was `1.66e-20`
(numerically zero) and standard deviation ~3.3% — exactly what you'd
expect from a quantity that's centered by construction (each stock's
forward return is compared against its peers, so on average across the
whole panel the ups and downs cancel out).

---

## 7. Point-in-time discipline and anti-leakage testing

`project_spec.md` section 13 names three specific ways a backtest can
lie to you:

- **Look-ahead bias** — using information that wasn't actually available
  at decision time (e.g. deciding after a close while assuming execution
  at that same close).
- **Survivorship bias** — covered in §3.3 (universe built with
  hindsight).
- **Data leakage** — future information sneaking into a feature or
  preprocessing step, e.g. fitting a scaler on the entire dataset before
  splitting into train/test periods, so the "training" statistics have
  already seen the test period.

Look-ahead bias and data leakage are the same failure mode wearing two
names, and they're exactly what the feature engine is built to prevent
by construction — every feature uses a rolling/trailing window ending at
row *t*. But "we designed it carefully" is a claim, not a proof.
`tests/features/test_no_lookahead.py` turns it into a proof:

1. Build a small synthetic panel (4 tickers, 2 sectors, 320 days) with a
   known price path.
2. Build a second copy of the same panel, but multiply every price by 5×
   starting at day 300 onward — a deliberately large, unmissable
   mutation, applied only to the future.
3. Recompute every feature and the target on both panels.
4. Assert every `FEATURE_COLUMNS` value **before day 300 is bit-for-bit
   identical** between the two runs. If any feature had accidentally
   used a `shift(-N)` instead of `shift(N)`, or a rolling window that
   wasn't properly bounded, this would catch it immediately — the
   mutated future prices would leak backward into an "earlier" feature
   value.
5. Assert the **target** value *does* change for rows whose 5-day
   forward window reaches into the mutated region (e.g. day 297, since
   297 + 5 ≥ 300) — proving the target genuinely depends on future data,
   as a label correctly should. This also catches the opposite bug: if
   someone accidentally made the target backward-looking, this assertion
   would fail because nothing would have changed.

This is the single most important test in the codebase, because it's
the one directly protecting against the failure mode most likely to
silently invalidate every future result.

---

## 8. Engineering practices behind these two stages

- **Parquet, not CSV.** Columnar storage, much smaller on disk, and
  preserves dtypes (a CSV would round-trip a `datetime64` through a
  string and force you to re-parse it every time).
- **Per-ticker files for raw data**, but **one merged table for
  features** — raw ingestion benefits from being able to redo a single
  ticker independently; the feature table is consumed as a whole by
  training code, so merging it once at build time is simpler downstream.
- **Pure functions for testability.** `build_feature_table(panel)` takes
  an in-memory DataFrame and returns one — no disk or network I/O inside
  it — specifically so `test_no_lookahead.py` can call it directly on
  synthetic data without needing real ingested files.
- **All tests are synthetic/known-answer with zero network calls (85 across data, features, models, portfolio, backtest and serving).** Every
  check has a hand-computed expected value (e.g. "20-day MA of 19 rows
  of 100 plus one row of 200 should be exactly 105") rather than a vague
  "does it run without crashing" test — per `project_spec.md` section 12
  ("unit and integration tests, including synthetic known-answer
  cases").

---

## 9. Known limitations (consolidated)

Full detail lives in `docs/limitations.md` and
`docs/data_validation_report.md`; the summary that matters for using
this data downstream:

1. **Survivorship bias** in the universe — accept as an upper bound on
   realistic historical performance, not a realistic estimate.
2. **Sector labels are current, not historical** — a small, unquantified
   corruption of sector-relative features/target for any ticker that
   changed GICS sector historically.
3. **5 tickers are missing 1–3 days each**, all in the most recent
   month — avoid training/eval windows whose boundary falls exactly in
   that gap until re-verified.
4. **`adj_close` is retroactively revised** by yfinance when new
   splits/dividends occur — two ingestion runs on different days can
   give slightly different historical `adj_close` values, which is why
   every row is stamped with `ingested_at`.
5. **This is v1's fixed, static universe** (55 names) — the spec's
   eventual target is 100–500; this will need revisiting before scaling
   up.
6. **Modeling, cost and evaluation limitations** added by the later stages
   (spec deviation on gradient boosting, simplified fills, a short and
   partly-seen holdout, beta against the universe rather than an index)
   are listed in `docs/limitations.md` §5–§9.

---

## 10. Reproducing all of this

```bash
source .venv/bin/activate

# 1. Ingest raw data (writes data/raw/*.parquet)
python -m mltrading.data.ingest

# 2. Validate it (prints a clean/unclean summary per ticker)
python -m mltrading.data.validate

# 3. Build features + target (writes data/processed/features.parquet)
python -m mltrading.features.build

# 4. Train + evaluate + backtest + robustness checks (writes results/v1/, appends docs/attempt_log.jsonl)
python -m mltrading.experiments.run --config configs/v1.toml --note "why this run"

# 5. Render tables/figures, benchmark, batch inference, endpoint
python -m mltrading.experiments.report
python -m mltrading.experiments.plots              # needs: pip install -e .[viz]
python -m mltrading.experiments.benchmark --label after --profile
python -m mltrading.models.infer --model ridge --allow-stale
python -m mltrading.models.serve --port 8000

# Run the full test suite (85 tests, no network calls)
python -m pytest -q
```

---

## 11. Stage 5 — Models (`src/mltrading/models/`)

### 11.1 Concept: a model progression, not a model

**Definition.** A model is a function from the day's features to a score per stock. The spec (§7) requires
building models in order of complexity and recording each as a benchmark **before** trying the next, so that
any added complexity has to *earn* its place against something simpler.

| Step | Model | What it tests |
|---|---|---|
| 1 | `mean` (constant) | the floor: predicts the training-set mean for every stock |
| – | `reversal_5d` | a **non-ML** one-liner: score = −(5-day sector-relative return). If an ML model can't beat this, the ML added nothing |
| 2 | OLS | is there a simple additive linear relationship? |
| 3 | Ridge (α = 1000) | the same, with an L2 penalty and a scaler |
| 4 | HGB | gradient-boosted trees: can non-linearity/interactions help? |

**Math (Ridge).** OLS minimizes Σ(yᵢ − xᵢβ)². Ridge adds α‖β‖²: large coefficients are penalized, which
shrinks estimates toward zero and stabilizes them when features are correlated (several features here are related by construction, e.g. `ret_20d`,
`dist_ma_20` and `sector_rel_ret_20d`). Because the penalty depends on coefficient size, features must be on a
common scale; the `StandardScaler` in the pipeline is fit on the **training fold only** (a global scaler would leak test-period means and
variances into training, the spec §13 leakage example; `test_ridge_scaler_is_fit_on_train_only` asserts it).

**Features to the models.** Each feature is replaced by its same-date cross-sectional percentile rank, centered to [−0.5, 0.5]
(`models/preprocess.py`). That removes scale differences (a 5-day return is ~0.02, dollar volume ~10⁹) and non-stationarity (dollar volume
trends up for a decade) without fitting anything across time, so nothing can leak.

**Deviation from the spec (disclosed).** §7 names XGBoost or LightGBM. Both need the system OpenMP library (`brew install libomp`), which this machine does not
have and I did not install without asking. `HistGradientBoostingRegressor` from scikit-learn is the same family (histogram-binned gradient-boosted trees) and
ships its own OpenMP. Swapping is a change in one factory function (`models/zoo.py`).

### 11.2 Failure mode and test

*Overfitting through tuning.* Every hyperparameter is fixed in `configs/v1.toml` before any result and never revisited. HGB is deliberately weak
and heavily regularized (depth 3, 500-sample leaves, learning rate 0.05) because daily return data is mostly noise.
*Look-ahead through the label.* `X = df[FEATURE_COLUMNS]` cannot include the label: labels live in columns that `FEATURE_COLUMNS` never lists.

### 11.3 Result and conclusion

Full-period rank IC (higher is better): mean —, reversal 0.011, OLS 0.009, Ridge 0.009, HGB 0.014. Only Ridge and OLS have positive out-of-sample R², and only barely
(+0.01–0.02%); HGB's is −0.24%. None of the ML models is clearly better than the one-line reversal heuristic. Details: `docs/results.md`.

---

## 12. Stage 6 — Walk-forward validation (`models/walk_forward.py`)

**Definition.** Evaluate the way you would have used the model: train on the past, predict the next block of time, then roll forward.
Never shuffle across time (a shuffled split lets the model train on Tuesday's data to "predict" Monday's).

**Logic.** Expanding window, one retrain per calendar year, first test year 2019 (train 2016–2018), through 2026:

```text
Train 2016–2018 → Test 2019      Train 2016–2019 → Test 2020   ...   Train 2016–2024 → Test 2025   Train 2016–2025 → Test 2026 (partial)
```

**The purge.** The label at date *t* is the return from *t* to *t+5* trading days. A training row dated *t = test_start − 3* has a label
that spans three days *inside* the test window, so the model would have "seen" test-period returns. Each fold therefore drops the last
5 trading dates before the test start from training. `test_no_training_label_window_reaches_test_period` asserts that the
label window of the last training row ends strictly before the test start, and `test_folds_are_chronological_and_purged` asserts exactly 5
dates sit between them.

**Holdout discipline.** Test years 2019–2024 are *development*; 2025-01-01 onward is the *holdout*, reported separately. That only means something if
configuration is fixed first and the number of attempts is recorded, which is why every full run appends to `docs/attempt_log.jsonl`
(and why `docs/results.md` discloses that the holdout was partly seen during smoke testing).

**Failure mode.** Overlapping labels: consecutive dates share 4 of 5 days of return, so daily ICs are strongly autocorrelated and a naive
t-statistic overstates significance by roughly √5. The system computes t-statistics on every 5th date.

---

## 13. Stage 7 — Evaluation metrics (`models/evaluate.py`)

| Metric | Definition | Why |
|---|---|---|
| IC | per-date Pearson correlation of prediction and realized label, averaged over dates | linear agreement |
| Rank IC | the same with Spearman (ranks) | the headline: the portfolio only uses **order** |
| MAE / MSE | mean absolute / squared error, pooled | magnitude accuracy |
| OOS R² | 1 − MSE(model) / MSE(constant predictor) | negative = worse than predicting the mean |
| Decile spread | mean label of the top predicted decile minus the bottom, per date | closest to what a long/short book earns before costs |

**Ranking versus regression error (ML checkpoint).** They measure different things. HGB has the *highest* full-period rank IC (0.014) and the *worst*
OOS R² (−0.24%): it orders stocks a little better than chance while being badly wrong about magnitudes. A model can also have good MSE and no ranking
skill. Picking a model by MSE would have chosen wrongly for a ranked portfolio, and picking by IC alone ignores calibration, which matters for the optimizer
because it treats predictions as expected returns.

**Why accuracy may not translate into PnL (ML checkpoint).** HGB has the best full-period rank IC but Ridge has the better baseline-portfolio Sharpe
(0.52 vs 0.27). Reasons visible in this data: PnL depends on the *extremes* of the ranking (6 names per side) rather than the average IC across all 55;
Ridge's baseline book carries beta 0.60 in a bull market; and costs (≈6% a year) are the same order as the whole edge.
Tests: perfect predictor → IC = +1, reversed → −1, constant predictions → undefined (`tests/models/test_models.py`).

---

## 14. Stage 8 — Portfolio construction (`src/mltrading/portfolio/`)

### 14.1 Baseline: long the top decile, short the bottom

**Definition.** Rank stocks by predicted return; hold the top 10% long and the bottom 10% short, equal-weight within each side. Long side sums to +1.0 of NAV,
short side to −1.0: **dollar-neutral** (net exposure 0) with **gross exposure** 2.0 (Σ|weights|). With 55 names that is 6 stocks per side. Ties break by ticker
so results are deterministic.

**Why "market-neutral" is only approximately true.** Net dollar exposure is zero, but *beta* need not be: if the longs are higher-beta than the shorts the book is
still long the market. Measured: baseline beta 0.28–0.60. This is the single most important interpretive fact in the results.

### 14.2 Risk-aware: a constrained optimizer

For each rebalance the optimizer solves (cvxpy, Clarabel):

```text
maximize    αᵀw  −  (γ/2)·H·wᵀΣw  −  c·‖w − w_prev‖₁
subject to  ‖w‖₁ ≤ 2.0            gross exposure
            |Σw| ≤ 0.02           net exposure
            |wᵢ| ≤ 0.10           concentration
            |Σ_{i∈sector} wᵢ| ≤ 0.10   per-sector net
            |βᵀw| ≤ 0.05          market beta
            ‖Fᵀw‖₂ ≤ σ_target/√252    ex-ante volatility ≤ 10%/yr   (Σ = FFᵀ)
            ‖w − w_prev‖₁ ≤ 0.6   turnover per rebalance
```

α = model's predicted 5-day sector-relative return; Σ = Ledoit-Wolf shrunk daily covariance of the last 60 days; H = 5 (holding period);
c = one-way cost as a fraction of notional (5 bp). Each term has a plain meaning: reward predicted return, penalize risk, penalize the *cost of changing the book*
(so it only trades when the predicted gain exceeds the cost), and hard limits rather than penalties for the things that must never be violated.

**Finance checkpoints touched:** *return* (P/P₀ − 1), *volatility* (σ√252), *correlation/covariance* (Σ, shrunk toward a scaled identity because 60 days for 55 stocks gives a noisy, near-singular sample covariance),
*beta* (cov(rᵢ, r_m)/var(r_m), against the equal-weight universe since the dataset has no index), *market neutrality* (β ≈ 0, not merely net ≈ 0), *gross/net exposure*, *turnover* (traded notional / NAV).

### 14.3 Tests and failure modes

* every constraint satisfied on random instances; zero alpha → zero position; high costs suppress trading; sector neutrality binds when alpha favors one sector; turnover limit respected;
* post-solve `check_constraints` runs at every rebalance in the backtest, and a violating book is **rejected** (current book held), never traded;
* the solver-failure policy: only genuine infeasibility relaxes the turnover limit (flagged); a numerical failure holds the book;
* **result:** 384/384 rebalances solved optimally, 0 violations, 0 relaxations per model.
* Limitation: realized beta (0.04–0.10) exceeds the 0.05 ex-ante limit because betas are estimated on 60 noisy days; the constraint controls *estimated* beta.

---

## 15. Stage 9 — Execution, costs and the backtester (`src/mltrading/backtest/`)

### 15.1 Concept

**Definition.** A backtest replays history through the same decision logic you would run live, with explicit assumptions about how orders fill. The
easiest way to make a strategy look profitable is to let it trade at a price it could not have known.

**Event-driven design.** State changes happen through typed events processed in order:

```text
MarketEvent(d)  → borrow cost accrues on the overnight book;  yesterday's orders fill at d's OPEN (FillEvents)
MarkEvent(d)    → mark to d's CLOSE, record NAV
SignalEvent(d)  → (rebalance days) strategy sees only data ≤ d, returns target weights
OrderEvent      → sized off d's close, queued for d+1's open
```

A decision made after the close of *t* can never trade at a price from *t* or earlier; the earliest fill is the next open.
`test_overnight_gap_gives_no_free_return` builds the classic trap (decide at close 100, opens at 110 tomorrow) and asserts NAV cannot rise on the gap;
an engine that filled at the decision-day close would show +10%.

### 15.2 The cost model (all explicit, none free)

| Component | Assumption | Mechanism |
|---|---|---|
| Commission | 1 bp of traded notional per side | cash debit |
| Bid/ask spread | 2 bp half-spread per side | buys pay mid×(1+…), sells receive mid×(1−…) |
| Slippage | 2 bp per side | same, added to the spread |
| Borrow | 50 bp/yr on short market value | accrued daily on the book held overnight |
| Liquidity | ≤ 5% of 20-day average dollar volume per order | larger orders partially fill |
| Missing bar | order lapses | counted; next rebalance re-decides |
| Whole shares | orders truncated to integers | small tracking error vs target |

**Slippage, turnover and drawdown, concretely.** *Slippage* is the gap between the price you decided on and the price you got; here it is a flat 2 bp plus the half-spread.
*Turnover* is traded notional as a multiple of NAV per year; it is the multiplier that turns a per-trade cost into an annual drag (traded notional already counts each buy and sell once: 116× × 5 bp ≈ 5.8% a year, plus ≈ 0.5% borrow, ≈ 6.3% total).
*Max drawdown* is the worst peak-to-trough fall of NAV.

### 15.3 Tests (all hand-computed) and a bug they caught

Fill timing; gap-up no free return; long/short P&L (+10% on a 50% long = +50,000); itemized costs on flat prices equal the exact NAV loss; buys pay up and sells receive
less; borrow accrual; liquidity truncation; lapsed orders; rebalance cadence and liquidation of dropped names; deterministic replay; max-drawdown and Sharpe known answers.
**Bug caught:** the first borrow accrual ran after the day's fills, so a short opened at the day-2 open was charged for the night before it existed (one extra day of borrow per position).
The test for the accrual schedule failed (350 ≠ 300) and the accrual moved to the start of the day.

### 15.4 Metrics (`backtest/metrics.py`)

*Net* = after all costs; *gross* = the same P&L with costs added back. Sharpe = mean(daily return)/std × √252 (zero risk-free rate, appropriate for a dollar-neutral book, ignoring cash interest);
Sortino uses downside deviation; hit rate = fraction of positive days (and of 5-day periods); exposures are averaged daily. **Alpha and beta** come from regressing daily strategy returns on the equal-weight
universe return: r = α + β·r_m + ε. Alpha is what is left after removing market exposure; it is reported with its t-statistic (|t| < 2 ≈ indistinguishable from zero).

---

## 16. Stage 10 — Robustness experiments (`experiments/`)

Each experiment exists to answer "could this result be an artifact?":

| Experiment | Question | Outcome |
|---|---|---|
| Label shuffle | Does the pipeline produce out-of-sample IC from **destroyed** labels? | No: rank IC −0.002 / −0.007 |
| 20 random-score portfolios | What does the same machinery earn with no information? | net Sharpe −0.71 ± 0.49 (≈ −11%/yr): costs alone are ruinous at this turnover |
| Cost sweep 0.5×–4× | How fast do costs erase the result? | baseline negative at 2–4×; risk-aware barely positive at 4× |
| Quantile / rebalance sweep | Is the result a knife-edge of one parameter? | yes: Sharpe 0.13–0.67 across variants, non-monotone |
| Per-year stability | Is it consistent? | no: rank IC negative in 2019, 2021, 2023 |
| Alpha/beta split | Is the PnL market exposure? | baseline mostly yes |

**Lesson.** Without the random-portfolio baseline a Sharpe of 0.5 looks fine; with it, you can see that the noise level of this whole setup is ±0.5.

---

## 17. Stage 11 — Systems: registry, inference, serving, observability, profiling

* **Model registry (`models/registry.py`).** `artifacts/<name>/<model_id>/{model.joblib, metadata.json}`, `model_id = name-trainEnd-configHash`. Metadata: feature list + version hash, training window,
  row count, config hash, data fingerprint, git commit, library versions. Every inference output row carries the `model_id`.
* **Batch versus online inference.** *Batch* (`models/infer.py`) scores the whole cross-section for the latest date in one call (~5–9 ms for 55 names; 0.18 s for the whole 10-year panel) and is what a daily
  process would run. The *endpoint* (`models/serve.py`: `/health`, `/predict`, `/metrics`) exposes the same code per request. **Latency** = time for one request (dominated by per-call overhead), **throughput** =
  rows per second in bulk (~800k/s for HGB). The endpoint caches the loaded model and features to avoid paying load time per request.
* **Failure handling for stale/missing data (`infer.py`).** Stale data → refuse (503); feature-version mismatch → refuse (409); missing/incomplete tickers → excluded and reported, never filled. Tested.
* **Deterministic replay.** The backtest is a pure function of (predictions, market data, config); tested by running twice and asserting identical frames. A full re-run on the same config reproduced predictions bit-for-bit.
* **State persistence.** Predictions, NAV, optimizer logs, metrics and a manifest (config, config hash, git commit, data fingerprint, timings, freshness) are written to `results/<name>/`; models to `artifacts/`.
* **Observability.** JSON logs; per-stage timers; data-freshness check (at the time of writing the data snapshot is 35 days old, so every run logs a stale-data warning, which is the monitor working); per-model, per-year prediction-distribution
  statistics to catch drift or degenerate outputs.
* **Profiling and caching.** `docs/benchmarks.md`: the bottleneck was cvxpy re-canonicalizing an identical problem structure on every rebalance (11 of 15 s). Compiling once (DPP) gave 2.7× per optimization and 1.9× on the
  backtest. Verifying it also surfaced a formulation problem that unit tests had accepted (solver `optimal_inaccurate`) — see that doc.

---

## 18. Results in one paragraph

See `docs/results.md` (narrative) and `docs/results_tables.md` (generated). The system finds a small ranking signal (rank IC ≈ 0.01) that is unstable across years and mostly a short-term reversal effect; after realistic costs and
beta adjustment no strategy shows statistically significant alpha, and none beats buying and holding the same stocks in 2019–2026. The risk-aware book is much more cost-robust than the naive decile book; the naive book's
apparent Sharpe is largely market beta. The valuable outputs are the checks, not the returns.

---

## 19. What's next (Phase 2 candidates, per `project_spec.md` §14)

In rough order of expected value:

1. **Point-in-time universe and sector data** (Norgate/Sharadar or similar) — removes the survivorship and sector-label limitations that bias every number here upward.
2. **A wider universe (100–500 names).** 55 names give only ~6 per side; the ranking signal cannot diversify. This is likely the biggest lever on both signal-to-noise and turnover.
3. **Lower-turnover design:** longer holding periods with the cost-aware optimizer; the parameter sweep hints at it but was not used to choose settings.
4. **Better risk model** (factor model, index/ETF beta rather than a universe average).
5. **XGBoost/LightGBM** (after `brew install libomp`), and a proper tuning protocol with nested walk-forward validation and a truly untouched holdout.
6. **Fresh data.** Re-ingest (the current snapshot ends 2026-08-17); note that vendor `adj_close` revisions will change historical values, so treat as a new experiment.
