# How This System Was Built — Architecture, Code, and the Finance Behind It

This is the project's learning log, in the format `project_spec.md`
section 15 asks for: for each piece of the system, a plain-language
definition of the concept, the math/logic behind it, the simplest
implementation, at least one failure mode or test, and an honest note on
limitations. It covers everything built so far — data ingestion,
validation, feature engineering, and target construction — in the order
the pipeline actually runs.

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

## 2. Pipeline built so far

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

Not yet built: model training, walk-forward evaluation, portfolio
construction, backtesting. Those are next — see §9.

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
- **19 tests, all synthetic/known-answer, zero network calls.** Every
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

# Run the full test suite (19 tests, no network calls)
python -m pytest -q
```

---

## 11. What's next

Per the pipeline in `project_spec.md` section 11, the next stage is
**walk-forward model training**, starting with the simplest possible
baseline (a constant/mean predictor) before anything more complex — per
section 7's explicit instruction not to skip straight to a fancier
model. That means:

1. A walk-forward train/test split respecting chronological order (never
   shuffling across time — the whole point of this exercise is testing
   whether the model would have worked in real time).
2. The baseline mean predictor, then linear regression, then ridge, then
   a gradient-boosted tree (XGBoost/LightGBM) — each one only introduced
   after the simpler one is recorded as a comparison point.
3. Evaluation with both ML metrics (MAE, MSE, Spearman rank correlation,
   information coefficient) and, once a portfolio exists, trading
   metrics (Sharpe, drawdown, turnover, hit rate).
