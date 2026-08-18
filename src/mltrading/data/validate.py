"""Raw-data validation checks for project_spec.md section 11
("Market-data ingestion and validation").

These checks catch the mechanical problems that would otherwise silently
corrupt every downstream feature and target: missing trading days,
stale/frozen prices, non-positive prices or volume, and single-day price
jumps large enough to suggest a bad tick or an unadjusted stock split.

This module reports problems; it does not decide how to fix them. Fixing
(dropping vs. forward-filling vs. re-ingesting a ticker) is a modeling
decision that belongs in a later stage, once a human has seen the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mltrading.data.ingest import RAW_DATA_DIR

# A single-day return larger than this is flagged for review. 50% is well
# above normal single-name daily moves and is meant to catch unadjusted
# splits or bad ticks, not to be a precise economic threshold.
JUMP_THRESHOLD = 0.50

# Flag a ticker if its close price does not change at all for this many
# consecutive trading days (a frozen/stale feed).
STALE_DAYS_THRESHOLD = 5


@dataclass
class TickerReport:
    ticker: str
    n_rows: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    missing_trading_days: list[pd.Timestamp] = field(default_factory=list)
    non_positive_price_rows: int = 0
    non_positive_volume_rows: int = 0
    large_jump_dates: list[pd.Timestamp] = field(default_factory=list)
    max_stale_run: int = 0
    duplicate_dates: int = 0

    @property
    def is_clean(self) -> bool:
        return not (
            self.missing_trading_days
            or self.non_positive_price_rows
            or self.non_positive_volume_rows
            or self.large_jump_dates
            or self.max_stale_run >= STALE_DAYS_THRESHOLD
            or self.duplicate_dates
        )


def _longest_flat_run(values: pd.Series) -> int:
    """Length (in rows) of the longest run of consecutive equal values.

    E.g. [100, 100, 100, 100, 100] is one run of length 5, not 4 — a
    diff()==0 mask undercounts by one because diff() has no predecessor
    for the first row of the run.
    """
    if values.empty:
        return 0
    changed = values.ne(values.shift())
    run_id = changed.cumsum()
    return int(values.groupby(run_id).size().max())


def validate_ticker(df: pd.DataFrame, ticker: str, calendar: pd.DatetimeIndex) -> TickerReport:
    df = df.sort_values("date")

    duplicate_dates = int(df["date"].duplicated().sum())

    expected = calendar[(calendar >= df["date"].min()) & (calendar <= df["date"].max())]
    missing = sorted(set(expected) - set(df["date"]))

    non_positive_price_rows = int(
        (df[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1).sum()
    )
    non_positive_volume_rows = int((df["volume"] <= 0).sum())

    daily_return = df["adj_close"].pct_change()
    large_jump_dates = df.loc[daily_return.abs() > JUMP_THRESHOLD, "date"].tolist()

    max_stale_run = _longest_flat_run(df["close"])

    return TickerReport(
        ticker=ticker,
        n_rows=len(df),
        date_min=df["date"].min(),
        date_max=df["date"].max(),
        missing_trading_days=missing,
        non_positive_price_rows=non_positive_price_rows,
        non_positive_volume_rows=non_positive_volume_rows,
        large_jump_dates=large_jump_dates,
        max_stale_run=max_stale_run,
        duplicate_dates=duplicate_dates,
    )


def build_reference_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all observed trading dates across the universe.

    Using the observed union (rather than an external exchange calendar)
    keeps this dependency-free; a ticker is only flagged "missing" a date
    if some other ticker in the universe actually traded on it.
    """
    all_dates = sorted(set().union(*(set(df["date"]) for df in frames.values())))
    return pd.DatetimeIndex(all_dates)


def load_raw(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(raw_dir.glob("*.parquet")):
        frames[path.stem] = pd.read_parquet(path)
    return frames


def validate_universe(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Run all checks across every ticker in `raw_dir` and return a
    one-row-per-ticker summary DataFrame, sorted worst-first.
    """
    frames = load_raw(raw_dir)
    if not frames:
        raise ValueError(f"No parquet files found in {raw_dir}")

    calendar = build_reference_calendar(frames)
    reports = [validate_ticker(df, ticker, calendar) for ticker, df in frames.items()]

    summary = pd.DataFrame([
        {
            "ticker": r.ticker,
            "n_rows": r.n_rows,
            "date_min": r.date_min,
            "date_max": r.date_max,
            "n_missing_days": len(r.missing_trading_days),
            "non_positive_price_rows": r.non_positive_price_rows,
            "non_positive_volume_rows": r.non_positive_volume_rows,
            "n_large_jumps": len(r.large_jump_dates),
            "max_stale_run": r.max_stale_run,
            "duplicate_dates": r.duplicate_dates,
            "is_clean": r.is_clean,
        }
        for r in reports
    ])
    return summary.sort_values(["is_clean", "n_missing_days"], ascending=[True, False]).reset_index(drop=True)


if __name__ == "__main__":
    summary = validate_universe()
    n_clean = int(summary["is_clean"].sum())
    print(f"{n_clean}/{len(summary)} tickers clean")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary)
