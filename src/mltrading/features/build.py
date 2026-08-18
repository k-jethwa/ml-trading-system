"""Build and persist the V1 feature + target table.

Pipeline (project_spec.md section 11, "Feature engine and persisted
feature tables"):

  1. Load the raw per-ticker panel (mltrading.features.panel).
  2. Compute per-ticker point-in-time features (price_features).
  3. Compute cross-sectional market-/sector-relative features
     (cross_sectional) — these only read each ticker's own
     already-point-in-time return columns as of the same date, so no
     new leakage risk is introduced by going cross-sectional.
  4. Compute the forward-looking label (target) in clearly named,
     clearly separated columns, never merged into FEATURE_COLUMNS.
  5. Persist the full table to data/processed/features.parquet.

`build_feature_table` is a pure function of an in-memory panel so it can
be unit-tested directly against synthetic data (see
tests/features/test_no_lookahead.py) without touching disk or network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mltrading.data.ingest import RAW_DATA_DIR
from mltrading.features.cross_sectional import leave_one_out_mean
from mltrading.features.panel import load_price_panel
from mltrading.features.price_features import add_price_features
from mltrading.features.target import add_target

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FEATURE_TABLE_PATH = PROCESSED_DATA_DIR / "features.parquet"

# Backward-looking return columns that get a market-relative and a
# sector-relative counterpart. Kept small and deliberate, per
# project_spec.md section 6's "V1 explicitly avoids a large, unexamined
# feature zoo."
_RELATIVE_RETURN_HORIZONS = ["ret_1d", "ret_5d", "ret_20d"]

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "mom_12_1",
    "dist_ma_20", "dist_ma_50",
    "realized_vol_20", "ewma_vol_20", "range_vol_20",
    "dollar_volume_20", "avg_volume_20", "volume_ratio_20",
    "range_position_20",
] + [f"mkt_rel_{c}" for c in _RELATIVE_RETURN_HORIZONS] + [
    f"sector_rel_{c}" for c in _RELATIVE_RETURN_HORIZONS
]

TARGET_COLUMN = "target_5d_sector_rel"
IDENTIFIER_COLUMNS = ["date", "ticker", "sector"]


def build_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: raw price panel -> full feature + target table."""
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Iterate tickers explicitly (rather than groupby().apply()) so each
    # sub-frame keeps its own "ticker" column with no ambiguity about
    # whether the grouping column survives the merge back together.
    per_ticker = [add_price_features(group) for _, group in panel.groupby("ticker", sort=True)]
    panel = pd.concat(per_ticker, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    for horizon in _RELATIVE_RETURN_HORIZONS:
        panel[f"mkt_rel_{horizon}"] = panel[horizon] - leave_one_out_mean(panel, horizon, ["date"])
        panel[f"sector_rel_{horizon}"] = panel[horizon] - leave_one_out_mean(panel, horizon, ["date", "sector"])

    panel = add_target(panel)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_and_build(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return build_feature_table(load_price_panel(raw_dir))


def write_feature_table(df: pd.DataFrame, out_path: Path = FEATURE_TABLE_PATH) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows x %d cols to %s", len(df), len(df.columns), out_path)
    return out_path


def run(raw_dir: Path = RAW_DATA_DIR) -> Path:
    logging.basicConfig(level=logging.INFO)
    table = load_and_build(raw_dir)
    return write_feature_table(table)


if __name__ == "__main__":
    run()
