"""Assemble the long-format price panel used by feature computation.

One row per (date, ticker). This is the join point between raw ingestion
(mltrading.data.ingest) and everything downstream — features, targets,
and eventually models all consume this shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mltrading.data.ingest import RAW_DATA_DIR

PANEL_COLUMNS = ["date", "ticker", "sector", "open", "high", "low", "close", "adj_close", "volume"]


def load_price_panel(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load every per-ticker raw Parquet file and concatenate into one
    panel, sorted by (ticker, date).
    """
    paths = sorted(Path(raw_dir).glob("*.parquet"))
    if not paths:
        raise ValueError(f"No raw parquet files found in {raw_dir}")

    frames = [pd.read_parquet(path, columns=PANEL_COLUMNS) for path in paths]
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)
