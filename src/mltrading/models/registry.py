"""Versioned model artifacts + metadata (project_spec.md sections 11-12).

Every saved model gets a directory

    artifacts/<name>/<model_id>/{model.joblib, metadata.json}

where `model_id = <name>-<train_end>-<config_hash>`; metadata.json
records enough to answer "which exact model made this prediction, and
what did it know?": the feature list and its version hash, the training
window and row count, the config hash, the dataset fingerprint, the git
commit and library versions. `artifacts/<name>/LATEST` names the newest
model for batch inference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from mltrading.config import PROJECT_ROOT
from mltrading.data.universe import UNIVERSE_VERSION
from mltrading.features.build import FEATURE_COLUMNS

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def feature_version(columns: list[str] = FEATURE_COLUMNS) -> str:
    return "f" + hashlib.sha256(",".join(columns).encode()).hexdigest()[:8]


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


@dataclass(frozen=True)
class SavedModel:
    model_id: str
    path: Path
    metadata: dict


def save_model(
    model,
    name: str,
    train_dates: pd.Series,
    n_train_rows: int,
    config_hash: str,
    data_fingerprint: str,
    params: dict,
    artifact_dir: Path = ARTIFACT_DIR,
) -> SavedModel:
    train_end = pd.Timestamp(train_dates.max()).date().isoformat()
    model_id = f"{name}-{train_end}-{config_hash}"
    out_dir = Path(artifact_dir) / name / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    metadata = {
        "model_id": model_id,
        "name": name,
        "params": params,
        "feature_columns": FEATURE_COLUMNS,
        "feature_version": feature_version(),
        "train_start": pd.Timestamp(train_dates.min()).date().isoformat(),
        "train_end": train_end,
        "n_train_rows": int(n_train_rows),
        "config_hash": config_hash,
        "universe_version": UNIVERSE_VERSION,
        "data_fingerprint": data_fingerprint,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "libraries": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (Path(artifact_dir) / name / "LATEST").write_text(model_id)
    return SavedModel(model_id, out_dir, metadata)


def load_latest(name: str, artifact_dir: Path = ARTIFACT_DIR) -> SavedModel:
    root = Path(artifact_dir) / name
    latest = root / "LATEST"
    if not latest.exists():
        raise FileNotFoundError(f"No trained model named {name!r} in {root}; run mltrading.models.train first")
    model_id = latest.read_text().strip()
    path = root / model_id
    return SavedModel(model_id, path, json.loads((path / "metadata.json").read_text()))


def load_model(saved: SavedModel):
    return joblib.load(saved.path / "model.joblib")
