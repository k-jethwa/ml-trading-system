"""Batch inference + serving: happy path and the documented failure modes."""

import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from mltrading.features.build import FEATURE_COLUMNS
from mltrading.models.infer import FeatureVersionMismatch, StaleDataError, score_latest
from mltrading.models.preprocess import build_model_frame
from mltrading.models.registry import save_model
from mltrading.models.serve import Service, make_handler
from mltrading.models.zoo import build_model

TICKERS = [f"T{i:02d}" for i in range(15)]
LAST = pd.Timestamp("2026-08-14")
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _features(seed=0, n_days=30):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range(end=LAST, periods=n_days):
        df = pd.DataFrame(rng.normal(size=(len(TICKERS), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        df["date"], df["ticker"] = d, TICKERS
        df["sector"] = ["A"] * 8 + ["B"] * 7
        df["target_5d_sector_rel"] = 0.01 * df["sector_rel_ret_5d"] + rng.normal(scale=0.01, size=len(TICKERS))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def trained(tmp_path):
    feats = _features()
    frame = build_model_frame(feats)
    model = build_model("ridge").fit(frame[FEATURE_COLUMNS], frame["target"])
    save_model(model, "ridge", frame["date"], len(frame), "cfg123", "fp", {}, artifact_dir=tmp_path)
    return feats, tmp_path


def test_scores_every_name_on_latest_date_with_model_id(trained):
    feats, art = trained
    res = score_latest(feats, "ridge", art, now=NOW)
    assert len(res.predictions) == 15 and set(res.predictions["ticker"]) == set(TICKERS)
    assert (res.predictions["date"] == LAST).all()
    assert res.predictions["model_id"].nunique() == 1 and res.predictions["model_id"].iloc[0].startswith("ridge-")
    assert res.health["status"] == "ok" and res.health["missing_tickers"] == []
    assert res.predictions["pred"].is_monotonic_decreasing


def test_stale_data_is_refused_unless_explicitly_allowed(trained):
    feats, art = trained
    late = datetime(2026, 9, 20, tzinfo=UTC)
    with pytest.raises(StaleDataError):
        score_latest(feats, "ridge", art, now=late)
    res = score_latest(feats, "ridge", art, now=late, allow_stale=True)
    assert res.health["status"] == "stale" and res.health["age_days"] == 37


def test_names_with_missing_features_are_excluded_and_reported(trained):
    feats, art = trained
    feats.loc[(feats["date"] == LAST) & (feats["ticker"] == "T03"), "ret_20d"] = np.nan
    feats = feats[~((feats["date"] == LAST) & (feats["ticker"] == "T07"))]  # no bar at all
    res = score_latest(feats, "ridge", art, now=NOW)
    assert res.health["missing_tickers"] == ["T03", "T07"] and res.health["n_scored"] == 13


def test_feature_version_mismatch_is_rejected(trained, monkeypatch):
    feats, art = trained
    from mltrading.models import infer

    monkeypatch.setattr(infer, "feature_version", lambda *a: "f_changed")
    with pytest.raises(FeatureVersionMismatch):
        score_latest(feats, "ridge", art, now=NOW)


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_latest(_features(), "hgb", tmp_path, now=NOW)


def _serve(tmp_path, feats, max_age_days):
    fpath = tmp_path / "features.parquet"
    feats.to_parquet(fpath)
    service = Service(fpath, tmp_path, max_age_days=max_age_days)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_endpoint_health_predict_and_error_codes(trained):
    feats, art = trained
    server, base = _serve(art, feats, max_age_days=100_000)  # never stale
    try:
        code, body = _get(base + "/health")
        assert code == 200 and body["status"] == "ok" and body["n_tickers"] == 15
        code, body = _get(base + "/predict?model=ridge&top=3")
        assert code == 200 and len(body["predictions"]) == 3 and body["health"]["model_id"].startswith("ridge-")
        assert _get(base + "/predict?model=bogus")[0] == 400
        assert _get(base + "/predict?model=hgb")[0] == 404          # registered nowhere -> not trained
        assert _get(base + "/nope")[0] == 404
        code, m = _get(base + "/metrics")
        assert code == 200 and m["requests"] == 6 and m["errors"] == 1 and m["latency_ms_p50"] is not None
    finally:
        server.shutdown()


def test_endpoint_returns_503_on_stale_data(trained):
    feats, art = trained
    server, base = _serve(art, feats, max_age_days=1)  # LAST is far in the past relative to real "now"
    try:
        code, body = _get(base + "/predict?model=ridge")
        assert code == 503 and body["error"] == "stale_data"
        assert _get(base + "/health")[1]["status"] == "stale"
    finally:
        server.shutdown()
