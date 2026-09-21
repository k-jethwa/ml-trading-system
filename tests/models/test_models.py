"""Known-answer tests for preprocessing, walk-forward folds, metrics and the registry."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from mltrading.config import ModelConfig
from mltrading.features.build import FEATURE_COLUMNS
from mltrading.models.evaluate import decile_spread, per_date_ic, summarize
from mltrading.models.preprocess import rank_features
from mltrading.models.registry import (
    feature_version,
    load_latest,
    load_model,
    save_model,
)
from mltrading.models.walk_forward import make_folds
from mltrading.models.zoo import build_model


def _dates(start="2016-01-04", n=1500):
    return pd.Series(pd.bdate_range(start, periods=n))


# ---------- preprocessing ----------

def test_rank_features_is_centered_and_same_date_only():
    df = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 4,
        "ret_1d": [1.0, 2.0, 3.0, 4.0, 400.0, 300.0, 200.0, 100.0],
    })
    r = rank_features(df, ["ret_1d"])["ret_1d"]
    assert r.tolist() == [-0.25, 0.0, 0.25, 0.5, 0.5, 0.25, 0.0, -0.25]


def test_rank_features_ignores_other_dates():
    """Changing d2 must not move any d1 rank (no cross-time information)."""
    a = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3, "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    b = a.copy()
    b.loc[b.date == "d2", "x"] = [-9.0, 99.0, 0.0]
    assert rank_features(a, ["x"]).loc[:2, "x"].tolist() == rank_features(b, ["x"]).loc[:2, "x"].tolist()


def test_rank_features_keeps_nan():
    df = pd.DataFrame({"date": ["d"] * 3, "x": [1.0, np.nan, 3.0]})
    assert rank_features(df, ["x"])["x"].isna().tolist() == [False, True, False]


# ---------- walk-forward ----------

def test_folds_are_chronological_and_purged():
    dates = _dates()
    folds = make_folds(dates, first_test_year=2019, embargo_days=5)
    unique = pd.DatetimeIndex(np.sort(dates.unique()))
    assert folds[0].test_year == 2019
    for f in folds:
        assert f.train_end < f.test_start <= f.test_end
        # exactly `embargo_days` trading dates sit strictly between train_end and test_start
        between = unique[(unique > f.train_end) & (unique < f.test_start)]
        assert len(between) == 5


def test_no_training_label_window_reaches_test_period():
    """A training row at date t has a label ending at t+5 trading days; that end must precede test_start."""
    dates = _dates()
    unique = pd.DatetimeIndex(np.sort(dates.unique()))
    for f in make_folds(dates, 2019, 5):
        last_train_idx = unique.get_loc(f.train_end)
        assert unique[last_train_idx + 5] < f.test_start


def test_folds_are_expanding_and_test_windows_disjoint():
    folds = make_folds(_dates(), 2019, 5)
    assert all(a.train_end < b.train_end for a, b in pairwise(folds))
    assert all(a.test_end < b.test_start for a, b in pairwise(folds))


def test_embargo_needing_more_history_than_exists_raises():
    with pytest.raises(ValueError):
        make_folds(pd.Series(pd.bdate_range("2019-01-01", periods=300)), 2019, 5)


# ---------- metrics ----------

def _panel(n_dates=30, n=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=n_dates):
        target = rng.normal(size=n)
        rows.append(pd.DataFrame({"date": d, "ticker": range(n), "target": target}))
    return pd.concat(rows, ignore_index=True)


def test_perfect_predictor_has_ic_one_and_reversed_has_minus_one():
    df = _panel()
    df["pred"] = df["target"]
    assert per_date_ic(df, "spearman").round(9).eq(1.0).all()
    assert per_date_ic(df, "pearson").round(9).eq(1.0).all()
    df["pred"] = -df["target"]
    assert per_date_ic(df, "spearman").round(9).eq(-1.0).all()


def test_constant_prediction_has_no_defined_ic():
    df = _panel()
    df["pred"] = 0.3
    assert per_date_ic(df).empty


def test_decile_spread_known_answer():
    df = pd.DataFrame({"date": "2020-01-01", "pred": np.arange(20.0), "target": np.arange(20.0)})
    # k = round(0.1*20) = 2: top two targets {18,19} mean 18.5, bottom {0,1} mean 0.5
    assert decile_spread(df).iloc[0] == pytest.approx(18.0)


def test_summarize_oos_r2_matches_hand_computation():
    df = pd.DataFrame({"date": ["2020-01-01"] * 4, "pred": [1.0, 2.0, 3.0, 4.0], "target": [1.0, 2.0, 3.0, 6.0]})
    s = summarize(df, mean_mse=2.0)
    assert s["mse"] == pytest.approx(1.0)    # (0+0+0+4)/4
    assert s["mae"] == pytest.approx(0.5)
    assert s["oos_r2"] == pytest.approx(0.5)  # 1 - 1/2


# ---------- model zoo / registry ----------

def _xy(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = 0.05 * X["sector_rel_ret_5d"] + rng.normal(scale=0.01, size=n)
    return X, y


@pytest.mark.parametrize("name", ["mean", "reversal_5d", "ols", "ridge", "hgb"])
def test_every_model_fits_and_predicts(name):
    X, y = _xy()
    m = build_model(name, ModelConfig(hgb_min_samples_leaf=20, hgb_max_iter=20)).fit(X, y)
    assert m.predict(X).shape == (len(X),)


def test_reversal_heuristic_is_minus_short_term_rank():
    X, y = _xy(50)
    pred = build_model("reversal_5d").fit(X, y).predict(X)
    assert np.allclose(pred, -X["sector_rel_ret_5d"].to_numpy())


def test_ridge_scaler_is_fit_on_train_only():
    """The scaler's mean must equal the TRAIN mean, not the mean of train+test."""
    X, y = _xy(200)
    model = build_model("ridge").fit(X.iloc[:100], y.iloc[:100])
    scaler = model.steps[0][1]
    assert np.allclose(scaler.mean_, X.iloc[:100].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X.mean().to_numpy())


def test_save_and_load_roundtrip(tmp_path):
    X, y = _xy()
    model = build_model("ridge").fit(X, y)
    saved = save_model(model, "ridge", pd.Series(pd.to_datetime(["2020-01-01", "2020-06-30"])), len(X),
                       "abc123", "fingerprint", {"alpha": 1000.0}, artifact_dir=tmp_path)
    assert saved.model_id == "ridge-2020-06-30-abc123"
    latest = load_latest("ridge", tmp_path)
    assert latest.metadata["feature_version"] == feature_version()
    assert latest.metadata["feature_columns"] == FEATURE_COLUMNS
    assert np.allclose(load_model(latest).predict(X), model.predict(X))


def test_load_latest_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest("nope", tmp_path)
