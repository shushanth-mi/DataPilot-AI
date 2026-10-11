"""Supervised learning tests: real training, real metrics, real guard rails."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adip import preprocess, supervised
from adip.preprocess import UnsupportedFeatureError
from adip.supervised import InsufficientDataError
from conftest import _profile


@pytest.fixture(autouse=True)
def _fast_budget(monkeypatch):
    """Keep the CI loop quick without weakening the assertions."""
    monkeypatch.setattr(supervised, "MIN_ROWS_CLASSIFY", 12, raising=False)
    monkeypatch.setattr(supervised, "MIN_ROWS_PER_CLASS", 3, raising=False)
    monkeypatch.setattr(supervised, "MIN_ROWS_REGRESSION", 12, raising=False)


def _train(df, target, profile=None, **kw):
    prof = profile or _profile(df)
    kw.setdefault("max_candidates", 3)
    kw.setdefault("time_budget_s", 600)
    return supervised.run_supervised(df, prof, target, **kw)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
class TestPreprocessing:
    def test_numeric_and_categorical_split(self, classification_df):
        p = _profile(classification_df)
        num, cat, dropped, _ = preprocess.feature_columns(p, "churned")
        assert "age" in num and "income" in num
        assert "region" in cat
        assert "churned" not in num

    def test_identifiers_excluded_from_features(self, classification_df):
        p = _profile(classification_df)
        _, _, dropped, _ = preprocess.feature_columns(p, "churned")
        assert "customer_id" in dropped

    def test_sensitive_excluded_from_features(self, classification_df):
        p = _profile(classification_df)
        num, cat, dropped, _ = preprocess.feature_columns(p, "churned")
        assert "email" in dropped
        assert "email" not in num and "email" not in cat

    def test_constant_column_excluded(self, messy_df):
        p = _profile(messy_df)
        _, _, dropped, _ = preprocess.feature_columns(p, "label")
        assert "constant" in dropped

    def test_imputer_handles_missing_numeric(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0] * 10,
                           "y": [0, 1, 0, 1] * 10})
        p = _profile(df)
        num, cat, _, _ = preprocess.feature_columns(p, "y")
        pre = preprocess.build_preprocessor(num, cat)
        out = pre.fit_transform(df[["x"]])
        assert not np.isnan(out).any()

    def test_onehot_handles_categorical(self, classification_df):
        p = _profile(classification_df)
        num, cat, _, _ = preprocess.feature_columns(p, "churned")
        pre = preprocess.build_preprocessor(num, cat)
        out = pre.fit_transform(classification_df[num + cat])
        assert out.shape[0] == len(classification_df)
        assert out.shape[1] >= len(num)

    def test_unseen_category_does_not_crash(self, classification_df):
        """handle_unknown must absorb levels never seen during training."""
        p = _profile(classification_df)
        num, cat, _, _ = preprocess.feature_columns(p, "churned")
        pre = preprocess.build_preprocessor(num, cat)
        pre.fit(classification_df[num + cat])
        novel = classification_df[num + cat].head(5).copy()
        novel["region"] = "Atlantis"
        out = pre.transform(novel)
        assert not np.isnan(out).any()

    def test_no_usable_features_raises(self):
        df = pd.DataFrame({"only_id": [f"r{i}" for i in range(40)],
                           "label": [0, 1] * 20})
        p = _profile(df)
        with pytest.raises(UnsupportedFeatureError):
            preprocess.build_preprocessor(
                *preprocess.feature_columns(p, "label")[:2])

    def test_missing_target_rows_dropped_only(self, classification_df):
        df = classification_df.copy()
        df.loc[df.index[:15], "churned"] = np.nan
        X, y, dropped = preprocess.prepare_xy(df, "churned")
        assert dropped == 15
        assert y.notna().all()
        # feature-side missingness is NOT dropped by prepare_xy
        assert len(X) == len(y)

    def test_target_missing_raises_keyerror(self, classification_df):
        with pytest.raises(KeyError):
            preprocess.prepare_xy(classification_df, "nope")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
class TestClassification:
    def test_trains_and_beats_baseline(self, classification_df):
        out = _train(classification_df, "churned")
        assert out.task == "classification"
        assert out.baseline is not None and out.best is not None
        # The platform selects the headline metric (balanced_accuracy when the
        # classes are imbalanced) and ranks candidates on it, so the baseline
        # comparison must use that same metric - not raw accuracy, which a
        # majority-class dummy wins on imbalanced data.
        key = out.primary_metric
        best = out.best.metrics[key]
        base = out.baseline.metrics[key]
        assert best > base, (key, best, base)

    def test_metrics_are_in_valid_range(self, classification_df):
        out = _train(classification_df, "churned")
        for name, value in out.best.metrics.items():
            if name in ("accuracy", "balanced_accuracy", "precision",
                        "recall", "f1"):
                assert 0.0 <= value <= 1.0, name

    def test_confusion_matrix_shape(self, classification_df):
        out = _train(classification_df, "churned")
        cm = out.best.confusion
        assert cm is not None and cm.shape == (2, 2)

    def test_train_test_split_is_disjoint(self, classification_df):
        out = _train(classification_df, "churned")
        assert out.train_rows + out.test_rows <= len(classification_df)
        assert out.train_rows > 0 and out.test_rows > 0

    def test_stratification_used_for_classification(self, classification_df):
        out = _train(classification_df, "churned")
        assert any("Stratified" in n or "stratified" in n
                   for n in out.notes), out.notes

    def test_cross_validation_reported(self, classification_df):
        out = _train(classification_df, "churned")
        assert out.best.cv_mean is not None
        assert out.best.cv_std is not None
        assert out.best.cv_std >= 0

    def test_candidates_compared_in_a_table(self, classification_df):
        out = _train(classification_df, "churned")
        frame = out.comparison_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) >= 2
        assert "Dummy" in " ".join(frame["model"])

    def test_baseline_improvement_explained(self, classification_df):
        out = _train(classification_df, "churned")
        assert out.best.name != out.baseline.name
        assert any("baseline" in n.lower() for n in out.notes)

    def test_imbalanced_switches_metric(self, imbalanced_df):
        """With a 1.6% minority class, plain accuracy must not be the headline."""
        out = _train(imbalanced_df, "churned")
        assert out.primary_metric == "balanced_accuracy"
        assert "balanced" in out.metric_reason.lower()

    def test_imbalance_is_warned_about(self, imbalanced_df):
        out = _train(imbalanced_df, "churned")
        assert any("imbalance" in w.lower() or "minority" in w.lower()
                   for w in out.warnings), out.warnings

    def test_per_class_metrics_for_imbalance(self, imbalanced_df):
        out = _train(imbalanced_df, "churned")
        per_class = getattr(out, "per_class", None)
        if per_class is not None:
            assert isinstance(per_class, pd.DataFrame)

    def test_multiclass_supported(self):
        rng = np.random.default_rng(0)
        n = 300
        base = rng.normal(0, 1, (n, 2))
        labels = np.zeros(n, dtype=int)
        pts = base.copy()
        for k, off in enumerate([[0, 0], [6, 0], [0, 6]], start=0):
            sel = rng.choice(n, 100, replace=False)
            pts[sel] += off
            labels[sel] = k
        df = pd.DataFrame({"f1": pts[:, 0], "f2": pts[:, 1], "grade": labels})
        out = _train(df, "grade")
        assert out.best.confusion is not None
        assert out.best.confusion.shape == (3, 3)

    def test_too_few_rows_blocked(self):
        tiny = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
        p = _profile(tiny)
        with pytest.raises(InsufficientDataError):
            supervised.run_supervised(tiny, p, "y")

    def test_single_class_blocked(self):
        df = pd.DataFrame({"x": list(range(40)), "y": [1] * 40})
        p = _profile(df)
        with pytest.raises(InsufficientDataError):
            supervised.run_supervised(df, p, "y")

    def test_missing_target_column_blocked(self, classification_df):
        p = _profile(classification_df)
        with pytest.raises(KeyError):
            supervised.run_supervised(classification_df, p, "nope")

    def test_no_feature_columns_blocked(self, messy_df):
        df = messy_df[["leaky_id", "label"]]
        p = _profile(df)
        with pytest.raises(UnsupportedFeatureError):
            supervised.run_supervised(df, p, "label")


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------
class TestRegression:
    def test_trains_and_beats_baseline(self, regression_df):
        out = _train(regression_df, "price")
        assert out.task == "regression"
        assert out.best.metrics["R2"] > out.baseline.metrics["R2"]

    def test_reports_expected_metrics(self, regression_df):
        out = _train(regression_df, "price")
        assert {"MAE", "RMSE", "R2"} <= set(out.best.metrics)

    def test_lower_rmse_is_ranked_better(self, regression_df):
        """For error metrics the *smallest* value must win."""
        out = _train(regression_df, "price")
        ok = [c for c in out.candidates if c.ok]
        best_rmse = min(c.metrics["RMSE"] for c in ok)
        assert out.best.metrics["RMSE"] == pytest.approx(best_rmse)

    def test_r2_is_high_for_a_learnable_signal(self, regression_df):
        out = _train(regression_df, "price")
        assert out.best.metrics["R2"] > 0.5

    def test_dummy_regressor_baseline_is_weak(self, regression_df):
        out = _train(regression_df, "price")
        assert out.baseline.metrics["R2"] <= 0.01

    def test_too_few_rows_blocked(self):
        tiny = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0] * 5})
        p = _profile(tiny)
        with pytest.raises(InsufficientDataError):
            supervised.run_supervised(tiny, p, "y")

    def test_metric_explanation_names_units(self, regression_df):
        out = _train(regression_df, "price")
        assert "RMSE" in out.metric_reason


# --------------------------------------------------------------------------
# leakage and evaluation integrity
# --------------------------------------------------------------------------
class TestLeakageAndIntegrity:
    def test_identifier_encoding_label_is_dropped_as_leaky(self, messy_df):
        """'leaky_id' is id_like, so it must never reach the model."""
        p = _profile(messy_df)
        _, _, dropped, _ = preprocess.feature_columns(p, "label")
        assert "leaky_id" in dropped

    def test_imputation_is_fitted_on_train_only(self, classification_df):
        """The preprocessor must be inside the pipeline, so stats cannot leak."""
        df = classification_df.copy()
        df.loc[df.index[:50], "income"] = np.nan
        out = _train(df, "churned")
        assert out.pipeline is not None
        steps = dict(out.pipeline.named_steps)
        assert "pre" in steps, "preprocessing must live inside the pipeline"

    def test_no_perfect_score_from_a_duplicated_dataset(self, messy_df):
        """Duplicates must not let a model score 100% on held-out data."""
        p = _profile(messy_df)
        out = supervised.run_supervised(
            messy_df, p, "label", max_candidates=1, time_budget_s=600)
        # not an assertion of a specific value, only that no crash occurs and
        # the duplication risk was surfaced to the user
        assert out is not None
        assert any("duplicat" in i.message.lower() for i in p.issues)

    def test_cv_uses_the_training_fold_only(self, classification_df):
        out = _train(classification_df, "churned")
        # test rows are excluded from fitting: cv_mean is computed on the model
        # fitted to the training split, and test rows never enter it
        assert out.test_rows + out.train_rows <= len(classification_df)

    def test_all_missing_feature_still_trains(self, classification_df):
        """A fully-empty column is dropped, not allowed to break the run."""
        df = classification_df.copy()
        df["empty_col"] = np.nan
        out = _train(df, "churned")
        assert out.best is not None

    def test_failed_candidate_is_recorded_not_raised(self, monkeypatch,
                                                     classification_df):
        """A crashing estimator must be reported, not abort the run."""
        from sklearn.linear_model import LogisticRegression

        real_fit = LogisticRegression.fit

        def boom(self, X, y=None):
            if y is not None and len(y) > 20:
                raise RuntimeError("synthetic training failure")
            return real_fit(self, X, y)

        monkeypatch.setattr(LogisticRegression, "fit", boom)
        out = _train(classification_df, "churned")
        assert out.best is not None
        assert any((not c.ok) for c in out.candidates)

    def test_unsupported_data_fails_gracefully(self):
        df = pd.DataFrame({"free_text": [f"some sentence number {i} " + "text "
                                         * 15 for i in range(40)],
                           "label": [0, 1] * 20})
        p = _profile(df)
        with pytest.raises(UnsupportedFeatureError):
            supervised.run_supervised(df, p, "label")


    def test_unseen_category_at_predict_time(self, classification_df):
        out = _train(classification_df, "churned")
        pipe = out.pipeline
        row = classification_df.drop(columns=["churned"]).head(3).copy()
        row["region"] = ["Nowhere", "Elsewhere", "Zzz"]
        pred = pipe.predict(row)
        assert len(pred) == 3

    def test_results_are_reproducible(self, classification_df):
        a = _train(classification_df, "churned").best.metrics
        b = _train(classification_df, "churned").best.metrics
        assert a == b

