"""Supervised learning: model zoo, honest evaluation, baselines, explainability.

Guarantees enforced here
------------------------
* Every candidate is compared against a dummy baseline on the *same* split.
* Preprocessing is fitted inside cross-validation (no leakage).
* Validation performance is never described as guaranteed future performance.
* Small samples and class imbalance produce explicit warnings.
* Training failures are recorded, never fatal to the whole run.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error,
                             precision_recall_fscore_support, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, GridSearchCV, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .preprocess import (UnsupportedFeatureError, build_preprocessor,
                         feature_columns, prepare_xy)

RANDOM_STATE = 42
MIN_ROWS_CLASSIFY = 30
MIN_ROWS_PER_CLASS = 8
MIN_ROWS_REGRESSION = 30
MAX_ONEHOT_CELLS = 400_000      # guards memory/time blow-ups on wide data
TIME_BUDGET_S = 90.0            # wall-clock ceiling for the candidate sweep


class InsufficientDataError(ValueError):
    """Raised when a dataset cannot support the requested task reliably."""


@dataclass
class ModelResult:
    name: str
    ok: bool = True
    error: str = ""
    fit_s: float = 0.0
    predict_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    cv_mean: float | None = None
    cv_std: float | None = None
    best_params: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    confusion: object = None             # ndarray (n_classes, n_classes) | None
    per_class: object = None             # DataFrame | None

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("confusion", None)
        d.pop("per_class", None)
        return d


@dataclass
class SupervisedOutcome:
    task: str                        # classification | regression
    target: str
    baseline: ModelResult | None = None
    best: ModelResult | None = None
    candidates: list[ModelResult] = field(default_factory=list)
    pipeline: Pipeline | None = None
    preprocessor: object | None = None
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    drop_reasons: list[str] = field(default_factory=list)
    primary_metric: str = ""
    metric_reason: str = ""
    train_rows: int = 0
    test_rows: int = 0
    dropped_target_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    classes: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None
    temporal: bool = False

    def comparison_frame(self) -> pd.DataFrame:
        rows = []
        for c in ([self.baseline] if self.baseline else []) + self.candidates:
            if not c.ok:
                continue
            rows.append({"model": c.name,
                         **{k: v for k, v in c.metrics.items()},
                         "cv_mean": c.cv_mean, "cv_std": c.cv_std,
                         "fit_s": round(c.fit_s, 3),
                         "params": str(c.best_params) if c.best_params else ""})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------
def _classification_zoo(n_rows: int, n_features: int,
                        minority: float) -> list[tuple[str, object, dict]]:
    """(name, estimator, small param grid) candidates for classification.

    Candidates are gated by dataset size so the sweep stays inside the time
    budget and never trains a model that cannot fit meaningfully.
    """
    zoo: list[tuple[str, object, dict]] = [
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), {}),
        ("Decision Tree",
         DecisionTreeClassifier(random_state=RANDOM_STATE),
         {"model__max_depth": [3, None]}),
        ("Gaussian NB", GaussianNB(), {}),
    ]
    if minority < 0.15:
        zoo.append(("Logistic Regression (balanced)",
                    LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE), {}))
    if n_rows >= 80:
        zoo.append(("Random Forest",
                    RandomForestClassifier(n_estimators=200,
                                           random_state=RANDOM_STATE, n_jobs=-1),
                    {"model__max_depth": [None, 10]}))
    if n_rows >= 150:
        zoo.append(("Extra Trees",
                    ExtraTreesClassifier(n_estimators=200,
                                         random_state=RANDOM_STATE, n_jobs=-1),
                    {"model__max_depth": [None, 10]}))
    if n_rows >= 100 and n_features <= 200:
        zoo.append(("Gradient Boosting",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                    {"model__n_estimators": [100]}))
    if 200 <= n_rows <= 20_000 and n_features <= 100:
        zoo.append(("SVM (RBF)", SVC(probability=True,
                                     random_state=RANDOM_STATE),
                    {"model__C": [1, 10]}))
    if n_rows >= 200 and n_features <= 400:
        zoo.append(("SVM (Linear)",
                    LinearSVC(random_state=RANDOM_STATE),
                    {"model__C": [0.1, 1]}))
    if n_rows >= 120:
        zoo.append(("k-NN", KNeighborsClassifier(),
                    {"model__n_neighbors": [5, 11]}))
    return zoo


def _regression_zoo(n_rows: int, n_features: int) -> list[tuple[str, object,
                                                                dict]]:
    zoo: list[tuple[str, object, dict]] = [
        ("Linear Regression", LinearRegression(), {}),
        ("Ridge", Ridge(random_state=RANDOM_STATE),
         {"model__alpha": [0.1, 1, 10]}),
        ("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_STATE),
         {"model__max_depth": [3, None]}),
    ]
    if n_rows >= 150:
        zoo.append(("Random Forest",
                    RandomForestRegressor(n_estimators=200,
                                          random_state=RANDOM_STATE, n_jobs=-1),
                    {"model__max_depth": [None, 10]}))
        zoo.append(("Extra Trees",
                    ExtraTreesRegressor(n_estimators=200,
                                        random_state=RANDOM_STATE, n_jobs=-1),
                    {"model__max_depth": [None, 10]}))
    if n_rows >= 200 and n_features <= 200:
        zoo.append(("Gradient Boosting",
                    GradientBoostingRegressor(random_state=RANDOM_STATE),
                    {"model__n_estimators": [100]}))
    if n_rows >= 200 and n_features <= 50:
        zoo.append(("Lasso", Lasso(random_state=RANDOM_STATE, max_iter=5000),
                    {"model__alpha": [0.01, 0.1]}))
    return zoo


def infer_task(y: pd.Series, hint: str | None = None) -> tuple[str, str]:
    """Infer classification vs regression from the target, with reasoning.

    A user-supplied ``hint`` always wins, because the user knows the business
    objective the prototype cannot see.
    """
    if hint in ("classification", "regression"):
        return hint, "Task type selected by the user."
    if y.dtype == bool:
        return "classification", "Target is Boolean, so it is a binary label."
    if pd.api.types.is_numeric_dtype(y):
        n_unique = int(y.nunique(dropna=True))
        if n_unique == 2:
            return "classification", (
                f"Target is numeric with exactly {n_unique} values, which "
                "behaves as a binary label.")
        if n_unique <= 10 and set(np.unique(y.dropna())) <= set(
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
            return "classification", (
                f"Target takes {n_unique} discrete whole-number values, which "
                "looks like an ordinal class count.")
        return "regression", (
            f"Target is continuous numeric with {n_unique:,} distinct values.")
    return "classification", (
        f"Target is non-numeric ({y.dtype}) with {y.nunique()} categories.")


def _estimate_onehot_cells(profile, categorical: list[str],
                           n_rows: int) -> int:
    """Estimate the size of the one-hot matrix, to guard memory and time."""
    total_levels = 0
    for name in categorical:
        c = profile.column(name)
        if c is not None:
            total_levels += min(c.n_unique, max(n_rows, 1))
    return total_levels * max(n_rows, 1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _primary_metric(task: str, y_train: pd.Series, minority: float) -> tuple[str,
                                                                             str]:
    """Choose the headline metric and explain why it fits this dataset."""
    if task == "regression":
        return "RMSE", ("RMSE is reported because it penalises large errors in "
                        "the same units as the target; MAE and R-squared are "
                        "shown alongside.")
    if minority < 0.2:
        return "balanced_accuracy", (
            f"Classes are imbalanced (minority {minority:.1%}), so balanced "
            "accuracy is preferred: plain accuracy would reward a model that "
            "always predicts the majority class.")
    return "accuracy", ("Classes are reasonably balanced, so overall accuracy "
                        "is a fair headline metric.")


#: Our display metric names are not all valid scikit-learn scorer names.
_SCORERS = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

#: Metrics where a *lower* value is better (used when ranking candidates).
_LOWER_IS_BETTER = {"RMSE", "MAE"}


def _scorer(metric: str) -> str:
    """Translate a display metric name into a scikit-learn scorer name."""
    try:
        return _SCORERS[metric]
    except KeyError as exc:
        raise ValueError(
            f"No scikit-learn scorer is defined for metric {metric!r}."
        ) from exc


def _cv_score_to_metric(cv_scores, metric: str) -> float:
    """Convert cross_val_score output back to the display metric's scale.

    scikit-learn returns *negated* values for error metrics, so RMSE and MAE
    are flipped back to positive before being reported.
    """
    mean = float(np.mean(cv_scores))
    return round(-mean if metric in _LOWER_IS_BETTER else mean, 4)


def classification_metrics(y_true, y_pred, y_score, classes) -> dict:
    out: dict = {}
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    out["balanced_accuracy"] = round(
        float(balanced_accuracy_score(y_true, y_pred)), 4)
    average = "binary" if len(classes) == 2 else "macro"
    out["precision"] = round(float(precision_score(
        y_true, y_pred, average=average, zero_division=0)), 4)
    out["recall"] = round(float(recall_score(
        y_true, y_pred, average=average, zero_division=0)), 4)
    out["f1"] = round(float(f1_score(y_true, y_pred, average=average,
                                     zero_division=0)), 4)
    if y_score is not None:
        try:
            if len(classes) == 2:
                out["roc_auc"] = round(float(roc_auc_score(
                    y_true, np.asarray(y_score)[:, 1])), 4)
            else:
                out["roc_auc"] = round(float(roc_auc_score(
                    y_true, np.asarray(y_score), multi_class="ovr",
                    average="macro")), 4)
        except Exception:
            out["roc_auc"] = None
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _make_split(task: str, y: pd.Series, temporal: bool, profile):
    """Return (cv splitter, strategy description)."""
    if temporal and profile.datetime:
        return TimeSeriesSplit(n_splits=3), (
            f"Time-aware split on '{profile.datetime[0]}': the model is "
            "validated on data chronologically later than its training data.")
    if task == "classification":
        counts = y.value_counts()
        if counts.min() >= 2:
            return StratifiedKFold(n_splits=min(5, int(counts.min())),
                                   shuffle=True,
                                   random_state=RANDOM_STATE), (
                "Stratified 5-fold cross-validation: every fold keeps the same "
                "class proportions as the full dataset.")
        return StratifiedKFold(n_splits=2, shuffle=True,
                               random_state=RANDOM_STATE), (
            "Stratified 2-fold cross-validation, reduced because a class has "
            "very few examples.")
    return KFold(n_splits=5, shuffle=True,
                 random_state=RANDOM_STATE), (
        "Shuffled 5-fold cross-validation on the full dataset.")


# --------------------------------------------------------------------------
# per-candidate fitting
# --------------------------------------------------------------------------
def _fit_one(name: str, pipe, X_train, y_train, X_test, y_test, task: str,
             primary: str, classes: list, fitted: bool = False) -> ModelResult:
    """Fit (unless already fitted), time it, and score on the held-out set.

    For classification this also materialises the confusion matrix and
    per-class metrics, and computes ROC-AUC when probabilities are available.
    """
    res = ModelResult(name=name)
    try:
        t0 = time.perf_counter()
        if not fitted:
            pipe.fit(X_train, y_train)
        res.fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        res.predict_s = time.perf_counter() - t0

        y_score = None
        if task == "classification":
            if hasattr(pipe, "predict_proba"):
                try:
                    y_score = pipe.predict_proba(X_test)
                except Exception:
                    y_score = None
            res.metrics = classification_metrics(y_test, y_pred, y_score,
                                                 classes)
            res.confusion = confusion_matrix(y_test, y_pred, labels=classes)
            res.per_class = per_class_frame(y_test, y_pred, classes)
        else:
            res.metrics = regression_metrics(y_test, y_pred)

        res.metrics["_y_pred"] = y_pred
        res.metrics["_y_score"] = y_score
    except Exception as exc:
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def per_class_frame(y_true, y_pred, classes) -> pd.DataFrame:
    """Per-class precision/recall/F1/support as a tidy DataFrame."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0)
    return pd.DataFrame({
        "class": [str(c) for c in classes],
        "precision": np.round(p, 4),
        "recall": np.round(r, 4),
        "f1": np.round(f, 4),
        "support": np.round(s, 4),
    })


def _safe_feature_names(pipe, numeric: list[str],
                        categorical: list[str]) -> list[str]:
    try:
        return [str(n) for n in pipe.named_steps["pre"]
                .get_feature_names_out()]
    except Exception:
        return list(numeric) + list(categorical)


def _compare_to_baseline(outcome: "SupervisedOutcome") -> None:
    """Record whether the winner actually beats the naive baseline."""
    key = outcome.primary_metric
    base = outcome.baseline.metrics.get(key) if outcome.baseline else None
    best = outcome.best.metrics.get(key)
    if base is None or best is None:
        return
    if key in _LOWER_IS_BETTER:
        delta = base - best            # lower RMSE/MAE is better
    else:
        delta = best - base            # higher accuracy/F1/R2 is better
    outcome.notes.append(
        f"Best model {outcome.best.name} scored {best:g} {key} versus "
        f"{base:g} for the dummy baseline "
        f"({'better' if delta > 0 else 'no better'}, delta {delta:+.4g}).")
    if delta <= 0:
        outcome.warnings.append(
            f"The recommended model did not beat the dummy baseline on {key}. "
            "Treat the result as inconclusive: the features carry little "
            "signal for this target.")



# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def run_supervised(df: pd.DataFrame, profile, target: str,
                   task_hint: str | None = None,
                   drop: list[str] | None = None,
                   temporal: bool = False,
                   max_candidates: int = 6,
                   time_budget_s: float = TIME_BUDGET_S,
                   random_state: int = RANDOM_STATE) -> SupervisedOutcome:
    """Train, evaluate and rank candidates against a dummy baseline.

    Raises ``InsufficientDataError`` when the dataset cannot support the task,
    and ``UnsupportedFeatureError`` when no usable features remain.
    """
    X, y, dropped_rows = prepare_xy(df, target)
    task, task_reason = infer_task(y, task_hint)
    numeric, categorical, dropped, drop_reasons = feature_columns(
        profile, target, drop)
    drop_reasons.append(f"{target}: used as the prediction target")

    if not numeric and not categorical:
        raise UnsupportedFeatureError(
            "No usable feature columns remain after profiling. "
            f"Dropped: {', '.join(dropped) if dropped else 'none'}.")

    counts = y.value_counts()
    minority = float(counts.min() / counts.sum()) \
        if task == "classification" else 0.0
    n_rows = len(X)

    warnings: list[str] = [task_reason]
    if dropped_rows:
        warnings.append(
            f"{dropped_rows:,} rows were dropped because the target value was "
            "missing; feature-side missingness is imputed instead.")
    if task == "classification":
        if len(counts) < 2:
            raise InsufficientDataError(
                f"The target column has only one distinct value "
                f"({counts.index[0]!r}). Classification needs at least two "
                "classes to compare against.")
        if n_rows < MIN_ROWS_CLASSIFY or counts.min() < 2:
            raise InsufficientDataError(
                f"Only {n_rows:,} usable rows with {counts.min()} example(s) "
                f"in the smallest class. A dependable classifier needs at "
                f"least {MIN_ROWS_CLASSIFY} rows and 2 examples per class.")
        if counts.min() < MIN_ROWS_PER_CLASS:
            warnings.append(
                f"The smallest class has only {counts.min()} example(s), so "
                "cross-validation folds are tiny and the metrics are noisy.")
        if minority < 0.2:
            warnings.append(
                f"Class imbalance detected: the smallest class is "
                f"{minority:.1%} of the data. Accuracy alone is misleading; a "
                "balanced variant is included as a candidate.")
    else:
        if n_rows < MIN_ROWS_REGRESSION:
            raise InsufficientDataError(
                f"Only {n_rows:,} usable rows; a dependable regression needs "
                f"at least {MIN_ROWS_REGRESSION}.")
        if float(y.std() or 0) == 0:
            warnings.append("The target has (near) zero variance, so no model "
                            "can learn anything useful from it.")

    cells = _estimate_onehot_cells(profile, categorical, n_rows)
    if cells > MAX_ONEHOT_CELLS:
        raise UnsupportedFeatureError(
            "One-hot encoding these categorical columns would create more than "
            f"{MAX_ONEHOT_CELLS:,} values. Group rare levels or drop the "
            "high-cardinality columns and try again.")

    can_strat = task == "classification" and counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state,
        stratify=y if can_strat else None)
    if len(X_train) == 0 or len(X_test) == 0:
        raise InsufficientDataError(
            "The dataset is too small to hold out a test set.")

    cv, cv_desc = _make_split(task, y, temporal, profile)
    primary, metric_reason = _primary_metric(task, y_train, minority)
    scorer = _scorer(primary)

    zoo = (_classification_zoo(len(X_train), len(numeric) + len(categorical),
                               minority) if task == "classification"
           else _regression_zoo(len(X_train), len(numeric) + len(categorical)))
    zoo = zoo[:max_candidates]

    outcome = SupervisedOutcome(
        task=task, target=target, primary_metric=primary,
        metric_reason=metric_reason, numeric=numeric, categorical=categorical,
        dropped=dropped, drop_reasons=drop_reasons, train_rows=len(X_train),
        test_rows=len(X_test), dropped_target_rows=dropped_rows,
        warnings=warnings, notes=[cv_desc], temporal=temporal,
        classes=list(counts.index) if task == "classification" else [],
        class_counts=({str(k): int(v) for k, v in counts.items()}
                      if task == "classification" else {}))

    Dummy = DummyClassifier if task == "classification" else DummyRegressor
    outcome.baseline = _fit_one(
        "Dummy baseline",
        Pipeline([("pre", build_preprocessor(numeric, categorical)),
                  ("model", Dummy(strategy="prior" if task == "classification"
                                  else "median"))]),
        X_train, y_train, X_test, y_test, task, primary, outcome.classes)

    deadline = time.time() + time_budget_s
    for name, est, grid in zoo:
        if time.time() > deadline:
            outcome.warnings.append(
                f"Time budget of {time_budget_s:.0f}s reached; remaining "
                f"candidates were skipped (from {name}).")
            break
        pipe = Pipeline([("pre", build_preprocessor(numeric, categorical)),
                         ("model", est)])
        best_params: dict = {}
        try:
            if grid:
                gs = GridSearchCV(pipe, grid, cv=cv, scoring=scorer,
                                  n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                cv_scores = cross_val_score(gs.best_estimator_, X_train,
                                            y_train, cv=cv, scoring=scorer,
                                            n_jobs=-1)
                best_params = dict(gs.best_params_)
                res = _fit_one(name, gs.best_estimator_, X_train, y_train,
                               X_test, y_test, task, primary, outcome.classes,
                               fitted=True)
            else:
                pipe.fit(X_train, y_train)
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                            scoring=scorer, n_jobs=-1)
                res = _fit_one(name, pipe, X_train, y_train, X_test, y_test,
                               task, primary, outcome.classes, fitted=True)
        except Exception as exc:
            outcome.candidates.append(ModelResult(
                name=name, ok=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        res.cv_mean = _cv_score_to_metric(cv_scores, primary)
        res.cv_std = round(float(np.std(cv_scores)), 4)
        res.best_params = best_params
        outcome.candidates.append(res)

    ok = [c for c in outcome.candidates if c.ok]
    if not ok:
        raise RuntimeError(
            "No candidate model could be trained on this dataset. Check the "
            "data-quality report for blocking issues.")

    higher_better = primary not in _LOWER_IS_BETTER
    outcome.best = sorted(ok, key=lambda c: c.metrics.get(
        primary, float("-inf")), reverse=higher_better)[0]

    for name, est, _grid in zoo:
        if name == outcome.best.name:
            final = Pipeline([("pre", build_preprocessor(numeric,
                                                         categorical)),
                              ("model", est)])
            if outcome.best.best_params:
                final.set_params(**outcome.best.best_params)
            final.fit(X_train, y_train)
            outcome.pipeline = final
            break
    if outcome.pipeline is None:                      # pragma: no cover
        outcome.pipeline = Pipeline([
            ("pre", build_preprocessor(numeric, categorical))])
        outcome.pipeline.fit(X_train, y_train)

    outcome.preprocessor = outcome.pipeline
    outcome.feature_names = _safe_feature_names(outcome.pipeline, numeric,
                                                categorical)
    outcome.y_pred = outcome.best.metrics.pop("_y_pred", None)
    outcome.y_score = outcome.best.metrics.pop("_y_score", None)
    for c in outcome.candidates:
        c.metrics.pop("_y_pred", None)
        c.metrics.pop("_y_score", None)
    if outcome.baseline:
        outcome.baseline.metrics.pop("_y_pred", None)
        outcome.baseline.metrics.pop("_y_score", None)
    if outcome.baseline and outcome.baseline.ok:
        _compare_to_baseline(outcome)
    return outcome


def permutation_explain(pipe, X_test, y_test, task: str,
                        n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on held-out data (never on training data)."""
    scoring = ("neg_root_mean_squared_error" if task == "regression"
               else "accuracy")
    try:
        r = permutation_importance(pipe, X_test, y_test, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE,
                                   scoring=scoring, n_jobs=1)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame({"feature": list(X_test.columns),
                         "importance": r.importances_mean,
                         "std": r.importances_std}).sort_values(
        "importance", ascending=False).reset_index(drop=True)

