"""Explainability: why did the model decide, and why do clusters look like this?

Everything here reuses the already-fitted objects produced by the supervised
and unsupervised modules. Two rules are enforced:

1. Permutation importance is always computed on **held-out** data. Explaining a
   model on the data it was fitted to describes memorisation, not signal.
2. Feature importance is explicitly labelled as *predictive association*, not
   causation. A model that scores well using ``region`` does not show that
   changing ``region`` changes the outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .supervised import permutation_explain


def native_importance(pipeline, feature_names: list[str],
                      task: str) -> tuple[pd.DataFrame, str]:
    """Model-internal importance (coefficients or impurity-based).

    Returns (frame, caveat). Linear models give signed coefficients; tree
    ensembles give impurity-based importance, which is biased toward
    high-cardinality features and must be labelled as such.
    """
    model = pipeline.named_steps.get("model") if pipeline else None
    if model is None or not feature_names:
        return pd.DataFrame(), ""
    rows: list[dict] = []
    caveat = ""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        mag = np.abs(coef).mean(axis=0) if coef.ndim > 1 else coef
        n = min(len(feature_names), mag.shape[0])
        rows = [{"feature": feature_names[i], "value": float(mag[i]),
                 "direction": ("increases outcome" if float(mag[i]) > 0
                               else "decreases outcome")}
                for i in range(n)]
        caveat = ("Coefficients are on the standardised scale, so they are "
                  "comparable to each other but not raw business units.")
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        n = min(len(feature_names), imp.shape[0])
        rows = [{"feature": feature_names[i], "value": float(imp[i]),
                 "direction": ""} for i in range(n)]
        caveat = ("Impurity-based importance favours features with many "
                  "levels and can hide the effect of correlated inputs.")
    if not rows:
        return pd.DataFrame(), ""
    df = pd.DataFrame(rows)
    df["abs"] = df["value"].abs()
    return (df.sort_values("abs", ascending=False).drop(columns="abs")
              .reset_index(drop=True), caveat)


def confusion(y_true, y_pred, classes: list) -> tuple[pd.DataFrame, str]:
    """Confusion matrix on held-out data. Rows = actual, columns = predicted."""
    if not len(classes):
        return pd.DataFrame(), ""
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=[f"actual {c}" for c in labels],
                      columns=[f"predicted {c}" for c in labels])
    note = ("Rows are the true class, columns are the predicted class; the "
            "off-diagonal cells are exactly where the model confuses classes.")
    return df, note


def residual_analysis(y_true, y_pred) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row residuals plus a summary of where the model misses most."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    res = yt - yp
    detail = pd.DataFrame({"actual": yt, "predicted": yp, "residual": res,
                           "abs_error": np.abs(res)})
    summary = pd.DataFrame([
        {"stat": "rows", "value": f"{len(res):,}"},
        {"stat": "mean residual", "value": f"{res.mean():.4f}"},
        {"stat": "residual std", "value": f"{res.std():.4f}"},
        {"stat": "largest over-prediction", "value": f"{res.min():.4f}"},
        {"stat": "largest under-prediction", "value": f"{res.max():.4f}"},
    ])
    return detail, summary

_IMPORTANCE_CAVEAT = ("Importance measures predictive association only. "
                      "Nothing here shows that changing an input would "
                      "change the outcome.")


def explain_supervised(outcome, X_test: pd.DataFrame,
                       n_repeats: int = 5) -> dict:
    """Bundle every supervised explanation available for this outcome."""
    out: dict = {"native": pd.DataFrame(), "native_caveat": "",
                 "permuted": pd.DataFrame(), "cm": pd.DataFrame(),
                 "cm_note": "", "residuals": pd.DataFrame(),
                 "residual_summary": pd.DataFrame(), "caveats": []}
    if outcome.pipeline is None or X_test is None or not len(X_test):
        out["caveats"].append(
            "No held-out evaluation data is available, so importance cannot "
            "be computed honestly.")
        return out
    native, caveat = native_importance(outcome.pipeline,
                                       outcome.feature_names, outcome.task)
    out["native"], out["native_caveat"] = native, caveat
    if caveat:
        out["caveats"].append(caveat)
    out["permuted"] = permutation_explain(outcome.pipeline, X_test,
                                          outcome.y_test, outcome.task,
                                          n_repeats=n_repeats)
    if outcome.task == "classification":
        cm, note = confusion(outcome.y_test, outcome.y_pred, outcome.classes)
        out["cm"], out["cm_note"] = cm, note
    else:
        res, summary = residual_analysis(outcome.y_test, outcome.y_pred)
        out["residuals"], out["residual_summary"] = res, summary
    out["caveats"].append(_IMPORTANCE_CAVEAT)
    return out


def explain_unsupervised(outcome) -> dict:
    """Explain clusters and anomalies using the objects already computed."""
    out: dict = {"profiles": {}, "anomaly_features": {},
                 "caveats": [(
                     "Cluster labels are discovered groupings, not classes. "
                     "Without ground truth there is no way to say a cluster "
                     "is 'correct'; these profiles describe what separates "
                     "the groups the algorithm found.")]}
    best = outcome.best_cluster()
    if best is not None:
        out["profiles"][best.algorithm] = outcome.cluster_profiles.get(
            best.algorithm, pd.DataFrame())
        if best.n_clusters > 1:
            out["caveats"].append(
                f"{best.algorithm} found {best.n_clusters} groups"
                + (f" (silhouette {best.silhouette:.3f}). "
                   if best.silhouette is not None else ". ")
                + "A positive silhouette means the groups are separable, not "
                  "that they are meaningful to your business.")
    for a in outcome.anomaly:
        if a.top_features:
            out["anomaly_features"][a.method] = pd.DataFrame(
                sorted(a.top_features.items(), key=lambda kv: -abs(kv[1])),
                columns=["feature", "mean_z_difference"])
    return out

