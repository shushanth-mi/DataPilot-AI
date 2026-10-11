"""Leakage-safe preprocessing built on scikit-learn Pipelines.

The preprocessor is fitted *inside* cross-validation by the callers, so the
statistics (imputation medians, scaling moments, one-hot categories) are never
learned from validation or test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .profiling import ColumnKind, DataProfile

DROP_KINDS = {ColumnKind.ID_LIKE, ColumnKind.TEXT}
# a categorical level is "rare" below this share of training rows
RARE_MIN_FREQ = 0.01
# one-hot columns above this ceiling fall back to frequency grouping
MAX_ONEHOT_LEVELS = 60


class UnsupportedFeatureError(ValueError):
    """Raised when a requested feature cannot be encoded safely."""


def feature_columns(profile: DataProfile, target: str,
                    drop: list[str] | None = None) -> tuple[list[str],
                                                             list[str],
                                                             list[str]]:
    """Split usable features into numeric / categorical / dropped.

    ``dropped`` carries the reason-free column names plus, for messaging, a
    parallel list of reasons is returned via ``drop_reasons``.
    """
    drop = set(drop or []) | {target}
    numeric, categorical, dropped, reasons = [], [], [], []
    for c in profile.columns:
        if c.name == target:
            continue
        if c.name in drop:
            dropped.append(c.name)
            reasons.append(f"{c.name}: excluded by user")
            continue
        if c.sensitive:
            dropped.append(c.name)
            reasons.append(f"{c.name}: sensitive column, never used as a feature")
            continue
        if c.kind in DROP_KINDS:
            dropped.append(c.name)
            reasons.append(f"{c.name}: {c.kind} column, not a model feature")
            continue
        if c.n_unique <= 1:
            dropped.append(c.name)
            reasons.append(f"{c.name}: constant column carries no information")
            continue
        if c.kind == ColumnKind.NUMERIC:
            numeric.append(c.name)
        elif c.kind == ColumnKind.DATETIME:
            dropped.append(c.name)
            reasons.append(f"{c.name}: datetime column, use the time-series "
                           "workflow instead")
        else:
            categorical.append(c.name)
    return numeric, categorical, dropped, reasons


def build_preprocessor(numeric: list[str], categorical: list[str],
                       *, scale: bool = True,
                       rare_min_freq: float = RARE_MIN_FREQ) -> ColumnTransformer:
    """Build a ColumnTransformer: impute -> (scale | one-hot)."""
    blocks = []
    if numeric:
        steps: list[tuple[str, object]] = [
            ("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        blocks.append(("num", Pipeline(steps), numeric))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     sparse_output=False,
                                     min_frequency=max(rare_min_freq, 1e-6))),
        ]), categorical))
    if not blocks:
        raise UnsupportedFeatureError(
            "No usable feature columns were found after profiling.")
    return ColumnTransformer(blocks, remainder="drop",
                             verbose_feature_names_out=False)


def feature_names(pre: ColumnTransformer) -> list[str]:
    """Resolve human-readable output feature names from a fitted transformer."""
    try:
        return [str(n) for n in pre.get_feature_names_out()]
    except Exception:                                        # pragma: no cover
        return []


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame,
                                                       pd.Series, int]:
    """Split into X/y, dropping rows whose *target* is missing.

    Feature-side missingness is handled by the pipeline, never by dropping.
    Returns (X, y, n_dropped_rows).
    """
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} is not in the dataset.")
    work = df.copy()
    before = len(work)
    work = work[work[target].notna()]
    dropped = before - len(work)
    y = work[target]
    X = work.drop(columns=[target])
    return X, y, dropped
