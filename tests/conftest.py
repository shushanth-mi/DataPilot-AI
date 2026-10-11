"""Automated tests for the Adaptive Data Intelligence Platform.

These tests exercise the real analytics engine, not the UI. Every assertion
checks computed behaviour - no test merely confirms that a file or symbol
exists.

Run with:
    python -m pytest tests -v
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adip import ingest, preprocess, profiling, supervised  # noqa: E402
from adip.discovery import run_discovery  # noqa: E402
from adip.explain import explain_supervised  # noqa: E402
from adip.nlq import ask  # noqa: E402
from adip.planner import build_plan  # noqa: E402
from adip.reliability import assess_supervised, assess_unsupervised  # noqa: E402,F401
from adip.reports import (build_pdf_report, export_table_csv,  # noqa: E402
                          export_table_excel, export_json)
from adip.sample_data import DATASETS, make_dataset  # noqa: E402
from adip.scenario import run_scenario  # noqa: E402
from adip.unsupervised import (DegenerateResultError,  # noqa: E402
                               run_unsupervised)

SEED = 42


# ==========================================================================
# fixtures
# ==========================================================================
@pytest.fixture
def classification_df() -> pd.DataFrame:
    """A labelled dataset with a genuine learnable signal."""
    rng = np.random.default_rng(SEED)
    n = 400
    age = rng.integers(18, 75, n)
    income = rng.normal(45000, 12000, n).round(0)
    tenure = rng.integers(1, 72, n)
    region = rng.choice(["North", "South", "East"], n)
    logit = (-2.0 + 0.04 * age - 0.00003 * income - 0.03 * tenure
             + (region == "North") * 0.8)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "age": age, "income": income, "tenure_months": tenure,
        "region": region, "churned": churn,
        "email": [f"user{i}@example.com" for i in range(n)],
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 350
    sqft = rng.normal(1500, 400, n).round(0)
    bedrooms = rng.integers(1, 6, n)
    age_years = rng.integers(0, 60, n)
    price = (50000 + 120 * sqft + 8000 * bedrooms - 900 * age_years
             + rng.normal(0, 15000, n)).round(0)
    return pd.DataFrame({"square_feet": sqft, "bedrooms": bedrooms,
                         "property_age_years": age_years, "price": price})


@pytest.fixture
def imbalanced_df(classification_df) -> pd.DataFrame:
    """Majority-negative: a naive accuracy score would look great but mean nothing."""
    df = classification_df.copy()
    df.loc[df.index[:330], "churned"] = 0
    df.loc[df.index[330:335], "churned"] = 1
    return df


@pytest.fixture
def clustered_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 300
    group = rng.integers(0, 3, n)
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-7.0, 6.0]])
    pts = centres[group] + rng.normal(0, 0.9, (n, 2))
    return pd.DataFrame({"feature_a": pts[:, 0].round(4),
                         "feature_b": pts[:, 1].round(4),
                         "true_group": group})


@pytest.fixture
def anomaly_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 400
    normal = rng.normal(50, 5, (n, 2))
    planted = np.full(n, False)
    planted[rng.choice(n, 15, replace=False)] = True
    normal[planted] += rng.normal(0, 1, (int(planted.sum()), 2)) * 25
    return pd.DataFrame({"sensor_x": normal[:, 0].round(3),
                         "sensor_y": normal[:, 1].round(3),
                         "planted_anomaly": planted})


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Missing values, duplicates, a constant column and a leaky identifier."""
    rng = np.random.default_rng(SEED)
    n = 200
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "with_gaps": rng.normal(0, 1, n),
        "half_empty": rng.normal(0, 1, n),
        "constant": ["same"] * n,
        "label": rng.integers(0, 2, n),
    })
    df.loc[df.index[:40], "with_gaps"] = np.nan
    df.loc[df.index[:100], "half_empty"] = np.nan
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    # a column that encodes the label exactly: guaranteed leakage
    df["leaky_id"] = [f"{v}_row{i}" for i, v in enumerate(df["label"])]
    return df


def _profile(df: pd.DataFrame) -> profiling.DataProfile:
    ds = ingest.load_frame(df, "test")
    return profiling.profile_dataset_obj(ds)
