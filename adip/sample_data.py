"""Reproducible synthetic demonstration datasets.

Every dataset is generated from a fixed seed and its planted patterns are
documented in ``DATASETS`` and echoed in each dataset's ``notes``. They are
synthetic on purpose: the demo must show real metrics and real structure
without misrepresenting any private data as real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


def _classification(n: int = 600) -> tuple[pd.DataFrame, str]:
    """Binary churn with 3 real signals + 1 noisy column + 1 identifier."""
    rng = np.random.default_rng(SEED)
    age = rng.integers(18, 70, n)
    usage = rng.normal(0, 1, n)
    support = rng.integers(0, 6, n)
    noise = rng.normal(0, 1, n)
    logit = 1.6 * usage - 0.04 * age + 0.9 * support - 0.5
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": age,
        "usage_score": usage.round(3),
        "support_calls": support,
        "noise_col": noise.round(3),
        "churned": churn,
    })
    notes = ("Binary churn planted from usage_score, age and support_calls "
             "(logistic signal). 'noise_col' is pure noise and must stay "
             "unimportant. 'customer_id' is an identifier and must be dropped.")
    return df, notes


def _regression(n: int = 600) -> tuple[pd.DataFrame, str]:
    """Numeric revenue driven by ad_spend and units, plus a region effect."""
    rng = np.random.default_rng(SEED)
    ad_spend = rng.uniform(0, 5000, n).round(2)
    units = (ad_spend / 40 + rng.normal(0, 12, n)).clip(0).round(1)
    region = rng.choice(["North", "South", "East"], n, p=[0.4, 0.35, 0.25])
    region_effect = np.select(
        [region == "North", region == "South", region == "East"],
        [200, -100, 400])
    revenue = (500 + 12 * units + 0.8 * ad_spend + region_effect
               + rng.normal(0, 150, n)).round(2)
    df = pd.DataFrame({
        "region": region,
        "ad_spend": ad_spend,
        "units": units,
        "noise_col": rng.normal(0, 1, n).round(3),
        "revenue": revenue,
    })
    notes = ("Revenue planted as a linear function of units and ad_spend, "
             "with an additive region effect (East +400, South -100). "
             "'noise_col' carries no signal.")
    return df, notes


def _clustering(n: int = 500) -> tuple[pd.DataFrame, str]:
    """Three well-separated Gaussian blobs plus a noise dimension."""
    rng = np.random.default_rng(SEED)
    centres = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    which = rng.integers(0, 3, n)
    pts = centres[which] + rng.normal(0, 0.8, (n, 2))
    df = pd.DataFrame({
        "feature_1": pts[:, 0].round(3),
        "feature_2": pts[:, 1].round(3),
        "noise_dim": rng.normal(0, 1, n).round(3),
    })
    notes = ("Three Gaussian blobs planted at (0,0), (10,0) and (0,10) in "
             "feature_1/feature_2. A good clustering recovers exactly 3 "
             "groups. 'noise_dim' is irrelevant to the structure.")
    return df, notes


def _anomaly(n: int = 500) -> tuple[pd.DataFrame, str]:
    """Normal cloud with a planted tail of outliers."""
    rng = np.random.default_rng(SEED)
    main = rng.normal(0, 1, (n - 25, 2))
    tail = np.array([[6.5, 6.5]] * 10 + [[-6, 6]] * 8 + [[7, -6]] * 7)
    pts = np.vstack([main, tail])
    rng.shuffle(pts)
    df = pd.DataFrame({
        "sensor_a": pts[:, 0].round(3),
        "sensor_b": pts[:, 1].round(3),
    })
    notes = ("500 rows; 25 outliers were planted far from the central Gaussian "
             "cloud at extreme coordinates. An anomaly method set to ~5% "
             "contamination should surface roughly these rows.")
    return df, notes


def _business(n: int = 800) -> tuple[pd.DataFrame, str]:
    """Interpretable SaaS-style table with planted segment and tier effects."""
    rng = np.random.default_rng(SEED)
    plan = rng.choice(["free", "pro", "enterprise"], n, p=[.5, .35, .15])
    region = rng.choice(["EMEA", "AMER", "APAC"], n, p=[.4, .45, .15])
    logins = np.clip(rng.gamma(4, 3, n), 0, 60).round(0)
    base = np.select([plan == "free", plan == "pro", plan == "enterprise"],
                     [10, 60, 200])
    spend = (base * (1 + 0.05 * logins) + rng.normal(0, 8, n)).clip(1).round(2)
    churn = (rng.random(n)
             < np.select([plan == "free", plan == "pro", plan == "enterprise"],
                         [0.35, 0.12, 0.05])).astype(int)
    df = pd.DataFrame({
        "account_id": [f"A{i:04d}" for i in range(n)],
        "plan": plan,
        "region": region,
        "monthly_logins": logins.astype(int),
        "monthly_spend": spend,
        "churned": churn,
    })
    notes = ("Churn rate is planted to differ strongly by plan "
             "(free 35%, pro 12%, enterprise 5%), and monthly_spend is driven "
             "by plan tier and logins. 'account_id' is an identifier.")
    return df, notes



def _timeseries(n: int = 730) -> tuple[pd.DataFrame, str]:
    """Daily series with trend, weekly seasonality and a level shift.

    Included so the profiler and trend discovery have realistic temporal
    structure. Forecasting itself is out of scope for this prototype; the
    platform must say so rather than pretend.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    weekly = 20 * np.sin(2 * np.pi * t / 7)
    level_shift = np.where(t >= n * 0.6, 60, 0)      # step change at 60%
    trend = 0.15 * t
    demand = (100 + trend + weekly + level_shift
              + rng.normal(0, 12, n)).round(1)
    df = pd.DataFrame({"date": dates, "demand": demand,
                       "promo": (rng.random(n) < 0.2).astype(int)})
    notes = ("Demand planted with a linear trend (+0.15/day), weekly "
             "seasonality and a +60 level shift at 60% of the window. "
             "Time-series *forecasting* is not implemented in this prototype; "
             "the platform reports trend and shift instead of inventing a "
             "forecast.")
    return df, notes


DATASETS = {
    "classification": {
        "title": "Customer churn (classification)",
        "file": "classification_churn.csv",
        "doc": "Predict the binary 'churned' label; noise and id must be ignored.",
        "builder": _classification,
    },
    "regression": {
        "title": "Revenue drivers (regression)",
        "file": "regression_revenue.csv",
        "doc": "Estimate numeric 'revenue' from spend, units and region.",
        "builder": _regression,
    },
    "clustering": {
        "title": "Cluster structure (unsupervised)",
        "file": "clustering_blobs.csv",
        "doc": "Unlabelled; three separable groups to discover.",
        "builder": _clustering,
    },
    "anomaly": {
        "title": "Sensor anomalies (anomaly detection)",
        "file": "anomaly_sensors.csv",
        "doc": "Unlabelled; 25 planted outliers to flag.",
        "builder": _anomaly,
    },
    "business": {
        "title": "SaaS accounts (business)",
        "file": "business_saas.csv",
        "doc": "Interpretable churn/spend patterns across plans and regions.",
        "builder": _business,
    },
    "timeseries": {
        "title": "Daily demand (time structure)",
        "file": "timeseries_demand.csv",
        "doc": ("Trend, weekly seasonality and a planted level shift. "
                "Forecasting is deliberately out of scope."),
        "builder": _timeseries,
    },
}


def make_dataset(key: str) -> tuple[pd.DataFrame, str]:
    """Build one synthetic dataset by key. Returns (frame, notes)."""
    return DATASETS[key]["builder"]()


def make_all() -> dict[str, tuple[pd.DataFrame, str]]:
    """Build every demo dataset."""
    return {k: DATASETS[k]["builder"]() for k in DATASETS}


def write_all(directory) -> dict[str, str]:
    """Write every dataset to ``directory`` as CSV; returns key -> path."""
    from pathlib import Path
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for key, (df, _notes) in make_all().items():
        target = out_dir / DATASETS[key]["file"]
        df.to_csv(target, index=False)
        written[key] = str(target)
    return written
