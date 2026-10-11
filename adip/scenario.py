"""What-if scenario exploration for fitted supervised models.

Honesty rules
-------------
* A prediction is compared against a *baseline* prediction so the user sees the
  change the model attributes to the change they made, not a bare number.
* Inputs that fall outside the training distribution are flagged: a model asked
  to extrapolate is reporting an assumption, not a measurement.
* Scenarios are never described as guaranteed outcomes or causal effects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A one-hot-encoded / standardised feature may legitimately exceed these
# z-score bounds; the check is against the *raw* training values instead.


@dataclass
class ScenarioResult:
    changed: list[dict] = field(default_factory=list)
    baseline_prediction: str = ""
    scenario_prediction: str = ""
    delta: str = ""
    out_of_range: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    baseline_frame: pd.DataFrame | None = None
    scenario_frame: pd.DataFrame | None = None

    def as_dict(self) -> dict:
        return {"changed": self.changed,
                "baseline_prediction": self.baseline_prediction,
                "scenario_prediction": self.scenario_prediction,
                "delta": self.delta, "out_of_range": self.out_of_range,
                "notes": self.notes}


def training_ranges(X: pd.DataFrame,
                    numeric: list[str]) -> dict[str, tuple[float, float]]:
    """Observed min/max per numeric feature, used for the shift check."""
    out: dict[str, tuple[float, float]] = {}
    for c in numeric:
        vals = pd.to_numeric(X[c], errors="coerce").dropna()
        if len(vals):
            out[c] = (float(vals.min()), float(vals.max()))
    return out


def _fmt(value, task: str) -> str:
    if task == "classification":
        return str(value)
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_scenario(pipeline, X: pd.DataFrame, row_index: int,
                 overrides: dict, numeric: list[str],
                 categorical: list[str], task: str,
                 ranges: dict | None = None) -> ScenarioResult:
    """Predict once for a real row, then again with ``overrides`` applied.

    ``overrides`` maps column name -> new value. The same row is reused so the
    only difference between the two predictions is the user's change.
    """
    ranges = ranges or training_ranges(X, numeric)
    if not 0 <= row_index < len(X):
        raise IndexError("That row is not in the dataset.")
    base_row = X.iloc[[row_index]].copy()
    scenario_row = base_row.copy()
    changed: list[dict] = []
    out_of_range: list[str] = []

    for col, new_val in overrides.items():
        if col not in scenario_row.columns:
            continue
        old_val = scenario_row.iloc[0][col]
        scenario_row.iloc[0, scenario_row.columns.get_loc(col)] = new_val
        changed.append({"column": col, "from": _fmt(old_val, task),
                        "to": _fmt(new_val, task)})
        if col in ranges:
            lo, hi = ranges[col]
            try:
                num = float(new_val)
            except (TypeError, ValueError):
                num = None
            if num is not None and not (lo <= num <= hi):
                out_of_range.append(
                    f"{col} = {num:,.4g} is outside the training range "
                    f"({lo:,.4g} to {hi:,.4g}); the model is extrapolating.")

    t0 = time.perf_counter()
    try:
        base_pred = pipeline.predict(base_row)[0]
        new_pred = pipeline.predict(scenario_row)[0]
    except Exception as exc:
        raise RuntimeError(
            f"The model could not score this row ({type(exc).__name__}: "
            f"{exc}). Check that the changed values match the column types."
        ) from exc
    elapsed = time.perf_counter() - t0

    base_s = _fmt(base_pred, task)
    new_s = _fmt(new_pred, task)
    delta_s = ""
    if task == "regression":
        try:
            diff = float(new_pred) - float(base_pred)
            delta_s = f"{diff:+,.4f}"
        except (TypeError, ValueError):
            delta_s = ""

    res = ScenarioResult(
        changed=changed, baseline_prediction=base_s, scenario_prediction=new_s,
        delta=delta_s, out_of_range=out_of_range,
        baseline_frame=base_row, scenario_frame=scenario_row)
    res.notes.append(
        "Both predictions come from the same fitted pipeline and the same "
        "record; only the values you changed differ.")
    res.notes.append(
        "This is a model prediction under a stated assumption, not a measured "
        "outcome and not proof that changing the input causes the change.")
    if not changed:
        res.notes.append("No values were changed, so both predictions are "
                         "identical by construction.")
    res.notes.append(f"Scored in {elapsed * 1000:.1f} ms.")
    return res


def editable_columns(numeric: list[str],
                     categorical: list[str]) -> tuple[list[str], list[str]]:
    """Which columns the UI should offer for editing, and why not the rest."""
    return list(numeric), list(categorical)
