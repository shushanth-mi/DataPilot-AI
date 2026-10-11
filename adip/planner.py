"""Automatic analysis planner.

Given a profile (and an optional user objective), the planner decides which
analytical workflows are *feasible* and *justified*, explains each choice, and
returns a reproducible plan. It never invents a target: when the objective is
ambiguous it presents a small set of defensible options instead of pretending
an arbitrary column is a label.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .profiling import (ColumnKind, DataProfile, TargetCandidate)

MIN_ROWS_SUPERVISED = 60
MIN_ROWS_CLUSTER = 30
MIN_ROWS_ANOMALY = 20


@dataclass
class TaskOption:
    """A feasible analytical task the user could run."""
    kind: str                            # classification | regression | ...
    title: str
    rationale: str
    feasible: bool = True
    blockers: list[str] = field(default_factory=list)
    needs_target: str | None = None      # candidate column name, if any

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        flag = "" if self.feasible else " [BLOCKED]"
        target = f" (target: {self.needs_target})" if self.needs_target else ""
        return f"{self.kind}: {self.title}{target}{flag}"

    def describe(self) -> str:
        """One-line rationale for display."""
        return f"{self.title} — {self.rationale}"


@dataclass
class AnalysisPlan:
    dataset_name: str
    fingerprint: str
    objective: str = ""                # free-text user objective, if any
    labelled: bool = False
    options: list = field(default_factory=list)      # list[TaskOption]
    recommended: str = ""             # kind of the recommended option
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    needs_user_choice: bool = False
    notes: list[str] = field(default_factory=list)

    def options_frame(self):
        import pandas as pd
        return pd.DataFrame([o.as_dict() for o in self.options])

    def feasible_options(self):
        return [o for o in self.options if o.feasible]

# --------------------------------------------------------------------------
# planning logic
# --------------------------------------------------------------------------
def _enough_rows(profile: DataProfile, floor: int) -> tuple[bool, str]:
    if profile.rows < floor:
        return False, (f"only {profile.rows:,} rows are available; this "
                       f"workflow needs at least {floor:,}")
    return True, ""


def _supervised_options(profile: DataProfile, objective: str
                        ) -> tuple[list, str | None, bool]:
    """Return supervised task options, a recommended target, and ambiguity."""
    options: list = []
    cands = profile.target_candidates
    cls = [c for c in cands if c.task == "classification"]
    reg = [c for c in cands if c.task == "regression"]

    if cls:
        top = cls[0]
        feasible, why = _enough_rows(profile, MIN_ROWS_SUPERVISED)
        imbalance = (top.minority_share is not None
                     and top.minority_share < 0.2)
        rationale = (f"'{top.column}' looks like a label ({top.reason}). "
                     "Classification predicts which class a new row falls "
                     "into, scored against a majority-class baseline."
                     + (" Classes are imbalanced, so a balanced metric will "
                        "be used." if imbalance else ""))
        options.append(TaskOption(
            "classification", f"Predict '{top.column}'", rationale,
            feasible=feasible, blockers=[] if feasible else [why],
            needs_target=top.column))
    if reg:
        top = reg[0]
        feasible, why = _enough_rows(profile, MIN_ROWS_SUPERVISED)
        options.append(TaskOption(
            "regression", f"Predict '{top.column}' (numeric)",
            f"'{top.column}' looks like a numeric quantity to estimate "
            f"({top.reason}). Regression scores predicted error "
            "(RMSE/MAE/R2) against a median-prediction baseline.",
            feasible=feasible, blockers=[] if feasible else [why],
            needs_target=top.column))

    recommended = ""
    if cls and _enough_rows(profile, MIN_ROWS_SUPERVISED)[0]:
        recommended = "classification"
    elif reg and _enough_rows(profile, MIN_ROWS_SUPERVISED)[0]:
        recommended = "regression"

    # Ambiguous when both task families are plausible and the user has not
    # named an objective: we must not silently pick one label.
    ambiguous = bool(cls and reg) and not objective
    return options, (cls[0].column if cls else (reg[0].column if reg else None)), ambiguous


def _unsupervised_options(profile: DataProfile) -> list:
    options: list = []
    ok, why = _enough_rows(profile, MIN_ROWS_CLUSTER)
    n_num = len(profile.numeric)
    if n_num >= 2:
        options.append(TaskOption(
            "clustering", "Discover natural groupings",
            f"There are {n_num} numeric columns and no single obvious label. "
            "Clustering groups similar rows so you can inspect what makes each "
            "group distinct. Groups are descriptive, not confirmed classes.",
            feasible=ok, blockers=[] if ok else [why]))
    ok, why = _enough_rows(profile, MIN_ROWS_ANOMALY)
    if n_num >= 2:
        options.append(TaskOption(
            "anomaly", "Flag unusual records",
            f"With {n_num} numeric columns, anomaly detection can rank how "
            "atypical each row is. Flagged rows are candidates to inspect, "
            "not confirmed errors or fraud.",
            feasible=ok, blockers=[] if ok else [why]))
    return options


def build_plan(profile: DataProfile, objective: str = "") -> AnalysisPlan:
    """Turn a dataset profile into a ranked, explained analysis plan.

    The plan lists every feasible task, names the single recommended one, and
    records assumptions, risks and missing information. It does NOT run any
    analysis; it only decides what *could* be run and why.
    """
    plan = AnalysisPlan(dataset_name=profile.dataset_name,
                        fingerprint=profile.fingerprint,
                        objective=objective)
    sup, target, ambiguous = _supervised_options(profile, objective)
    options = sup + _unsupervised_options(profile)

    # Data-quality driven caveats apply to every task.
    high_issues = [i for i in profile.issues if i.severity == "high"]
    for i in high_issues:
        plan.risks.append(f"{i.column or 'dataset'}: {i.message}")
    if profile.duplicate_pct > 1:
        plan.risks.append(
            f"{profile.duplicate_pct}% of rows are exact duplicates; "
            "duplicates inflate apparent performance and must be removed "
            "before training.")
    if profile.sensitive_columns:
        plan.missing_info.append(
            "Sensitive-looking columns were detected "
            f"({', '.join(profile.sensitive_columns)}); confirm they may be "
            "used before including them.")

    # Assumptions and next steps depend on whether the data is labelled.
    if target:
        plan.labelled = True
        plan.assumptions.append(
            f"'{target}' is assumed to be the outcome to predict. This is an "
            "inference from the data shape, not a guarantee.")
    else:
        plan.notes.append(
            "No column looks like a clear label, so only unsupervised "
            "workflows are recommended. This is stated rather than guessing a "
            "target.")

    if ambiguous:
        plan.needs_user_choice = True
        plan.notes.append(
            "Both a classification label and a regression target are "
            "plausible. Choose which question you want to answer; the platform "
            "will not pick one for you.")

    feasible = [o for o in options if o.feasible]
    if not feasible:
        plan.notes.append(
            "No workflow is feasible at the current dataset size. See the "
            "blockers on each option.")
    else:
        plan.recommended = feasible[0].kind
        plan.steps.append(
            f"Recommended first step: {feasible[0].title.lower()}.")
        plan.steps.append(
            "1) Review the data-quality report and act on high-severity "
            "issues.")
        plan.steps.append(
            "2) Run the recommended workflow and read its reliability panel.")
        plan.steps.append(
            "3) Inspect findings and drill into any that look material.")

    if not profile.numeric:
        plan.risks.append(
            "No numeric columns were found, which limits modelling and "
            "charting to categorical summaries.")
    plan.options = options
    return plan

