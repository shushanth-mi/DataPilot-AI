"""Explicit, rule-based reliability assessment.

Deliberate design choice: there is **no single universal "confidence score"**.
Averaging data quality with predictive accuracy would let a clean dataset with
a useless model look reliable. Each dimension is reported separately, and the
overall status is decided by the weakest dimension, using documented rules.

Status ladder (ordered worst to best)
-------------------------------------
INSUFFICIENT -> LIMITED -> PROMISING -> SUITABLE_FOR_EXPLORATION
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Statuses are ordered; `_worst()` picks by this rank.
_STATUS_RANK = ["INSUFFICIENT", "LIMITED", "PROMISING",
                "SUITABLE_FOR_EXPLORATION"]

SUITABLE_FOR_EXPLORATION = "SUITABLE_FOR_EXPLORATION"
PROMISING = "PROMISING"
LIMITED = "LIMITED"
INSUFFICIENT = "INSUFFICIENT"

#: Plain-language meaning of each status, shown verbatim in the UI.
STATUS_MEANING = {
    SUITABLE_FOR_EXPLORATION: (
        "Evidence is adequate for exploration and hypothesis generation. "
        "This is not a guarantee of future performance."),
    PROMISING: (
        "Signals are encouraging but not yet conclusive. Validate on more "
        "or newer data before acting on the result."),
    LIMITED: (
        "Data quality or sample size restricts what can be concluded. "
        "Treat results as indicative only."),
    INSUFFICIENT: (
        "The available evidence cannot support a dependable conclusion."),
}

# Thresholds are explicit so reviewers can audit exactly where a verdict
# came from, rather than trusting an opaque formula.
MIN_ROWS_SUPERVISED = 60
MIN_ROWS_WEAK = 30
MIN_TEST_ROWS = 20
MIN_PER_CLASS = 5
MIN_CLASSES = 2
SEVERE_MISSING_PCT = 40.0
MODERATE_MISSING_PCT = 10.0
DUP_PCT_HIGH = 5.0
IMBALANCE_RATIO = 0.10        # minority share below this is "imbalanced"
HIGH_CARDINALITY_LEVELS = 100
STRONG_BEAT_BASELINE = 0.10   # margin over baseline that counts as real signal
MEAN_BEAT_BASELINE = 0.02


@dataclass
class ReliabilityReport:
    status: str
    headline: str
    meaning: str
    dimensions: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status, "headline": self.headline,
            "meaning": self.meaning, "dimensions": self.dimensions,
            "reasons": self.reasons, "limitations": self.limitations,
            "next_steps": self.next_steps, "rules": self.rules,
        }


def _worst(statuses) -> str:
    """Return the weakest status present, using the documented ladder."""
    present = [s for s in statuses if s in _STATUS_RANK]
    if not present:
        return SUITABLE_FOR_EXPLORATION
    return min(present, key=_STATUS_RANK.index)


def _rule(text: str) -> str:
    return text

# --------------------------------------------------------------------------
# data-quality dimension (dataset level, applies to every analysis)
# --------------------------------------------------------------------------
def _data_quality(profile) -> tuple[str, list[str], list[str]]:
    """Rate data quality from the profiler's issue list."""
    issues = profile.issues
    high = [i for i in issues if i.severity == "high"]
    med = [i for i in issues if i.severity == "medium"]
    reasons, limits = [], []

    worst_missing = max([c.missing_pct for c in profile.columns] or [0.0])
    n_severe = sum(1 for c in profile.columns
                   if c.missing_pct >= SEVERE_MISSING_PCT)
    n_moderate = sum(1 for c in profile.columns
                     if MODERATE_MISSING_PCT <= c.missing_pct
                     < SEVERE_MISSING_PCT)

    if n_severe:
        return INSUFFICIENT, [
            f"{n_severe} column(s) are missing {SEVERE_MISSING_PCT:.0f}%+ of "
            f"values (worst: {worst_missing:.1f}%)."], [
            "Columns with very high missingness are mostly guesswork after "
            "imputation and can silently distort any result."]
    if high:
        reasons.append(f"{len(high)} high-severity data-quality issue(s): "
                       + "; ".join(i.message for i in high[:3]))
    if profile.duplicate_pct > DUP_PCT_HIGH:
        limits.append(f"{profile.duplicate_pct:.1f}% duplicate rows risk "
                      "inflating measured performance.")
    if n_moderate:
        reasons.append(f"{n_moderate} column(s) missing 10-40% of values.")

    if high or profile.duplicate_pct > DUP_PCT_HIGH:
        return LIMITED, reasons, limits
    if med or n_moderate:
        return PROMISING, (reasons or
                           [f"{len(med)} medium-severity issue(s) to review."]), \
            limits
    return SUITABLE_FOR_EXPLORATION, [
        "No blocking data-quality problems detected."], limits


def _headline(status: str, subject: str) -> str:
    return {
        SUITABLE_FOR_EXPLORATION:
            f"The analysis of {subject} is adequate for exploration.",
        PROMISING:
            f"Early signals for {subject} are encouraging but need validation.",
        LIMITED:
            f"Conclusions about {subject} are restricted by the available data.",
        INSUFFICIENT:
            f"The data does not currently support a dependable conclusion "
            f"about {subject}.",
    }[status]


# --------------------------------------------------------------------------
# supervised reliability
# --------------------------------------------------------------------------
def assess_supervised(profile, outcome) -> ReliabilityReport:
    """Assess a supervised run across explicit dimensions."""
    dims: dict[str, str] = {}
    reasons: list[str] = []
    limits: list[str] = []
    steps: list[str] = []

    q_status, q_reasons, q_limits = _data_quality(profile)
    dims["Data quality"] = q_status
    reasons += q_reasons
    limits += q_limits

    # ---- sample size --------------------------------------------------
    rows, test_rows = profile.rows, outcome.test_rows
    if rows < MIN_ROWS_WEAK or test_rows < MIN_TEST_ROWS:
        dims["Sample size"] = INSUFFICIENT
        limits.append(
            f"Only {rows:,} rows total ({test_rows:,} held out for testing). "
            f"A dependable estimate needs more than {MIN_ROWS_SUPERVISED} rows "
            "and a non-trivial test set.")
        steps.append("Collect more labelled examples before trusting the score.")
    elif rows < MIN_ROWS_SUPERVISED:
        dims["Sample size"] = LIMITED
        limits.append(f"{rows:,} rows is below the {MIN_ROWS_SUPERVISED}-row "
                      "guideline, so the estimate carries wide uncertainty.")
        steps.append("Add data and re-run to see whether the score is stable.")
    else:
        dims["Sample size"] = SUITABLE_FOR_EXPLORATION

    # ---- class imbalance ----------------------------------------------
    if outcome.task == "classification":
        n_cls = max(len(outcome.classes), 0)
        counts = outcome.class_counts or {}
        minority = (min(counts.values()) / rows) if counts and rows else 0.0
        if n_cls < MIN_CLASSES:
            dims["Class balance"] = INSUFFICIENT
            limits.append("Fewer than two classes are present, so "
                          "classification cannot be evaluated.")
        elif minority < IMBALANCE_RATIO:
            dims["Class balance"] = LIMITED
            limits.append(f"Strong class imbalance: the rarest class is "
                          f"{minority:.1%} of rows. Accuracy is misleading and "
                          "the model may ignore that class entirely.")
            steps.append("Review per-class precision/recall, and consider "
                         "collecting more minority examples or rebalancing.")
        else:
            dims["Class balance"] = SUITABLE_FOR_EXPLORATION
    else:
        dims["Class balance"] = SUITABLE_FOR_EXPLORATION

    # ---- out-of-sample performance vs baseline ------------------------
    best, base = outcome.best, outcome.baseline
    if best is None or base is None:
        dims["Out-of-sample performance"] = INSUFFICIENT
        limits.append("No model could be trained and scored on held-out data.")
    else:
        key = outcome.primary_metric
        b_val, d_val = best.metrics.get(key), base.metrics.get(key)
        if b_val is None or d_val is None:
            dims["Out-of-sample performance"] = INSUFFICIENT
        else:
            lower_better = key in ("RMSE", "MAE")
            delta = (d_val - b_val) if lower_better else (b_val - d_val)
            span = abs(d_val) if abs(d_val) > 1e-12 else 1.0
            rel = delta / span
            dims["Out-of-sample performance"] = (
                PROMISING if rel >= MEAN_BEAT_BASELINE
                else LIMITED if rel > 0 else INSUFFICIENT)
            if rel >= STRONG_BEAT_BASELINE:
                dims["Out-of-sample performance"] = SUITABLE_FOR_EXPLORATION
            if rel <= 0:
                limits.append(f"The best model did not beat the trivial "
                              f"baseline on {key}, which means it has learned "
                              "no usable signal from these features.")
                steps.append("Re-examine the feature set: the current columns "
                             "may not carry information about the target.")

    # ---- stability across folds ---------------------------------------
    if best is not None and best.cv_std is not None \
            and best.cv_mean is not None:
        cv_rel = abs(best.cv_std) / (abs(best.cv_mean) or 1.0)
        dims["Stability across folds"] = (
            SUITABLE_FOR_EXPLORATION if cv_rel < 0.10
            else PROMISING if cv_rel < 0.25 else LIMITED)
        if cv_rel >= 0.25:
            limits.append(f"Cross-validation scores vary widely between folds "
                          f"(mean {best.cv_mean:.3f} +/- {best.cv_std:.3f}), so "
                          "the single reported score is not dependable.")
    else:
        dims["Stability across folds"] = LIMITED
        limits.append("Fold-to-fold variation could not be measured.")

    # ---- leakage -------------------------------------------------------
    leaked = [r for r in outcome.drop_reasons if "id_like" in r]
    dims["Potential leakage"] = (
        PROMISING if leaked else SUITABLE_FOR_EXPLORATION)
    if leaked:
        limits.append(f"{len(leaked)} identifier-like column(s) were excluded "
                      "from features to prevent leakage: "
                      + ", ".join(r.split(":")[0] for r in leaked[:3]))

# --------------------------------------------------------------------------
# unsupervised reliability
# --------------------------------------------------------------------------
def assess_unsupervised(profile, outcome) -> ReliabilityReport:
    """Assess an unsupervised run.

    The decisive honesty rule: with no ground truth, predictive accuracy is
    *never* reported, and the status can never exceed PROMISING.
    """
    dims: dict[str, str] = {}
    reasons: list[str] = []
    limits: list[str] = []
    steps: list[str] = []

    q_status, q_reasons, q_limits = _data_quality(profile)
    dims["Data quality"] = q_status
    reasons += q_reasons
    limits += q_limits

    rows = profile.rows
    dims["Sample size"] = (
        SUITABLE_FOR_EXPLORATION if rows >= 100
        else LIMITED if rows >= MIN_ROWS_CLUSTER else INSUFFICIENT)
    if rows < 100:
        limits.append(f"{rows:,} rows: internal clustering scores and "
                      "stability estimates are unstable at this size.")

    best = outcome.best_cluster()
    if best is None:
        dims["Cluster structure"] = INSUFFICIENT
        limits.append("No algorithm produced more than one group, so there is "
                      "no cluster structure to report.")
        steps.append("Clustering may be the wrong tool for this dataset; "
                     "try profiling or anomaly detection instead.")
    else:
        sil = best.silhouette or 0.0
        dims["Cluster structure"] = (
            SUITABLE_FOR_EXPLORATION if sil >= 0.5
            else PROMISING if sil >= 0.25 else LIMITED)
        reasons.append(f"{best.algorithm} produced {best.n_clusters} groups "
                       f"(silhouette {sil:.3f}).")
        if sil < 0.25:
            limits.append("Separation between groups is weak (silhouette "
                          f"{sil:.3f}); the partition may not reflect real "
                          "structure.")

    stab = best.stability if best is not None else None
    if stab is None:
        dims["Stability"] = LIMITED
        limits.append("Cluster stability across resampling was not measured.")
    else:
        dims["Stability"] = (SUITABLE_FOR_EXPLORATION if stab >= 0.75
                             else PROMISING if stab >= 0.5 else LIMITED)
        if stab < 0.5:
            limits.append(f"Clusters are unstable across resamples "
                          f"(mean agreement {stab:.2f}), so group membership "
                          "should not be over-interpreted.")

    # This dimension is permanent for label-free data.
    dims["Ground-truth accuracy"] = INSUFFICIENT
    limits.append("No ground-truth labels exist for this dataset, so "
                  "classification accuracy cannot be calculated and is "
                  "deliberately not reported. Clusters are descriptive "
                  "groups, not validated classes.")

    status = _worst(dims.values())
    if _STATUS_RANK.index(status) > _STATUS_RANK.index(PROMISING):
        status = PROMISING      # unverifiable evidence never reads as final

    return ReliabilityReport(
        status=status,
        headline=_headline(status, "the unsupervised structure"),
        meaning=STATUS_MEANING[status] + " Clusters are not classes.",
        dimensions=dims, reasons=reasons, limitations=limits,
        next_steps=steps or [
            "Inspect cluster profiles, then validate the most interesting "
            "group with domain knowledge before acting on it."],
        rules=[
            "No ground truth: status is capped at PROMISING and no accuracy "
            "is reported.",
            "Silhouette >= 0.50 supports structure; 0.25-0.50 is weak; "
            "below 0.25 suggests no real partition.",
            "Stability >= 0.75 mean adjusted Rand index across resamples.",
            "Overall status is the WEAKEST dimension, not an average."])

    reasons.append("Preprocessing was fitted inside each validation fold, so "
                   "imputation and scaling statistics never leaked from the "
                   "test set.")

    status = _worst(dims.values())
    return ReliabilityReport(
        status=status, headline=_headline(status, outcome.target),
        meaning=STATUS_MEANING[status], dimensions=dims, reasons=reasons,
        limitations=limits, next_steps=steps or [
            "Report the metric together with the test-set size and the "
            "cross-validation spread, never as a bare accuracy number."],
        rules=[
            f"Sample size: >={MIN_ROWS_SUPERVISED} rows and "
            f">={MIN_TEST_ROWS} test rows to be suitable.",
            f"Class imbalance: minority share below "
            f"{IMBALANCE_RATIO:.0%} downgrades to LIMITED.",
            "Performance: must beat the dummy baseline on the primary metric.",
            "Stability: fold-to-fold relative spread under 10% is suitable.",
            "Overall status is the WEAKEST dimension, not an average."])

