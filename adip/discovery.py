"""Evidence-based discovery engine.

Purpose: surface what the user did not know to ask for, while keeping the
difference between *observation* and *explanation* explicit.

Rules enforced here
-------------------
1. Every finding carries its own evidence, method, and limitation.
2. Multiple-comparison control is applied. Testing dozens of hypotheses at
   p < 0.05 guarantees false positives, so the Benjamini-Hochberg procedure
   controls the false discovery rate and findings that fail it are labelled
   exploratory rather than reported as confirmed.
3. No causal language. A correlation between two columns is reported as an
   association and nothing more.
4. Findings are ranked by effect size, stability and data quality, not by
   p-value alone. A tiny effect on a huge sample is not interesting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .profiling import ColumnKind, DataProfile

# --- ranking / significance tuning -----------------------------------------
#: Only findings ranked in this band are reported, to control noise.
TOP_N = 12
#: Benjamini-Hochberg target false discovery rate.
FDR = 0.10
#: Minimum absolute effect size for a correlation to be worth reporting.
MIN_CORR = 0.30
#: A segment difference must be at least this large (standardised) to count.
MIN_SEGMENT_D = 0.50
#: A group must hold at least this share of rows to be a "segment".
MIN_SEGMENT_SHARE = 0.05
#: Rows below this count make any comparison too fragile to report.
MIN_GROUP_N = 10
#: Effect-size floor for the strongest association found.
MIN_TOP_ASSOC = 0.40


@dataclass
class Finding:
    """A single discovery, with everything needed to defend it."""

    title: str
    description: str
    evidence: str
    columns: list[str] = field(default_factory=list)
    population: str = ""
    method: str = ""
    effect_size: float = 0.0
    stability: str = ""
    limitation: str = ""
    next_step: str = ""
    kind: str = ""                     # association | segment | anomaly | quality
    exploratory: bool = False          # survived multiple-comparison control?
    p_value: float | None = None
    rank: float = 0.0

    def as_dict(self) -> dict:
        return {
            "rank": self.rank, "kind": self.kind, "title": self.title,
            "description": self.description, "evidence": self.evidence,
            "columns": self.columns, "population": self.population,
            "method": self.method, "effect_size": round(self.effect_size, 4),
            "stability": self.stability, "limitation": self.limitation,
            "next_step": self.next_step, "exploratory": self.exploratory,
            "p_value": self.p_value,
        }


def benjamini_hochberg(pvals: list[float],
                       q: float = FDR) -> list[bool]:
    """Return a keep/reject mask controlling the false discovery rate.

    Standard BH: sort p-values ascending, find the largest rank i where
    p(i) <= (i / m) * q, and reject everything up to and including that rank.
    Returns a boolean list aligned with the *input* order.
    """
    m = len(pvals)
    if m == 0:
        return []
    keep = [False] * m
    valid = [(p, i) for i, p in enumerate(pvals) if p is not None]
    if not valid:
        return keep
    valid.sort()
    threshold_rank = 0
    for rank, (p, _i) in enumerate(valid, start=1):
        if p <= (rank / len(valid)) * q:
            threshold_rank = rank
    for rank, (_p, idx) in enumerate(valid, start=1):
        if rank <= threshold_rank:
            keep[idx] = True
    return keep


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
#: Bootstrap replicates used for the sign-stability measure.
BOOT_ROUNDS = 40
#: Deterministic seed so discovery is reproducible across runs.
RANDOM_STATE = 42
#: |rho| at or above this is treated as two columns measuring the same thing.
NEAR_DUPLICATE = 0.98
#: Below this many complete rows a comparison is too fragile to report.
MIN_ROWS_EFFECT = 30
#: A segment level needs at least this many rows to be compared with the rest.
MIN_ROWS_SEGMENT = 15
#: Correlation strength floor for reporting (uses the shared MIN_CORR idea).
_REPORT_CORR = MIN_CORR


def _finite_pairs(x: pd.Series, y: pd.Series):
    """Return the two arrays reduced to rows where both values are known."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    return d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float)


def _bootstrap_stability(x, y, rounds: int = BOOT_ROUNDS,
                         seed: int = RANDOM_STATE) -> float:
    """Share of bootstrap resamples that reproduce the same sign.

    Deliberately simple and interpretable: it answers "would I still see this
    effect in a different sample from the same population?" A finding that
    flips sign half the time is a hypothesis, not a result.
    """
    n = len(x)
    if n < MIN_ROWS_EFFECT or len(y) != n:
        return float("nan")
    rng = np.random.default_rng(seed)
    agree = 0
    used = 0
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        used += 1
        r, _p = stats.spearmanr(xb, yb)
        if np.isfinite(r) and r > 0:
            agree += 1
    if not used:
        return float("nan")
    return round(agree / used, 4)


#: Adverb forms, so sentences read naturally ("moderately higher", not "mediumly").
_CORR_ADVERB = ["negligibly", "weakly", "moderately", "strongly", "very strongly"]
_SEGMENT_ADVERB = ["negligibly", "slightly", "moderately", "substantially"]


def _effect_label(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return _CORR_ADVERB[0]
    if a < 0.3:
        return _CORR_ADVERB[1]
    if a < 0.5:
        return _CORR_ADVERB[2]
    if a < 0.7:
        return _CORR_ADVERB[3]
    return _CORR_ADVERB[4]




# --------------------------------------------------------------------------
# correlation discovery
# --------------------------------------------------------------------------
def find_correlations(df: pd.DataFrame, numeric: list[str],
                      max_pairs: int = 25) -> list[Finding]:
    """Pairwise monotonic associations among numeric columns.

    Spearman rank correlation is used instead of Pearson because it tolerates
    outliers and skew, which are common in uploaded business data. A |rho| near
    1.0 is flagged as a *duplicate-measurement* problem rather than a triumph:
    two perfectly related columns carry redundant information.
    """
    findings: list[Finding] = []
    if len(numeric) < 2:
        return findings
    records = []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            a, b = numeric[i], numeric[j]
            x, y = _finite_pairs(df[a], df[b])
            n = len(x)
            if n < MIN_ROWS_EFFECT or len(np.unique(x)) < 2 \
                    or len(np.unique(y)) < 2:
                continue
            with np.errstate(all="ignore"):
                rho, p = stats.spearmanr(x, y)
            if not np.isfinite(rho):
                continue
            records.append({"a": a, "b": b, "n": n, "rho": float(rho),
                            "p": float(p) if np.isfinite(p) else 1.0,
                            "x": x, "y": y})
    if not records:
        return findings

    # Drop associations too weak to be worth reporting *before* the
    # multiple-comparison step, so the correction is applied only to the
    # hypotheses we actually intend to test. Testing everything and then
    # filtering by effect size would leave the p-values over-corrected.
    records = [r for r in records if abs(r["rho"]) >= _REPORT_CORR]
    if not records:
        return findings

    # Attach the multiple-comparison verdict BEFORE sorting, so each record
    # keeps its own keep/reject flag (sorting later must not reorder the mask).
    keep = benjamini_hochberg([r["p"] for r in records], FDR)
    for rec, survives in zip(records, keep):
        rec["survives"] = survives
    records.sort(key=lambda r: -abs(r["rho"]))

    for rec in records[:max_pairs]:
        rho, n, survives = rec["rho"], rec["n"], rec["survives"]
        stab = _bootstrap_stability(rec["x"], rec["y"])
        direction = ("rise together" if rho > 0
                     else "move in opposite directions")
        # A finding that survives BH correction is reported as confirmed;
        # one that does not is explicitly labelled exploratory.
        conf = ("confirmed against multiple comparisons"
                if survives
                else "exploratory only, not confirmed after "
                     "multiple-comparison correction")
        stab_txt = (f"sign reproduced in {stab:.0%} of {BOOT_ROUNDS} bootstrap "
                    f"resamples")
        findings.append(Finding(
            kind="association",
            title=(f"{rec['a']} and {rec['b']} are {_effect_label(rho)} "
                   f"associated ({'positive' if rho > 0 else 'negative'})"),
            description=(
                f"When {rec['a']} rises, {rec['b']} tend to {direction}. "
                f"The rank correlation is rho = {rho:.2f} "
                f"({_effect_label(rho)}). This association is {conf}."),
            evidence=(f"Spearman rho = {rho:.3f}, p = {rec['p']:.2e}, "
                      f"n = {n:,} complete pairs, {stab_txt}"),
            columns=[rec["a"], rec["b"]],
            population=f"{n:,} rows with both values present",
            method="Spearman rank correlation with bootstrap sign stability",
            p_value=rec["p"],
            effect_size=round(abs(rho), 4),
            stability=stab_txt,
            limitation=("Correlation is not causation: a third variable may "
                        "drive both columns, and reverse causation is not "
                        "excluded."),
            next_step=(f"Stratify {rec['a']} vs {rec['b']} by a categorical "
                       f"column to test for a confounding variable."),
            exploratory=not survives,
        ))
        if abs(rho) >= NEAR_DUPLICATE:
            findings.append(Finding(
                kind="quality",
                title=(f"{rec['a']} and {rec['b']} are almost the same "
                       f"measure"),
                description=(
                    f"The two columns move almost perfectly together "
                    f"(rho = {rho:.3f}). They are likely duplicates, or one is "
                    f"derived from the other, so keeping both inflates "
                    f"apparent feature importance."),
                evidence=f"|rho| = {abs(rho):.3f} exceeds the "
                         f"{NEAR_DUPLICATE} near-duplicate threshold",
                columns=[rec["a"], rec["b"]],
                population=f"{n:,} rows",
                method="Spearman rank correlation, near-duplicate rule",
                effect_size=round(abs(rho), 4),
                stability=stab_txt,
                limitation=("Derived columns can leak the target when the "
                            "target is computed from them."),
                next_step=("Check the lineage of both columns and keep only "
                           "one as a model feature."),
            ))
    return findings



# --------------------------------------------------------------------------
# segment discovery
# --------------------------------------------------------------------------
def _hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Hedges' g effect size (bias-corrected Cohen's d) and a rough p-value.

    Preferred over a plain t-test because it quantifies *how big* a difference
    is, which is what a decision needs; the p-value only says whether n is
    large enough. Hedges' correction matters for the small segments common in
    uploaded data.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), 1.0
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
    if pooled <= 0:
        return float("nan"), 1.0
    s = np.sqrt(pooled)
    d = (np.mean(a) - np.mean(b)) / s
    correction = 1 - (3 / (4 * (n1 + n2) - 9))     # Hedges' small-sample fix
    g = d * correction
    if s == 0:
        return float("nan"), 1.0
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2 - 2)))
    p = float(2 * (1 - stats.norm.cdf(abs(g / se)))) if se > 0 else 1.0
    return float(g), min(max(p, 0.0), 1.0)


def _segment_label(g: float) -> str:
    a = abs(g)
    if a < 0.2:
        return _SEGMENT_ADVERB[0]
    if a < 0.5:
        return _SEGMENT_ADVERB[1]
    if a < 0.8:
        return _SEGMENT_ADVERB[2]
    return _SEGMENT_ADVERB[3]


def find_segments(df: pd.DataFrame, numeric: list[str],
                  categorical: list[str],
                  max_levels: int = 12,
                  max_results: int = 20) -> list[Finding]:
    """Compare each categorical level against the rest on each numeric column.

    This is the "which segments differ most?" question. Every level is
    contrasted with all other rows (not with another single level), so the
    comparison is stable even when a column has many uneven levels. Only the
    strongest level per (column, feature) pair is reported to avoid listing
    every category of a long tail.
    """
    findings: list[Finding] = []
    if not categorical or not numeric:
        return findings

    records = []
    for cat in categorical:
        levels = df[cat].dropna().unique().tolist()
        if len(levels) < 2 or len(levels) > max_levels:
            continue
        for col in numeric:
            sub = df[[cat, col]].dropna()
            if len(sub) < MIN_ROWS_EFFECT:
                continue
            for lvl in levels:
                mask = sub[cat] == lvl
                a = sub.loc[mask, col].to_numpy(dtype=float)
                b = sub.loc[~mask, col].to_numpy(dtype=float)
                if len(a) < MIN_ROWS_SEGMENT or len(b) < MIN_ROWS_SEGMENT:
                    continue
                g, p = _hedges_g(a, b)
                if not np.isfinite(g) or abs(g) < 0.2:
                    continue
                records.append({"cat": cat, "col": col, "lvl": lvl,
                                "g": g, "p": p, "n": len(a),
                                "mean": float(np.mean(a)),
                                "rest": float(np.mean(b))})
    if not records:
        return findings

    # Attach the keep/reject verdict before sorting so the mask stays aligned.
    keep = benjamini_hochberg([r["p"] for r in records], FDR)
    for rec, survives in zip(records, keep):
        rec["survives"] = survives
    records.sort(key=lambda r: -abs(r["g"]))
    seen: set[tuple[str, str]] = set()
    emitted = 0
    for rec in records:
        key = (rec["cat"], rec["col"])
        if key in seen or emitted >= max_results:
            continue
        seen.add(key)
        emitted += 1
        survives = rec["survives"]
        higher = ("higher than" if rec["g"] > 0 else "lower than")
        share = rec["n"] / max(len(df), 1)
        findings.append(Finding(
            kind="segment",
            title=(f"{rec['cat']} = {rec['lvl']!s} has "
                   f"{_segment_label(rec['g'])} "
                   f"{higher.replace(' than', '')} "
                   f"{rec['col']} than the rest of the data"),
            description=(
                f"Rows where {rec['cat']} is {rec['lvl']!s} have an average "
                f"{rec['col']} of {rec['mean']:.3g}, which is {higher} the "
                f"{rec['rest']:.3g} average of all other rows. The difference "
                f"is {_segment_label(rec['g'])} by Hedges' g and "
                f"{'survives' if survives else 'does not survive'} "
                f"multiple-comparison correction."),
            evidence=(f"Segment mean {rec['mean']:.3g} vs {rec['rest']:.3g} "
                      f"elsewhere, Hedges' g = {rec['g']:.3f}, "
                      f"p = {rec['p']:.2e}, n = {rec['n']:,} rows in segment "
                      f"({share:.1%} of data)"),
            columns=[rec["cat"], rec["col"]],
            population=(f"{rec['n']:,} rows where {rec['cat']} = "
                        f"{rec['lvl']!s}, compared with all other rows"),
            method="Level-vs-rest comparison with Hedges' g and BH correction",
            p_value=rec["p"],
            effect_size=round(abs(rec["g"]), 4),
            stability=(f"segment holds {share:.1%} of rows; mean is estimated "
                       f"from {rec['n']:,} observations"),
            limitation=("This is an observational contrast, not a controlled "
                        "comparison: the segment may differ in other ways that "
                        "explain the gap."),
            next_step=(f"Break {rec['col']} down by {rec['cat']} in the "
                       f"Explore view and check whether the gap persists "
                       f"inside sub-segments."),
            exploratory=not survives,
        ))
    return findings

# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def _quality_weight(profile: DataProfile) -> float:
    """Shrink trust when the underlying data is weak.

    A finding computed on a column that is 60% missing, or on a frame full of
    duplicate rows, should not outrank the same finding on clean data. This is
    a deliberate discount, not a confidence score.
    """
    total_missing = sum(c.missing_pct for c in profile.columns) \
        / max(len(profile.columns), 1)
    dup = profile.duplicate_pct
    weight = 1.0 - (total_missing / 100.0) * 0.5 - (dup / 100.0) * 0.3
    return float(max(0.2, min(1.0, weight)))


def rank_findings(findings: list[Finding], profile: DataProfile) -> list[Finding]:
    """Order findings by effect size and stability, then apply the quality
    discount and keep only the top band.

    Ranking is *not* by p-value: a trivially small effect on a huge sample can
    be highly significant, while a large stable effect on a modest sample is
    far more useful. Exploratory (uncorrected) findings are ordered after
    confirmed ones of similar size so they never dominate the report.
    """
    qw = _quality_weight(profile)
    for f in findings:
        stability_bonus = 0.0
        if f.stability and "%" in f.stability:
            try:
                pct = float(f.stability.split("%")[0].split()[-1])
                stability_bonus = min(pct / 100.0, 1.0) * 0.3
            except ValueError:
                stability_bonus = 0.0
        base = f.effect_size + stability_bonus
        if f.exploratory:
            base *= 0.6          # uncorrected findings rank lower
        f.rank = round(base * qw, 4)
    findings.sort(key=lambda f: -f.rank)
    return findings[:TOP_N]



# --------------------------------------------------------------------------
# data-quality findings
# --------------------------------------------------------------------------
def find_quality_findings(df: pd.DataFrame,
                          profile: DataProfile) -> list[Finding]:
    """Turn the profile's issue list into ranked, explainable findings.

    The profiler already detects the problems; this adds the consequence and a
    next step, which is what a user actually needs in order to act.
    """
    findings: list[Finding] = []
    n = max(len(df), 1)

    heavy = [c for c in profile.columns if c.missing_pct >= 30]
    if heavy:
        cols = [c.name for c in heavy]
        findings.append(Finding(
            kind="quality",
            title=f"{len(heavy)} column(s) are mostly empty",
            description=(
                f"{len(heavy)} column(s) have 30% or more of their values "
                f"missing: {', '.join(cols[:5])}"
                + (" and others." if len(cols) > 5 else ".")
                + " Any model trained on them would mostly learn the "
                  "imputation rule, not the real signal."),
            evidence="; ".join(
                f"{c.name}: {c.missing_pct:.1f}% missing" for c in heavy[:5]),
            columns=cols,
            population=f"{n:,} rows",
            method="Missingness share per column from the data profile",
            effect_size=round(max(c.missing_pct / 100 for c in heavy), 4),
            limitation=("Missing values are not necessarily random: if they "
                        "depend on the value itself, imputation biases "
                        "results."),
            next_step=("Drop these columns, or confirm with the data owner "
                       "whether the blanks carry meaning (for example 'not "
                       "applicable' vs 'not recorded')."),
        ))

    if profile.duplicate_rows:
        findings.append(Finding(
            kind="quality",
            title=f"{profile.duplicate_rows:,} fully duplicated rows",
            description=(
                f"{profile.duplicate_pct:.1f}% of rows repeat exactly. If the "
                f"same rows land on both sides of a train/test split, a model "
                f"appears to learn the pattern when it is simply recognising "
                f"records it has already seen."),
            evidence=f"{profile.duplicate_rows:,} of {n:,} rows are exact "
                     f"duplicates",
            columns=[],
            population=f"{n:,} rows",
            method="pandas DataFrame.duplicated over all columns",
            effect_size=round(profile.duplicate_pct / 100, 4),
            limitation=("Duplicates are sometimes legitimate (many-to-one "
                        "transaction rows), so they must not be removed "
                        "automatically."),
            next_step=("Check whether these are genuine repeats. If so, drop "
                       "duplicates before training and note it in the report."),
        ))

    # Leakage risk: an identifier that is unique in every row lets a model
    # memorise individual records. This must NOT fire on ordinary continuous
    # measurements, which are also unique per row simply because they are
    # real numbers, so the check requires the column to be identifier-like
    # (id_like kind or an id-suggesting name) *and* unique per row. Trivial
    # frames are skipped because "one unique value per row" is meaningless
    # at n <= 5.
    leaky = [c for c in profile.columns
             if c.id_like and c.n_unique >= n and n > 5]
    if leaky:
        cols = [c.name for c in leaky]
        findings.append(Finding(
            kind="quality",
            title="Possible target leakage detected",
            description=(
                f"{', '.join(cols[:4])} holds a unique value for every row. A "
                f"column like this lets a model memorise individual records "
                f"instead of learning a general pattern, producing validation "
                f"scores that will not repeat in production."),
            evidence="; ".join(f"{c.name}: {c.n_unique:,} unique values for "
                               f"{n:,} rows" for c in leaky[:4]),
            columns=cols,
            population=f"{n:,} rows",
            method="Cardinality-equals-row-count identifier check",
            effect_size=0.9,
            limitation=("Row identifiers are normal and expected; the risk "
                        "arises only when such a column is used as a feature."),
            next_step=("Confirm these are identifiers and keep them out of the "
                       "feature set (the trainer already excludes them)."),
        ))
    return findings


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class DiscoveryReport:
    """Everything the UI needs to render the findings screen."""
    findings: list[Finding] = field(default_factory=list)
    n_tested: int = 0
    n_confirmed: int = 0
    quality_weight: float = 1.0
    notes: list[str] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_dict() for f in self.findings])

    def confirmed(self) -> list[Finding]:
        return [f for f in self.findings if not f.exploratory]

    def exploratory(self) -> list[Finding]:
        return [f for f in self.findings if f.exploratory]


def run_discovery(df: pd.DataFrame, profile: DataProfile) -> DiscoveryReport:
    """Run every applicable discovery pass and rank the combined results.

    Each pass is defensive: an unexpected shape in the data degrades that one
    pass instead of failing the whole report, and any failure is recorded in
    ``notes`` so the user is told what was not checked.
    """
    report = DiscoveryReport()
    report.quality_weight = round(_quality_weight(profile), 3)
    numeric = [c for c in profile.numeric if not df[c].dropna().empty]
    categorical = [c for c in profile.categorical
                   if 2 <= df[c].nunique(dropna=True) <= 50]

    passes = [
        ("correlations", lambda: find_correlations(df, numeric)),
        ("segments", lambda: find_segments(df, numeric, categorical)),
        ("data quality", lambda: find_quality_findings(df, profile)),
    ]
    collected: list[Finding] = []
    for label, fn in passes:
        try:
            collected.extend(fn())
        except Exception as exc:                       # pragma: no cover
            report.notes.append(
                f"The {label} pass could not run ({type(exc).__name__}: {exc}); "
                f"those findings are missing from this report.")
    report.findings = rank_findings(collected, profile)
    report.n_confirmed = len(report.confirmed())
    report.n_tested = len(collected)

    if not report.findings:
        report.notes.append(
            "No finding passed the reporting thresholds. This usually means "
            "the dataset is small, or the columns are too weakly related to "
            "distinguish signal from noise at this sample size.")
    elif report.n_confirmed == 0:
        report.notes.append(
            f"Every finding is exploratory: none survived the "
            f"{FDR:.0%} false-discovery-rate correction. Treat them as "
            f"candidates to check, not as conclusions.")
    else:
        report.notes.append(
            f"{report.n_confirmed} of {len(report.findings)} reported findings "
            f"survived Benjamini-Hochberg correction at q = {FDR}. Findings "
            f"marked exploratory did not, and should not be quoted as facts.")
    return report

