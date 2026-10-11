"""Dataset profiling: schema inference, data-quality issues, target candidates.

Everything here is deterministic and offline. No value is ever modified; the
profiler only *observes* the frame and describes what it found.
"""
from __future__ import annotations

import re
import warnings as _warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

_SENSITIVE_RE = re.compile(
    r"(email|e-mail|phone|mobile|ssn|social|passport|address|zip|postcode|"
    r"dob|birth|credit|card_?number|surname|firstname|lastname|fullname)",
    re.I)
_ID_RE = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|^row|_no$|^no$|_number$|_key$|^index$)", re.I)
_TARGET_HINTS = re.compile(
    r"(churn|fail|target|label|class|outcome|risk|fraud|default|survive|"
    r"attrit|convert|buy|accept|approve|^y$|diagnosis|spam|click)", re.I)
_VALUE_HINTS = re.compile(
    r"(sales|price|revenue|cost|amount|salary|value|demand|spend|profit|"
    r"margin|score|rating|quantity|qty|volume|rate|age|tenure)", re.I)


class ColumnKind:
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ID_LIKE = "id_like"
    TEXT = "text"


@dataclass
class ColumnProfile:
    name: str
    kind: str
    dtype: str
    n_unique: int
    missing: int
    missing_pct: float
    top_share: float                     # share of the most frequent value
    outlier_count: int = 0               # IQR fence, numeric only
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skew: float | None = None
    top_values: str = ""                 # compact preview for categorical
    sensitive: bool = False
    id_like: bool = False
    mixed_types: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityIssue:
    severity: str                        # high | medium | low
    code: str
    column: str | None                   # None = dataset-level issue
    message: str
    remedy: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TargetCandidate:
    column: str
    task: str                            # classification | regression
    reason: str
    n_classes: int | None = None
    minority_share: float | None = None
    score: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataProfile:
    dataset_name: str
    fingerprint: str
    rows: int
    cols: int
    columns: list[ColumnProfile] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0
    id_like_columns: list[str] = field(default_factory=list)
    sensitive_columns: list[str] = field(default_factory=list)
    target_candidates: list[TargetCandidate] = field(default_factory=list)
    memory_mb: float = 0.0

    def names(self, kind: str) -> list[str]:
        return [c.name for c in self.columns if c.kind == kind]

    @property
    def numeric(self) -> list[str]:
        return self.names(ColumnKind.NUMERIC)

    @property
    def categorical(self) -> list[str]:
        return [c.name for c in self.columns
                if c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)]

    @property
    def datetime(self) -> list[str]:
        return self.names(ColumnKind.DATETIME)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.columns])

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.as_dict() for i in self.issues])

# --------------------------------------------------------------------------
# column typing
# --------------------------------------------------------------------------
def _try_datetime(s: pd.Series) -> bool:
    """Heuristic datetime detection for string columns (sample-based)."""
    sample = s.dropna()
    if sample.empty:
        return False
    sample = sample.astype(str).head(200)
    if int(sample.str.len().max()) > 40:
        return False
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def _infer_kind(s: pd.Series, n_rows: int) -> str:
    if pd.api.types.is_bool_dtype(s):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnKind.DATETIME
    if pd.api.types.is_numeric_dtype(s):
        return ColumnKind.NUMERIC
    non_null = s.dropna()
    if non_null.empty:
        return ColumnKind.CATEGORICAL
    if _try_datetime(non_null):
        return ColumnKind.DATETIME
    n_unique = int(non_null.nunique())
    ratio = n_unique / max(n_rows, 1)
    as_str = non_null.astype(str)
    mean_len = float(as_str.str.len().mean())
    # Free text first: unique long strings are prose, not identifiers.
    if mean_len > 60 and n_unique >= 20:
        return ColumnKind.TEXT
    # An identifier is unique per row, or has far more distinct values than a
    # plausible category count.
    if (n_unique == n_rows and n_rows > 1 and ratio >= 0.98) \
            or (ratio >= 0.5 and n_unique >= 50):
        return ColumnKind.ID_LIKE
    return ColumnKind.CATEGORICAL


def _mixed_types(s: pd.Series) -> bool:
    nn = s.dropna()
    if nn.empty or pd.api.types.is_numeric_dtype(s):
        return False
    try:
        return nn.map(type).nunique() > 1
    except Exception:                                        # pragma: no cover
        return False


def _profile_column(name: str, s: pd.Series, n_rows: int) -> ColumnProfile:
    n_missing = int(s.isna().sum())
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    top_share = float(non_null.value_counts(normalize=True).iloc[0]) \
        if n_unique else 0.0
    kind = _infer_kind(s, n_rows)
    id_like = kind == ColumnKind.ID_LIKE or bool(_ID_RE.search(str(name)))
    sensitive = bool(_SENSITIVE_RE.search(str(name)))

    cp = ColumnProfile(
        name=str(name), kind=kind, dtype=str(s.dtype), n_unique=n_unique,
        missing=n_missing,
        missing_pct=round(100 * n_missing / max(n_rows, 1), 2),
        top_share=round(top_share, 4), sensitive=sensitive, id_like=id_like,
        mixed_types=_mixed_types(s),
    )
    if kind == ColumnKind.NUMERIC and n_unique:
        vals = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(vals):
            q1, q3 = float(vals.quantile(.25)), float(vals.quantile(.75))
            iqr = q3 - q1
            cp.outlier_count = int(((vals < q1 - 1.5 * iqr)
                                    | (vals > q3 + 1.5 * iqr)).sum())
            cp.mean, cp.median = float(vals.mean()), float(vals.median())
            cp.std = float(vals.std()) if len(vals) > 1 else 0.0
            cp.min, cp.max = float(vals.min()), float(vals.max())
            if len(vals) > 2:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    cp.skew = float(stats.skew(vals))
    else:
        vc = non_null.value_counts().head(3)
        cp.top_values = "; ".join(f"{k} ({v})" for k, v in vc.items())
    return cp


# --------------------------------------------------------------------------
# quality issues
# --------------------------------------------------------------------------
def _quality_issues(df: pd.DataFrame, cols: list[ColumnProfile],
                    n_rows: int) -> tuple[list[QualityIssue], int]:
    issues: list[QualityIssue] = []
    # Duplicates are counted on the *substantive* columns only. A row-level
    # identifier (e.g. "row0", "row1", ...) is unique by construction, so
    # including it would hide genuine duplicate records entirely.
    dup_cols = [c.name for c in cols if not c.id_like] or [c.name for c in cols]
    dup = int(df[dup_cols].duplicated().sum())
    if dup:
        issues.append(QualityIssue(
            "medium", "duplicate_rows", None,
            f"{dup:,} fully duplicated rows ({100 * dup / n_rows:.1f}% of data)"
            " once identifiers are ignored.",
            "Remove duplicates before training; they inflate apparent "
            "performance through train/test contamination.",
        ))
    for c in cols:
        if c.missing_pct >= 30:
            issues.append(QualityIssue(
                "high", "heavy_missing", c.name,
                f"{c.missing_pct:.1f}% of values are missing.",
                "Consider dropping the column or modelling missingness "
                "explicitly; imputation here is mostly guesswork.",
            ))
        elif c.missing_pct >= 5:
            issues.append(QualityIssue(
                "medium", "missing_values", c.name,
                f"{c.missing_pct:.1f}% of values are missing.",
                "Use median/mode imputation inside the pipeline and record "
                "that results are sensitive to the imputation rule.",
            ))
        if c.kind == ColumnKind.NUMERIC and n_rows > 10 \
                and c.outlier_count / n_rows > 0.05:
            issues.append(QualityIssue(
                "medium", "outliers", c.name,
                f"{c.outlier_count:,} values "
                f"({100 * c.outlier_count / n_rows:.1f}%) fall outside the "
                "1.5x IQR fence.",
                "Investigate whether these are genuine extremes or data-entry "
                "errors; scale-based models are sensitive to them.",
            ))
        if c.n_unique == 1:
            issues.append(QualityIssue(
                "low", "constant_column", c.name,
                "Column has a single unique value.",
                "Drop it: it carries no information for any model.",
            ))
        elif c.n_unique == 2 and c.top_share > 0.95:
            issues.append(QualityIssue(
                "medium", "near_constant", c.name,
                f"One value covers {100 * c.top_share:.1f}% of rows.",
                "Rare-event learning here is dominated by the majority class; "
                "check class-imbalance handling.",
            ))
        if c.kind in (ColumnKind.CATEGORICAL, ColumnKind.ID_LIKE) \
                and c.n_unique > 100:
            issues.append(QualityIssue(
                "medium", "high_cardinality", c.name,
                f"{c.n_unique:,} distinct values.",
                "One-hot encoding explodes here; group rare levels or use a "
                "target-free frequency encoding.",
            ))
        if c.id_like and c.kind != ColumnKind.ID_LIKE:
            issues.append(QualityIssue(
                "high", "suspected_identifier", c.name,
                "Name or content pattern suggests an identifier.",
                "Exclude identifiers from model features to avoid leakage.",
            ))
        if c.sensitive:
            issues.append(QualityIssue(
                "high", "sensitive_column", c.name,
                "Column name suggests personal data.",
                "Values are intentionally not displayed. Do not export this "
                "column; consider removing it before analysis.",
            ))
        if c.mixed_types:
            issues.append(QualityIssue(
                "medium", "mixed_types", c.name,
                "More than one Python type stored in this column.",
                "Coerce to text and clean formatting; mixed types often "
                "indicate broken data entry.",
            ))
    return issues, dup


# --------------------------------------------------------------------------
# target candidates
# --------------------------------------------------------------------------
def _target_candidates(df: pd.DataFrame,
                       cols: list[ColumnProfile]) -> list[TargetCandidate]:
    out: list[TargetCandidate] = []
    for c in cols:
        if c.id_like or c.sensitive or c.kind in (ColumnKind.TEXT,
                                                  ColumnKind.DATETIME):
            continue
        hint = 0.3 if _TARGET_HINTS.search(c.name) else 0.0
        discrete = (c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)
                    or (c.kind == ColumnKind.NUMERIC and c.n_unique <= 10))
        if discrete:
            if 2 <= c.n_unique <= 30:
                shares = (df[c.name].value_counts(normalize=True)
                          .sort_values())
                minority = float(shares.iloc[0])
                if minority >= 0.01:
                    bal = 1.0 - abs(minority - 0.5) * 2
                    out.append(TargetCandidate(
                        column=c.name, task="classification",
                        reason=(f"Categorical with {c.n_unique} classes, "
                                f"minority share {minority:.1%}"
                                + (", name hints at a label" if hint else "")),
                        n_classes=c.n_unique,
                        minority_share=round(minority, 4),
                        score=round(0.55 * bal + hint + 0.15, 3)))
        elif c.kind == ColumnKind.NUMERIC and c.n_unique > 10:
            sk = abs(c.skew or 0.0)
            spread = 0.0
            if c.std and c.mean is not None and abs(c.mean) > 1e-9:
                spread = min(1.0, abs(c.std / c.mean))
            valhint = 0.3 if _VALUE_HINTS.search(c.name) else 0.0
            out.append(TargetCandidate(
                column=c.name, task="regression",
                reason=(f"Continuous numeric ({c.n_unique:,} distinct values)"
                        + (", name hints at a business value" if valhint
                           else "")),
                score=round(min(0.85, 0.45 + 0.25 * spread
                                + 0.10 * (1 - min(sk / 3, 1)) + valhint), 3)))
    out.sort(key=lambda t: -t.score)
    return out[:8]


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame, name: str = "dataset",
                    fingerprint: str = "") -> DataProfile:
    """Profile a DataFrame: schema, quality issues, target candidates."""
    n_rows = int(df.shape[0])
    cols = [_profile_column(str(c), df[c], n_rows) for c in df.columns]
    issues, dup = _quality_issues(df, cols, n_rows)
    targets = _target_candidates(df, cols)
    try:
        mem = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    except Exception:                                        # pragma: no cover
        mem = 0.0
    return DataProfile(
        dataset_name=name, fingerprint=fingerprint, rows=n_rows,
        cols=int(df.shape[1]), columns=cols, issues=issues,
        duplicate_rows=dup,
        duplicate_pct=round(100 * dup / max(n_rows, 1), 2),
        id_like_columns=[c.name for c in cols if c.id_like],
        sensitive_columns=[c.name for c in cols if c.sensitive],
        target_candidates=targets, memory_mb=round(mem, 2),
    )


def profile_dataset_obj(dataset) -> DataProfile:
    """Profile an ``adip.ingest.Dataset`` object."""
    return profile_dataset(dataset.df, dataset.name, dataset.fingerprint)

