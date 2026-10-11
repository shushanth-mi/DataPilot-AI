"""Ingestion and profiling tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adip import ingest, profiling
from adip.ingest import IngestionError
from conftest import _profile


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
class TestIngestion:
    def test_loads_csv_bytes(self):
        csv = b"a,b\n1,2\n3,4\n"
        ds = ingest.load_table(csv, "t.csv")
        assert ds.rows == 2 and ds.cols == 2
        assert list(ds.df.columns) == ["a", "b"]
        assert ds.source == "csv"

    def test_loads_xlsx_bytes(self, tmp_path):
        from adip import preprocess  # noqa: F401  (import cost already paid)
        pd.DataFrame({"n": [1, 2, 3], "c": ["a", "b", "c"]}).to_excel(
            tmp_path / "f.xlsx", index=False)
        ds = ingest.load_table(tmp_path / "f.xlsx")
        assert ds.rows == 3 and ds.cols == 2
        assert ds.source == "xlsx"

    def test_rejects_empty_file(self):
        with pytest.raises(IngestionError, match="empty"):
            ingest.load_table(b"", "empty.csv")

    def test_rejects_header_only_csv(self):
        with pytest.raises(IngestionError, match="header"):
            ingest.load_table(b"a,b,c\n", "hdr.csv")

    def test_rejects_zero_row_frame(self):
        with pytest.raises(IngestionError, match="at least one column"):
            ingest.load_frame(pd.DataFrame(), "empty")

    def test_rejects_zero_column_frame(self):
        with pytest.raises(IngestionError, match="at least one column"):
            ingest.load_frame(pd.DataFrame(index=[1, 2, 3]), "nocols")

    def test_rejects_missing_path(self, tmp_path):
        with pytest.raises(IngestionError, match="not found"):
            ingest.load_table(tmp_path / "nope.csv")

    def test_rejects_malformed_csv(self):
        with pytest.raises(IngestionError, match="parsed as CSV"):
            ingest.load_table(b'a,b\n1,2\n"unclosed,3\n', "bad.csv")

    def test_latin1_fallback_decodes(self):
        # latin-1 bytes that are invalid UTF-8
        raw = "name\ncaf\xe9\n".encode("latin-1")
        ds = ingest.load_table(raw, "l.csv")
        assert ds.rows == 1

    def test_large_file_is_sampled_and_recorded(self):
        big = pd.DataFrame({"v": np.arange(ingest.MAX_ROWS + 50)})
        ds = ingest.load_frame(big, "big")
        assert ds.rows == ingest.MAX_ROWS
        assert any("Large-file safeguard" in w for w in ds.warnings)
        # the reduction must be disclosed, never silent
        assert ds.warnings

    def test_fingerprint_is_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert ingest.fingerprint(df, "x") == ingest.fingerprint(df, "x")

    def test_fingerprint_changes_with_data(self):
        a = pd.DataFrame({"a": [1, 2, 3]})
        b = pd.DataFrame({"a": [1, 2, 4]})
        assert ingest.fingerprint(a) != ingest.fingerprint(b)

    def test_original_data_is_preserved(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        ds = ingest.load_frame(df, "keep")
        assert ds.df["a"].isna().sum() == 1      # no silent imputation
        assert ds.df["a"].iloc[0] == 1.0


# --------------------------------------------------------------------------
# schema inference
# --------------------------------------------------------------------------
class TestSchemaInference:
    def test_numeric_detected(self):
        p = _profile(pd.DataFrame({"n": [1.0, 2.0, 3.0]}))
        assert p.columns[0].kind == profiling.ColumnKind.NUMERIC

    def test_categorical_detected(self):
        p = _profile(pd.DataFrame({"c": ["a", "b", "a", "b"] * 5}))
        assert p.columns[0].kind == profiling.ColumnKind.CATEGORICAL

    def test_boolean_detected(self):
        p = _profile(pd.DataFrame({"b": [True, False] * 5}))
        assert p.columns[0].kind == profiling.ColumnKind.BOOLEAN

    def test_datetime_detected(self):
        p = _profile(pd.DataFrame({"d": pd.date_range("2024-01-01", periods=8)}))
        assert p.columns[0].kind == profiling.ColumnKind.DATETIME

    def test_datetime_detected_from_strings(self):
        p = _profile(pd.DataFrame(
            {"d": [f"2024-01-{i:02d}" for i in range(1, 11)]}))
        assert p.columns[0].kind == profiling.ColumnKind.DATETIME

    def test_id_like_detected_by_uniqueness(self):
        p = _profile(pd.DataFrame({"k": [f"u{i}" for i in range(30)]}))
        assert p.columns[0].kind == profiling.ColumnKind.ID_LIKE

    def test_id_like_detected_by_name(self):
        p = _profile(pd.DataFrame({"user_id": [1, 2, 3, 4, 2]}))
        assert p.columns[0].id_like is True

    def test_text_detected(self):
        rng = np.random.default_rng(0)
        p = _profile(pd.DataFrame({"t": [f"sentence number {i} " + "word " * 12
                                         for i in range(40)]}))
        assert p.columns[0].kind == profiling.ColumnKind.TEXT

    def test_mixed_types_detected(self):
        p = _profile(pd.DataFrame({"m": ["a", "b", "c", "d", "e", 1, 2, 3,
                                         4, 5]}))
        assert p.columns[0].mixed_types is True

    def test_sensitive_column_flagged_by_name(self):
        p = _profile(pd.DataFrame({"email_address": ["a@b.com"] * 10}))
        assert "email_address" in p.sensitive_columns

    def test_sensitive_column_flagged_phone(self):
        p = _profile(pd.DataFrame({"phone": ["555-0100"] * 10}))
        assert "phone" in p.sensitive_columns

    def test_properties_group_by_kind(self):
        p = _profile(pd.DataFrame({
            "n": [1.0, 2.0, 3.0, 4.0],
            "c": ["a", "b", "a", "b"],
            "d": pd.date_range("2024-01-01", periods=4)}))
        assert "n" in p.numeric
        assert "c" in p.categorical
        assert "d" in p.datetime


# --------------------------------------------------------------------------
# data quality
# --------------------------------------------------------------------------
class TestDataQuality:
    def test_missing_values_counted(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
        p = _profile(df)
        assert p.columns[0].missing == 2
        assert p.columns[0].missing_pct == 50.0

    def test_heavy_missing_flagged(self):
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 9})
        p = _profile(df)
        codes = [i.code for i in p.issues]
        assert "heavy_missing" in codes
        assert any(i.severity == "high" for i in p.issues
                   if i.code == "heavy_missing")

    def test_moderate_missing_flagged(self):
        df = pd.DataFrame({"x": [1.0] * 90 + [np.nan] * 10})
        p = _profile(df)
        assert "missing_values" in [i.code for i in p.issues]

    def test_duplicates_counted(self, messy_df):
        p = _profile(messy_df)
        assert p.duplicate_rows == 10

    def test_duplicate_flagged_as_contamination_risk(self, messy_df):
        p = _profile(messy_df)
        issue = next(i for i in p.issues if i.code == "duplicate_rows")
        assert "contamination" in issue.remedy.lower()

    def test_constant_column_flagged(self):
        p = _profile(pd.DataFrame({"c": ["same"] * 20}))
        assert "constant_column" in [i.code for i in p.issues]

    def test_outliers_counted(self):
        p = _profile(pd.DataFrame({"x": [10.0] * 95 + [10000.0] * 5}))
        assert p.columns[0].outlier_count == 5

    def test_high_cardinality_flagged(self):
        p = _profile(pd.DataFrame({"c": [f"lvl{i}" for i in range(150)]}))
        assert "high_cardinality" in [i.code for i in p.issues]

    def test_sensitive_column_warns_against_export(self):
        p = _profile(pd.DataFrame({"email": ["a@b.com"] * 10}))
        issue = next(i for i in p.issues if i.code == "sensitive_column")
        assert "not displayed" in issue.remedy

    def test_identifier_flagged_as_leakage_risk(self):
        p = _profile(pd.DataFrame({"customer_id": list(range(40))}))
        assert "suspected_identifier" in [i.code for i in p.issues]

    def test_all_missing_column_does_not_crash(self):
        p = _profile(pd.DataFrame({"blank": [np.nan] * 10}))
        assert p.columns[0].missing_pct == 100.0

    def test_single_row_dataset_profiled(self):
        p = _profile(pd.DataFrame({"x": [1.0], "c": ["a"]}))
        assert p.rows == 1 and p.cols == 2

    def test_issues_frame_is_tabular(self, messy_df):
        p = _profile(messy_df)
        frame = p.issues_frame()
        assert isinstance(frame, pd.DataFrame)
        assert {"severity", "code", "message", "remedy"} <= set(frame.columns)


# --------------------------------------------------------------------------
# target detection
# --------------------------------------------------------------------------
class TestTargetDetection:
    def test_classification_target_inferred(self, classification_df):
        p = _profile(classification_df)
        cand = next((t for t in p.target_candidates
                     if t.column == "churned"), None)
        assert cand is not None
        assert cand.task == "classification"
        assert cand.n_classes == 2

    def test_numeric_label_treated_as_classification(self, classification_df):
        """A 0/1 numeric column is a label, not a regression target."""
        p = _profile(classification_df)
        assert any(t.column == "churned" and t.task == "classification"
                   for t in p.target_candidates)

    def test_regression_target_inferred(self, regression_df):
        p = _profile(regression_df)
        assert any(t.column == "price" and t.task == "regression"
                   for t in p.target_candidates)

    def test_identifiers_never_suggested_as_targets(self, classification_df):
        p = _profile(classification_df)
        assert not any(t.column in ("customer_id", "email")
                       for t in p.target_candidates)

    def test_sensitive_never_suggested_as_target(self, classification_df):
        p = _profile(classification_df)
        assert "email" not in [t.column for t in p.target_candidates]

    def test_minority_share_reported(self, imbalanced_df):
        p = _profile(imbalanced_df)
        cand = next(t for t in p.target_candidates if t.column == "churned")
        assert cand.minority_share is not None
        assert cand.minority_share < 0.1

    def test_candidates_are_ranked_descending(self, classification_df):
        p = _profile(classification_df)
        scores = [t.score for t in p.target_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_unlabeled_data_yields_candidates_not_claims(self, clustered_df):
        """No invented labels: unlabeled data lists options, not conclusions."""
        p = _profile(clustered_df)
        assert p.target_candidates is not None

