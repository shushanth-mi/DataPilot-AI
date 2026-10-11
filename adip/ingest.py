"""Safe dataset ingestion: CSV / XLSX with validation and safeguards.

Design rules
------------
* Never silently change the meaning of data. Every transform that alters the
  frame (truncation, sheet selection) is recorded in ``Dataset.warnings``.
* Malformed, empty or oversized inputs raise ``IngestionError`` carrying a
  message that is safe to show directly to a user.
* No network access, no macro execution, no code evaluation happens here.
* Uploaded bytes are only ever parsed by pandas readers.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, BinaryIO, Union

import pandas as pd

MAX_ROWS = 200_000          # deterministic-sample safeguard
MAX_COLS = 1_000
MAX_BYTES = 100 * 1024 * 1024   # 100 MB upload safeguard
_SAMPLE_SEED = 42
Source = Union[str, Path, bytes, BinaryIO, Any]


class IngestionError(ValueError):
    """A user-facing ingestion failure (safe to display in the UI)."""


@dataclass
class Dataset:
    """An ingested dataset exactly as provided, plus ingestion metadata."""

    df: pd.DataFrame
    name: str
    source: str                 # "csv" | "xlsx" | "sample"
    fingerprint: str
    warnings: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.df.shape[0])

    @property
    def cols(self) -> int:
        return int(self.df.shape[1])


def fingerprint(df: pd.DataFrame, name: str = "", n: int = 200) -> str:
    """Deterministic short fingerprint: shape + columns + first n rows."""
    h = sha1()
    h.update(f"{name}|{df.shape}|{'|'.join(map(str, df.columns))}".encode())
    try:
        h.update(df.head(n).to_csv(index=False).encode())
    except Exception:                                   # pragma: no cover
        h.update(repr(list(df.columns)).encode())
    return h.hexdigest()[:16]


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
        except pd.errors.ParserError as exc:
            raise IngestionError(
                "This file could not be parsed as CSV "
                f"({exc}). Check for unbalanced quotes or inconsistent "
                "column counts."
            ) from exc
        except pd.errors.EmptyDataError as exc:
            # Distinguish a header-only file (column names, no data rows)
            # from a genuinely empty file, because the fix differs.
            names = pd.read_csv(io.BytesIO(data), encoding=enc,
                                nrows=0).columns
            if len(names):
                raise IngestionError(
                    "The CSV file has a header row but no data rows. Please "
                    "upload a file with at least one row of data."
                ) from exc
            raise IngestionError(
                "The CSV file is empty. Please upload a file with a header row "
                "and at least one data row."
            ) from exc
    raise IngestionError(
        "Could not decode the CSV text (unknown encoding). Try saving it as "
        "UTF-8 CSV."
    ) from last

def _read_xlsx_bytes(data: bytes, sheet_name: Any = 0) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name,
                             engine="openpyxl")
    except IngestionError:                              # pragma: no cover
        raise
    except Exception as exc:
        raise IngestionError(
            f"This workbook could not be read ({type(exc).__name__}: {exc}). "
            "Supported format: .xlsx (Excel 2010+)."
        ) from exc


def _safeguards(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Enforce size limits and record every alteration that was applied."""
    if df.shape[1] == 0:
        raise IngestionError(
            "The dataset has no columns: it must contain at least one column "
            "and one row of data."
        )
    if df.shape[0] == 0:
        raise IngestionError(
            "The file has a header row but no data rows. Please upload a file "
            "with at least one row of data."
        )
    if df.shape[1] > MAX_COLS:
        raise IngestionError(
            f"This dataset has {df.shape[1]:,} columns; the prototype supports "
            f"up to {MAX_COLS:,}."
        )
    if df.shape[0] > MAX_ROWS:
        before = df.shape[0]
        df = df.sample(n=MAX_ROWS, random_state=_SAMPLE_SEED).sort_index()
        warnings.append(
            f"Large-file safeguard: {before:,} rows reduced to {MAX_ROWS:,} "
            f"using a deterministic sample (seed {_SAMPLE_SEED})."
        )
    mangled = [c for c in df.columns
               if isinstance(c, str) and re.search(r"\.\d+$", c)]
    if mangled:
        warnings.append(
            "Column names were auto-renamed by the parser because of "
            f"duplicates: {', '.join(map(str, mangled[:5]))}"
            + (" ..." if len(mangled) > 5 else "")
        )
    return df


def load_table(source: Source, name: str | None = None,
               sheet_name: Any = 0) -> Dataset:
    """Load a CSV or XLSX from a path, raw bytes or a file-like object."""
    warnings: list[str] = []
    data: bytes | None = None
    inferred = "csv"

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise IngestionError(f"File not found: {p}")
        if p.stat().st_size > MAX_BYTES:
            raise IngestionError(
                f"File is larger than the {MAX_BYTES // (1024 * 1024)} MB limit."
            )
        inferred = "xlsx" if p.suffix.lower() in (".xlsx", ".xlsm") else "csv"
        data = p.read_bytes()
        name = name or p.name
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = name or "uploaded-file"
    else:                                    # file-like (e.g. Streamlit)
        raw = getattr(source, "getvalue", None)
        data = raw() if callable(raw) else source.read()
        name = name or getattr(source, "name", "uploaded-file") or "uploaded-file"
        if str(name).lower().endswith((".xlsx", ".xlsm")):
            inferred = "xlsx"

    if not data:
        raise IngestionError(
            "The uploaded file is empty (0 bytes). Please upload a file with a "
            "header row and at least one data row."
        )

    df = _read_xlsx_bytes(data, sheet_name) if inferred == "xlsx" \
        else _read_csv_bytes(data)
    df = _safeguards(df, warnings)
    return Dataset(df=df, name=str(name), source=inferred,
                   fingerprint=fingerprint(df, str(name)), warnings=warnings)


def load_frame(df: pd.DataFrame, name: str = "dataframe",
               source: str = "sample") -> Dataset:
    """Wrap an in-memory DataFrame (used by tests and the sample generator)."""
    warnings: list[str] = []
    df = _safeguards(df.copy(), warnings)
    return Dataset(df=df, name=name, source=source,
                   fingerprint=fingerprint(df, name), warnings=warnings)

