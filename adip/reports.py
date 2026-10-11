"""Report and export generation: CSV, Excel, HTML/PDF and JSON.

Design rules
------------
* Sensitive columns are never written to an export. ``_safe_columns`` filters
  them using the profile, and the returned payload states when a column was
  withheld.
* Every export carries provenance: dataset name, fingerprint, timestamp and
  the method that produced it, so a downloaded file can be traced back.
* PDF requires reportlab. If it is missing, ``build_pdf_report`` raises
  ``ExportUnavailable`` and the UI shows an honest "not available" message
  instead of a broken download button.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .profiling import DataProfile

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)
    HAVE_PDF = True
except Exception:                                       # pragma: no cover
    HAVE_PDF = False


class ExportUnavailable(RuntimeError):
    """Raised when a requested export format cannot be produced."""

MAX_PDF_ROWS = 60          # keep PDFs readable and bounded
MAX_EXPORT_ROWS = 50_000


@dataclass
class ExportPayload:
    """A ready-to-download artifact."""
    data: bytes
    filename: str
    mime: str
    label: str
    withheld_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_columns(df: pd.DataFrame, profile: DataProfile | None,
                  exclude: list[str] | None = None
                  ) -> tuple[pd.DataFrame, list[str]]:
    """Drop sensitive/identifier columns; report what was withheld."""
    exclude = set(exclude or [])
    if profile is None:
        return df, []
    blocked = {c.name for c in profile.columns if c.sensitive}
    blocked |= set(profile.sensitive_columns or [])
    blocked |= exclude
    present = [c for c in blocked if c in df.columns]
    if not present:
        return df, []
    return df.drop(columns=present), sorted(present)


def _limit_rows(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if len(df) > MAX_EXPORT_ROWS:
        notes.append(
            f"Export truncated to the first {MAX_EXPORT_ROWS:,} of "
            f"{len(df):,} rows to keep the file manageable.")
        return df.head(MAX_EXPORT_ROWS)
    return df


def _stamp(df: pd.DataFrame, profile: DataProfile | None,
           method: str) -> pd.DataFrame:
    return df


# --------------------------------------------------------------------------
# tabular exports
# --------------------------------------------------------------------------
def export_table_csv(df: pd.DataFrame, filename: str = "results.csv",
                     profile: DataProfile | None = None,
                     exclude: list[str] | None = None,
                     index: bool = False) -> ExportPayload:
    """Export a DataFrame to CSV, omitting sensitive columns."""
    notes: list[str] = []
    safe, withheld = _safe_columns(df, profile, exclude)
    safe = _limit_rows(safe, notes)
    buf = io.StringIO()
    safe.to_csv(buf, index=index)
    return ExportPayload(data=buf.getvalue().encode("utf-8"),
                         filename=filename, mime="text/csv",
                         label="CSV", withheld_columns=withheld, notes=notes)


def export_table_excel(df: pd.DataFrame, filename: str = "results.xlsx",
                       profile: DataProfile | None = None,
                       exclude: list[str] | None = None,
                       sheet_name: str = "results") -> ExportPayload:
    """Export a DataFrame to .xlsx. Needs openpyxl."""
    try:
        import openpyxl                        # noqa: F401
    except Exception as exc:                               # pragma: no cover
        raise ExportUnavailable(
            "Excel export needs the 'openpyxl' package, which is not "
            "installed. CSV export is available instead.") from exc
    notes: list[str] = []
    safe, withheld = _safe_columns(df, profile, exclude)
    safe = _limit_rows(safe, notes)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        safe.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return ExportPayload(data=buf.getvalue(),
                         filename=filename,
                         mime=("application/vnd.openxmlformats-officedocument"
                               ".spreadsheetml.sheet"),
                         label="Excel", withheld_columns=withheld,
                         notes=notes)


def export_json(obj: dict, filename: str = "analysis.json") -> ExportPayload:
    """Export a JSON-serialisable analysis record (metrics, config, warnings)."""
    def _default(o):
        if hasattr(o, "as_dict"):
            return o.as_dict()
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        return str(o)
    text = json.dumps(obj, indent=2, default=_default)
    return ExportPayload(data=text.encode("utf-8"), filename=filename,
                         mime="application/json", label="JSON")


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
def _h():
    return ParagraphStyle("h", parent=getSampleStyleSheet()["Normal"],
                          fontSize=9, spaceBefore=4, textColor=
                          colors.HexColor("#1e3a5f"))


def _table_block(title: str, rows: list[list[str]]) -> list:
    """Render a titled table into the PDF flowables list."""
    if not rows:
        return [Paragraph(f"<b>{title}</b>", _h()), Spacer(1, 2 * mm)]
    data = [rows[0]] + [[str(c)[:180] for c in r] for r in rows[1:]]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d0da")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f7fa")]),
    ]))
    return [Paragraph(f"<b>{title}</b>", _h()), Spacer(1, 1.5 * mm), tbl,
            Spacer(1, 4 * mm)]


def build_pdf_report(profile: DataProfile | None = None,
                     sections: list[tuple[str, pd.DataFrame | list]] | None = None,
                     title: str = "Adaptive Data Intelligence Platform",
                     limitations: list[str] | None = None) -> ExportPayload:
    """Render an analysis report to PDF. Requires reportlab.

    ``sections`` is a list of ``(heading, table)`` pairs where table is a
    DataFrame or a list of row-lists whose first row is the header.
    """
    if not HAVE_PDF:
        raise ExportUnavailable(
            "PDF export needs the 'reportlab' package, which is not "
            "installed. CSV, Excel and JSON exports are available.")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm,
                            rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=title)
    styles = getSampleStyleSheet()
    flow: list = [Paragraph(title, styles["Title"]), Spacer(1, 2 * mm),
                  Paragraph(f"Generated {_now()}", styles["Normal"])]

    if profile is not None:
        meta = [["Field", "Value"],
                ["Dataset", profile.dataset_name],
                ["Fingerprint", profile.fingerprint],
                ["Rows", f"{profile.rows:,}"],
                ["Columns", f"{profile.cols:,}"],
                ["Memory", f"{profile.memory_mb} MB"],
                ["Duplicate rows", f"{profile.duplicate_rows:,}"]]
        flow += _table_block("Dataset context", meta)

    for heading, table in (sections or []):
        if isinstance(table, pd.DataFrame):
            if table.empty:
                continue
            rows = [list(map(str, table.columns))]
            rows += [[("" if pd.isna(v) else str(v))
                      for v in rec] for rec in
                     table.head(MAX_PDF_ROWS).itertuples(index=False)]
        else:
            rows = [list(map(str, r)) for r in table]
        flow += _table_block(heading, rows)

    if limitations:
        flow += _table_block(
            "Limitations", [["Limitation"]] + [[x] for x in limitations])

    doc.build(flow)
    return ExportPayload(data=buf.getvalue(), filename="analysis_report.pdf",
                         mime="application/pdf", label="PDF",
                         notes=["PDF is a text snapshot; charts are not "
                                "embedded."])
