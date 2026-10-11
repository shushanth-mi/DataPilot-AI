"""Reusable presentational helpers.

Everything here is presentation only: it never mutates data, never calls the
analytics engine, and never fabricates a value. Each helper returns HTML that
is styled by a class we own in :mod:`ui.theme`, so the DOM structure stays
stable across Streamlit versions.

All dynamic text is HTML-escaped before insertion, because column names and
values come from user uploads and could contain markup.
"""
from __future__ import annotations

import html
import math
from typing import Iterable, Sequence

import pandas as pd

from . import theme


def esc(value) -> str:
    """HTML-escape any value for safe interpolation."""
    return html.escape(str(value), quote=True)


def _fmt_number(v) -> str:
    """Human-readable number formatting without overstating precision."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _html(markup: str) -> None:
    """Write raw HTML into the page via Streamlit's sanctioned bridge."""
    import streamlit as st
    st.markdown(markup, unsafe_allow_html=True)


def page_head(title: str, subtitle: str = "") -> None:
    """Render a consistent page title + optional subtitle."""
    sub = f'<p class="adip-page-sub">{esc(subtitle)}</p>' if subtitle else ""
    _html(f'<div class="adip-page-head"><h1 class="adip-page-title">'
          f'{esc(title)}</h1>{sub}</div>')


def section_label(text: str) -> None:
    """A small uppercase label that groups related controls."""
    _html(f'<p class="adip-section-label">{esc(text)}</p>')


def card_open(title: str = "", sub: str = "", icon: str = "",
              tight: bool = False) -> None:
    """Open a bordered card. Always pair with :func:`card_close`."""
    cls = "adip-card adip-card-tight" if tight else "adip-card"
    head = ""
    if title or icon:
        ic = f'<span aria-hidden="true">{esc(icon)}</span>' if icon else ""
        head += f'<p class="adip-card-title">{ic}{esc(title)}</p>'
    if sub:
        head += f'<p class="adip-card-sub">{esc(sub)}</p>'
    _html(f'<div class="{cls}">{head}')


def card_close() -> None:
    _html("</div>")


def divider() -> None:
    _html('<hr class="adip-divider" />')


def kpis(items: Sequence[tuple]) -> None:
    """Render a responsive row of KPI cards.

    ``items`` are ``(label, value, note)`` tuples; ``note`` may be empty.
    """
    cells = []
    for label, value, note in items:
        n = f'<div class="adip-kpi-note">{esc(note)}</div>' if note else ""
        cells.append(
            f'<div class="adip-kpi"><div class="adip-kpi-label">'
            f'{esc(label)}</div><div class="adip-kpi-value">'
            f'{esc(_fmt_number(value))}</div>{n}</div>')
    _html(f'<div class="adip-kpis">{"".join(cells)}</div>')


def tag(text: str, tone: str = "muted") -> str:
    """Return an inline pill tag (span), e.g. for statuses or task types."""
    return (f'<span class="adip-tag adip-tag-{esc(tone)}">'
            f'{esc(text)}</span>')





def banner(title: str, body: str = "", tone: str = "limited",
           bullets: Iterable[str] = ()) -> None:
    """A coloured status banner. ``tone`` maps to :data:`theme.STATUS_COLORS`.

    Used for reliability verdicts, warnings, and errors. ``bullets`` render as
    a list under the body.
    """
    text, bg, border = theme.status_colors(tone)
    lis = "".join(f"<li>{esc(b)}</li>" for b in bullets)
    ul = f"<ul>{lis}</ul>" if lis else ""
    b = f"<div>{esc(body)}</div>" if body else ""
    _html(
        f'<div class="adip-banner" style="color:{text};background:{bg};'
        f'border-color:{border}">'
        f'<div class="adip-banner-title">{esc(title)}</div>{b}{ul}</div>')


def finding(title: str, body: str, evidence: str = "", method: str = "",
            columns: str = "", uncertainty: str = "", limitation: str = "",
            next_step: str = "") -> None:
    """Render a discovery finding with its evidence and caveats.

    A finding is only as strong as the evidence shown, so every field the
    caller passes is surfaced and nothing is invented here.
    """
    meta = []
    if method:
        meta.append(f"<b>Method</b> {esc(method)}")
    if columns:
        meta.append(f"<b>Columns</b> {esc(columns)}")
    if uncertainty:
        meta.append(f"<b>Stability</b> {esc(uncertainty)}")
    if limitation:
        meta.append(f"<b>Limitation</b> {esc(limitation)}")
    if next_step:
        meta.append(f"<b>Next step</b> {esc(next_step)}")
    meta_html = (f'<div class="adip-finding-meta">{" · ".join(meta)}</div>'
                 if meta else "")
    ev = (f'<div class="adip-evidence">{esc(evidence)}</div>'
          if evidence else "")
    _html(
        f'<div class="adip-finding"><div class="adip-finding-title">'
        f'{esc(title)}</div><div class="adip-finding-body">{esc(body)}'
        f'</div>{ev}{meta_html}</div>')


def empty_state(title: str, body: str, icon: str = "🗂️") -> None:
    """A friendly, honest empty state used before a dataset is loaded."""
    _html(
        f'<div class="adip-empty"><div class="adip-empty-icon">'
        f'{esc(icon)}</div><div class="adip-empty-title">{esc(title)}</div>'
        f'<div class="adip-empty-body">{esc(body)}</div></div>')


def note(text: str) -> None:
    """A small italic caveat line under a chart or table."""
    _html(f'<div class="adip-note">{esc(text)}</div>')


def metric_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> pd.DataFrame:
    """Return a display copy of a results table with numbers formatted.

    Does not alter the caller's frame; floats are rounded to a readable
    precision so long decimals do not dominate the layout.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else float_fmt.format(v))
    return out

