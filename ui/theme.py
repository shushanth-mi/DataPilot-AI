"""Design tokens and the single stylesheet for the whole application.

One stylesheet is injected once per run. Every rule is scoped to a custom
class that we own, never to Streamlit's internal DOM structure, so a
Streamlit upgrade cannot silently break the layout.
"""
from __future__ import annotations

import streamlit as st

# --- palette ---------------------------------------------------------------
NAVY = "#0f172a"          # sidebar / chrome
NAVY_SOFT = "#1e293b"
INK = "#0f172a"           # primary text
INK_MUTED = "#64748b"     # secondary text
SURFACE = "#ffffff"
CANVAS = "#f6f8fb"        # app background
BORDER = "#e2e8f0"
BLUE = "#2563eb"          # primary accent
BLUE_SOFT = "#eff6ff"
PURPLE = "#7c3aed"        # restrained secondary accent
TEAL = "#0d9488"          # tertiary accent
AMBER = "#b45309"
AMBER_SOFT = "#fffbeb"
RED = "#b91c1c"
RED_SOFT = "#fef2f2"
GREEN = "#047857"
GREEN_SOFT = "#ecfdf5"

#: status -> (text, background, border) for reliability banners
STATUS_COLORS = {
    "suitable": (GREEN, GREEN_SOFT, "#a7f3d0"),
    "promising": (BLUE, BLUE_SOFT, "#bfdbfe"),
    "limited": (AMBER, AMBER_SOFT, "#fde68a"),
    "insufficient": (RED, RED_SOFT, "#fecaca"),
}

#: Plotly sequence used by every chart so colours stay consistent
CHART_COLORS = [BLUE, TEAL, PURPLE, "#0891b2", "#db2777", "#ca8a04",
                "#4f46e5", "#059669"]

RADIUS = "10px"


def inject() -> None:
    """Inject the application stylesheet once per session."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def status_colors(status: str) -> tuple[str, str, str]:
    """Return (text, bg, border) for a reliability status, defaulting safely."""
    return STATUS_COLORS.get((status or "").lower(),
                             STATUS_COLORS["limited"])


_CSS = f"""
:root {{
  --navy: {NAVY}; --ink: {INK}; --ink-muted: {INK_MUTED};
  --surface: {SURFACE}; --canvas: {CANVAS}; --border: {BORDER};
  --blue: {BLUE}; --blue-soft: {BLUE_SOFT}; --purple: {PURPLE};
  --teal: {TEAL}; --radius: {RADIUS};
}}
html, body, [class*="css"] {{
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}}
.stApp {{ background: {CANVAS}; }}
section[data-testid="stSidebar"] > div {{ background: {NAVY};
  padding-top: 1.25rem; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.6rem; font-weight: 700; }}
h2 {{ font-size: 1.15rem; font-weight: 650; }}
h3 {{ font-size: 1rem; font-weight: 650; }}
p, li {{ color: {INK}; font-size: 0.92rem; line-height: 1.55; }}
small, .adip-muted {{ color: {INK_MUTED}; font-size: 0.82rem; }}

.adip-brand {{ display:flex; align-items:center; gap:.6rem;
  padding:.25rem .25rem 1rem; }}
.adip-logo {{ width:32px; height:32px; border-radius:9px; background:{BLUE};
  display:flex; align-items:center; justify-content:center; color:#fff;
  font-weight:700; font-size:.95rem; }}
.adip-brand-name {{ color:#fff; font-size:1.05rem; font-weight:700;
  line-height:1.1; }}
.adip-brand-sub {{ color:#94a3b8; font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; }}
.adip-nav-label {{ color:#94a3b8; font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; padding:.5rem .5rem .35rem; font-weight:600; }}
.adip-status {{ background:{NAVY_SOFT}; border:1px solid #334155;
  border-radius:{RADIUS}; padding:.6rem .7rem; margin:.5rem 0 1rem; }}
.adip-status-row {{ display:flex; align-items:center; gap:.45rem;
  font-size:.8rem; color:#e2e8f0; }}
.adip-dot {{ width:8px; height:8px; border-radius:50%; flex:none; }}

.adip-card {{ background:{SURFACE}; border:1px solid {BORDER};
  border-radius:{RADIUS}; padding:1.05rem 1.15rem; margin-bottom:.85rem; }}
.adip-card-tight {{ padding:.8rem .9rem; }}
.adip-card-title {{ font-size:.95rem; font-weight:650; color:{INK};
  margin:0 0 .35rem; display:flex; align-items:center; gap:.45rem; }}
.adip-card-sub {{ color:{INK_MUTED}; font-size:.8rem; margin:0 0 .7rem;
  line-height:1.5; }}
.adip-page-head {{ margin-bottom:1.1rem; }}
.adip-page-title {{ font-size:1.5rem; font-weight:700; color:{INK};
  margin:0; letter-spacing:-.02em; }}
.adip-page-sub {{ color:{INK_MUTED}; font-size:.9rem; margin:.25rem 0 0;
  line-height:1.5; }}
.adip-kpis {{ display:grid; gap:.7rem;
  grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }}
.adip-kpi {{ background:{SURFACE}; border:1px solid {BORDER};
  border-radius:{RADIUS}; padding:.85rem .9rem; }}
.adip-kpi-label {{ color:{INK_MUTED}; font-size:.72rem; font-weight:600;
  letter-spacing:.04em; text-transform:uppercase; }}
.adip-kpi-value {{ color:{INK}; font-size:1.5rem; font-weight:700;
  line-height:1.25; margin-top:.15rem; }}
.adip-kpi-note {{ color:{INK_MUTED}; font-size:.75rem; margin-top:.1rem; }}
.adip-banner {{ border-radius:{RADIUS}; padding:.8rem .95rem; border:1px solid;
  margin-bottom:.85rem; font-size:.88rem; line-height:1.55; }}
.adip-banner-title {{ font-weight:650; margin-bottom:.2rem; display:flex;
  align-items:center; gap:.4rem; }}
.adip-banner ul {{ margin:.35rem 0 0; padding-left:1.15rem; }}
.adip-banner li {{ font-size:.85rem; margin-bottom:.2rem; }}
"""

_CSS += f"""
.adip-tag {{ display:inline-block; padding:.14rem .5rem; border-radius:999px;
  font-size:.72rem; font-weight:600; border:1px solid; margin:0 .3rem .3rem 0; }}
.adip-tag-blue {{ color:{BLUE}; background:{BLUE_SOFT}; border-color:#bfdbfe; }}
.adip-tag-teal {{ color:{TEAL}; background:#f0fdfa; border-color:#99f6e4; }}
.adip-tag-purple {{ color:{PURPLE}; background:#f5f3ff; border-color:#ddd6fe; }}
.adip-tag-red {{ color:{RED}; background:{RED_SOFT}; border-color:#fecaca; }}
.adip-tag-amber {{ color:{AMBER}; background:{AMBER_SOFT};
  border-color:#fde68a; }}
.adip-tag-green {{ color:{GREEN}; background:{GREEN_SOFT};
  border-color:#a7f3d0; }}
.adip-tag-muted {{ color:{INK_MUTED}; background:#f8fafc;
  border-color:{BORDER}; }}

.adip-finding {{ background:{SURFACE}; border:1px solid {BORDER};
  border-left:3px solid {BLUE}; border-radius:{RADIUS};
  padding:.9rem 1rem; margin-bottom:.7rem; }}
.adip-finding-title {{ font-weight:650; font-size:.93rem; color:{INK};
  margin:0 0 .3rem; }}
.adip-finding-body {{ color:{INK}; font-size:.87rem; line-height:1.55;
  margin:0 0 .45rem; }}
.adip-evidence {{ background:#f8fafc; border:1px solid {BORDER};
  border-radius:8px; padding:.55rem .7rem; font-size:.82rem; color:{INK};
  margin-bottom:.45rem; }}
.adip-finding-meta {{ font-size:.78rem; color:{INK_MUTED};
  border-top:1px dashed {BORDER}; padding-top:.45rem; margin-top:.4rem; }}
.adip-finding-meta b {{ color:{INK}; font-weight:600; }}

.adip-empty {{ border:1.5px dashed #cbd5e1; border-radius:12px;
  padding:2.2rem 1.5rem; text-align:center; background:{SURFACE}; }}
.adip-empty-icon {{ font-size:1.9rem; margin-bottom:.4rem; }}
.adip-empty-title {{ font-size:1.05rem; font-weight:650; color:{INK};
  margin:0 0 .3rem; }}
.adip-empty-body {{ color:{INK_MUTED}; font-size:.88rem; max-width:44ch;
  margin:0 auto; line-height:1.55; }}

.stButton > button {{ border-radius:9px; font-weight:600; font-size:.87rem;
  border:1px solid {BORDER}; padding:.42rem 1rem; }}
.stButton > button[kind="primary"] {{ background:{BLUE}; color:#fff;
  border-color:{BLUE}; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER};
  border-radius:{RADIUS}; padding:.7rem .8rem; }}

.adip-table {{ font-size:.85rem; }}
.adip-note {{ color:{INK_MUTED}; font-size:.78rem; font-style:italic;
  margin-top:.3rem; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER};
  border-radius:{RADIUS}; overflow:hidden; }}
.adip-divider {{ border:none; border-top:1px solid {BORDER};
  margin:1.1rem 0; }}
.adip-section-label {{ font-size:.7rem; font-weight:700; color:{INK_MUTED};
  letter-spacing:.09em; text-transform:uppercase; margin:0 0 .5rem; }}
@media (max-width: 720px) {{
  .adip-kpis {{ grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); }}
  .adip-page-title {{ font-size:1.3rem; }}
}}
"""

