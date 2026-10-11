"""Adaptive Data Intelligence Platform (ADIP) - Streamlit application.

This file is presentation and orchestration only. Every number, model,
finding and reliability verdict comes from the :mod:`adip` analytics
package; nothing here computes analytics or fabricates values. The UI's job
is to load a dataset, run the analysis engines the user selects, and present
the results honestly with their evidence and limitations.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adip import (discovery, explain, ingest, nlq, planner, preprocess,
                  profiling, reliability, reports, sample_data, scenario,
                  supervised, unsupervised)
from ui import components as C
from ui import theme

st.set_page_config(
    page_title="Adaptive Data Intelligence Platform",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()

# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------
_DEFAULTS = {
    "page": "Overview",
    "ds": None,              # ingest.Dataset
    "profile": None,         # profiling.DataProfile
    "plan": None,            # planner.AnalysisPlan
    "objective": "",
    "sup": None,             # supervised.SupervisedOutcome
    "sup_rel": None,         # reliability.ReliabilityReport
    "sup_expl": None,        # explain.explain_supervised dict
    "unsup": None,           # unsupervised.UnsupervisedOutcome
    "unsup_rel": None,       # reliability.ReliabilityReport
    "disc": None,            # discovery.DiscoveryReport
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PAGES = ["Overview", "Analysis Plan", "Supervised Learning",
         "Unsupervised Discovery", "Findings", "Explainability",
         "Scenario Explorer", "Ask Your Data", "Reports & Exports"]


def _reset_results() -> None:
    """Clear cached analysis when the dataset changes; results no longer apply."""
    for k in ("profile", "plan", "sup", "sup_rel", "sup_expl", "unsup",
              "unsup_rel", "disc"):
        st.session_state[k] = None


def _load_df(df: pd.DataFrame, name: str) -> None:
    """Adopt an in-memory DataFrame as the working dataset and re-profile."""
    ds = ingest.load_frame(df, name=name)
    _reset_results()
    st.session_state["ds"] = ds
    st.session_state["profile"] = profiling.profile_dataset_obj(ds)


def _dataframe() -> pd.DataFrame | None:
    ds = st.session_state.get("ds")
    return None if ds is None else ds.df


def _profile():
    return st.session_state.get("profile")


def _require_data() -> bool:
    """Guard pages that need a dataset; renders an empty state otherwise."""
    if st.session_state.get("ds") is None:
        C.empty_state(
            "No dataset loaded",
            "Upload a CSV or Excel file, or load a bundled sample dataset "
            "from the sidebar, to start an analysis.",
            icon="🗂️")
        return False
    return True

# --------------------------------------------------------------------------
# sidebar: branding, dataset status, load actions
# --------------------------------------------------------------------------
def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="adip-brand">'
            '<span class="adip-brand-mark" aria-hidden="true">◆</span>'
            '<span class="adip-brand-text">Adaptive Data<br>Intelligence'
            "</span></div>", unsafe_allow_html=True)
        st.caption("Profile · discover · model · verify")

        st.session_state["page"] = st.radio(
            "Navigate", PAGES, index=PAGES.index(st.session_state["page"]),
            label_visibility="collapsed")

        st.markdown("---")
        _sidebar_dataset()
        st.markdown("---")
        _sidebar_upload()

        st.caption(
            "Every figure shown is computed locally from your data. No "
            "network calls, no external AI service.")


def _sidebar_dataset() -> None:
    ds = st.session_state.get("ds")
    C.section_label("Dataset")
    if ds is None:
        st.caption("No dataset loaded yet.")
        return
    prof = _profile()
    st.markdown(f"**{C.esc(ds.name)}**")
    rows = f"{ds.rows:,}"
    cols = f"{ds.cols:,}"
    n_high = sum(1 for i in (prof.issues if prof else [])
                 if i.severity == "high")
    status = ("limited" if n_high else "promising")
    st.markdown(
        f'{C.tag("loaded", "promising")}&nbsp;{C.tag(rows + " rows", "muted")}'
        f'&nbsp;{C.tag(cols + " cols", "muted")}', unsafe_allow_html=True)
    if n_high:
        st.caption(f"{n_high} high-severity data-quality "
                   f"issue{'s' if n_high != 1 else ''} detected.")
    st.markdown(
        f'<p class="adip-fingerprint">fingerprint '
        f"{C.esc(ds.fingerprint)}</p>", unsafe_allow_html=True)
    st.caption("Run analyses from the pages listed above.")


def _sidebar_upload() -> None:
    C.section_label("Load data")
    with st.expander("Upload CSV or Excel", icon="📤"):
        up = st.file_uploader(
            "CSV or XLSX", type=["csv", "xlsx", "xlsm"],
            label_visibility="collapsed", key="_uploader",
            help="Parsed locally. Nothing leaves this machine.")
        if up is not None:
            _ingest_upload(up)
    with st.expander("Or load a sample dataset", icon="🧪"):
        for key, meta in sample_data.DATASETS.items():
            if st.button(meta["title"], key=f"_sample_{key}",
                         use_container_width=True):
                df, notes = sample_data.make_dataset(key)
                _load_df(df, f"{key} (sample)")
                st.session_state["_sample_notes"] = notes
                st.session_state["page"] = "Analysis Plan"
                st.rerun()

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    C.page_head("Overview",
                "What is in this dataset, and how much can it be trusted?")
    if not _require_data():
        _overview_empty()
        return

    prof = _profile()
    df = _dataframe()
    for w in st.session_state.get("_ingest_warnings", []) or []:
        C.banner("Ingestion notice", w, tone="limited")
    st.session_state["_ingest_warnings"] = []
    notes = st.session_state.get("_sample_notes")
    if notes:
        C.banner("Sample dataset", notes, tone="limited")

    _overview_kpis(prof)
    _overview_quality(prof)
    _overview_schema(prof)
    _overview_preview(df, prof)
    _overview_distributions(df, prof)


def _overview_empty() -> None:
    C.card_open("How this platform works", icon="🧭")
    st.markdown(
        "1. **Profile** the dataset - schema, quality, identifiers, targets.\n"
        "2. **Plan** - decide which analyses the data can actually support.\n"
        "3. **Model** - train and evaluate supervised models, or discover "
        "unsupervised structure.\n"
        "4. **Verify** - report reliability per result, with limitations.\n"
        "5. **Explain** - show the evidence behind each finding.")
    st.markdown(
        "Load a dataset from the sidebar to begin. A bundled synthetic "
        "dataset demonstrates each workflow.")
    C.card_close()


def _overview_kpis(prof) -> None:
    miss_total = sum(c.missing for c in prof.columns)
    miss_pct = 100 * miss_total / max(prof.rows * prof.cols, 1)
    numeric = prof.numeric
    C.section_label("Dataset at a glance")
    C.kpis([
        ("Rows", f"{prof.rows:,}", "records"),
        ("Columns", f"{prof.cols:,}",
         f"{len(numeric)} numeric · {len(prof.categorical)} categorical"),
        ("Missing cells", f"{miss_total:,}", f"{miss_pct:.2f}% of all cells"),
        ("Duplicate rows", f"{prof.duplicate_rows:,}",
         f"{prof.duplicate_pct:.2f}% of rows"),
        ("Memory", f"{prof.memory_mb:.1f} MB", "in-memory footprint"),
    ])
    if numeric:
        means = [c for c in prof.columns
                 if c.kind == "numeric" and c.mean is not None]
        if means:
            C.card_open("Numeric summaries", sub="Computed from the loaded "
                        "data; no values are imputed or altered.", icon="∑")
            st.dataframe(C.metric_table(prof.to_frame()), use_container_width=True,
                         hide_index=True)
            C.card_close()


def _overview_quality(prof) -> None:
    C.section_label("Data quality")
    if not prof.issues:
        C.banner("No data-quality issues detected",
                 "Profiling found no missing values, duplicates, constants, "
                 "outliers, identifiers or sensitive columns. That does not "
                 "guarantee the data is correct - only that these automated "
                 "checks found nothing to flag.", tone="promising")
    else:
        n_high = sum(1 for i in prof.issues if i.severity == "high")
        frame = prof.issues_frame()
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if n_high:
            C.note(f"{n_high} high-severity issue"
                   f"{'s' if n_high != 1 else ''} will constrain which "
                   "models are reliable. See the Analysis Plan page.")
    if prof.sensitive_columns:
        C.banner(
            "Sensitive columns detected",
            "These columns are never used as model features and their values "
            "are not displayed or exported: "
            + ", ".join(f"`{c}`" for c in prof.sensitive_columns),
            tone="limited")


def _overview_distributions(df, prof) -> None:
    """Plot the distributions that answer one question: is anything odd?"""
    numeric = prof.numeric[:3]
    cats = prof.categorical[:2]
    if not numeric and not cats:
        return
    C.section_label("Distributions")
    C.card_open("What do the values look like?",
                sub="Each panel answers one question about spread, skew or "
                    "imbalance.", icon="📊")
    if numeric:
        cols = st.columns(len(numeric))
        for col, name in zip(cols, numeric):
            with col:
                fig = go.Figure(go.Histogram(
                    x=df[name].dropna(), nbinsx=30,
                    marker_color="#2563eb", opacity=0.85))
                fig.update_layout(
                    title=dict(text=name, font_size=12), height=230,
                    margin=dict(l=8, r=8, t=36, b=8),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(size=10, color="#334155"),
                    xaxis=dict(title=""), yaxis=dict(title="rows", showgrid=True,
                                                     gridcolor="#eef2f7"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
    for name in cats:
        vc = df[name].value_counts().head(10)
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str), y=vc.values, marker_color="#7c3aed"))
        fig.update_layout(
            title=dict(text=f"{name} — top {len(vc)} levels", font_size=12),
            height=260, margin=dict(l=8, r=8, t=36, b=8),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(size=10, color="#334155"),
            xaxis=dict(title=""), yaxis=dict(title="rows", showgrid=True,
                                             gridcolor="#eef2f7"))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
    C.card_close()


# --------------------------------------------------------------------------
# Analysis plan
# --------------------------------------------------------------------------
def page_plan() -> None:
    C.page_head("Analysis Plan",
                "Which analyses this dataset can actually support, and why.")
    if not _require_data():
        return
    prof = _profile()

    C.card_open("What do you want to learn?", icon="🎯", tight=True)
    objective = st.text_input(
        "Objective (optional)", value=st.session_state.get("objective", ""),
        label_visibility="collapsed",
        placeholder="e.g. predict which customers will churn")
    if st.button("Build plan", type="primary"):
        st.session_state["objective"] = objective
        with st.spinner("Assessing the dataset..."):
            st.session_state["plan"] = planner.build_plan(prof, objective)
    C.card_close()

    plan = st.session_state.get("plan")
    if plan is None:
        C.empty_state("No plan built yet",
                      "Optionally describe what you want to learn, then build "
                      "the plan. It selects feasible analyses and explains "
                      "each choice.", icon="🧭")
        return
    _render_plan(plan)


def _render_plan(plan) -> None:
    C.section_label("Feasible analyses")
    feasible = [o for o in plan.options if o.feasible]
    blocked = [o for o in plan.options if not o.feasible]

    for opt in feasible:
        rec = opt.kind == plan.recommended
        C.card_open(
            opt.title + ("  —  recommended" if rec else ""),
            sub=opt.rationale,
            icon="✅" if rec else "•")
        if opt.needs_target:
            st.markdown(f"Target column: `{C.esc(opt.needs_target)}`")
        if rec:
            st.markdown(
                f'<p class="adip-rec">Recommended because it fits the '
                f"dataset's size, target type and data quality.</p>",
                unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if opt.kind in ("classification", "regression"):
                    if st.button("Run this analysis", type="primary",
                                 key="_go_sup"):
                        st.session_state["page"] = "Supervised Learning"
                        st.rerun()
            with c2:
                if st.button("Run discovery first", key="_go_disc"):
                    st.session_state["page"] = "Findings"
                    st.rerun()
        C.card_close()

    for opt in blocked:
        C.card_open(opt.title, icon="⛔")
        for b in opt.blockers:
            st.markdown(f"- {C.esc(b)}")
        C.card_close()

    if plan.assumptions:
        C.card_open("Assumptions this plan relies on", icon="📌")
        for a in plan.assumptions:
            st.markdown(f"- {C.esc(a)}")
        C.card_close()
    if plan.risks:
        C.card_open("Risks and limitations", icon="⚠️")
        for r in plan.risks:
            st.markdown(f"- {C.esc(r)}")
        C.card_close()
    if plan.missing_info:
        C.card_open("Information that would improve the analysis", icon="❓")
        for m in plan.missing_info:
            st.markdown(f"- {C.esc(m)}")
        C.card_close()
    if plan.steps:
        C.card_open("Reproducible execution steps", icon="🪜")
        for i, s in enumerate(plan.steps, 1):
            st.markdown(f"{i}. {C.esc(s)}")
        C.card_close()


# --------------------------------------------------------------------------
# Supervised learning
# --------------------------------------------------------------------------
def page_supervised() -> None:
    C.page_head("Supervised Learning",
                "Train, evaluate and compare models against a baseline.")
    if not _require_data():
        return
    prof = _profile()
    df = _dataframe()
    cands = prof.target_candidates

    if not cands:
        C.banner(
            "No usable target column detected",
            "Supervised learning needs a label or value to predict. This "
            "dataset has no candidate target, so try the Unsupervised "
            "Discovery page instead.", tone="limited")
        return

    C.card_open("1 · Choose what to predict", icon="🎯", tight=True)
    options = [f"{c.column}  →  {c.task}" for c in cands]
    pick = st.selectbox("Target column", options, label_visibility="collapsed")
    target = cands[options.index(pick)].column

    task_hint = "regression" if "→  regression" in pick else "classification"
    st.caption(cands[options.index(pick)].reason)

    exclude = st.multiselect(
        "Exclude extra columns from features",
        [c.name for c in prof.columns if c.name != target],
        default=[c for c in prof.sensitive_columns if c != target])
    temporal = st.checkbox("Time-aware split (data is time-ordered)", value=False)
    budget = st.slider("Training time budget (seconds)", 10, 300, 120, step=10)
    if st.button("2 · Train and evaluate", type="primary"):
        _run_supervised(df, prof, target, task_hint, exclude, temporal, budget)
    C.card_close()

    if st.session_state.get("sup") is None:
        C.empty_state("Nothing trained yet",
                      "Choose a target and run the evaluation to see real "
                      "held-out metrics.", icon="🧪")
        return

    out = st.session_state["sup"]
    rel = st.session_state.get("sup_rel")
    _render_reliability(rel)
    _render_supervised_results(out)


def _run_supervised(df, prof, target, task_hint, exclude, temporal,
                    budget) -> None:
    try:
        with st.spinner("Training candidates against a baseline..."):
            out = supervised.run_supervised(
                df, prof, target, task_hint=task_hint, drop=exclude,
                temporal=temporal, time_budget_s=float(budget))
            rel = reliability.assess_supervised(prof, out)
    except supervised.InsufficientDataError as exc:
        C.banner("This dataset cannot support that model",
                 str(exc) + " Reduce the task's ambition or collect more data.",
                 tone="insufficient")
        st.session_state["sup"] = None
        return
    except (KeyError, supervised.UnsupportedFeatureError, ValueError) as exc:
        C.banner("Could not train on this selection", str(exc),
                 tone="limited")
        st.session_state["sup"] = None
        return
    except Exception as exc:                                # pragma: no cover
        C.banner("Training failed unexpectedly",
                 f"{type(exc).__name__}: {exc}", tone="limited")
        st.session_state["sup"] = None
        return
    st.session_state["sup"] = out
    st.session_state["sup_rel"] = rel
    st.session_state["sup_expl"] = None
    st.rerun()


def _render_supervised_results(out) -> None:
    C.section_label("Model comparison")
    C.card_open("Candidates versus baseline",
                sub=f"Task: {out.task} · primary metric: "
                    f"{out.primary_metric}", icon="⚖️")
    st.caption(out.metric_reason)
    rows = []
    for r in ([out.baseline] if out.baseline else []) + out.candidates:
        if r is None:
            continue
        d = {"model": r.name, "ok": r.ok, "fit_s": round(r.fit_s, 3)}
        d.update(r.metrics)
        d["cv_mean"] = r.cv_mean
        d["cv_std"] = r.cv_std
        if not r.ok:
            d["error"] = r.error
        rows.append(d)
    comp = pd.DataFrame(rows)
    st.dataframe(C.metric_table(comp), use_container_width=True, hide_index=True)
    C.card_close()

    if out.best:
        C.section_label("Recommended model")
        C.card_open(out.best.name, sub="Selected on validation evidence, not "
                    "training accuracy.", icon="🏆")
        st.markdown("**Held-out test metrics**")
        st.dataframe(pd.DataFrame(
            [{"metric": k, "value": v} for k, v in out.best.metrics.items()]),
            use_container_width=True, hide_index=True)
        if out.best.best_params:
            C.note("Tuned hyperparameters: "
                   + ", ".join(f"{k}={v}" for k, v in
                               out.best.best_params.items()))
        C.note(f"Train rows: {out.train_rows:,} · test rows: {out.test_rows:,}"
               + (f" · {out.dropped_target_rows:,} rows dropped for a missing "
                  "target." if out.dropped_target_rows else ""))
        for n in out.notes:
            C.note(n)
        C.card_close()

    if out.warnings:
        C.card_open("Warnings from this run", icon="⚠️")
        for w in out.warnings:
            st.markdown(f"- {C.esc(w)}")
        C.card_close()

    if out.drop_reasons:
        with st.expander("Columns excluded from features and why", icon="🚫"):
            for r in out.drop_reasons:
                st.markdown(f"- {C.esc(r)}")

    _render_predictions(out)


def _render_predictions(out) -> None:
    if out.y_pred is None or out.y_test is None:
        return
    C.section_label("Held-out predictions")
    C.card_open("Actual versus predicted", sub="Every test row, from the "
                "single held-out split.", icon="🎯")
    frame = pd.DataFrame({"actual": out.y_test, "predicted": out.y_pred})
    st.dataframe(frame.head(200), use_container_width=True)
    if len(frame) > 200:
        C.note(f"Showing 200 of {len(frame):,} held-out rows. Full results are "
               "in Reports & Exports.")
    C.card_close()



def _overview_schema(prof) -> None:
    with st.expander("Detected schema and column types", icon="🧬"):
        st.caption("Types are inferred from content, not file metadata.")
        st.dataframe(C.metric_table(prof.to_frame()),
                     use_container_width=True, hide_index=True)
        if prof.id_like_columns:
            C.note("Identifier-like columns (excluded from modelling): "
                   + ", ".join(f"`{c}`" for c in prof.id_like_columns))


def _overview_preview(df, prof) -> None:
    with st.expander("Preview the loaded rows", icon="🔍"):
        hidden = set(prof.sensitive_columns)
        show = df.drop(columns=[c for c in df.columns if c in hidden])
        if hidden:
            C.note("Sensitive columns are hidden from this preview.")
        st.dataframe(show.head(50), use_container_width=True)
        if len(show) > 50:
            C.note(f"Showing the first 50 of {len(show):,} rows.")



def _ingest_upload(file) -> None:
    """Parse an uploaded file, surfacing ingestion errors as user-facing text."""
    try:
        with st.spinner("Reading and profiling..."):
            ds = ingest.load_table(file, name=file.name)
    except ingest.IngestionError as exc:
        C.banner("Could not read that file", str(exc), tone="limited")
        return
    _load_df(ds.df, ds.name)
    st.session_state["_ingest_warnings"] = list(ds.warnings)
    st.session_state["page"] = "Analysis Plan"
    st.rerun()

