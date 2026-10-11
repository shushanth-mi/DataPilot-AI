"""Natural-language analysis assistant: deterministic intent handling.

This is deliberately **not** an LLM integration. Questions are matched to a
small set of constrained analytical tools that operate on real computed
results, so the assistant works fully offline and cannot invent numbers.

Contract enforced by every tool:
* answers cite the computed evidence they came from;
* anything not supported by the data returns an explicit "not answerable"
  result rather than a guess;
* computed facts, predictions and hypotheses are labelled separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

#: Every intent maps to (regex, tool-name). Order matters: the first match wins.
_INTENTS: list[tuple[str, str]] = [
    (r"\bunusual|outlier|anomal|strange|weird|odd\b", "anomalies"),
    (r"\bcluster|segment|group(?:s|ing)? of (?:rows|records|customers)\b",
     "clusters"),
    (r"\bmiss(?:ing|ingness)|null|blank|incomplete|quality\b", "quality"),
    (r"\bcolumn|field|feature|schema|shape|dimension|how many rows|row count\b",
     "schema"),
    (r"\bdiffer|compar(?:e|ison)|higher|lower|between .* and\b", "segments"),
    (r"\brel(?:ated|ationship)|correlat|associat|drive|influence\b",
     "associations"),
    (r"\bpredict|forecast|reliab|accura|how well|trust|dependable|"
     r"can this dataset support\b", "reliability"),
    (r"\bbest model|which model|compare(?: the)? models?\b", "models"),
    (r"\blimitation|weakness|caveat|risk|what can(?:'| no)t\b", "limitations"),
    (r"\bchanged|change|shift|trend|over time|between .* period\b", "trend"),
    (r"\bfinding|insight|what(?:'s| is) (?:in|going on)|summar", "findings"),
]

_NOT_SUPPORTED = ("not_answerable",)


@dataclass
class Answer:
    """A single assistant response, with its provenance attached."""

    question: str
    intent: str = ""
    answered: bool = False
    headline: str = ""
    body: str = ""
    evidence: list[str] = field(default_factory=list)
    kind: str = ""              # computed | prediction | hypothesis | refused
    tables: dict = field(default_factory=dict)
    limitation: str = ""
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent,
                "answered": self.answered, "headline": self.headline,
                "body": self.body, "evidence": self.evidence,
                "kind": self.kind, "limitation": self.limitation}


def classify(question: str) -> str:
    """Map a question to an intent name (or ``not_answerable``)."""
    q = (question or "").lower().strip()
    if not q:
        return "not_answerable"
    for pattern, intent in _INTENTS:
        if re.search(pattern, q):
            return intent
    return "not_answerable"


def example_questions(profile, has_supervised: bool = False,
                      has_unsupervised: bool = False,
                      has_findings: bool = False) -> list[str]:
    """Suggest questions that this dataset and these results can answer."""
    out: list[str] = []
    n_num = len(profile.numeric) if profile else 0
    if profile:
        out.append(f"How many rows and columns are in {profile.dataset_name}?")
        out.append("Are there missing values or data-quality problems?")
    if n_num >= 2:
        out.append("Which numeric columns are related to each other?")
        if profile.categorical:
            col = profile.categorical[0]
            out.append(f"Which segments differ most on {profile.numeric[0]}?")
    if has_unsupervised:
        out.append("What are the most unusual records?")
        out.append("What did the clustering find?")
    if has_supervised:
        out.append("Which model performs best on unseen data?")
        out.append("Can this dataset support a reliable prediction?")
    if has_findings:
        out.append("What are the main limitations of this analysis?")
        out.append("Summarise the key findings.")
    return out[:10]

_REFUSAL = ("I could not match that question to a supported analytical tool, "
            "so I will not guess. I answer from computed results only — try "
            "one of the suggested questions, or run an analysis first.")


def _no_analysis(a: "Answer", what: str, action: str) -> "Answer":
    a.headline = f"Run {what} first"
    a.body = (f"{what.capitalize()} has not been executed on this dataset, so "
              "there is nothing to report yet.")
    a.kind = "refused"
    a.limitation = f"Action needed: {action}."
    return a


def ask(question: str, *, df: pd.DataFrame | None = None, profile=None,
        plan=None, reliability=None, discovery=None, supervised=None,
        unsupervised=None) -> Answer:
    """Answer a question using only results that were actually computed.

    Each branch reads a real object produced elsewhere in the pipeline. When
    the required analysis has not been run yet, the answer says so instead of
    fabricating a result.
    """
    q = (question or "").strip()
    if not q:
        return Answer(question=q, headline="Ask me something about your data.",
                      body=_REFUSAL, kind="refused",
                      suggestions=example_questions(profile))
    intent = classify(q)
    a = Answer(question=q, intent=intent)

    if intent == "not_answerable":
        a.headline = "Not answerable from the current results"
        a.body = _REFUSAL
        a.kind = "refused"
        a.suggestions = example_questions(profile, bool(supervised),
                                          bool(unsupervised), bool(discovery))
        return a

    if profile is None or df is None:
        a.headline = "No dataset loaded"
        a.body = ("Upload or load a dataset first; every answer is derived "
                  "from the data you provide.")
        a.kind = "refused"
        return a

    if intent == "schema":
        a.answered = True
        a.kind = "computed"
        a.headline = (f"{profile.rows:,} rows x {profile.cols:,} columns "
                      f"({profile.memory_mb} MB)")
        a.body = ("Detected "
                  f"{len(profile.numeric)} numeric, "
                  f"{len(profile.categorical)} categorical and "
                  f"{len(profile.datetime)} datetime columns.")
        a.evidence = [f"Dataset: {profile.dataset_name}",
                      f"Fingerprint: {profile.fingerprint}",
                      f"Duplicate rows: {profile.duplicate_rows:,}"]
        a.tables["schema"] = profile.to_frame()
        return a

    if intent == "quality":
        a.answered = True
        a.kind = "computed"
        issues = profile.issues
        a.headline = (f"{len(issues)} data-quality "
                      f"{'issue' if len(issues) == 1 else 'issues'} detected")
        a.body = ("These are observed facts about the file, not judgements "
                  "about your process." if issues else
                  "No structural data-quality problems were detected.")
        a.evidence = [f"{i.severity.upper()}: {i.message} ({i.remedy})"
                      for i in issues[:8]]
        a.tables["issues"] = profile.issues_frame()
        a.limitation = ("The profiler detects structural problems it can "
                        "verify. It cannot tell whether a value is "
                        "business-wrong if it is well-formed.")
        return a

    if intent in ("segments", "associations", "trend"):
        if discovery is None:
            return _no_analysis(a, "discovery", "open the Findings tab")
        want = {"segments": "segment", "associations": "association",
                "trend": "trend"}[intent]
        hits = [f for f in discovery.findings if f.kind == want]
        if not hits:
            a.headline = f"No {want} findings survived filtering"
            a.body = (f"The discovery pass ran {discovery.n_tested} tests and "
                      "found nothing of this type that survived "
                      "multiple-comparison control.")
            a.kind = "hypothesis"
            a.limitation = ("Absence of a finding is not evidence of absence; "
                            "it may reflect sample size or data quality.")
            return a
        a.answered = True
        a.kind = "computed"
        top = hits[0]
        a.headline = top.title
        a.body = top.description
        a.evidence = [f"{f.title}: {f.evidence}" for f in hits[:5]]
        a.tables["findings"] = discovery.frame()
        a.limitation = (top.limitation or
                        "These are associations in observed data, not causal "
                        "effects.")
        return a

    if intent == "anomalies":
        if unsupervised is None or not unsupervised.anomaly:
            return _no_analysis(a, "unsupervised anomaly detection",
                                "open the Discovery tab")
        a.answered = True
        a.kind = "computed"
        res = unsupervised.anomaly[0]
        a.headline = (f"{res.n_flagged:,} rows flagged by {res.method} "
                      f"({res.pct_flagged:.1f}%)")
        a.body = ("Flagged rows are statistical outliers under the method's "
                  "assumptions, not proven errors.")
        a.evidence = [f"Threshold: {res.threshold:.4f}"] + list(res.notes)
        a.limitation = ("A flag is not a verdict. Without ground truth the "
                        "false-positive rate cannot be measured.")
        return a

