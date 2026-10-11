"""Unsupervised learning: clustering, anomaly detection, dimensionality reduction.

Honesty rules enforced here
---------------------------
* No ground truth is invented. Cluster labels are *not* classes, and the UI
  must say so.
* Metrics are only reported for algorithms they are valid for (internal
  metrics need distance/connectivity structure that DBSCAN noise breaks).
* Degenerate results (one cluster, all noise) are surfaced as failures with
  an explanation instead of being scored as if they were meaningful.
* Stability is measured across resamples so a "good" silhouette score is not
  mistaken for a robust partition.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                             estimate_bandwidth)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .preprocess import build_preprocessor, feature_columns

RANDOM_STATE = 42
MAX_PCA_COMPONENTS = 2
MAX_K = 10
MIN_ROWS_CLUSTER = 30
MIN_ROWS_ANOMALY = 20
STABILITY_ROUNDS = 5
# parsimony tolerance for choosing k: smallest k within this of the best score
SELECT_TOL = 0.05
GAP_ROUNDS = 10
# A larger k must beat the best smaller k by this much to be selected.
# Silhouette grows with k, so an unpenalised argmax over-splits clusters.
K_GAIN = 0.03

try:                                            # optional dependency
    import umap as _umap
    HAVE_UMAP = True
except Exception:                               # pragma: no cover
    _umap = None
    HAVE_UMAP = False


class DegenerateResultError(ValueError):
    """Raised when an unsupervised method produced no usable structure."""


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette: float | None = None
    davies_bouldin: float | None = None
    calinski_harabasz: float | None = None
    balance: float | None = None             # min/max cluster size ratio
    sizes: dict = field(default_factory=dict)
    stability: float | None = None          # mean ARI across resamples
    fit_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    centres: object = None                  # cluster profile frame

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("labels", None)
        d.pop("centres", None)
        return d


@dataclass
class AnomalyResult:
    method: str
    scores: np.ndarray
    flagged: np.ndarray                  # boolean mask
    threshold: float
    n_flagged: int
    pct_flagged: float
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fit_s: float = 0.0
    top_features: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("scores", None)
        d.pop("flagged", None)
        return d


@dataclass
class UnsupervisedOutcome:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    drop_reasons: list[str] = field(default_factory=list)
    rows: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    clustering: list[ClusterResult] = field(default_factory=list)
    anomaly: list[AnomalyResult] = field(default_factory=list)
    embedding: pd.DataFrame | None = None
    embedding_method: str = ""
    X: object = None                       # processed feature matrix
    feature_names: list[str] = field(default_factory=list)
    cluster_profiles: dict = field(default_factory=dict)
    no_ground_truth: bool = True

    def best_cluster(self) -> ClusterResult | None:
        ok = [c for c in self.clustering if c.n_clusters > 1]
        if not ok:
            return None
        return sorted(ok, key=lambda c: -(c.silhouette
                                         if c.silhouette is not None
                                         else -1))[0]

    def cluster_frame(self) -> pd.DataFrame:
        rows = [c.as_dict() for c in self.clustering]
        return pd.DataFrame(rows)

    def anomaly_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.as_dict() for a in self.anomaly])


# --------------------------------------------------------------------------
# feature preparation
# --------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, profile, target: str | None = None
                     ) -> tuple[pd.DataFrame, UnsupervisedOutcome]:
    """Build the numeric/categorical feature split for unsupervised work."""
    numeric, categorical, dropped, reasons = feature_columns(
        profile, target or "__none__")
    out = UnsupervisedOutcome(numeric=numeric, categorical=categorical,
                              dropped=dropped, drop_reasons=reasons,
                              rows=int(df.shape[0]))
    if not numeric and not categorical:
        raise DegenerateResultError(
            "No usable feature columns remain after profiling. "
            f"Dropped: {', '.join(dropped) or 'none'}.")
    cols = numeric + categorical
    X = df[cols].copy()
    return X, out


def _fit_matrix(pre, X: pd.DataFrame) -> np.ndarray:
    Z = pre.fit_transform(X)
    return np.asarray(Z, dtype=float)


# --------------------------------------------------------------------------
# internal quality + stability
# --------------------------------------------------------------------------
def _internal_scores(Z: np.ndarray, labels: np.ndarray
                     ) -> tuple[float | None, float | None, float | None]:
    """Internal metrics, only where they are mathematically valid.

    * silhouette needs >= 2 clusters and excludes DBSCAN noise labels (-1),
      which are not a cluster.
    * Davies-Bouldin and Calinski-Harabasz are computed on the non-noise
      subset for the same reason.
    """
    mask = labels >= 0
    lab = labels[mask]
    Zs = Z[mask]
    uniq = np.unique(lab)
    if len(uniq) < 2 or len(Zs) <= len(uniq):
        return None, None, None
    sil = round(float(silhouette_score(Zs, lab)), 4)
    try:
        db = round(float(davies_bouldin_score(Zs, lab)), 4)
    except Exception:                                       # pragma: no cover
        db = None
    try:
        ch = round(float(calinski_harabasz_score(Zs, lab)), 4)
    except Exception:                                       # pragma: no cover
        ch = None
    return sil, db, ch


def _estimate_bandwidth(Z: np.ndarray, n: int) -> float | None:
    """Estimate a usable DBSCAN radius, tolerant of sklearn API changes.

    ``estimate_bandwidth`` renamed its sampling argument from ``subsample``
    (older releases) to ``n_samples`` (newer ones), so both are tried.
    """
    sample = int(min(500, n))
    for kw in ("n_samples", "subsample"):
        try:
            return estimate_bandwidth(Z, **{kw: sample},
                                      random_state=RANDOM_STATE)
        except TypeError:                                   # pragma: no cover
            continue
    return None


def _cv_gap(Z: np.ndarray, k: int, rng: np.random.Generator,
            rounds: int = GAP_ROUNDS):
    """Gap statistic with its standard deviation (Tibshirani et al., 2001).

    Fits K-Means on the real data and on uniform reference datasets drawn
    from the same bounding box, then compares log within-cluster dispersion.
    A large gap means the real data is far more clustered than random noise,
    so that k captures genuine structure rather than splitting a single blob.
    Returns ``(mean_gap, std_gap)`` or ``(None, None)`` when it cannot be
    computed; the standard deviation is what makes the 1-SE rule possible.
    """
    n = Z.shape[0]
    try:
        log_wk = np.log(KMeans(n_clusters=k, n_init=3,
                               random_state=RANDOM_STATE).fit(Z).inertia_ + 1e-9)
    except Exception:                                       # pragma: no cover
        return None, None
    gaps = []
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    try:
        pca = PCA(n_components=min(Z.shape[1], 10), random_state=RANDOM_STATE)
        T = pca.fit_transform(Z)
        pc_lo = T.min(axis=0)
        pc_span = T.max(axis=0) - pc_lo
        pca_ok = True
    except Exception:                                       # pragma: no cover
        pca_ok = False
    for _ in range(rounds):
        try:
            if pca_ok:
                ref_T = pc_lo + rng.random((n, T.shape[1])) * pc_span
                ref = pca.inverse_transform(ref_T)
            else:
                ref = lo + rng.random((n, Z.shape[1])) * span
            wk = KMeans(n_clusters=k, n_init=3,
                        random_state=RANDOM_STATE).fit(ref).inertia_
        except Exception:                                   # pragma: no cover
            continue
        gaps.append(np.log(wk + 1e-9) - log_wk)
    if not gaps:
        return None, None
    return round(float(np.mean(gaps)), 4), round(float(np.std(gaps)), 4)


def _cluster_sizes(labels: np.ndarray) -> tuple[dict, float]:
    vals, counts = np.unique(labels[labels >= 0], return_counts=True)
    sizes = {int(v): int(c) for v, c in zip(vals, counts)}
    if not counts.size:
        return sizes, 0.0
    return sizes, round(float(counts.min() / counts.max()), 4)


def _stability(Z: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
               rounds: int = STABILITY_ROUNDS) -> float | None:
    """Adjusted Rand Index between labels on bootstrap resamples.

    A stable partition keeps ARI close to 1.0; an unstable one drifts towards
    0.0. This guards against over-reading a single lucky choice of k.
    """
    n = Z.shape[0]
    if n < MIN_ROWS_CLUSTER:
        return None
    from sklearn.metrics import adjusted_rand_score
    k = len(np.unique(labels[labels >= 0]))
    if k < 2:
        return None
    scores = []
    for _ in range(rounds):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < MIN_ROWS_CLUSTER // 2:
            continue
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            km = KMeans(n_clusters=k, n_init=3,
                        random_state=int(rng.integers(1 << 30)))
            km.fit(Z[idx])
            scores.append(adjusted_rand_score(labels[idx], km.labels_))
        except Exception:                                   # pragma: no cover
            continue
    return round(float(np.mean(scores)), 4) if scores else None


def _select_k(swept):
    """Pick k by the gap statistic using the original one-standard-error rule.

    ``swept`` holds ``(k, gap, gap_sd, silhouette, result)`` tuples.

    Tibshirani's rule: choose the smallest k such that
    ``gap(k) >= gap(k+1) - sd(k+1)``. Raw argmax gap is *not* usable for model
    selection because the gap statistic grows monotonically with k, so it would
    always pick the largest k tested and over-split obvious structure. Silhouette
    is the fallback when gap statistics are unavailable.
    """
    usable = [t for t in swept if t[2] is not None or t[3] is not None]
    if not usable:
        return None
    have_gap = (len(usable) > 1
                and all(t[1] is not None and t[2] is not None for t in usable))
    if have_gap:
        rows = sorted(usable, key=lambda t: t[0])
        for i, (k, gap, sd, _sil, res) in enumerate(rows[:-1]):
            nxt_gap, nxt_sd = rows[i + 1][1], rows[i + 1][2]
            if gap >= nxt_gap - nxt_sd:
                return res
        return rows[-1][4]
    scored = [(t[0], t[3], t[4]) for t in usable if t[3] is not None]
    if not scored:
        return None
    best = max(s[1] for s in scored)
    within = [s for s in scored if s[1] >= best - SELECT_TOL]
    return min(within, key=lambda s: s[0])[2]


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def run_clustering(X: pd.DataFrame, pre, profile, time_budget_s: float = 120.0
                   ) -> tuple[list[ClusterResult], np.ndarray, list[str],
                              list[str]]:
    """Run a bounded set of clustering algorithms and score them honestly."""
    notes: list[str] = []
    warnings: list[str] = []
    results: list[ClusterResult] = []
    rng = np.random.default_rng(RANDOM_STATE)
    t0 = time.perf_counter()
    Z = _fit_matrix(pre, X)

    n = Z.shape[0]
    if n < MIN_ROWS_CLUSTER:
        raise DegenerateResultError(
            f"Clustering needs at least {MIN_ROWS_CLUSTER} rows to be "
            f"meaningful; this dataset has {n}.")
    k_max = int(min(MAX_K, max(2, n // 10)))

    # ---- K-Means over a small k sweep -----------------------------------
    # Silhouette alone rises with k, so a raw argmax would pick the largest k
    # we tried (splitting one real blob into slivers). We therefore rank k by
    # the gap statistic (Tibshirani) and apply the original one-standard-error
    # rule, which stops the search as soon as adding a cluster no longer beats
    # noise. Silhouette is the fallback when gap is unavailable, and when the
    # two disagree the disagreement is reported instead of hidden.
    swept: list[tuple] = []
    k_selection: list[dict] = []
    for k in range(2, k_max + 1):
        if time.perf_counter() - t0 > time_budget_s:
            warnings.append("Clustering time budget reached; the k sweep was "
                            "truncated.")
            break
        labels = KMeans(n_clusters=k, n_init=5,
                        random_state=RANDOM_STATE).fit_predict(Z)
        sil, db, ch = _internal_scores(Z, labels)
        sizes, balance = _cluster_sizes(labels)
        gap, gap_sd = _cv_gap(Z, k, rng)
        r = ClusterResult(
            algorithm=f"K-Means (k={k})", labels=labels, n_clusters=k,
            n_noise=int((labels < 0).sum()), silhouette=sil,
            davies_bouldin=db, calinski_harabasz=ch, balance=balance,
            sizes=sizes, fit_s=round(time.perf_counter() - t0, 3))
        swept.append((k, gap, gap_sd, sil, r))
        k_selection.append({"k": k, "gap": gap, "gap_sd": gap_sd,
                            "silhouette": sil})

    best_kmeans = _select_k(swept)
    if os.environ.get("ADIP_DEBUG"):
        for (k, gap, gap_sd, sil, _r) in swept:
            print(f"  [dbg] k={k} gap={gap} sd={gap_sd} sil={sil}")
    sils = [(t[0], t[3]) for t in swept if t[3] is not None]
    k_sil = max(sils, key=lambda t: t[1])[0] if sils else None
    if best_kmeans is not None:
        best_kmeans.stability = _stability(Z, best_kmeans.labels, rng)
        best_kmeans.notes.append(
            "k was chosen by the gap statistic (one-standard-error rule: the "
            f"smallest k whose gap is not beaten by the next k's "
            "gap minus one standard deviation). Silhouette was the fallback.")
        results.append(best_kmeans)
        notes.append(f"K-Means swept k=2..{k_max}; {best_kmeans.algorithm} was "
                     f"kept (silhouette {best_kmeans.silhouette}).")
        if k_sil is not None and k_sil != best_kmeans.n_clusters:
            warnings.append(
                f"The automatic selection criteria disagree: the gap statistic "
                f"favours k={best_kmeans.n_clusters} while the best silhouette "
                f"score is at k={k_sil}. Neither is proof of the true number of "
                "groups, and this dataset has no ground truth. Compare the "
                "candidates below before interpreting any cluster as real.")

    # ---- Agglomerative (Ward) -------------------------------------------
    try:
        labels = AgglomerativeClustering(
            n_clusters=min(3, k_max)).fit_predict(Z)
        sil, db, ch = _internal_scores(Z, labels)
        sizes, balance = _cluster_sizes(labels)
        results.append(ClusterResult(
            algorithm="Agglomerative (Ward)", labels=labels,
            n_clusters=int(len(np.unique(labels))), n_noise=0,
            silhouette=sil, davies_bouldin=db, calinski_harabasz=ch,
            balance=balance, sizes=sizes,
            fit_s=round(time.perf_counter() - t0, 3),
            notes=["Hierarchical clustering with 3 groups for comparison."]))
    except Exception as exc:                                # pragma: no cover
        warnings.append(f"Agglomerative clustering skipped: "
                        f"{type(exc).__name__}.")

    # ---- DBSCAN (density, produces noise labels) ------------------------
    try:
        bw = _estimate_bandwidth(Z, n)
        eps = float(bw) if bw and bw > 0 else None
        min_samples = int(max(5, n // 50))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        sizes, balance = _cluster_sizes(labels)
        noise = int((labels < 0).sum())
        k = int(len(np.unique(labels[labels >= 0])))
        dres = ClusterResult(
            algorithm="DBSCAN", labels=labels, n_clusters=k, n_noise=noise,
            balance=balance, sizes=sizes,
            fit_s=round(time.perf_counter() - t0, 3),
            notes=["Density-based; it does not force every point into a "
                   "cluster and marks sparse points as noise (label -1)."])
        if k < 2:
            dres.warnings.append(
                f"DBSCAN found no density-based clusters at the chosen "
                f"radius ({noise}/{n} points marked as noise). Internal "
                "scores are therefore not reported, because they are "
                "undefined for a single cluster.")
        else:
            sil, db, ch = _internal_scores(Z, labels)
            dres.silhouette, dres.davies_bouldin = sil, db
            dres.calinski_harabasz = ch
        results.append(dres)
    except Exception as exc:                                # pragma: no cover
        warnings.append(f"DBSCAN skipped: {type(exc).__name__}.")

    # ---- Gaussian Mixture (soft assignment) -----------------------------
    try:
        gm = GaussianMixture(n_components=min(3, k_max), n_init=2,
                             random_state=RANDOM_STATE)
        labels = gm.fit_predict(Z)
        sil, db, ch = _internal_scores(Z, labels)
        sizes, balance = _cluster_sizes(labels)
        results.append(ClusterResult(
            algorithm="Gaussian Mixture", labels=labels,
            n_clusters=int(len(np.unique(labels))), n_noise=0,
            silhouette=sil, davies_bouldin=db, calinski_harabasz=ch,
            balance=balance, sizes=sizes,
            fit_s=round(time.perf_counter() - t0, 3),
            notes=["Soft-probability model; kept for comparison against "
                   "hard clusterings."]))
    except Exception as exc:                                # pragma: no cover
        warnings.append(f"Gaussian Mixture skipped: {type(exc).__name__}.")

    if not results:
        raise DegenerateResultError(
            "No clustering algorithm produced a result on this dataset.")
    return results, Z, notes, warnings


# --------------------------------------------------------------------------
# anomaly detection
# --------------------------------------------------------------------------
def _robust_z(X: pd.DataFrame) -> np.ndarray:
    """Median / MAD based multivariate outlier score (no training needed)."""
    vals = X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if vals.size == 0:
        return np.zeros(len(X))
    med = np.nanmedian(vals, axis=0)
    mad = np.nanmedian(np.abs(vals - med), axis=0)
    mad = np.where(mad == 0, 1.0, mad)
    z = np.abs((np.nan_to_num(vals) - med) / mad)
    return z.mean(axis=1)


def run_anomaly_detection(X: pd.DataFrame, pre, contamination: float = 0.05
                          ) -> tuple[list[AnomalyResult], list[str], list[str]]:
    """Isolation Forest, Local Outlier Factor and a robust-z baseline."""
    notes: list[str] = []
    warnings: list[str] = []
    results: list[AnomalyResult] = []
    Z = _fit_matrix(pre, X)
    n = Z.shape[0]
    if n < MIN_ROWS_ANOMALY:
        raise DegenerateResultError(
            f"Anomaly detection needs at least {MIN_ROWS_ANOMALY} rows; this "
            f"dataset has {n}.")

    notes.append("The contamination rate is an assumption, not a measured "
                 f"truth: {contamination:.0%} of rows are flagged by design. "
                 "A flagged row is unusual relative to the others, not "
                 "proven to be an error or fraud.")

    def _build(method: str, scores: np.ndarray, decision: np.ndarray,
               fitted_s: float, note: str) -> AnomalyResult:
        """Wrap a scorer into a result.

        ``decision`` follows scikit-learn's convention: -1 = outlier.
        """
        if method == "Robust z-score (median/MAD)":
            thr = float(np.quantile(scores, 1 - contamination))
            flagged = scores >= thr
        else:
            thr = float(np.quantile(scores, contamination))
            flagged = (decision == -1)
        nf = int(flagged.sum())
        return AnomalyResult(
            method=method, scores=np.round(scores, 4), flagged=flagged,
            threshold=round(thr, 4), n_flagged=nf,
            pct_flagged=round(100 * nf / max(n, 1), 2), notes=[note],
            fit_s=round(fitted_s, 3))

    t = time.perf_counter()
    try:
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=RANDOM_STATE)
        decision = iso.fit_predict(Z)
        scores = -iso.score_samples(Z)
        results.append(_build("Isolation Forest", scores, decision,
                              time.perf_counter() - t,
                              "Trees isolate points; short average path "
                              "length implies an unusual record."))
    except Exception as exc:                                # pragma: no cover
        warnings.append(f"Isolation Forest skipped: {type(exc).__name__}.")

    t = time.perf_counter()
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, max(5, n // 10)),
                                 contamination=contamination)
        decision = lof.fit_predict(Z)
        scores = -lof.negative_outlier_factor_
        results.append(_build("Local Outlier Factor", scores, decision,
                              time.perf_counter() - t,
                              "Compares local density against neighbours; "
                              "catches records odd only in their own "
                              "neighbourhood."))
    except Exception as exc:                                # pragma: no cover
        warnings.append(f"Local Outlier Factor skipped: {type(exc).__name__}.")

    t = time.perf_counter()
    scores = _robust_z(X)
    thr = float(np.quantile(scores, 1 - contamination))
    results.append(_build("Robust z-score (median/MAD)", scores,
                          scores >= thr, time.perf_counter() - t,
                          "Reference baseline: distance from the median in "
                          "median-absolute-deviation units."))
    return results, notes, warnings


# --------------------------------------------------------------------------
# embedding + profiles
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def run_unsupervised(df: pd.DataFrame, profile, target: str | None = None,
                     contamination: float = 0.05,
                     time_budget_s: float = 120.0) -> UnsupervisedOutcome:
    """Cluster, score clusters, detect anomalies and project the structure."""
    X, out = prepare_features(df, profile, target)
    pre = build_preprocessor(out.numeric, out.categorical)
    out.warnings.append(
        "These clusters are descriptive groups, not validated classes. "
        "No ground truth exists in this dataset, so classification accuracy "
        "cannot be calculated.")
    clusters, Z, notes, warns = run_clustering(X, pre, profile, time_budget_s)
    out.clustering, out.X = clusters, Z
    out.notes.extend(notes)
    out.warnings.extend(warns)

    anomalies, anotes, awarns = run_anomaly_detection(X, pre, contamination)
    out.anomaly = anomalies
    out.notes.extend(anotes)
    out.warnings.extend(awarns)

    out.embedding, out.embedding_method = make_embedding(Z)
    out.notes.append(
        f"Structure is visualised with {out.embedding_method} for "
        "inspection. A 2-D picture of clusters is not proof that they are "
        "meaningful; the internal scores and stability are.")

    best = out.best_cluster()
    if best is not None:
        prof = cluster_profiles(X, best.labels, out.numeric, out.categorical)
        out.cluster_profiles[best.algorithm] = prof
    return out


def cluster_profiles(X: pd.DataFrame, labels: np.ndarray,
                     numeric: list[str], categorical: list[str]
                     ) -> pd.DataFrame:
    """Mean (numeric) and mode (categorical) per cluster, on ORIGINAL scale."""
    work = X.copy()
    work["_cluster"] = np.where(labels < 0, "noise", labels.astype(str))
    parts = []
    if numeric:
        means = work.groupby("_cluster")[numeric].mean().round(3)
        parts.append(means.add_prefix("mean_"))
    if categorical:
        for c in categorical[:5]:
            mode = work.groupby("_cluster")[c].agg(
                lambda s: s.mode().iat[0] if not s.mode().empty else "")
            parts.append(mode.to_frame(f"mode_{c}"))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
def make_embedding(Z: np.ndarray,
                   method: str = "pca") -> tuple[pd.DataFrame, str]:
    """2-D projection for inspection. UMAP is used only if it is installed."""
    n = Z.shape[0]
    if method == "umap" and HAVE_UMAP:
        try:
            nn = min(15, max(2, n // 10))
            emb = _umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                             n_neighbors=nn).fit_transform(Z)
            return (pd.DataFrame(emb, columns=["UMAP 1", "UMAP 2"]), "UMAP")
        except Exception:                                   # pragma: no cover
            pass
    k = int(min(MAX_PCA_COMPONENTS, Z.shape[1], n))
    pca = PCA(n_components=max(k, 2), random_state=RANDOM_STATE)
    emb = pca.fit_transform(Z)
    if emb.shape[1] < 2:
        emb = np.column_stack([emb[:, 0], emb[:, 0]])
    return pd.DataFrame(emb[:, :2], columns=["PC1", "PC2"]), "PCA"

