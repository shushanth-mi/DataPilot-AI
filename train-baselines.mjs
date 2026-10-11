/**
 * train-baselines.mjs
 * ---------------------------------------------------------------------------
 * Trains the two baseline classifiers (Decision Tree + SVM/RBF) on the REAL
 * AI4I 2020 Predictive Maintenance dataset and writes fitted artifacts to
 * model-artifacts.js so they can run live in the browser prototype.
 *
 * Pipeline mirrors the documented study (MODEL_METRICS.pipeline):
 *   features  AirTemp, ProcTemp, RPM, Torque, ToolWear, one-hot Type_H/L/M
 *   excluded  FailureType (label leakage)
 *   split     stratified 80/20, class_weight = "balanced"
 *   scaling   StandardScaler fitted on the training split (SVM only)
 *   tuning    stratified k-fold CV scored on F1
 *
 * Usage:  node train-baselines.mjs [path-to-ai4i2020.csv]
 * Output: model-artifacts.js
 * Pure Node - no dependencies, so the artifacts are reproducible anywhere.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const CSV_PATH = resolve(process.argv[2] || process.env.AI4I_CSV || "");
const SEED = 42;
const COLS = ["AirTemp", "ProcTemp", "RPM", "Torque", "ToolWear", "Type_H", "Type_L", "Type_M"];
const log = (...a) => console.log(...a);

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function loadCsv(path) {
  const lines = readFileSync(path, "utf8").replace(/\r/g, "").split("\n").filter((l) => l.trim());
  const head = lines[0].split(",");
  const ix = {};
  head.forEach((h, i) => (ix[h.trim()] = i));
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(",");
    const t = c[ix["Type"]].trim();
    rows.push({
      x: [
        +c[ix["Air temperature [K]"]],
        +c[ix["Process temperature [K]"]],
        +c[ix["Rotational speed [rpm]"]],
        +c[ix["Torque [Nm]"]],
        +c[ix["Tool wear [min]"]],
        t === "H" ? 1 : 0, t === "L" ? 1 : 0, t === "M" ? 1 : 0
      ],
      y: +c[ix["Machine failure"]] === 1 ? 1 : 0
    });
  }
  return rows;
}

function stratifiedSplit(rows, testFrac, seed) {
  const rng = mulberry32(seed);
  const byClass = { 0: [], 1: [] };
  rows.forEach((r, i) => byClass[r.y].push(i));
  const testSet = new Set();
  for (const cls of [0, 1]) {
    const idxs = shuffle(byClass[cls].slice(), rng);
    const nTest = Math.floor(idxs.length * testFrac);
    for (let i = 0; i < nTest; i++) testSet.add(idxs[i]);
  }
  const train = [], test = [];
  rows.forEach((r, i) => (testSet.has(i) ? test : train).push(r));
  return { train, test };
}
function kFolds(train, k, seed) {
  const rng = mulberry32(seed);
  const byClass = { 0: [], 1: [] };
  train.forEach((r, i) => byClass[r.y].push(i));
  const foldOf = new Array(train.length);
  for (const cls of [0, 1]) shuffle(byClass[cls].slice(), rng).forEach((idx, pos) => (foldOf[idx] = pos % k));
  const folds = [];
  for (let f = 0; f < k; f++) {
    folds.push({ tr: train.filter((_, i) => foldOf[i] !== f), va: train.filter((_, i) => foldOf[i] === f) });
  }
  return folds;
}
/* sklearn class_weight="balanced": w_c = n / (2 * n_c) */
function balancedWeights(rows) {
  const n = rows.length;
  const nPos = rows.reduce((s, r) => s + r.y, 0);
  const wPos = n / (2 * nPos), wNeg = n / (2 * (n - nPos));
  return rows.map((r) => (r.y ? wPos : wNeg));
}

function prf1(yTrue, scores, thr) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (scores[i] >= thr) { if (yTrue[i]) tp++; else fp++; }
    else if (yTrue[i]) fn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1, tp, fp, fn, tn: yTrue.length - tp - fp - fn };
}
function rocAuc(yTrue, scores) {
  const pairs = scores.map((s, i) => [s, yTrue[i]]).sort((a, b) => a[0] - b[0]);
  let rank = 1, i = 0, sumRankPos = 0, nPos = 0;
  while (i < pairs.length) {
    let j = i;
    while (j + 1 < pairs.length && pairs[j + 1][0] === pairs[i][0]) j++;
    const mid = (rank + (rank + (j - i))) / 2;
    for (let t = i; t <= j; t++) if (pairs[t][1] === 1) { sumRankPos += mid; nPos++; }
    rank += j - i + 1;
    i = j + 1;
  }
  const nNeg = pairs.length - nPos;
  if (!nPos || !nNeg) return NaN;
  return (sumRankPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}
/* ============================ DECISION TREE ===============================
 * CART with Gini impurity and balanced sample weights. Weighted class
 * fractions drive both the split criterion and the leaf probability, which
 * matches DecisionTreeClassifier(class_weight="balanced"). The fitted tree is
 * exported as a nested literal so the browser can walk it directly.
 * -------------------------------------------------------------------------- */
function buildTree(rows, weights, maxDepth, minSamplesLeaf) {
  const gini = (p) => 1 - p * p - (1 - p) * (1 - p);
  return grow(rows.map((_, i) => i), 0);

  function grow(idxs, depth) {
    const n = idxs.length;
    let wPos = 0, wTot = 0;
    for (const i of idxs) { const w = weights[i]; wTot += w; if (rows[i].y) wPos += w; }
    const pPos = wTot > 0 ? wPos / wTot : 0;
    const leaf = (p) => ({ leaf: true, p: +p.toFixed(6) });

    if (depth >= maxDepth || n < 2 * minSamplesLeaf || pPos <= 0 || pPos >= 1) return leaf(pPos);

    let best = null;
    const parentGini = 1 - pPos * pPos - (1 - pPos) * (1 - pPos);

    for (let f = 0; f < COLS.length; f++) {
      const order = idxs.slice().sort((a, b) => rows[a].x[f] - rows[b].x[f]);
      let lwPos = 0, lwTot = 0;
      for (let k = 0; k < n - 1; k++) {
        const i = order[k];
        lwPos += rows[i].y ? weights[i] : 0;
        lwTot += weights[i];
        const thr = rows[i].x[f];
        if (rows[order[k + 1]].x[f] <= thr) continue;        // no gap -> no valid split
        const rTot = wTot - lwTot;
        if (lwTot < minSamplesLeaf || rTot < minSamplesLeaf) continue;
        const lp = lwPos / lwTot, rp = (wPos - lwPos) / rTot;
        const gain = parentGini - (lwTot / wTot) * gini(lp) - (rTot / wTot) * gini(rp);
        if (!best || gain > best.gain) {
          best = { gain, f, thr: +((thr + rows[order[k + 1]].x[f]) / 2).toFixed(6) };
        }
      }
    }
    if (!best || best.gain <= 1e-9) return leaf(pPos);

    const li = [], ri = [];
    for (const i of idxs) (rows[i].x[best.f] <= best.thr ? li : ri).push(i);
    if (!li.length || !ri.length) return leaf(pPos);
    return { leaf: false, f: COLS[best.f], thr: best.thr, l: grow(li, depth + 1), r: grow(ri, depth + 1) };
  }
}
function dtPredictTree(node, x) {
  let cur = node;
  while (!cur.leaf) cur = x[COLS.indexOf(cur.f)] <= cur.thr ? cur.l : cur.r;
  return cur.p;
}
const countNodes = (n) => (n.leaf ? 1 : 1 + countNodes(n.l) + countNodes(n.r));
const treeDepth = (n) => (n.leaf ? 1 : 1 + Math.max(treeDepth(n.l), treeDepth(n.r)));

/* ============================ STANDARD SCALER ============================ */
function fitScaler(rows) {
  const d = COLS.length;
  const mean = new Array(d).fill(0), scale = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r.x[j];
  for (let j = 0; j < d; j++) mean[j] /= rows.length;
  for (const r of rows) for (let j = 0; j < d; j++) scale[j] += (r.x[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) scale[j] = Math.sqrt(scale[j] / rows.length) || 1;
  return { mean, scale };
}
const scaleOne = (x, sc) => x.map((v, j) => (v - sc.mean[j]) / sc.scale[j]);
const scaleRows = (rows, sc) => rows.map((r) => ({ x: scaleOne(r.x, sc), y: r.y }));
/* sklearn gamma="scale" = 1 / (n_features * variance of the scaled features) */
function scaledVariance(rows) {
  const d = COLS.length, n = rows.length;
  const m = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) m[j] += r.x[j];
  for (let j = 0; j < d; j++) m[j] /= n;
  let s = 0;
  for (const r of rows) for (let j = 0; j < d; j++) s += (r.x[j] - m[j]) ** 2;
  return s / (n * d);
}
/* ============================== SVM (RBF) ================================
 * Binary RBF-kernel SVC solved with Sequential Minimal Optimization using
 * per-class bounds from class_weight="balanced". The kernel matrix is
 * precomputed (n is kept tractable by balanced subsampling, see MAIN), so
 * every SMO step is O(n) with O(1) kernel lookups and the solver converges
 * properly. Probabilities come from Platt's sigmoid, as SVC(probability=
 * True) does.
 * -------------------------------------------------------------------------- */
function trainSVC(rows, C, gamma, opts) {
  const o = opts || {};
  const tol = o.tol || 1e-4;
  const maxIter = o.maxIter || 60000;
  const n = rows.length;
  const X = rows.map((r) => r.x);
  const y = new Int8Array(n);
  let nPos = 0;
  for (let i = 0; i < n; i++) { y[i] = rows[i].y ? 1 : -1; if (y[i] === 1) nPos++; }
  const nNeg = n - nPos;
  const wPos = n / (2 * nPos), wNeg = n / (2 * nNeg);
  const bounds = new Float64Array(n);
  for (let i = 0; i < n; i++) bounds[i] = C * (y[i] === 1 ? wPos : wNeg);

  /* precomputed RBF kernel matrix */
  const K = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    K[i * n + i] = 1;
    const xi = X[i];
    for (let j = i + 1; j < n; j++) {
      const xj = X[j];
      let s = 0;
      for (let k = 0; k < xi.length; k++) { const d = xi[k] - xj[k]; s += d * d; }
      const v = Math.exp(-gamma * s);
      K[i * n + j] = v; K[j * n + i] = v;
    }
  }

  const alpha = new Float64Array(n);
  const f = new Float64Array(n);          // decision values without the bias term
  let b = 0;

  /* Vapnik-style warm start: give every sample a small non-zero alpha so the
     decision function carries class information from the first iteration, then
     centre the two classes. Without this the bias never gets set, because with
     class_weight="balanced" almost every alpha immediately saturates. */
  {
    for (let i = 0; i < n; i++) alpha[i] = Math.min(bounds[i], 1e-3);
    for (let i = 0; i < n; i++) {
      const row = i * n;
      let s = 0;
      for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * y[j] * K[row + j];
      f[i] = s;
    }
    /* centre the classes: mean decision over positives = minus that over negatives */
    let mp = 0, mn = 0, cp = 0, cn = 0;
    for (let i = 0; i < n; i++) { if (y[i] === 1) { mp += f[i]; cp++; } else { mn += f[i]; cn++; } }
    b = -0.5 * (mp / Math.max(1, cp) + mn / Math.max(1, cn));
  }

  for (let iter = 0; iter < maxIter; iter++) {
    /* 1. most violated sample */
    let iSel = -1, maxViol = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f[i] + b - y[i];
      const r = y[i] * Ei;
      const viol = (alpha[i] < bounds[i] && r < -tol) ? -r : (alpha[i] > 0 && r > tol) ? r : 0;
      if (viol > maxViol) { maxViol = viol; iSel = i; }
    }
    if (iSel < 0) break;

    /* 2. partner j. For a pair step the dual objective changes by
   dW(t) = L*t + (1 - K_ij) * t^2, because
     - opposite labels: alphas move together  -> L = -(y_i*E0_i + y_j*E0_j)
     - equal labels:    alphas move oppositely -> L = -y_i*(E0_i - E0_j)
   with E0 = f - y (the gradient dW/d(alpha) = 1 - y*f uses f WITHOUT bias).
   dW is convex in t, so its maximum over the feasible box sits at an endpoint. */
    const Ei0 = f[iSel] - y[iSel];
    let jSel = -1, bestGain = 0, bestT = 0;
    for (let j = 0; j < n; j++) {
      if (j === iSel) continue;
      const quad = 1 - K[iSel * n + j];
      if (quad <= 1e-12) continue;                     // identical points
      let tLo, tHi;
      if (y[iSel] !== y[j]) {
        tLo = Math.max(-alpha[iSel], -alpha[j]);
        tHi = Math.min(bounds[iSel] - alpha[iSel], bounds[j] - alpha[j]);
      } else {
        const s = alpha[iSel] + alpha[j];
        tLo = Math.max(-alpha[iSel], s - alpha[iSel] - bounds[j]);
        tHi = Math.min(bounds[iSel] - alpha[iSel], s - alpha[iSel]);
      }
      if (tHi <= tLo) continue;
      const Ej0 = f[j] - y[j];
      const L = (y[iSel] !== y[j])
        ? -(y[iSel] * Ei0 + y[j] * Ej0)
        : -y[iSel] * (Ei0 - Ej0);
      const gLo = L * tLo + quad * tLo * tLo;
      const gHi = L * tHi + quad * tHi * tHi;
      const gain = Math.max(gLo, gHi);
      if (gain > bestGain) { bestGain = gain; jSel = j; bestT = gHi >= gLo ? tHi : tLo; }
    }
    if (jSel < 0) break;

    /* 3. two-variable update */
    const j = jSel, i = iSel;
    const ai = alpha[i], aj = alpha[j];
    const nAi = ai + bestT;
    const nAj = y[i] !== y[j] ? aj + bestT : ai + aj - nAi;
    const dA = nAi - ai, dB = nAj - aj;
    alpha[i] = nAi; alpha[j] = nAj;
    const rowI = i * n, rowJ = j * n;
    const yi = y[i], yj = y[j];
    for (let k = 0; k < n; k++) f[k] += yi * dA * K[rowI + k] + yj * dB * K[rowJ + k];
    /* Bias from the KKT condition of the pair: a free (0 < alpha < C) support
       vector must satisfy G = f + b - y = 0, hence b = y - f. Taking it from a
       bound variable has no information, so b is then left untouched. */
    const iFree = nAi > 1e-12 && nAi < bounds[i] - 1e-12;
    const jFree = nAj > 1e-12 && nAj < bounds[j] - 1e-12;
    if (iFree) b = yi - f[i];
    else if (jFree) b = yj - f[j];
  }

  const svIdx = [];
  for (let i = 0; i < n; i++) if (alpha[i] > 1e-7) svIdx.push(i);
  const svs = svIdx.map((i) => X[i].slice());
  const dualCoef = svIdx.map((i) => alpha[i] * y[i]);

  const decision = (x) => {
    let s = 0;
    for (let k = 0; k < svIdx.length; k++) {
      const sv = svs[k];
      let d = 0;
      for (let i = 0; i < x.length; i++) { const t = x[i] - sv[i]; d += t * t; }
      s += dualCoef[k] * Math.exp(-gamma * d);
    }
    return s + b;
  };

  const dec = new Float64Array(n);
  for (let i = 0; i < n; i++) dec[i] = decision(X[i]);
  /* Platt targets use the class counts of the ORIGINAL training split, not the
     balanced subsample. The subsample deliberately over-represents failures
     (1:6 instead of 1:29), so calibrating on its own counts would inflate every
     probability and shift the 0.5 decision boundary. */
  let calPos = nPos, calTotal = n;
  if (Number.isFinite(o.calibPos) && Number.isFinite(o.calibTotal)) {
    calPos = o.calibPos; calTotal = o.calibTotal;
  }
  const sig = fitPlatt(dec, y, (calPos + 1) / (calTotal + 2), 1 / (calTotal + 2));
  return {
    svIdx, svs, dualCoef, b, gamma, A: sig.A, B: sig.B, decision,
    prob: (x) => 1 / (1 + Math.exp(-(decision(x) * sig.A + sig.B)))
  };
}

/* ---- Platt sigmoid calibration (as used by SVC(probability=True)) ------ */
function fitPlatt(dec, y, tPos, tNeg) {
  const n = dec.length;
  /* Platt's smoothed targets, supplied by the caller so they can follow the
     original class balance rather than a resampled one */
  const hiT = tPos, loT = tNeg;
  const t = new Float64Array(n);
  for (let i = 0; i < n; i++) t[i] = y[i] === 1 ? hiT : loT;

  /* Standardise the decision values first. Raw SVM margins can be tiny, and
     the Newton fit below is badly conditioned in that case (the Hessian
     collapses onto the 1e-12 ridge and the step explodes). The scale is
     folded back into A/B afterwards, so the result is identical. */
  let mu = 0;
  for (let i = 0; i < n; i++) mu += dec[i];
  mu /= n;
  let vr = 0;
  for (let i = 0; i < n; i++) vr += (dec[i] - mu) * (dec[i] - mu);
  const sd = Math.sqrt(vr / n) || 1;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = (dec[i] - mu) / sd;

  const softplus = (v) => (v > 0 ? v + Math.log1p(Math.exp(-v)) : Math.log1p(Math.exp(v)));
  const objective = (A, B) => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const fv = z[i] * A + B;
      /* -log likelihood: t*softplus(-f) + (1-t)*softplus(f) */
      s += t[i] * softplus(-fv) + (1 - t[i]) * softplus(fv);
    }
    return s;
  };
  let A = 0, B = Math.log(loT / hiT);        // prior odds at A = 0
  for (let iter = 0; iter < 100; iter++) {
    let h11 = 1e-12, h22 = 1e-12, h21 = 0, g1 = 0, g2 = 0;
    for (let i = 0; i < n; i++) {
      const fv = z[i] * A + B;
      const p = 1 / (1 + Math.exp(-fv));   // sigma(fv), matching the objective above
      const q = p * (1 - p);
      const d = p - t[i];                   // d(objective)/d(f_i)
      g1 += z[i] * d; g2 += d;
      h11 += z[i] * z[i] * q; h22 += q; h21 += z[i] * q;
    }
    const det = h11 * h22 - h21 * h21;
    if (!(Math.abs(det) > 1e-14)) break;
    /* g1/g2 are the true gradient of the objective; Newton step = -H^-1 g */
    const dA = -(h22 * g1 - h21 * g2) / det;
    const dB = -(h11 * g2 - h21 * g1) / det;
    let step = 1, moved = false;
    while (step >= 1e-10) {
      const nA = A + step * dA, nB = B + step * dB;
      if (objective(nA, nB) < objective(A, B)) { A = nA; B = nB; moved = true; break; }
      step /= 2;
    }
    if (!moved) break;
    if (Math.abs(g1) < 1e-8 && Math.abs(g2) < 1e-8) break;
  }
  /* undo the standardisation: A*(dec) + B == A*(z) + B' */
  return { A: A / sd, B: B - (A * mu) / sd };
}
/* ---- evaluation helpers ---- */
function bestThreshold(yTrue, scores) {
  let best = { f1: -1, thr: 0.5 };
  for (let i = 1; i < 99; i++) {
    const thr = i / 100;
    const f1 = prf1(yTrue, scores, thr).f1;
    if (f1 > best.f1) best = { f1, thr };
  }
  return best;
}
function evaluate(name, probFn, thr) {
  const yTrue = test.map((r) => r.y);
  const scores = test.map(probFn);
  const atThr = prf1(yTrue, scores, thr);
  const atHalf = prf1(yTrue, scores, 0.5);
  const auc = rocAuc(yTrue, scores);
  log(`  ${name.padEnd(4)} AUC=${auc.toFixed(4)}  F1=${atThr.f1.toFixed(4)} P=${atThr.precision.toFixed(4)} R=${atThr.recall.toFixed(4)} @thr ${thr.toFixed(2)}   (F1@0.50 = ${atHalf.f1.toFixed(4)})   CM=[[${atThr.tn},${atThr.fp}],[${atThr.fn},${atThr.tp}]]`);
  return {
    f1: +atThr.f1.toFixed(4), precision: +atThr.precision.toFixed(4), recall: +atThr.recall.toFixed(4),
    rocAuc: +auc.toFixed(4), threshold: thr, f1AtHalf: +atHalf.f1.toFixed(4), testRecords: test.length,
    confusion: { tn: atThr.tn, fp: atThr.fp, fn: atThr.fn, tp: atThr.tp }
  };
}

/* ================================= MAIN ================================== */
log("Loading dataset:", CSV_PATH);
const rows = loadCsv(CSV_PATH);
const failures = rows.reduce((s, r) => s + r.y, 0);
log(`  rows=${rows.length}  failures=${failures} (${(100 * failures / rows.length).toFixed(2)}%)`);

const { train, test } = stratifiedSplit(rows, 0.2, SEED);
log(`  stratified 80/20 split (seed ${SEED}): train=${train.length} test=${test.length}`);

/* ---- Decision Tree: 5-fold CV, selected on ROC-AUC ---- */
let bestDt = null;
const dtFolds = kFolds(train, 5, SEED);
for (const maxDepth of [3, 5, 7, 10]) {
  for (const minLeaf of [1, 5, 10, 20]) {
    const oofY = [], oofS = [];
    for (const fold of dtFolds) {
      const tree = buildTree(fold.tr, balancedWeights(fold.tr), maxDepth, minLeaf);
      fold.va.forEach((r) => { oofY.push(r.y); oofS.push(dtPredictTree(tree, r.x)); });
    }
    const auc = rocAuc(oofY, oofS);
    const bt = bestThreshold(oofY, oofS);
    log(`  DT   depth=${String(maxDepth).padStart(2)} leaf=${String(minLeaf).padStart(2)}  CV AUC = ${auc.toFixed(4)}  CV F1 = ${bt.f1.toFixed(4)} @${bt.thr.toFixed(2)}`);
    if (!bestDt || auc > bestDt.auc) bestDt = { maxDepth, minLeaf, auc, thr: bt.thr, cv: bt.f1 };
  }
}
log(`  -> best DT: max_depth=${bestDt.maxDepth} min_samples_leaf=${bestDt.minLeaf} (CV AUC ${bestDt.auc.toFixed(4)}, F1 ${bestDt.cv.toFixed(4)} @${bestDt.thr.toFixed(2)})`);

const t0 = Date.now();
const dtTree = buildTree(train, balancedWeights(train), bestDt.maxDepth, bestDt.minLeaf);
log(`  fitted DT in ${Date.now() - t0} ms -> ${countNodes(dtTree)} nodes, depth ${treeDepth(dtTree)}`);

/* ---- SVM: scaler + RBF SMO, 3-fold CV grid search on F1 ----
   The kernel matrix is precomputed, so the solver needs a tractable n.
   We therefore fit on a class-balanced subsample of the training split:
   every failure plus NEG_RATIO non-failures per failure. This is a
   standard approach for SVMs on strongly imbalanced data and is applied
   identically inside every CV fold, so the reported numbers stay honest. */
const NEG_RATIO = 6;
function balancedSubsample(rows, seed) {
  const rng = mulberry32(seed);
  const pos = rows.filter((r) => r.y === 1);
  const neg = shuffle(rows.filter((r) => r.y === 0).slice(), rng).slice(0, pos.length * NEG_RATIO);
  return shuffle(pos.concat(neg), rng);
}

const scaler = fitScaler(train);
const gammaScale = 1 / (COLS.length * scaledVariance(scaleRows(train, scaler)));
log(`  StandardScaler mean = [${scaler.mean.map((v) => v.toFixed(2)).join(", ")}]`);
log(`  gamma="scale" = 1 / (8 * Var) = ${gammaScale.toFixed(6)}`);
log(`  SVM training set = balanced subsample (all failures + ${NEG_RATIO}x non-failures)`);

const svmFolds = kFolds(train, 3, SEED + 1);
let bestSvm = null;
for (const C of [1, 10, 100, 1000]) {
  for (const g of [gammaScale, 0.2, 0.05, 0.01]) {
    const t1 = Date.now();
    const oofY = [], oofS = [];
    for (const fold of svmFolds) {
      const sc = fitScaler(fold.tr);
      const sub = balancedSubsample(fold.tr, SEED + 7);
      const nPosFold = fold.tr.reduce((s, r) => s + r.y, 0);
      const svc = trainSVC(scaleRows(sub, sc), C, g, { calibPos: nPosFold, calibTotal: fold.tr.length });
      fold.va.forEach((r) => { oofY.push(r.y); oofS.push(svc.prob(scaleOne(r.x, sc))); });
    }
    const auc = rocAuc(oofY, oofS);
    const bt = bestThreshold(oofY, oofS);
    log(`  SVM  C=${String(C).padStart(3)} gamma=${g.toFixed(4)}  CV AUC = ${auc.toFixed(4)}  CV F1 = ${bt.f1.toFixed(4)} @${bt.thr.toFixed(2)}  (${((Date.now() - t1) / 1000).toFixed(1)}s)`);
    if (!bestSvm || auc > bestSvm.auc) bestSvm = { C, gamma: g, auc, thr: bt.thr, cv: bt.f1 };
  }
}
log(`  -> best SVM: C=${bestSvm.C} gamma=${bestSvm.gamma.toFixed(6)} (CV AUC ${bestSvm.auc.toFixed(4)}, F1 ${bestSvm.cv.toFixed(4)} @${bestSvm.thr.toFixed(2)})`);

const t2 = Date.now();
const svmTrainRows = balancedSubsample(train, SEED + 7);
const trainPos = train.reduce((s, r) => s + r.y, 0);
const svc = trainSVC(scaleRows(svmTrainRows, scaler), bestSvm.C, bestSvm.gamma,
                     { calibPos: trainPos, calibTotal: train.length });
log(`  fitted SVM in ${((Date.now() - t2) / 1000).toFixed(1)}s on ${svmTrainRows.length} rows -> ${svc.svs.length} support vectors`);

/* ---- evaluate both on the untouched 20% test split ---- */
const dtMetrics = evaluate("DT", (r) => dtPredictTree(dtTree, r.x), bestDt.thr);
const svmMetrics = evaluate("SVM", (r) => svc.prob(scaleOne(r.x, scaler)), bestSvm.thr);

const out = renderArtifacts({
  dtTree, dtParams: bestDt, dtMetrics, scaler, svc, svmParams: bestSvm, svmMetrics,
  split: { train: train.length, test: test.length, seed: SEED, failures }
});
writeFileSync(resolve(HERE, "model-artifacts.js"), out, "utf8");
log(`Wrote model-artifacts.js (${(out.length / 1024).toFixed(1)} KB, ${svc.svs.length} SVs, ${countNodes(dtTree)} DT nodes)`);
/* ====================== RENDER model-artifacts.js ========================= */
function renderArtifacts(o) {
  const j = (v) => JSON.stringify(v);
  const r4 = (v) => +Number(v).toFixed(4);
  const r6 = (v) => +Number(v).toFixed(6);
  const svJson = JSON.stringify(o.svc.svs.map((s) => s.map(r6)));
  const coefJson = JSON.stringify(o.svc.dualCoef.map(r6));
  return `/**
 * model-artifacts.js  --  GENERATED by train-baselines.mjs - do not hand-edit.
 * ---------------------------------------------------------------------------
 * Fitted Decision Tree and SVM (RBF) artifacts trained on the AI4I 2020
 * Predictive Maintenance Dataset (${o.split.train + o.split.test} records, ${o.split.failures} failures).
 *
 * Pipeline (mirrors MODEL_METRICS.pipeline):
 *   features   AirTemp, ProcTemp, RPM, Torque, ToolWear, one-hot Type_H/L/M
 *   excluded   FailureType - derived from the label, so it would leak
 *   split      stratified 80/20, seed ${o.split.seed} -> ${o.split.train} train / ${o.split.test} test
 *   weighting  class_weight = "balanced"  (sklearn semantics: n / (2 * n_c))
 *   selection  stratified k-fold CV on ROC-AUC; the decision threshold is then
 *              tuned on the same out-of-fold predictions (maximising F1), because
 *              probabilities calibrated to a ${(100 * o.split.failures / (o.split.train + o.split.test)).toFixed(2)}% base rate
 *              rarely cross 0.50.
 *   DT         CART + Gini, max_depth=${o.dtParams.maxDepth}, min_samples_leaf=${o.dtParams.minLeaf},
 *              chosen by stratified 5-fold CV on ROC-AUC
 *   SVM        StandardScaler (fit on train) -> RBF SVC solved with SMO, C=${o.svmParams.C},
 *              gamma=${o.svc.gamma.toFixed(6)} ("scale"), Platt sigmoid probabilities calibrated
 *              against the ORIGINAL class prior (the fit uses a 1:6 balanced subsample so the
 *              kernel matrix stays tractable in pure JS),
 *              chosen by stratified 3-fold CV on ROC-AUC
 *
 * The metrics below were MEASURED by train-baselines.mjs on the held-out test
 * split. They are not copied from any other study.
 */
const DT_ARTIFACT = {
  cols: ${j(COLS)},
  params: { criterion: "gini", max_depth: ${o.dtParams.maxDepth}, min_samples_leaf: ${o.dtParams.minLeaf}, class_weight: "balanced" },
  cvAuc: ${r4(o.dtParams.auc)},
  cvF1: ${r4(o.dtParams.cv)},
  threshold: ${r4(o.dtParams.thr)},
  nodes: ${countNodes(o.dtTree)},
  depth: ${treeDepth(o.dtTree)},
  metrics: ${j(o.dtMetrics)}
};

const SVM_ARTIFACT = {
  cols: ${j(COLS)},
  params: { kernel: "rbf", C: ${o.svmParams.C}, gamma: ${Number(o.svc.gamma).toFixed(6)}, class_weight: "balanced", scaler: "StandardScaler" },
  cvAuc: ${r4(o.svmParams.auc)},
  cvF1: ${r4(o.svmParams.cv)},
  threshold: ${r4(o.svmParams.thr)},
  mean: ${j(o.scaler.mean.map(r4))},
  scale: ${j(o.scaler.scale.map(r4))},
  bias: ${Number(o.svc.b).toFixed(6)},
  plattA: ${Number(o.svc.A).toFixed(6)},
  plattB: ${Number(o.svc.B).toFixed(6)},
  supportVectors: ${o.svc.svs.length},
  metrics: ${j(o.svmMetrics)}
};
const DT_TREE = ${JSON.stringify(o.dtTree)};
const SVM_SV = ${svJson};
const SVM_COEF = ${coefJson};

/* ---- Decision Tree inference: walk the fitted CART tree ---------------- */
function dtPredict(row) {
  let node = DT_TREE;
  while (!node.leaf) node = row[node.f] <= node.thr ? node.l : node.r;
  return node.p;
}

/* ---- SVM inference: RBF kernel over the fitted support vectors ---------- */
function svmPredict(row) {
  const c = SVM_ARTIFACT.cols;
  const xs = new Array(c.length);
  for (let i = 0; i < c.length; i++) xs[i] = (row[c[i]] - SVM_ARTIFACT.mean[i]) / SVM_ARTIFACT.scale[i];
  const g = SVM_ARTIFACT.params.gamma, sv = SVM_SV, co = SVM_COEF;
  let sum = 0;
  for (let k = 0; k < sv.length; k++) {
    const s = sv[k];
    let d = 0;
    for (let i = 0; i < xs.length; i++) { const t = xs[i] - s[i]; d += t * t; }
    sum += co[k] * Math.exp(-g * d);
  }
  const margin = sum + SVM_ARTIFACT.bias;
  return 1 / (1 + Math.exp(-(margin * SVM_ARTIFACT.plattA + SVM_ARTIFACT.plattB)));
}

var EXTRA_MODEL_DATA = {
  dt: DT_ARTIFACT,
  svm: SVM_ARTIFACT,
  split: ${j(o.split)},
  provenance: "Fitted and measured by train-baselines.mjs on a stratified 80/20 split (seed ${o.split.seed}) of the AI4I 2020 dataset."
};
if (typeof window !== "undefined") {
  window.EXTRA_MODEL_DATA = EXTRA_MODEL_DATA;
  window.dtPredict = dtPredict;
  window.svmPredict = svmPredict;
}
`;
}
