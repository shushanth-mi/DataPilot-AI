/**
 * verify-artifacts.mjs
 * ---------------------------------------------------------------------------
 * Independent end-to-end check of the GENERATED model-artifacts.js.
 *
 * It rebuilds the exact stratified 80/20 split (seed 42) used during training,
 * then scores the held-out test split using ONLY the exported artifacts
 * (dtPredict / svmPredict from model-artifacts.js) and compares the numbers
 * with the metrics baked into the file.
 *
 * This proves the browser predictions are the same model that was measured -
 * it does not re-train anything.
 *
 * Usage:  node verify-artifacts.mjs [path-to-ai4i2020.csv]
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import vm from "node:vm";

const CSV_PATH = resolve(process.argv[2] || process.env.AI4I_CSV || "");
const ARTIFACTS = resolve(process.argv[3] || "model-artifacts.js");

/* ---------- load the exported artifacts in an isolated context ---------- */
const sandbox = { window: {}, console };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync(ARTIFACTS, "utf8"), sandbox, { filename: ARTIFACTS });
const { dtPredict, svmPredict, EXTRA_MODEL_DATA } = sandbox.window;
if (typeof dtPredict !== "function" || typeof svmPredict !== "function") {
  console.error("FAIL: model-artifacts.js did not export dtPredict / svmPredict");
  process.exit(1);
}

/* ---------- rebuild the dataset and the identical split ------------------ */
const COLS = ["AirTemp", "ProcTemp", "RPM", "Torque", "ToolWear", "Type_H", "Type_L", "Type_M"];
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
const lines = readFileSync(CSV_PATH, "utf8").replace(/\r/g, "").split("\n").filter((l) => l.trim());
const head = lines[0].split(",");
const ix = {};
head.forEach((h, i) => (ix[h.trim()] = i));
const rows = [];
for (let i = 1; i < lines.length; i++) {
  const c = lines[i].split(",");
  const t = c[ix["Type"]].trim();
  rows.push({
    row: {
      AirTemp: +c[ix["Air temperature [K]"]],
      ProcTemp: +c[ix["Process temperature [K]"]],
      RPM: +c[ix["Rotational speed [rpm]"]],
      Torque: +c[ix["Torque [Nm]"]],
      ToolWear: +c[ix["Tool wear [min]"]],
      Type_H: t === "H" ? 1 : 0, Type_L: t === "L" ? 1 : 0, Type_M: t === "M" ? 1 : 0
    },
    y: +c[ix["Machine failure"]] === 1 ? 1 : 0
  });
}
const rng = mulberry32(42);
const byClass = { 0: [], 1: [] };
rows.forEach((r, i) => byClass[r.y].push(i));
const testSet = new Set();
for (const cls of [0, 1]) {
  const idxs = shuffle(byClass[cls].slice(), rng);
  const nTest = Math.floor(idxs.length * 0.2);
  for (let i = 0; i < nTest; i++) testSet.add(idxs[i]);
}
const test = rows.filter((_, i) => testSet.has(i));
console.log(`rebuilt split: train=${rows.length - test.length} test=${test.length}`);

function prf1(y, s, thr) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < y.length; i++) {
    if (s[i] >= thr) { if (y[i]) tp++; else fp++; } else if (y[i]) fn++;
  }
  const p = tp + fp ? tp / (tp + fp) : 0;
  const r = tp + fn ? tp / (tp + fn) : 0;
  return { precision: p, recall: r, f1: p + r ? (2 * p * r) / (p + r) : 0, tp, fp, fn, tn: y.length - tp - fp - fn };
}
function rocAuc(y, s) {
  const pairs = s.map((v, i) => [v, y[i]]).sort((a, b) => a[0] - b[0]);
  let rank = 1, i = 0, sumPos = 0, nPos = 0;
  while (i < pairs.length) {
    let j = i;
    while (j + 1 < pairs.length && pairs[j + 1][0] === pairs[i][0]) j++;
    const mid = (rank + (rank + (j - i))) / 2;
    for (let t = i; t <= j; t++) if (pairs[t][1] === 1) { sumPos += mid; nPos++; }
    rank += j - i + 1; i = j + 1;
  }
  const nNeg = pairs.length - nPos;
  return nPos && nNeg ? (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg) : NaN;
}

const y = test.map((r) => r.y);
const dtS = test.map((r) => dtPredict(r.row));
const svmS = test.map((r) => svmPredict(r.row));

let failures = 0;
function check(name, actual, expected, tol) {
  const ok = Math.abs(actual - expected) <= tol;
  if (!ok) failures++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}: measured ${actual.toFixed(4)} vs artifact ${expected.toFixed(4)}`);
  return ok;
}
for (const [key, scores, art] of [["DT", dtS, EXTRA_MODEL_DATA.dt], ["SVM", svmS, EXTRA_MODEL_DATA.svm]]) {
  const m = art.metrics;
  const at = prf1(y, scores, m.threshold);
  console.log(`${key}  (threshold ${m.threshold})`);
  check(`${key} F1`, at.f1, m.f1, 0.005);
  check(`${key} precision`, at.precision, m.precision, 0.005);
  check(`${key} recall`, at.recall, m.recall, 0.005);
  check(`${key} ROC-AUC`, rocAuc(y, scores), m.rocAuc, 0.005);
  const cmOk = at.tp === m.confusion.tp && at.fp === m.confusion.fp &&
               at.fn === m.confusion.fn && at.tn === m.confusion.tn;
  if (!cmOk) failures++;
  console.log(`  ${cmOk ? "PASS" : "FAIL"}  ${key} confusion matrix [[${at.tn},${at.fp}],[${at.fn},${at.tp}]]`);
}
console.log(failures === 0
  ? "\nALL ARTIFACT CHECKS PASSED - the exported models reproduce the measured metrics."
  : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);