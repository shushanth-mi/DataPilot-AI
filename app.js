/* ============================================================================
   MACHINE FAILURE PREDICTION V4.0 — Predictive Maintenance Control Center
   ----------------------------------------------------------------------------
   REAL ML        : Random Forest (300 trees, class_weight=balanced) — the real
                    serialized model in model-data.js drives every risk number.
   ADVISORY/RULES : TWF / HDF / PWF / OSF / RNF indicators are rule-based
                    (documented AI4I failure definitions), NOT model outputs.
   SIMULATION     : Telemetry stream, scenarios, demo mode and the machine
                    fleet are simulated. This is an offline academic prototype.
   No fabricated ML outputs: all three models are real fitted artifacts.
   Random Forest (300 trees) ships in model-data.js; the fitted Decision Tree
   and RBF-SVM ship in model-artifacts.js, produced by train-baselines.mjs and
   re-checked against a held-out split by verify-artifacts.mjs.
   ============================================================================ */
"use strict";
(function () {
  const MD = window.MODEL_DATA || null;
  const EX = window.MODEL_EXTRA || null;

  /* ------------------------------ helpers ------------------------------- */
  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmt = (n, d = 1) => (Number.isFinite(Number(n)) ? Number(n).toFixed(d) : "--");
  const fmtInt = (n) => (Number.isFinite(Number(n)) ? Math.round(Number(n)).toLocaleString("en-US") : "--");
  const nowStamp = () => new Date().toLocaleTimeString("en-GB", { hour12: false });
  const nowFull = () => new Date().toLocaleString("en-GB", { hour12: false });
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const toC = (k) => (Number(k) - 273.15).toFixed(1);
  const typeLabel = (t) => (t === "M" ? "Medium (M)" : t === "H" ? "High (H)" : "Low (L)");

  /* ------------------------------ storage ------------------------------- */
  const Storage = (() => {
    let ok = false;
    try {
      const k = "__mfp_probe__";
      window.localStorage.setItem(k, "1");
      window.localStorage.removeItem(k);
      ok = true;
    } catch (e) { ok = false; }
    const mem = Object.create(null);
    return {
      available: ok,
      get(key, fallback) {
        try {
          const raw = ok ? window.localStorage.getItem(key) : mem[key];
          if (raw == null) return fallback;
          return JSON.parse(raw);
        } catch (e) { return fallback; }
      },
      set(key, value) {
        try {
          const raw = JSON.stringify(value);
          if (ok) window.localStorage.setItem(key, raw); else mem[key] = raw;
        } catch (e) { /* quota or blocked — ignore */ }
      },
      remove(key) {
        try { if (ok) window.localStorage.removeItem(key); else delete mem[key]; } catch (e) {}
      }
    };
  })();
  const LS_SETTINGS = "mfp_v4_settings";
  const LS_HISTORY = "mfp_v4_history";

  /* ------------------------------ settings ------------------------------ */
  const DEFAULT_SETTINGS = {
    simInterval: 700,   // ms
    simSpeed: 1,        // 0.5 | 1 | 2
    noise: 1,           // 0..3 multiplier on simulation noise
    autoRun: false,     // start telemetry simulation on load
    warnThreshold: 28,  // risk % — warning
    critThreshold: 65,  // risk % — critical
    compact: false      // compact UI mode
  };
  function loadSettings() {
    const raw = Storage.get(LS_SETTINGS, {}) || {};
    const s = Object.assign({}, DEFAULT_SETTINGS, raw);
    s.simInterval = clamp(Number(s.simInterval) || 700, 200, 5000);
    s.simSpeed = [0.5, 1, 2].indexOf(Number(s.simSpeed)) >= 0 ? Number(s.simSpeed) : 1;
    s.noise = clamp(Number.isFinite(Number(s.noise)) ? Number(s.noise) : 1, 0, 3);
    s.warnThreshold = clamp(Number(s.warnThreshold) || 28, 1, 99);
    s.critThreshold = clamp(Number(s.critThreshold) || 65, 2, 100);
    if (s.critThreshold <= s.warnThreshold) s.critThreshold = Math.min(100, s.warnThreshold + 1);
    s.autoRun = !!s.autoRun;
    s.compact = !!s.compact;
    return s;
  }
  function saveSettings() { Storage.set(LS_SETTINGS, state.settings); }

  /* -------------------------------- state ------------------------------- */
  const state = {
    machine: { id: "CNC-MILL-08" },
    sensors: { productType: "M", airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 1540, torque: 39.5, toolWear: 85 },
    prevSensors: null,
    prediction: null,          // output of MODEL_DATA.predictFailure (REAL Random Forest + advisory rules)
    contributions: [],         // real ablation-based contributions from the trained forest
    risk: 0,                   // percent
    health: 100,               // 0..100
    status: { key: "normal", label: "OPTIMAL OPERATION", tag: "LOW RISK", cls: "status-optimal" },
    alerts: [],
    alertSeq: 0,
    alertState: {},            // condition -> active flag (for transition detection)
    unreadAlerts: 0,
    history: [],
    telemetry: { air: [], proc: [], rpm: [], torque: [], wear: [], risk: [], max: 45 },
    sparks: { air: [], proc: [], rpm: [], torque: [], wear: [], risk: [], health: [], max: 30 },
    sim: { running: false, timer: null, cycle: 0 },
    demo: { active: false, playing: false, step: -1, timer: null },
    dataset: { rows: null, name: "", stats: null },
    scenario: { selected: null, running: null },
    fleet: { selected: null, cache: {} },
    view: "dashboard",
    lastRecordTs: 0,
    lastTelemetrySample: 0,
    booted: false
  };
  state.settings = loadSettings();
  /* ============================ MODEL READINESS ============================ */
  function modelReady() { return !!(MD && typeof MD.predictFailure === "function"); }
  function modelUnavailable(msg) {
    const chip = $("mlEngineChip");
    if (chip) {
      chip.classList.add("chip-bad");
      chip.classList.remove("chip-warn");
      chip.innerHTML = '<span class="sys-dot"></span> ML ENGINE UNAVAILABLE';
    }
    if (!state.modelWarned) {
      state.modelWarned = true;
      toast(msg || "Model unavailable for live inference. Check that model-data.js loaded correctly.", "error");
    }
  }

  /* ========================= SENSOR VALIDATION ============================= */
  const SENSOR_LIMITS = {
    airTemperature: { min: 295, max: 305, def: 298.5 },
    processTemperature: { min: 305, max: 315, def: 308.8 },
    rotationalSpeed: { min: 1100, max: 2900, def: 1540 },
    torque: { min: 3.5, max: 80, def: 39.5 },
    toolWear: { min: 0, max: 260, def: 85 }
  };
  function sanitizeSensors(src) {
    const type = ["L", "M", "H"].indexOf(src.productType) >= 0 ? src.productType : "M";
    const out = { productType: type };
    Object.keys(SENSOR_LIMITS).forEach((k) => {
      const lim = SENSOR_LIMITS[k];
      const num = Number(src[k]);
      out[k] = clamp(Number.isFinite(num) ? num : lim.def, lim.min, lim.max);
    });
    return out;
  }
  function sensorsEqual(a, b) {
    if (!a || !b) return false;
    return a.productType === b.productType && a.airTemperature === b.airTemperature &&
      a.processTemperature === b.processTemperature && a.rotationalSpeed === b.rotationalSpeed &&
      a.torque === b.torque && a.toolWear === b.toolWear;
  }
  function celsius(k) {
    if (MD && MD.features && MD.features.airTemperature && typeof MD.features.airTemperature.toCelsius === "function") {
      return MD.features.airTemperature.toCelsius(k);
    }
    return toC(k);
  }

  /* =========================== CLASSIFICATION ============================== */
  function statusFromRisk(risk) {
    const warn = state.settings.warnThreshold;
    const crit = state.settings.critThreshold;
    if (risk >= crit) return { key: "critical", label: "CRITICAL RISK", tag: "CRITICAL RISK", cls: "status-critical" };
    if (risk >= warn) return { key: "warning", label: "ATTENTION REQUIRED", tag: "MODERATE RISK", cls: "status-warning" };
    return { key: "normal", label: "OPTIMAL OPERATION", tag: "LOW RISK", cls: "status-optimal" };
  }
  function statusWord(key) {
    if (key === "critical") return "CRITICAL";
    if (key === "warning") return "WARNING";
    return "NORMAL";
  }
  // Advisory indicator level (rule-engine output, NOT a model output)
  function indicatorLevel(pct) {
    if (pct >= 50) return { key: "critical", label: "Critical" };
    if (pct >= 25) return { key: "warning", label: "Elevated" };
    return { key: "normal", label: "Normal" };
  }
  function indicators() {
    const p = state.prediction && state.prediction.failureModeProbabilities;
    if (!p) return null;
    return [
      { code: "TWF", label: "Tool Wear Indicator", pct: p.twf },
      { code: "HDF", label: "Heat Dissipation Indicator", pct: p.hdf },
      { code: "PWF", label: "Power Indicator", pct: p.pwf },
      { code: "OSF", label: "Overstrain Indicator", pct: p.osf },
      { code: "RNF", label: "Random Failure Indicator", pct: p.rnf }
    ];
  }
  /* ============================ MEMORY BUFFERS ============================= */
  function pushSample(arr, value, max) {
    arr.push(value);
    if (arr.length > max) arr.splice(0, arr.length - max);
  }
  function pushSamples() {
    const s = state.sensors, t = state.telemetry, k = state.sparks;
    pushSample(t.air, s.airTemperature, t.max);
    pushSample(t.proc, s.processTemperature, t.max);
    pushSample(t.rpm, s.rotationalSpeed, t.max);
    pushSample(t.torque, s.torque, t.max);
    pushSample(t.wear, s.toolWear, t.max);
    pushSample(t.risk, state.risk, t.max);
    pushSample(k.air, s.airTemperature, k.max);
    pushSample(k.proc, s.processTemperature, k.max);
    pushSample(k.rpm, s.rotationalSpeed, k.max);
    pushSample(k.torque, s.torque, k.max);
    pushSample(k.wear, s.toolWear, k.max);
    pushSample(k.risk, state.risk, k.max);
    pushSample(k.health, state.health, k.max);
  }

  /* ========================== PREDICTION HISTORY =========================== */
  const FORCE_RECORD_REASONS = { init: true, scenario: true, demo: true, fleet: true, preset: true, apply: true, manual: true, reset: true };
  function decideRecord(reason, force) {
    if (force === true) return true;
    if (force === false) return false;
    const now = Date.now();
    const last = state.history.length ? state.history[state.history.length - 1] : null;
    if (FORCE_RECORD_REASONS[reason]) return true;
    if (last && last.status !== statusWord(state.status.key)) return true;
    return now - state.lastRecordTs >= 3000;
  }
  function recordHistory(source) {
    const s = state.sensors;
    const row = {
      ts: Date.now(), time: nowStamp(), machine: state.machine.id,
      productType: s.productType, airTemperature: s.airTemperature, processTemperature: s.processTemperature,
      rotationalSpeed: s.rotationalSpeed, torque: s.torque, toolWear: s.toolWear,
      risk: Number(state.risk.toFixed(1)), health: Math.round(state.health),
      status: statusWord(state.status.key), source: source || "manual",
      model: "Random Forest (300 trees)"
    };
    state.history.push(row);
    if (state.history.length > 500) state.history = state.history.slice(-500);
    state.lastRecordTs = row.ts;
    Storage.set(LS_HISTORY, state.history);
    updateHistoryView();
    updateAnalyticsView();
  }
  function loadHistory() {
    const rows = Storage.get(LS_HISTORY, []);
    state.history = Array.isArray(rows)
      ? rows.filter((r) => r && Number.isFinite(Number(r.risk))).slice(-500)
      : [];
    state.lastRecordTs = state.history.length ? Number(state.history[state.history.length - 1].ts) || 0 : 0;
  }
  function clearHistory() {
    state.history = [];
    state.lastRecordTs = 0;
    Storage.remove(LS_HISTORY);
    updateHistoryView();
    updateAnalyticsView();
  }

  /* ============================ CENTRAL PIPELINE =========================== */
  function runPipeline(reason, opts) {
    const options = opts || {};
    if (!modelReady()) { modelUnavailable(); return; }
    const changed = !sensorsEqual(state.sensors, state.prevSensors);
    state.prevSensors = Object.assign({}, state.sensors);
    state.sensors = sanitizeSensors(state.sensors);
    let prediction;
    try {
      prediction = MD.predictFailure(state.sensors); // REAL Random Forest inference
    } catch (err) {
      console.error("Inference failed:", err);
      modelUnavailable("Inference failed: " + err.message);
      return;
    }
    state.prediction = prediction;
    state.risk = clamp(Number(prediction.probabilityPercentage) || 0, 0, 100);
    state.health = Number.isFinite(Number(prediction.healthIndex))
      ? Number(prediction.healthIndex)
      : Math.max(0, Math.round(100 - state.risk));
    state.status = statusFromRisk(state.risk);
    state.contributions = (EX && typeof EX.computeContributions === "function")
      ? EX.computeContributions(state.sensors)
      : [];
    state.changed = changed;
    pushSamples();
    updateAlerts(reason);
    if (decideRecord(reason, options.record)) recordHistory(reason);
    renderAll(reason);
  }
  /* ============================== ALERTS ENGINE ============================ */
  /* Alerts are derived from the current REAL Random Forest risk plus the
     advisory rule engine (documented AI4I-style indicator definitions).
     This is a prototype alert feed — not an industrial safety system. */
  function overstrainThreshold(type) {
    if (type === "L") return 11000;
    if (type === "H") return 13000;
    return 12000; // Type M
  }
  function alertConditions() {
    const p = state.prediction;
    if (!p) return [];
    const s = state.sensors;
    const pm = p.physicsMetrics || {};
    const conds = [];
    if (state.status.key === "critical") {
      conds.push({
        key: "risk", level: "critical", view: "prediction",
        title: "Critical machine failure risk detected",
        detail: "Random Forest failure risk is " + fmt(state.risk, 1) + "% (critical threshold " + state.settings.critThreshold + "%).",
        clearMsg: "Failure risk returned to " + fmt(state.risk, 1) + "% — below the critical threshold."
      });
    } else if (state.status.key === "warning") {
      conds.push({
        key: "risk", level: "warning", view: "prediction",
        title: "Elevated machine failure risk",
        detail: "Random Forest failure risk is " + fmt(state.risk, 1) + "% (warning threshold " + state.settings.warnThreshold + "%).",
        clearMsg: "Failure risk returned to " + fmt(state.risk, 1) + "% — below the warning threshold."
      });
    }
    if (s.toolWear >= 200) {
      const lvl = s.toolWear >= 240 ? "critical" : "warning";
      conds.push({
        key: "twf", level: lvl, view: "maintenance",
        title: lvl === "critical" ? "Tool wear at hard advisory limit" : "Tool wear approaching advisory limit",
        detail: "Tool wear is " + s.toolWear + " min. Advisory tool-wear limit is 200 min (AI4I TWF definition) — schedule an insert replacement.",
        clearMsg: "Tool wear back below the 200 min advisory limit (currently " + s.toolWear + " min)."
      });
    }
    if (Number(pm.tempDelta) < 8.6 && s.rotationalSpeed < 1380) {
      conds.push({
        key: "hdf", level: "warning", view: "prediction",
        title: "Heat dissipation margin reduced",
        detail: "Temperature delta is " + fmt(pm.tempDelta, 1) + " K at " + s.rotationalSpeed + " rpm — below the advisory heat-dissipation margin (8.6 K / 1380 rpm).",
        clearMsg: "Heat dissipation conditions returned to the advisory operating band."
      });
    }
    if (Number(pm.powerWatts) > 9000 || Number(pm.powerWatts) < 3500) {
      conds.push({
        key: "pwf", level: "warning", view: "prediction",
        title: "Calculated spindle power outside advisory band",
        detail: "Calculated power is " + fmtInt(pm.powerWatts) + " W — outside the advisory 3,500–9,000 W band (AI4I PWF definition).",
        clearMsg: "Calculated spindle power returned to the advisory operating band."
      });
    }
    const osfThreshold = overstrainThreshold(s.productType);
    if (Number(pm.overstrain) >= osfThreshold) {
      conds.push({
        key: "osf", level: "warning", view: "prediction",
        title: "Overstrain product exceeds advisory threshold",
        detail: "Overstrain metric is " + fmtInt(pm.overstrain) + " min·Nm — at or above the advisory " + fmtInt(osfThreshold) + " min·Nm threshold for Type " + s.productType + " (AI4I OSF definition).",
        clearMsg: "Overstrain product returned below the advisory threshold."
      });
    }
    return conds;
  }
  /* ========================= RENDER DISPATCHER ============================= */
  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value;
  }
  function renderAll(reason) {
    const jobs = [
      updateHeaderStatus, updateDashboardOverview, updateKpiCards, updateConsensus,
      syncUIFromState, updatePredictionReadout,
      updateTelemetryRows, updateWaveform, updateRiskTrends, updateXai,
      updateMaintenance, updateScenarioIndicator, updateFleetMetrics, updateViz3D,
      updateHistoryView, updateAnalyticsView, updateModelsView, updateDatasetView, updateExportNote
    ];
    jobs.forEach((fn) => {
      try {
        if (typeof fn === "function") fn(reason);
      } catch (err) {
        console.error("Renderer failed:", fn ? fn.name : "?", err);
      }
    });
    try { ccSync(reason); } catch (err) { /* V6 presentation layer is strictly optional */ }
  }

  /* ============================ HEADER STATUS ============================== */
  function updateHeaderStatus() {
    const eng = $("mlEngineChip");
    if (eng) {
      const ready = modelReady();
      eng.classList.toggle("chip-bad", !ready);
      eng.classList.remove("chip-warn");
      eng.innerHTML = '<span class="sys-dot"></span> ' + (ready ? "ML ENGINE READY" : "ML ENGINE UNAVAILABLE");
    }
    const simPill = $("simStatePill");
    if (simPill) simPill.textContent = "Simulation: " + (state.sim.running ? "RUNNING" : "OFF");
    const label = $("simBtnLabel");
    if (label) label.textContent = "Telemetry Sim: " + (state.sim.running ? "RUNNING" : "OFF");
    const btn = $("btnToggleSim");
    if (btn) {
      btn.classList.toggle("active", state.sim.running);
      btn.setAttribute("aria-pressed", state.sim.running ? "true" : "false");
    }
    setText("activeAssetId", state.machine.id);
    setText("dashUpdatedAt", "Last pipeline run: " + nowStamp() + " (simulated)");
  }

  /* ========================== DASHBOARD OVERVIEW =========================== */
  function updateDashboardOverview() {
    setText("dashMachineId", state.machine.id);
    setText("dashHealthScore", String(Math.round(state.health)));
    setText("dashRisk", fmt(state.risk, 1));
    const st = $("dashStatus");
    if (st) {
      st.textContent = statusWord(state.status.key);
      st.className = "hov-value hov-status hov-" + state.status.key;
    }
    setText("dashModelStatus", modelReady() ? "Random Forest — live inference" : "Unavailable");
    setText("dashDataStatus", state.dataset.rows ? "Loaded AI4I CSV (user file)" : (state.sim.running ? "Simulated telemetry — streaming" : "Simulated telemetry"));
    const bar = $("dashHealthBar");
    if (bar) {
      bar.style.width = clamp(state.health, 0, 100) + "%";
      bar.className = "hov-bar-fill hov-bar-" + state.status.key;
    }
  }

  /* ============================ MODEL CONSENSUS ============================ */
  function modelMeta() {
    const fallback = [
      { key: "dt", name: "Decision Tree", live: false, artifact: "none" },
      { key: "rf", name: "Random Forest", live: true, artifact: "Full serialized 300-tree forest bundled in model-data.js" },
      { key: "svm", name: "SVM (RBF)", live: false, artifact: "none" }
    ];
    if (EX && EX.MODEL_METRICS && Array.isArray(EX.MODEL_METRICS.models)) return EX.MODEL_METRICS.models;
    return fallback;
  }
  function buildConsensusRows() {
    const host = $("consensusList");
    if (!host) return;
    host.innerHTML = modelMeta().map((m) => `
      <div class="consensus-row" id="consRow-${m.key}">
        <span class="cons-name">${esc(m.name)}</span>
        <span class="cons-verdict" id="consVerdict-${m.key}">--</span>
        <span class="cons-prob" id="consProb-${m.key}">--</span>
      </div>`).join("");
  }
  function updateConsensus() {
    if (!$("consensusList")) return;
    if (!$("consRow-rf")) buildConsensusRows();
    const models = modelMeta();
    const per = (state.prediction && state.prediction.perModelPct) || null;
    let available = 0;
    models.forEach((m) => {
      const verdict = $("consVerdict-" + m.key);
      const prob = $("consProb-" + m.key);
      const row = $("consRow-" + m.key);
      if (!verdict || !prob || !row) return;
      const value = per && Number.isFinite(per[m.key]) ? per[m.key] : null;
      if (m.live && modelReady() && value !== null) {
        available++;
        verdict.textContent = "LIVE INFERENCE";
        verdict.className = "cons-verdict cons-live";
        prob.textContent = fmt(value, 1) + "% · " + (m.key === "rf" ? statusWord(state.status.key) : "measured");
        row.className = "consensus-row cons-active";
      } else if (m.live && modelReady()) {
        available++;
        verdict.textContent = "LIVE INFERENCE";
        verdict.className = "cons-verdict cons-live";
        prob.textContent = fmt(state.risk, 1) + "% · " + statusWord(state.status.key);
        row.className = "consensus-row cons-active";
      } else {
        verdict.textContent = "ARTIFACT NOT CONNECTED";
        verdict.className = "cons-verdict cons-absent";
        prob.textContent = "No live prediction";
        row.className = "consensus-row";
      }
    });
    setText("consensusCount", available + " / " + models.length + " live");
    /* live readouts on the AI risk-engine diagram in the Overview hero */
    setText("nnDtVal", per && Number.isFinite(per.dt) ? fmt(per.dt, 1) + "%" : "--");
    setText("nnSvmVal", per && Number.isFinite(per.svm) ? fmt(per.svm, 1) + "%" : "--");
    const agree = state.prediction && state.prediction.agreement;
    if (available === 3 && agree) {
      setText("consensusNote", "All three fitted models score every reading live in this browser. "
        + "Mean risk " + fmt(agree.mean, 1) + "%, spread " + fmt(agree.spread, 1)
        + " pp, " + agree.votes + " of 3 above the 28% advisory line"
        + (agree.unanimous ? " (models agree)." : " (models disagree - treat the band as uncertain).")
        + " The headline figure stays the Random Forest, which every documented scenario value in this project was measured against.");
    } else if (available === 1) {
      setText("consensusNote", "Only the Random Forest artifact is loaded. Decision Tree and SVM run when model-artifacts.js is present.");
    } else {
      setText("consensusNote", "Model availability changed - check that model-artifacts.js loaded.");
    }
  }
  function pushAlert(level, title, detail, view) {
    state.alerts.unshift({ id: ++state.alertSeq, level: level, title: title, detail: detail, view: view || "prediction", time: nowStamp() });
    if (state.alerts.length > 60) state.alerts.length = 60;
    state.unreadAlerts = Math.min(99, state.unreadAlerts + 1);
  }
  function updateAlerts(reason) {
    const active = {};
    alertConditions().forEach((c) => { active[c.key] = c; });
    Object.keys(active).forEach((k) => {
      const prev = state.alertState[k];
      const now = active[k];
      if (!prev || !prev.active || prev.level !== now.level) {
        pushAlert(now.level, now.title, now.detail, now.view);
      }
    });
    Object.keys(state.alertState).forEach((k) => {
      const prev = state.alertState[k];
      if (prev.active && !active[k]) {
        pushAlert("normal", "Condition cleared", prev.clearMsg, prev.view);
      }
    });
    const next = {};
    Object.keys(active).forEach((k) => {
      next[k] = { active: true, level: active[k].level, clearMsg: active[k].clearMsg, view: active[k].view };
    });
    state.alertState = next;
    if (!state.alerts.length && reason === "init") {
      pushAlert("normal", "Machine operating normally", "Baseline pipeline run completed. No advisory conditions are active.", "dashboard");
    }
    renderAlerts();
  }
  function renderAlerts() {
    const badge = $("alertCount");
    if (badge) badge.textContent = state.unreadAlerts > 9 ? "9+" : String(state.unreadAlerts);
    const list = $("alertList");
    if (list) {
      list.innerHTML = state.alerts.length ? state.alerts.map((a) => `
        <button type="button" class="alert-item alert-${a.level}" data-alert-view="${esc(a.view)}">
          <span class="alert-item-head"><span class="alert-level-tag">${a.level.toUpperCase()}</span><span class="alert-time">${esc(a.time)}</span></span>
          <span class="alert-item-title">${esc(a.title)}</span>
          <span class="alert-item-detail">${esc(a.detail)}</span>
        </button>`).join("")
        : '<div class="muted-note">No alerts yet. Run a prediction, a scenario or the telemetry simulation.</div>';
    }
    const feed = $("dashboardAlerts");
    if (feed) {
      const recent = state.alerts.slice(0, 4);
      feed.innerHTML = recent.length ? recent.map((a) => `
        <div class="alert-feed-item alert-${a.level}">
          <span class="alert-level-tag">${a.level.toUpperCase()}</span>
          <div class="alert-feed-body"><span class="alert-item-title">${esc(a.title)}</span>
          <span class="alert-item-detail">${esc(a.detail)}</span></div>
          <span class="alert-time">${esc(a.time)}</span>
        </div>`).join("")
        : '<div class="muted-note">No alerts yet. Alerts are raised by this prototype\'s own thresholds and advisory indicators.</div>';
    }
  }
  function acknowledgeAlerts() {
    state.unreadAlerts = 0;
    renderAlerts();
  }
  function toggleAlertPanel(force) {
    const panel = $("alertPanel");
    if (!panel) return;
    const show = typeof force === "boolean" ? force : panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !show);
    if (show) acknowledgeAlerts();
  }
  /* ============================== KPI CARDS =============================== */
  const KPI_ICONS = {
    risk: '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    health: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    air: '<path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/>',
    proc: '<path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>',
    rpm: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1"/>',
    torque: '<path d="M21 12a9 9 0 1 1-9-9"/><polyline points="21 3 21 9 15 9"/>',
    wear: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'
  };
  function riskColor(risk) {
    if (risk >= state.settings.critThreshold) return "var(--status-critical)";
    if (risk >= state.settings.warnThreshold) return "var(--status-warning)";
    return "var(--status-optimal)";
  }
  function kpiDefs() {
    const ind = indicators() || [];
    const pctOf = (codes) => {
      let max = 0;
      codes.forEach((c) => {
        const f = ind.filter((x) => x.code === c)[0];
        if (f && f.pct > max) max = f.pct;
      });
      return max;
    };
    const lvl = (codes) => indicatorLevel(pctOf(codes));
    const s = state.sensors;
    return [
      { key: "risk", label: "Failure Risk", unit: "%", value: state.risk, decimals: 1, level: state.status.key, levelLabel: statusWord(state.status.key), spark: state.sparks.risk, color: riskColor(state.risk), note: "Random Forest output" },
      { key: "health", label: "Machine Health", unit: "/ 100", value: state.health, decimals: 0, level: state.status.key, levelLabel: statusWord(state.status.key), spark: state.sparks.health.length ? state.sparks.health : state.sparks.risk.map((r) => Math.max(0, 100 - r)), color: riskColor(state.risk), note: "Derived from RF risk" },
      { key: "air", label: "Air Temperature", unit: "K", value: s.airTemperature, decimals: 1, level: lvl(["HDF"]).key, levelLabel: lvl(["HDF"]).label, spark: state.sparks.air, color: "var(--accent-cyan)", note: "Linked advisory: HDF" },
      { key: "proc", label: "Process Temperature", unit: "K", value: s.processTemperature, decimals: 1, level: lvl(["HDF"]).key, levelLabel: lvl(["HDF"]).label, spark: state.sparks.proc, color: "var(--accent-blue)", note: "Linked advisory: HDF" },
      { key: "rpm", label: "Rotational Speed", unit: "rpm", value: s.rotationalSpeed, decimals: 0, level: lvl(["HDF", "PWF"]).key, levelLabel: lvl(["HDF", "PWF"]).label, spark: state.sparks.rpm, color: "var(--accent-indigo)", note: "Linked advisory: HDF / PWF" },
      { key: "torque", label: "Torque", unit: "Nm", value: s.torque, decimals: 1, level: lvl(["PWF", "OSF"]).key, levelLabel: lvl(["PWF", "OSF"]).label, spark: state.sparks.torque, color: "var(--status-warning)", note: "Linked advisory: PWF / OSF" },
      { key: "wear", label: "Tool Wear", unit: "min", value: s.toolWear, decimals: 0, level: lvl(["TWF"]).key, levelLabel: lvl(["TWF"]).label, spark: state.sparks.wear, color: "var(--status-critical)", note: "Linked advisory: TWF" }
    ];
  }
  function buildKpiCards() {
    const host = $("kpiGrid");
    if (!host) return;
    host.innerHTML = kpiDefs().map((d) => `
      <div class="kpi-card" id="kpiCard-${d.key}" title="${esc(d.note)}">
        <div class="kpi-head">
          <span class="kpi-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${KPI_ICONS[d.key] || ""}</svg></span>
          <span class="kpi-label">${esc(d.label)}</span>
        </div>
        <div class="kpi-value-row"><span class="kpi-value" id="kpiVal-${d.key}">--</span><span class="kpi-unit">${esc(d.unit)}</span></div>
        <div class="kpi-foot">
          <span class="kpi-status" id="kpiStatus-${d.key}">--</span>
          <span class="kpi-trend" id="kpiTrend-${d.key}">--</span>
        </div>
        <canvas class="kpi-spark" id="kpiSpark-${d.key}" width="240" height="48" aria-hidden="true"></canvas>
      </div>`).join("");
  }
  function trendOf(series, invert) {
    if (!series || series.length < 6) return { arrow: "—", cls: "trend-flat", delta: null };
    const last = series[series.length - 1];
    const prev = series[series.length - 6];
    const delta = last - prev;
    const eps = Math.abs(prev) * 0.001 + 0.0001;
    if (Math.abs(delta) <= eps) return { arrow: "→", cls: "trend-flat", delta: 0 };
    const rising = delta > 0;
    const good = invert ? !rising : rising;
    return { arrow: rising ? "↑" : "↓", cls: good ? "trend-good" : "trend-bad", delta: delta };
  }
  function updateKpiCards() {
    const host = $("kpiGrid");
    if (!host) return;
    if (!$("kpiCard-risk")) buildKpiCards();
    kpiDefs().forEach((d) => {
      const card = $("kpiCard-" + d.key);
      if (!card) return;
      setText("kpiVal-" + d.key, fmt(d.value, d.decimals));
      card.className = "kpi-card kpi-" + d.level;
      const st = $("kpiStatus-" + d.key);
      if (st) {
        st.textContent = d.levelLabel;
        st.className = "kpi-status kpi-status-" + d.level;
      }
      const tr = $("kpiTrend-" + d.key);
      if (tr) {
        const invert = d.key === "health";
        const t = trendOf(d.spark, invert);
        tr.textContent = t.delta == null ? "—" : t.arrow + " " + fmt(Math.abs(t.delta), d.decimals);
        tr.className = "kpi-trend " + t.cls;
      }
      const cv = $("kpiSpark-" + d.key);
      if (cv && typeof drawSparkline === "function") {
        drawSparkline(cv, d.spark.slice(-30), d.color);
      }
    });
  }
  /* ======================== PREDICTION VIEW READOUT ======================== */
  function updatePredictionReadout() {
    const result = state.prediction;
    if (!result) return;
    const pm = result.physicsMetrics || {};
    setText("riskPercentage", String(result.probabilityPercentage));
    setText("riskLevelTag", state.status.tag);
    setText("healthIndexVal", Math.round(state.health) + " / 100");
    const gauge = $("gaugeFill");
    if (gauge) {
      const circumference = 440;
      const offset = circumference - (circumference * Number(result.probability));
      gauge.style.strokeDashoffset = Math.max(0, offset);
      gauge.style.stroke = state.status.key === "critical" ? "var(--status-critical)"
        : state.status.key === "warning" ? "var(--status-warning)" : "var(--status-optimal)";
    }
    const card = $("predictionOverviewCard");
    if (card) card.className = "card prediction-overview-card " + state.status.cls;
    const badge = $("machineStatusBadge");
    if (badge) {
      badge.style.color = state.status.key === "critical" ? "#fb7185"
        : state.status.key === "warning" ? "#fbbf24" : "#34d399";
    }
    setText("machineStatusText", result.status);
    setText("metricPower", fmtInt(pm.powerWatts) + " W");
    setText("metricTempDelta", fmt(pm.tempDelta, 1) + " K");
    setText("metricOverstrain", fmtInt(pm.overstrain) + " min·Nm");
    setText("predStamp", "Prediction time: " + nowStamp() + " · Primary model: Random Forest (300 trees) · Inference: browser-local · Result recorded in history");
    renderIndicators(result.failureModeProbabilities);
    renderPrescriptions(result.recommendations);
  }
  function renderIndicators(fm) {
    if (!fm) return;
    const map = [["TWF", fm.twf], ["HDF", fm.hdf], ["PWF", fm.pwf], ["OSF", fm.osf], ["RNF", fm.rnf]];
    map.forEach((pair) => {
      const code = pair[0], value = Number(pair[1]) || 0;
      const pctEl = $("pct" + code), barEl = $("bar" + code), lvlEl = $("lvl" + code);
      if (pctEl) pctEl.textContent = fmt(value, 1) + "%";
      if (barEl) {
        barEl.style.width = clamp(value, 1, 100) + "%";
        barEl.className = "mode-progress-bar";
        if (value >= 50) barEl.classList.add("bar-critical");
        else if (value >= 25) barEl.classList.add("bar-warning");
      }
      if (lvlEl) {
        const lv = indicatorLevel(value);
        lvlEl.textContent = lv.label;
        lvlEl.className = "mode-level mode-level-" + lv.key;
      }
    });
    const note = $("indicatorNote");
    if (note) {
      const active = map.filter((p) => Number(p[1]) >= 25).map((p) => p[0]);
      note.textContent = active.length
        ? "Elevated advisory indicators: " + active.join(", ") + ". These values are rule-based advisory scores (documented AI4I definitions) — they are NOT Random Forest outputs."
        : "All advisory indicators are within their normal bands. These values are rule-based (documented AI4I definitions) — they are NOT Random Forest outputs.";
    }
  }
  function renderPrescriptions(recommendations) {
    const host = $("prescriptionsList");
    if (!host) return;
    if (!recommendations || !recommendations.length) {
      host.innerHTML = '<div class="presc-reason">No active prescriptive alerts. Operating parameters are within the advisory bands of this prototype.</div>';
      return;
    }
    host.innerHTML = recommendations.map((r) => `
      <div class="prescription-card priority-${esc(r.priority)}">
        <div class="presc-icon" aria-hidden="true">${getPrescriptionIcon(r.priority)}</div>
        <div class="presc-details">
          <div class="presc-action-row">
            <span class="presc-badge">${esc(r.priority)}</span>
            <span class="presc-action">${esc(r.action)}</span>
          </div>
          <div class="presc-reason">${esc(r.reason)}</div>
        </div>
      </div>`).join("");
  }
  function getPrescriptionIcon(priority) {
    if (priority === "CRITICAL" || priority === "URGENT") {
      return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    }
    return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>';
  }
  /* ==================== PRESETS + UI SYNCHRONIZATION ====================== */
  function renderPresets() {
    const host = $("presetsList");
    if (!host || !MD || !MD.presetScenarios) return;
    host.innerHTML = MD.presetScenarios.map((preset) => `
      <button type="button" class="preset-chip" data-preset-id="${esc(preset.id)}" title="${esc(preset.description)}">
        <span class="chip-tag ${esc(preset.badgeClass)}">${esc(preset.badge)}</span>
        <span>${esc(preset.name)}</span>
      </button>`).join("");
  }
  function loadPreset(presetId) {
    const preset = (MD && MD.presetScenarios ? MD.presetScenarios : []).filter((p) => p.id === presetId)[0];
    if (!preset) return;
    state.sensors.productType = preset.values.productType;
    state.sensors.airTemperature = preset.values.airTemperature;
    state.sensors.processTemperature = preset.values.processTemperature;
    state.sensors.rotationalSpeed = preset.values.rotationalSpeed;
    state.sensors.torque = preset.values.torque;
    state.sensors.toolWear = preset.values.toolWear;
    document.querySelectorAll(".preset-chip").forEach((c) => {
      c.classList.toggle("active", c.getAttribute("data-preset-id") === presetId);
    });
    syncUIFromState();
    runPipeline("preset");
  }
  function clearActivePresetHighlight() {
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
  }
  function syncUIFromState() {
    const s = state.sensors;
    const setVal = (id, value) => { const n = $(id); if (n) n.value = value; };
    const setNum = (id, value) => { const n = $(id); if (n && document.activeElement !== n) n.value = value; };
    setVal("sliderAirTemp", s.airTemperature);
    setText("valAirTemp", s.airTemperature.toFixed(1));
    setText("valAirTempC", "(" + celsius(s.airTemperature) + "°C)");
    setVal("sliderProcTemp", s.processTemperature);
    setText("valProcTemp", s.processTemperature.toFixed(1));
    setText("valProcTempC", "(" + celsius(s.processTemperature) + "°C)");
    setVal("sliderSpeed", s.rotationalSpeed);
    setText("valSpeed", String(Math.round(s.rotationalSpeed)));
    setVal("sliderTorque", s.torque);
    setText("valTorque", s.torque.toFixed(1));
    setVal("sliderWear", s.toolWear);
    setText("valWear", String(Math.round(s.toolWear)));
    setNum("numAirTemp", s.airTemperature.toFixed(1));
    setNum("numProcTemp", s.processTemperature.toFixed(1));
    setNum("numSpeed", String(Math.round(s.rotationalSpeed)));
    setNum("numTorque", s.torque.toFixed(1));
    setNum("numWear", String(Math.round(s.toolWear)));
    document.querySelectorAll(".type-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-type") === s.productType);
    });
    setText("badgeProductType", "Type " + s.productType);
  }

  /* ==================== TELEMETRY REGISTRY TABLE ========================== */
  /* Reference rows from the training-study notes; risk is RECOMPUTED live by
     the real Random Forest (never the stored value). */
  function renderHistoricalTable() {
    const body = $("telemetryTableBody");
    if (!body) return;
    const logs = (MD && MD.historicalLogs) || [];
    if (!logs.length || !modelReady()) {
      body.innerHTML = '<tr><td colspan="10" style="color:var(--text-dim);">Reference telemetry unavailable.</td></tr>';
      return;
    }
    body.innerHTML = logs.map((log) => {
      let risk = Number(log.risk);
      let word = log.status;
      let statusClass = "table-status-normal";
      try {
        const r = MD.predictFailure({
          productType: log.type, airTemperature: log.airTemp, processTemperature: log.procTemp,
          rotationalSpeed: log.speed, torque: log.torque, toolWear: log.wear
        });
        risk = Number(r.probabilityPercentage);
        const st = statusFromRisk(risk);
        word = statusWord(st.key).charAt(0) + statusWord(st.key).slice(1).toLowerCase();
        statusClass = st.key === "critical" ? "table-status-critical" : st.key === "warning" ? "table-status-warning" : "table-status-normal";
      } catch (e) { /* keep stored values if the model is unavailable */ }
      const rowClass = word === "Critical" ? "row-critical" : "";
      const indColor = log.failureType === "None" ? "var(--text-dim)" : "#fb7185";
      return `
        <tr class="${rowClass}" data-log-id="${esc(log.id)}">
          <td style="font-family:var(--font-mono);font-weight:600;color:#38bdf8;">${esc(log.id)}</td>
          <td><span class="chip-tag" style="background:rgba(255,255,255,0.06);color:#f1f5f9;">Type ${esc(log.type)}</span></td>
          <td>${Number(log.airTemp).toFixed(1)} K</td>
          <td>${Number(log.procTemp).toFixed(1)} K</td>
          <td>${log.speed} rpm</td>
          <td>${Number(log.torque).toFixed(1)} Nm</td>
          <td>${log.wear} min</td>
          <td style="font-family:var(--font-mono);font-weight:700;">${risk.toFixed(1)}%</td>
          <td><span class="table-status-pill ${statusClass}">${esc(word)}</span></td>
          <td style="font-family:var(--font-mono);font-size:0.75rem;color:${indColor};">${esc(log.failureType)}</td>
        </tr>`;
    }).join("");
  }
  /* ======================= CONTROL BINDINGS (INPUT) ======================= */
  function bindPredictionControls() {
    const sliders = [
      { id: "sliderAirTemp", key: "airTemperature", isFloat: true },
      { id: "sliderProcTemp", key: "processTemperature", isFloat: true },
      { id: "sliderSpeed", key: "rotationalSpeed", isFloat: false },
      { id: "sliderTorque", key: "torque", isFloat: true },
      { id: "sliderWear", key: "toolWear", isFloat: false }
    ];
    sliders.forEach((cfg) => {
      const node = $(cfg.id);
      if (!node) return;
      node.addEventListener("input", (e) => {
        state.sensors[cfg.key] = cfg.isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
        clearActivePresetHighlight();
        syncUIFromState();
        runPipeline("slider");
      });
    });
    document.querySelectorAll(".type-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const type = btn.getAttribute("data-type");
        state.sensors.productType = type;
        document.querySelectorAll(".type-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        setText("badgeProductType", "Type " + type);
        clearActivePresetHighlight();
        runPipeline("manual");
      });
    });
    const presetsHost = $("presetsList");
    if (presetsHost) {
      presetsHost.addEventListener("click", (e) => {
        const chip = e.target.closest(".preset-chip");
        if (!chip) return;
        loadPreset(chip.getAttribute("data-preset-id"));
      });
    }
    const reset = $("btnResetDefaults");
    if (reset) reset.addEventListener("click", () => loadPreset("nominal"));
    const tableBody = $("telemetryTableBody");
    if (tableBody) {
      tableBody.addEventListener("click", (e) => {
        const tr = e.target.closest("tr");
        if (!tr) return;
        const logId = tr.getAttribute("data-log-id");
        const record = ((MD && MD.historicalLogs) || []).filter((r) => r.id === logId)[0];
        if (!record) return;
        state.sensors.productType = record.type;
        state.sensors.airTemperature = Number(record.airTemp);
        state.sensors.processTemperature = Number(record.procTemp);
        state.sensors.rotationalSpeed = Number(record.speed);
        state.sensors.torque = Number(record.torque);
        state.sensors.toolWear = Number(record.wear);
        state.machine.id = record.id;
        clearActivePresetHighlight();
        syncUIFromState();
        runPipeline("manual");
        toast("Loaded reference telemetry " + record.id + " (simulated batch record).", "ok");
      });
    }
    const applyBtn = $("btnApplySensors");
    if (applyBtn) applyBtn.addEventListener("click", applyNumericInputs);
    ["numAirTemp", "numProcTemp", "numSpeed", "numTorque", "numWear"].forEach((id) => {
      const node = $(id);
      if (!node) return;
      node.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); applyNumericInputs(); } });
    });
  }
  function applyNumericInputs() {
    const map = [
      { id: "numAirTemp", key: "airTemperature", label: "Air Temperature", min: 295, max: 305 },
      { id: "numProcTemp", key: "processTemperature", label: "Process Temperature", min: 305, max: 315 },
      { id: "numSpeed", key: "rotationalSpeed", label: "Rotational Speed", min: 1100, max: 2900 },
      { id: "numTorque", key: "torque", label: "Torque", min: 3.5, max: 80 },
      { id: "numWear", key: "toolWear", label: "Tool Wear", min: 0, max: 260 }
    ];
    const problems = [];
    map.forEach((m) => {
      const node = $(m.id);
      if (!node) return;
      const raw = String(node.value).trim();
      const num = Number(raw);
      if (raw === "" || !Number.isFinite(num)) {
        problems.push(m.label + " was empty — kept previous value");
        return;
      }
      if (num < m.min || num > m.max) {
        problems.push(m.label + " " + num + " adjusted to " + clamp(num, m.min, m.max) + " (" + m.min + "–" + m.max + ")");
      }
      state.sensors[m.key] = clamp(num, m.min, m.max);
    });
    clearActivePresetHighlight();
    syncUIFromState();
    runPipeline("apply");
    if (problems.length) toast(problems.join(" · "), "warn");
    else toast("Sensor values applied — Random Forest inference complete.", "ok");
  }
  /* ======================= TELEMETRY SIMULATION =========================== */
  /* The stream is a documented SIMULATION of a telemetry feed — it is not a
     live factory connection. Every reading it produces is still scored by the
     real Random Forest through the central pipeline. */
  function startSimulation() {
    if (state.sim.running) return;
    if (!modelReady()) { modelUnavailable(); return; }
    state.sim.running = true;
    state.sim.cycle = 0;
    state.sim.timer = window.setInterval(simTick, state.settings.simInterval);
    toast("Simulated telemetry stream started (interval " + state.settings.simInterval + " ms).", "ok");
    if (state.view !== "live") switchView("live");
    runPipeline("sim", { record: false });
  }
  function stopSimulation(silent) {
    if (state.sim.timer) window.clearInterval(state.sim.timer);
    state.sim.timer = null;
    state.sim.running = false;
    if (!silent) toast("Simulated telemetry stream stopped.", "info");
    renderAll("sim");
  }
  function toggleSimulation() {
    if (state.sim.running) stopSimulation(); else startSimulation();
  }
  function simTick() {
    state.sim.cycle += 1;
    const noise = clamp(Number(state.settings.noise), 0, 3);
    const speed = clamp(Number(state.settings.simSpeed) || 1, 0.5, 2);
    const s = state.sensors;
    s.rotationalSpeed = Math.round(clamp(s.rotationalSpeed + (Math.random() - 0.5) * 22 * noise * speed, 1150, 2850));
    s.torque = Number(clamp(s.torque + (Math.random() - 0.5) * 1.8 * noise * speed, 5, 78).toFixed(1));
    s.airTemperature = Number(clamp(s.airTemperature + (Math.random() - 0.5) * 0.15 * noise * speed, 296, 304).toFixed(1));
    if (state.sim.cycle % 4 === 0 && s.toolWear < 255) s.toolWear += 1;
    syncUIFromState();
    runPipeline("sim");
  }

  /* ======================= TELEMETRY CHANNEL ROWS ========================= */
  function channelDefs() {
    const s = state.sensors;
    const prev = state.prevSensors;
    const ind = indicators() || [];
    const pctOf = (codes) => {
      let max = 0;
      codes.forEach((c) => {
        const f = ind.filter((x) => x.code === c)[0];
        if (f && f.pct > max) max = f.pct;
      });
      return max;
    };
    const mk = (key, label, value, unit, decimals, prevValue, codes) => {
      const lv = indicatorLevel(pctOf(codes));
      return { key: key, label: label, unit: unit, decimals: decimals, value: value, prev: prevValue, status: lv, codes: codes.join(" / ") };
    };
    return [
      mk("air", "Air Temperature", s.airTemperature, "K", 1, prev ? prev.airTemperature : null, ["HDF"]),
      mk("proc", "Process Temperature", s.processTemperature, "K", 1, prev ? prev.processTemperature : null, ["HDF"]),
      mk("rpm", "Rotational Speed", s.rotationalSpeed, "rpm", 0, prev ? prev.rotationalSpeed : null, ["HDF", "PWF"]),
      mk("torque", "Torque", s.torque, "Nm", 1, prev ? prev.torque : null, ["PWF", "OSF"]),
      mk("wear", "Tool Wear", s.toolWear, "min", 0, prev ? prev.toolWear : null, ["TWF"])
    ];
  }
  function buildTelemetryRows() {
    const host = $("telemetryRows");
    if (!host) return;
    host.innerHTML = `
      <div class="telemetry-row telemetry-head">
        <span class="tel-label">Channel</span><span class="tel-value">Value</span>
        <span class="tel-delta">Δ vs previous</span><span class="tel-trend">Trend</span>
        <span class="tel-status">Linked advisory indicator</span>
      </div>` + channelDefs().map((d) => `
      <div class="telemetry-row" id="telRow-${d.key}">
        <span class="tel-label">${esc(d.label)}</span>
        <span class="tel-value" id="telVal-${d.key}">--</span>
        <span class="tel-delta" id="telDelta-${d.key}">—</span>
        <span class="tel-trend" id="telTrend-${d.key}">—</span>
        <span class="tel-status" id="telStatus-${d.key}">--</span>
      </div>`).join("");
  }
  function updateTelemetryRows() {
    const host = $("telemetryRows");
    if (!host) return;
    if (!$("telRow-air")) buildTelemetryRows();
    channelDefs().forEach((d) => {
      setText("telVal-" + d.key, fmt(d.value, d.decimals) + " " + d.unit);
      const deltaEl = $("telDelta-" + d.key);
      if (deltaEl) {
        if (d.prev == null || !state.changed) {
          deltaEl.textContent = "—";
          deltaEl.className = "tel-delta trend-flat";
        } else {
          const delta = d.value - d.prev;
          deltaEl.textContent = (delta > 0 ? "+" : "") + fmt(delta, d.decimals);
          deltaEl.className = "tel-delta " + (Math.abs(delta) < 0.05 ? "trend-flat" : delta > 0 ? "trend-up" : "trend-down");
        }
      }
      const trendEl = $("telTrend-" + d.key);
      if (trendEl) {
        const t = trendOf(state.sparks[d.key], false);
        trendEl.textContent = t.arrow;
        trendEl.className = "tel-trend " + t.cls;
      }
      const stEl = $("telStatus-" + d.key);
      if (stEl) {
        stEl.textContent = d.status.label + " · " + d.codes;
        stEl.className = "tel-status tel-" + d.status.key;
      }
    });
  }
  /* ============================ CHART UTILITIES =========================== */
  /* Lightweight canvas charting (no external libraries). */
  const COLOR_MAP = {
    "var(--status-optimal)": "#10b981", "var(--status-warning)": "#f59e0b", "var(--status-critical)": "#f43f5e",
    "var(--accent-cyan)": "#06b6d4", "var(--accent-blue)": "#3b82f6", "var(--accent-indigo)": "#6366f1"
  };
  function hex(color) { return COLOR_MAP[color] || color; }
  function fade(color, alpha) {
    const c = hex(color);
    if (c.charAt(0) === "#" && c.length === 7) {
      const r = parseInt(c.substr(1, 2), 16), g = parseInt(c.substr(3, 2), 16), b = parseInt(c.substr(5, 2), 16);
      return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
    }
    return c;
  }
  function canvasCtx(canvas) {
    if (!canvas || !canvas.getContext) return null;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : null;
    const cssW = Math.max(120, Math.round((rect && rect.width) || canvas.clientWidth || canvas.width || 240));
    const cssH = Math.max(60, Math.round((rect && rect.height) || canvas.clientHeight || canvas.height || 120));
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: cssW, h: cssH };
  }
  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = String(text).split(" ");
    const lines = [];
    let line = "";
    words.forEach((word) => {
      const test = line ? line + " " + word : word;
      if (ctx.measureText(test).width > maxWidth && line) { lines.push(line); line = word; }
      else line = test;
    });
    if (line) lines.push(line);
    const startY = y - ((lines.length - 1) * lineHeight) / 2;
    lines.forEach((l, i) => ctx.fillText(l, x, startY + i * lineHeight));
  }
  function drawEmpty(canvas, text) {
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    ctx.fillStyle = "rgba(148,163,184,0.8)";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    wrapText(ctx, text, c.w / 2, c.h / 2, c.w - 24, 16);
  }
  function drawSparkline(canvas, values, color) {
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    if (!values || values.length < 2) return;
    const min = Math.min.apply(null, values);
    const max = Math.max.apply(null, values);
    const span = (max - min) || 1;
    const stepX = w / (values.length - 1);
    const yOf = (v) => h - 4 - ((v - min) / span) * (h - 8);
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = i * stepX, y = yOf(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = fade(color, 0.12);
    ctx.fill();
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = i * stepX, y = yOf(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = hex(color);
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
  function drawLineChart(canvas, opts) {
    if (!canvas) return;
    const values = (opts.values || []).map(Number).filter((v) => Number.isFinite(v));
    if (!values.length) { drawEmpty(canvas, opts.emptyText || "No data yet."); return; }
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    const padL = 30, padR = 8, padT = 10, padB = 16;
    const yMax = Number.isFinite(opts.yMax) ? opts.yMax : Math.max.apply(null, values);
    const yMin = Number.isFinite(opts.yMin) ? opts.yMin : 0;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.fillStyle = "rgba(148,163,184,0.7)";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(String(Math.round(yMax - ((yMax - yMin) / 4) * i)), padL - 5, y);
    }
    const stepX = values.length > 1 ? plotW / (values.length - 1) : 0;
    const yOf = (v) => padT + plotH - ((v - yMin) / ((yMax - yMin) || 1)) * plotH;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = padL + i * stepX, y = yOf(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(padL + (values.length - 1) * stepX, padT + plotH);
    ctx.lineTo(padL, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = fade(opts.color || "#38bdf8", 0.12);
    ctx.fill();
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = padL + i * stepX, y = yOf(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = hex(opts.color || "#38bdf8");
    ctx.lineWidth = 2;
    ctx.stroke();
    if (opts.thresholds) {
      opts.thresholds.forEach((t) => {
        if (!Number.isFinite(t.value)) return;
        const y = yOf(t.value);
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = hex(t.color);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
        ctx.restore();
      });
    }
    const last = values[values.length - 1];
    const lx = padL + (values.length - 1) * stepX, ly = yOf(last);
    ctx.beginPath();
    ctx.arc(lx, ly, 3.4, 0, Math.PI * 2);
    ctx.fillStyle = hex(opts.color || "#38bdf8");
    ctx.fill();
  }
  /* ======================= SENSOR DYNAMIC WAVEFORM ======================== */
  /* Preserved from V3: same traces, same scaling, same glow on the risk line.
     Now redrawn on data change instead of a 60 fps requestAnimationFrame loop. */
  function waveformSamples() {
    const t = state.telemetry;
    if (t.risk.length >= 2) {
      return {
        speed: t.rpm.map((v) => v / 30),
        torque: t.torque.slice(),
        wear: t.wear.map((v) => v / 3),
        risk: t.risk.slice()
      };
    }
    // Startup: render the current reading as a flat baseline
    const n = 45, s = state.sensors;
    const flat = (v) => new Array(n).fill(v);
    return {
      speed: flat(s.rotationalSpeed / 30), torque: flat(s.torque),
      wear: flat(s.toolWear / 3), risk: flat(state.risk)
    };
  }
  function updateWaveform() {
    const canvas = $("telemetryChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width, height = canvas.height;
    ctx.fillStyle = "rgba(10, 15, 28, 0.95)";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    const gridXSteps = 10, gridYSteps = 4;
    for (let i = 0; i <= gridXSteps; i++) {
      const x = (width / gridXSteps) * i;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let j = 0; j <= gridYSteps; j++) {
      const y = (height / gridYSteps) * j;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    const samples = waveformSamples();
    const count = samples.risk.length;
    if (count < 2) return;
    const traces = [
      { key: "speed", color: "#06b6d4", max: 100 },
      { key: "torque", color: "#f59e0b", max: 80 },
      { key: "wear", color: "#10b981", max: 90 },
      { key: "risk", color: "#f43f5e", max: 100, glow: true }
    ];
    const stepX = width / (count - 1);
    traces.forEach((trace) => {
      const series = samples[trace.key];
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = trace.color;
      ctx.lineWidth = trace.glow ? 2.5 : 1.7;
      if (trace.glow) { ctx.shadowColor = trace.color; ctx.shadowBlur = 10; }
      series.forEach((value, index) => {
        const x = index * stepX;
        const normalized = Math.min(trace.max, Math.max(0, value)) / trace.max;
        const y = height - (normalized * (height - 24)) - 12;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    });
  }

  /* ====================== RISK TREND (HISTORY SERIES) ===================== */
  function computeTrendLabel() {
    const v = state.history.slice(-10).map((r) => Number(r.risk));
    if (v.length < 4) return "No trend yet";
    const half = Math.floor(v.length / 2);
    const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const delta = avg(v.slice(half)) - avg(v.slice(0, half));
    if (Math.abs(delta) < 0.5) return "Stable";
    return delta > 0 ? "Increasing" : "Decreasing";
  }
  function updateRiskTrends() {
    const rows = state.history.slice(-40);
    const values = rows.map((r) => Number(r.risk));
    const label = computeTrendLabel();
    setText("dashTrendLabel", label);
    setText("liveTrendLabel", label);
    const empty = "No recorded prediction events yet. Run a scenario, a preset or the telemetry simulation.";
    const opts = {
      values: values, color: "#f43f5e", yMax: 100, emptyText: empty,
      thresholds: [
        { value: state.settings.warnThreshold, color: "var(--status-warning)" },
        { value: state.settings.critThreshold, color: "var(--status-critical)" }
      ]
    };
    if ($("dashRiskTrend")) drawLineChart($("dashRiskTrend"), opts);
    if ($("liveRiskTrend")) drawLineChart($("liveRiskTrend"), opts);
  }
  function drawBarChart(canvas, items, opts) {
    if (!canvas) return;
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    const options = opts || {};
    const max = Number.isFinite(options.max) ? options.max : 1;
    const padL = options.padL || 104, padR = options.padR || 118, padT = 6, padB = 6;
    const n = Math.max(1, items.length);
    const rowH = (h - padT - padB) / n;
    ctx.font = "11px Inter, sans-serif";
    items.forEach((item, i) => {
      const y = padT + i * rowH;
      const cy = y + rowH / 2;
      const barH = Math.min(14, Math.max(8, rowH - 12));
      ctx.fillStyle = "rgba(148,163,184,0.9)";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(item.label, padL - 8, cy);
      const trackX = padL, trackW = Math.max(20, w - padL - padR);
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(trackX, cy - barH / 2, trackW, barH);
      if (Number.isFinite(item.value)) {
        const bw = Math.max(2, (item.value / max) * trackW);
        ctx.fillStyle = hex(item.color || "#38bdf8");
        ctx.fillRect(trackX, cy - barH / 2, bw, barH);
        ctx.fillStyle = "#e2e8f0";
        ctx.textAlign = "left";
        ctx.fillText(item.display || String(item.value), trackX + bw + 6, cy);
      } else {
        ctx.save();
        ctx.strokeStyle = "rgba(148,163,184,0.45)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(trackX + 0.5, cy - barH / 2 + 0.5, trackW - 1, barH - 1);
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(148,163,184,0.85)";
        ctx.textAlign = "left";
        ctx.fillText("N/A", trackX + 8, cy);
        ctx.restore();
      }
    });
  }
  function drawDonut(canvas, segments, opts) {
    if (!canvas) return;
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    const options = opts || {};
    const total = segments.reduce((sum, s) => sum + (Number(s.value) || 0), 0);
    const cx = w / 2, cy = h / 2;
    const r = Math.min(w, h) / 2 - 8;
    const ring = Math.max(10, r * 0.32);
    if (!total) { drawEmpty(canvas, options.emptyText || "No data."); return; }
    let angle = -Math.PI / 2;
    segments.forEach((seg) => {
      const frac = (Number(seg.value) || 0) / total;
      const sweep = frac * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, angle, angle + sweep);
      ctx.strokeStyle = hex(seg.color);
      ctx.lineWidth = ring;
      ctx.stroke();
      angle += sweep;
    });
    ctx.fillStyle = "#f1f5f9";
    ctx.font = "600 15px Outfit, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(options.centerText || "", cx, cy - 6);
    ctx.fillStyle = "rgba(148,163,184,0.9)";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(options.centerSub || "", cx, cy + 12);
  }
  function drawHistogram(canvas, values, opts) {
    if (!canvas) return;
    const nums = (values || []).map(Number).filter((v) => Number.isFinite(v));
    const options = opts || {};
    if (nums.length < 2) { drawEmpty(canvas, options.emptyText || "No data loaded."); return; }
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    const bins = Math.max(4, options.bins || 12);
    const min = Math.min.apply(null, nums);
    const max = Math.max.apply(null, nums);
    const span = (max - min) || 1;
    const counts = new Array(bins).fill(0);
    nums.forEach((v) => {
      const idx = Math.min(bins - 1, Math.floor(((v - min) / span) * bins));
      counts[idx] += 1;
    });
    const maxCount = Math.max.apply(null, counts) || 1;
    const padL = 26, padR = 8, padT = 8, padB = 18;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.fillStyle = "rgba(148,163,184,0.7)";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 3; i++) {
      const y = padT + (plotH / 3) * i;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(String(Math.round(maxCount - (maxCount / 3) * i)), padL - 4, y);
    }
    const barW = plotW / bins;
    counts.forEach((count, i) => {
      const bh = (count / maxCount) * plotH;
      ctx.fillStyle = fade(options.color || "#06b6d4", 0.8);
      ctx.fillRect(padL + i * barW + 1, padT + plotH - bh, Math.max(1, barW - 2), bh);
    });
    ctx.fillStyle = "rgba(148,163,184,0.8)";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(fmt(min, options.decimals || 1), padL, h - 12);
    ctx.textAlign = "right";
    ctx.fillText(fmt(max, options.decimals || 1), w - padR, h - 12);
    ctx.textAlign = "center";
    ctx.fillText(options.xLabel || "", padL + plotW / 2, h - 12);
  }
  function renderMatrix(host, matrix, opts) {
    if (!host) return;
    const options = opts || {};
    if (!matrix) {
      host.innerHTML = '<div class="na-panel"><strong>Confusion matrix &mdash; N/A</strong><br>' +
        esc(options.reason || "Not documented for this model. No serialized artifact or saved evaluation output was provided with this project, so no values are shown.") + '</div>';
      return;
    }
    const cells = [
      { key: "tn", label: "True Negative", value: matrix.tn, cls: "cm-ok" },
      { key: "fp", label: "False Positive", value: matrix.fp, cls: "cm-warn" },
      { key: "fn", label: "False Negative", value: matrix.fn, cls: "cm-bad" },
      { key: "tp", label: "True Positive", value: matrix.tp, cls: "cm-ok" }
    ];
    host.innerHTML = (options.caption ? '<div class="muted-note" style="margin:0 0 8px">' + esc(options.caption) + '</div>' : "")
      + '<div class="cm-grid">' + cells.map((cell) => `
      <div class="cm-cell ${cell.cls}"><span class="cm-value">${Number.isFinite(cell.value) ? fmtInt(cell.value) : "N/A"}</span>
      <span class="cm-label">${cell.label}</span></div>`).join("") + "</div>";
  }
  /* ============================ MODELS VIEW =============================== */
  function metricsData() { return (EX && EX.MODEL_METRICS) ? EX.MODEL_METRICS : null; }
  function metricValue(modelKey, metric) {
    const data = metricsData();
    if (!data) return null;
    const m = (data.models || []).filter((x) => x.key === modelKey)[0];
    if (!m || !m.metrics) return null;
    const v = m.metrics[metric];
    return Number.isFinite(v) ? v : null;
  }
  function metricCell(value) {
    return Number.isFinite(value)
      ? '<td style="font-family:var(--font-mono);font-weight:700;color:#e2e8f0;">' + value.toFixed(3) + "</td>"
      : '<td class="na-cell">N/A</td>';
  }
  function updateModelsView() {
    if (state.view !== "models") return;
    const data = metricsData();
    if (!data) return;
    ["dt", "rf", "svm"].forEach((key) => {
      const map = { dt: ["dtF1", "dtPrec", "dtRec", "dtAuc"], rf: ["rfF1", "rfPrec", "rfRec", "rfAuc"], svm: ["svmF1", "svmPrec", "svmRec", "svmAuc"] };
      const ids = map[key];
      const keys = ["f1", "precision", "recall", "rocAuc"];
      ids.forEach((id, i) => {
        const v = metricValue(key, keys[i]);
        const node = $(id);
        if (!node) return;
        node.textContent = Number.isFinite(v) ? v.toFixed(3) : "N/A";
        node.className = "metric-val" + (Number.isFinite(v) ? " metric-doc" : " metric-na");
      });
    });
    const rfOut = $("rfLiveOut");
    if (rfOut) {
      rfOut.textContent = modelReady() ? fmt(state.risk, 1) + "% · " + statusWord(state.status.key) : "Model unavailable";
    }
    const rfBadge = $("rfLive");
    if (rfBadge) {
      rfBadge.className = "model-status " + (modelReady() ? "status-live" : "status-absent");
      rfBadge.innerHTML = modelReady() ? '<span class="sys-dot"></span> LIVE INFERENCE' : "MODEL UNAVAILABLE";
    }
    /* Decision Tree + SVM live outputs and badges */
    const per = (state.prediction && state.prediction.perModelPct) || null;
    [["dt", "dtLiveOut", "dtLive"], ["svm", "svmLiveOut", "svmLive"]].forEach(function (pair) {
      const key = pair[0], outId = pair[1], badgeId = pair[2];
      const out = $(outId), badge = $(badgeId);
      const meta = (data.models || []).filter(function (x) { return x.key === key; })[0];
      const live = !!(meta && meta.live && modelReady());
      const value = per && Number.isFinite(per[key]) ? per[key] : null;
      if (out) out.textContent = live && value !== null ? fmt(value, 1) + "% failure risk" : (live ? "Ready" : "Artifact not loaded");
      if (badge) {
        badge.className = "model-status " + (live ? "status-live" : "status-absent");
        badge.innerHTML = live ? '<span class="sys-dot"></span> LIVE INFERENCE' : "ARTIFACT NOT LOADED";
      }
    });
    // Class balance
    const ds = data.dataset || {};
    if (Number.isFinite(ds.records) && Number.isFinite(ds.failures)) {
      setText("cbNormal", fmtInt(ds.records - ds.failures) + " records");
      setText("cbFail", fmtInt(ds.failures) + " records");
      setText("cbRate", fmt(ds.failureRate, 2) + "%");
      setText("cbPill", fmtInt(ds.records) + " records");
      drawDonut($("classBalanceChart"), [
        { label: "Normal", value: ds.records - ds.failures, color: "#10b981" },
        { label: "Failures", value: ds.failures, color: "#f43f5e" }
      ], { centerText: fmt(ds.failureRate, 2) + "%", centerSub: "failures" });

      /* ---- Imbalance-handling block ----
         Values come straight from MODEL_METRICS.pipeline and the measured
         per-model metrics. Nothing here is estimated: a metric the project
         did not measure renders as "N/A". */
      const pipe = data.pipeline || {};
      setText("cbSplit", pipe.split || "N/A");
      setText("cbCv", pipe.cv || "N/A");
      setText("cbScoring", pipe.scoring || "N/A");
      setText("cbImb", pipe.imbalance || "N/A");
      document.querySelectorAll("[data-cb-m]").forEach(function (row) {
        const key = row.getAttribute("data-cb-m");
        row.querySelectorAll("[data-cb]").forEach(function (cell) {
          const v = metricValue(key, cell.getAttribute("data-cb"));
          cell.textContent = Number.isFinite(v) ? v.toFixed(3) : "N/A";
          cell.className = Number.isFinite(v) ? "" : "cb-na";
        });
      });
      const measured = ["dt", "rf", "svm"].filter(function (k) {
        return Number.isFinite(metricValue(k, "precision")) && Number.isFinite(metricValue(k, "recall"));
      });
      const testRows = ["dt", "rf", "svm"].map(function (k) {
        const m = (data.models || []).filter(function (x) { return x.key === k; })[0];
        return m && m.metrics && Number.isFinite(m.metrics.testRecords) ? m.metrics.testRecords : null;
      }).filter(function (v) { return v !== null; })[0];
      setText("cbTradeNote", measured.length
        ? "Measured on the held-out test split"
          + (testRows ? " of " + fmtInt(testRows) + " records" : "")
          + ". The precision/recall spread across models is why F1, not accuracy, was the tuning metric."
        : "Values are read from the saved model metrics. Any metric the project did not measure is shown as N/A rather than estimated.");
    }
    // Comparison charts (documented values only; missing = N/A placeholder)
    const models = modelMeta();
    const barItems = (metric) => models.map((m) => {
      const v = metricValue(m.key, metric);
      return { label: m.name, value: v, display: Number.isFinite(v) ? v.toFixed(3) : "N/A", color: m.key === "rf" ? "#06b6d4" : "#64748b" };
    });
    drawBarChart($("chartF1"), barItems("f1"), { max: 1, padL: 110, padR: 70 });
    drawBarChart($("chartROC"), barItems("rocAuc"), { max: 1, padL: 110, padR: 70 });
    drawBarChart($("chartPR"), [
      { label: "Precision (RF)", value: metricValue("rf", "precision"), display: "N/A", color: "#64748b" },
      { label: "Recall (RF)", value: metricValue("rf", "recall"), display: "N/A", color: "#64748b" }
    ], { max: 1, padL: 110, padR: 70 });
    // Metric table
    const body = $("cmpTableBody");
    if (body) {
      body.innerHTML = models.map((m) => {
        const cells = ["f1", "precision", "recall", "rocAuc"].map((k) => metricCell(metricValue(m.key, k))).join("");
        const live = (m.live && modelReady())
          ? '<span class="table-status-pill table-status-normal">LIVE</span>'
          : '<span class="na-cell">not connected</span>';
        const artifact = m.artifact && m.artifact !== "none" ? "Bundled" : "None";
        return "<tr><td><strong>" + esc(m.name) + "</strong></td>" + cells +
          "<td>" + live + "</td><td>" + esc(artifact) + "</td></tr>";
      }).join("");
    }
    renderConfusionMatrix(state.cmpModel || "dt");
  }
  function renderConfusionMatrix(selectedKey) {
    const host = $("cmHost");
    if (!host) return;
    const m = modelMeta().filter((x) => x.key === selectedKey)[0];
    const label = m ? m.name : "Model";
    /* Real matrices exist for every model whose metrics were measured in this
       project's own pipeline. Only fall back to the N/A panel when the model
       genuinely has no numbers (the Random Forest, whose study printed the
       matrix but never saved it). */
    if (m && m.confusion) {
      renderMatrix(host, m.confusion, {
        caption: label + " - measured on the held-out 20% test split"
      });
      setText("cmNote", "Selected: " + label + ". " + (m.provenance || "Measured in this project.")
        + " Decision threshold " + m.threshold + " (tuned on the cross-validated out-of-fold predictions).");
      return;
    }
    renderMatrix(host, null, {
      reason: "No confusion matrix is available for " + label + ". The training script printed matrices to console, but none were saved into this project's outputs, and no artifact exists to recompute them offline. Values would be invented — so none are shown."
    });
    setText("cmNote", "Selected: " + label + ". Matrices were printed by the study script but not persisted, and this model has no bundled artifact to recompute them from, so the panel stays N/A by design.");
  }
  /* ============================== XAI VIEW ================================
     MACHINE INTELLIGENCE V9.0 — EXPLAINABLE AI INTELLIGENCE CENTER
     ------------------------------------------------------------------
     HONESTY CONTRACT for this whole block:

     model-data.js and model-artifacts.js are FROZEN and never edited.
     Everything below READS the fitted artifacts already loaded in the page.
     Nothing is hard-coded, estimated or interpolated, and no SHAP value is
     fabricated anywhere.

     Genuine capability, established by inspecting the actual artifacts:

       DECISION TREE — DT_TREE (34 internal splits, depth 7) is a real fitted
       CART tree, so the EXACT decision path for the current input can be
       extracted: every split feature, the real threshold, the branch taken
       and the leaf probability.
       NOT available: sklearn's impurity-weighted feature_importances_,
       because per-node sample counts were never serialized. Not faked.

       RANDOM FOREST — FOREST (300 trees / 46,418 internal splits) is
       reachable, so split-usage frequency is real and we can trace which
       feature actually gated each of the 300 trees for THIS input.
       Reported as RELATIVE MODEL SPLIT USAGE, never as causal impact.

       SVM (RBF) — an RBF kernel has NO per-feature coefficient vector.
       SVM_COEF holds 1900 dual coefficients alpha_i, one per support vector.
       They are NOT feature weights. Presenting them as feature importance
       would be fabrication, so the SVM panel states feature-level
       explanation is unavailable and shows what IS genuine: probability,
       raw decision score and dominant support vectors.

     Artifact access: DT_TREE / FOREST / SVM_ARTIFACT / SVM_SV / SVM_COEF are
     top-level `const` bindings living in the global lexical scope rather than
     on `window`, so they are read through guarded typeof probes. If a probe
     fails the UI degrades to "NOT AVAILABLE" instead of inventing anything.
     ====================================================================== */
  const XA = { model: "rf", feature: null, mode: false, history: [], cache: {} };
  /* ---- guarded access to the frozen artifacts ------------------------- */
  function xaArtifact(name) {
    try { return (0, eval)(name); } catch (err) { return undefined; }
  }
  function xaDtTree() {
    if (XA.cache.dtTree !== undefined) return XA.cache.dtTree;
    const t = xaArtifact("DT_TREE");
    XA.cache.dtTree = (t && typeof t === "object" && ("f" in t || "leaf" in t)) ? t : null;
    return XA.cache.dtTree;
  }
  function xaForest() {
    if (XA.cache.forest !== undefined) return XA.cache.forest;
    const f = xaArtifact("FOREST");
    XA.cache.forest = (f && Array.isArray(f.trees) && f.trees.length) ? f : null;
    return XA.cache.forest;
  }
  function xaSvmArt() {
    if (XA.cache.svmArt !== undefined) return XA.cache.svmArt;
    let a = xaArtifact("SVM_ARTIFACT") || null;
    if ((!a || !a.mean || !a.scale) && window.EXTRA_MODEL_DATA) a = window.EXTRA_MODEL_DATA.svm || a;
    XA.cache.svmArt = a; return a;
  }
  function xaSvmVectors() {
    if (XA.cache.svmSV !== undefined) return XA.cache.svmSV;
    const sv = xaArtifact("SVM_SV");
    XA.cache.svmSV = Array.isArray(sv) && sv.length ? sv : null;
    return XA.cache.svmSV;
  }
  function xaSvmCoef() {
    if (XA.cache.svmCoef !== undefined) return XA.cache.svmCoef;
    const c = xaArtifact("SVM_COEF");
    XA.cache.svmCoef = Array.isArray(c) && c.length ? c : null;
    return XA.cache.svmCoef;
  }
  const XA_COLMAP = {
    AirTemp: "airTemperature", ProcTemp: "processTemperature", RPM: "rotationalSpeed",
    Torque: "torque", ToolWear: "toolWear"
  };
  const XA_FEATURES = [
    { key: "toolWear", col: "ToolWear", label: "Tool Wear", unit: "min", dec: 0, twin: "wear", zone: "tool area" },
    { key: "torque", col: "Torque", label: "Torque", unit: "Nm", dec: 1, twin: "torque", zone: "motor / spindle" },
    { key: "rotationalSpeed", col: "RPM", label: "Rotational Speed", unit: "rpm", dec: 0, twin: "rpm", zone: "rotating shaft" },
    { key: "processTemperature", col: "ProcTemp", label: "Process Temperature", unit: "K", dec: 1, twin: "proc", zone: "thermal / cutting zone" },
    { key: "airTemperature", col: "AirTemp", label: "Air Temperature", unit: "K", dec: 1, twin: "air", zone: "cooling / ambient" },
    { key: "productType", col: "Type_*", label: "Product Variant", unit: "", dec: 0, twin: null, zone: "product grade" }
  ];
  function xaFeat(key) { return XA_FEATURES.filter(function (f) { return f.key === key; })[0]; }
  function xaRow(features) {
    const s = sanitizeSensors(features || state.sensors);
    return {
      AirTemp: s.airTemperature, ProcTemp: s.processTemperature, RPM: s.rotationalSpeed,
      Torque: s.torque, ToolWear: s.toolWear,
      Type_H: s.productType === "H" ? 1 : 0, Type_L: s.productType === "L" ? 1 : 0, Type_M: s.productType === "M" ? 1 : 0
    };
  }
  /* ---- genuine structural statistics (cached once, never per frame) ---- */
  function xaSplitUsage(model) {
    const key = "usage_" + model;
    if (XA.cache[key]) return XA.cache[key];
    let trees = null;
    if (model === "rf") { const f = xaForest(); trees = f ? f.trees : null; }
    else if (model === "dt") { const t = xaDtTree(); trees = t ? [t] : null; }
    const out = { ok: false, rows: [], splits: 0, nodes: 0, trees: 0 };
    if (trees) {
      const counts = Object.create(null);
      let splits = 0, nodes = 0;
      for (let i = 0; i < trees.length; i++) {
        const stack = [trees[i]];
        while (stack.length) {
          const nd = stack.pop();
          nodes++;
          if (nd.leaf) continue;
          splits++;
          counts[nd.f] = (counts[nd.f] || 0) + 1;
          if (nd.l) stack.push(nd.l);
          if (nd.r) stack.push(nd.r);
        }
      }
      out.rows = Object.keys(counts).map(function (c) {
        const f = XA_FEATURES.filter(function (x) { return x.col === c; })[0];
        return {
          col: c, key: f ? f.key : null, label: f ? f.label : c,
          count: counts[c], share: splits ? counts[c] / splits : 0
        };
      }).sort(function (a, b) { return b.count - a.count; });
      out.ok = true; out.splits = splits; out.nodes = nodes; out.trees = trees.length;
    }
    XA.cache[key] = out;
    return out;
  }
  /* ---- REAL decision path through the fitted Decision Tree ------------- */
  function xaDtPath(features) {
    const tree = xaDtTree();
    if (!tree) return null;
    const row = xaRow(features);
    const steps = [];
    let node = tree;
    let guard = 0;
    while (node && !node.leaf && guard++ < 32) {
      const v = Number(row[node.f]);
      const wentLeft = v <= Number(node.thr);
      const f = XA_FEATURES.filter(function (x) { return x.col === node.f; })[0];
      steps.push({
        col: node.f, key: f ? f.key : null, label: f ? f.label : node.f,
        threshold: Number(node.thr), value: v, left: wentLeft,
        categorical: node.f.indexOf("Type_") === 0
      });
      node = wentLeft ? node.l : node.r;
    }
    if (!node || !node.leaf) return null;
    const leaf = Number(node.p);
    let p = leaf;
    try { if (typeof window.dtPredict === "function") p = Number(window.dtPredict(row)); } catch (e) { /* probe failed */ }
    return { steps: steps, leaf: leaf, prob: p, depth: steps.length };
  }
  /* ---- REAL per-input trace across all 300 forest trees ---------------- */
  function xaForestTrace(features) {
    const f = xaForest();
    if (!f) return null;
    const row = xaRow(features);
    /* Count TREES (a tree contributes at most once per feature) and split
       NODES separately. Reporting nodes as "trees" could exceed the 300 total
       and would misstate the statistic. */
    const treesUsing = Object.create(null);
    const nodes = Object.create(null);
    const depthSum = Object.create(null);
    let high = 0;
    for (let i = 0; i < f.trees.length; i++) {
      let nd = f.trees[i];
      let d = 0, guard = 0;
      const seenHere = new Set();
      while (nd && !nd.leaf && guard++ < 48) {
        if (!seenHere.has(nd.f)) { seenHere.add(nd.f); treesUsing[nd.f] = (treesUsing[nd.f] || 0) + 1; }
        nodes[nd.f] = (nodes[nd.f] || 0) + 1;
        depthSum[nd.f] = (depthSum[nd.f] || 0) + d;
        nd = Number(row[nd.f]) <= Number(nd.thr) ? nd.l : nd.r;
        d++;
      }
      if (nd && nd.leaf && Number(nd.p) > 0.5) high++;
    }
    const total = f.trees.length;
    const rows = Object.keys(treesUsing).map(function (c) {
      const ft = XA_FEATURES.filter(function (x) { return x.col === c; })[0];
      return {
        col: c, key: ft ? ft.key : null, label: ft ? ft.label : c,
        trees: treesUsing[c], nodes: nodes[c], share: treesUsing[c] / (total || 1),
        avgDepth: nodes[c] ? depthSum[c] / nodes[c] : 0
      };
    }).sort(function (a, b) { return b.trees - a.trees || b.nodes - a.nodes; });
    return { rows: rows, total: total, highRiskTrees: high };
  }
  /* ---- GENUINE SVM explanation (deliberately no feature weights) ------- */
  function xaSvmExplain(features) {
    const art = xaSvmArt(), sv = xaSvmVectors(), co = xaSvmCoef();
    if (!art || !sv || !co || !art.mean || !art.scale) return null;
    const cols = art.cols || (xaForest() ? xaForest().cols : null);
    if (!cols) return null;
    const row = xaRow(features || state.sensors);
    const xs = cols.map(function (c, i) {
      return (Number(row[c]) - Number(art.mean[i])) / Number(art.scale[i]);
    });
    const g = Number(art.params && art.params.gamma) || 0;
    let sum = 0;
    const terms = [];
    for (let k = 0; k < sv.length; k++) {
      const s = sv[k];
      let d = 0;
      for (let i = 0; i < xs.length; i++) { const t = xs[i] - Number(s[i]); d += t * t; }
      const kernel = Math.exp(-g * d);
      const a = Number(co[k]);
      const term = a * kernel;
      sum += term;
      terms.push({ k: k, a: a, kernel: kernel, term: term, dist: Math.sqrt(d) });
    }
    const margin = sum + Number(art.bias);
    const sorted = terms.slice().sort(function (x, y) { return Math.abs(y.term) - Math.abs(x.term); });
    const top = sorted.slice(0, 6);
    const dominance = top.reduce(function (acc, t) { return acc + Math.abs(t.term); }, 0);
    let prob = null;
    try { if (typeof window.svmPredict === "function") prob = Number(window.svmPredict(row)); } catch (e) { /* probe failed */ }
    return {
      margin: margin, prob: prob, gamma: g, supportVectors: sv.length,
      xs: xs, cols: cols, top: top, dominance: dominance,
      totalAbs: Math.abs(sum), platt: { a: art.plattA, b: art.plattB },
      threshold: art.threshold, metrics: art.metrics
    };
  }
  /* ---- availability registry: computed, never assumed ------------------ */
  function xaCapability(model) {
    if (model === "dt") {
      return {
        prob: typeof window.dtPredict === "function",
        path: !!xaDtTree(),
        importance: false,
        importanceWhy: "Impurity-weighted importances are not stored in the DT artifact (no per-node sample counts), so none is shown.",
        score: false
      };
    }
    if (model === "svm") {
      return {
        prob: typeof window.svmPredict === "function",
        path: false,
        importance: false,
        importanceWhy: "An RBF kernel has no per-feature coefficient vector, so feature importance cannot be derived for this model.",
        score: !!(xaSvmExplain(null) !== null)
      };
    }
    return {
      prob: !!xaForest(),
      path: false,
      importance: false,
      importanceWhy: "Impurity-weighted importances are not stored in the forest artifact (no per-node sample counts); split-usage frequency is shown instead.",
      score: false
    };
  }
  function xaModelLabel(key) {
    return key === "dt" ? "Decision Tree" : key === "svm" ? "SVM (RBF)" : "Random Forest";
  }
  function xaLevelOf(risk) { return wiLevelOf(clamp(Number(risk) || 0, 0, 100)); }
  function xaBandClass(key) {
    return key === "critical" ? "xa-crit" : key === "warning" ? "xa-warn" : "xa-ok";
  }
  /* Ordinal strength word derived from a RELATIVE share — never a claim of
     physical severity, only of how much the model relies on the feature. */
  function xaStrengthWord(share) {
    if (share >= 0.2) return { word: "HIGH", cls: "s-high" };
    if (share >= 0.1) return { word: "MEDIUM", cls: "s-med" };
    if (share > 0) return { word: "LOW", cls: "s-low" };
    return { word: "UNUSED", cls: "s-none" };
  }
  function interpretationText(contribs) {
    const rising = contribs.filter((c) => c.deltaPp > 1).slice(0, 3).map((c) => c.label);
    const reducing = contribs.filter((c) => c.deltaPp < -1).slice(0, 3).map((c) => c.label);
    const parts = [];
    if (rising.length) parts.push("Risk is being pushed above the baseline mainly by " + rising.join(", ") + ".");
    else parts.push("No feature is currently pushing risk materially above the nominal baseline.");
    if (reducing.length) parts.push("Holding risk down: " + reducing.join(", ") + ".");
    parts.push("Current status: " + statusWord(state.status.key) + " at " + fmt(state.risk, 1) + "%.");
    return parts.join(" ");
  }
  /* ---- picks which features drive the current explanation -------------- */
  function xaActiveFeatures() {
    if (XA.model === "dt") {
      const path = xaDtPath(state.sensors);
      if (path && path.steps.length) {
        return path.steps.map(function (s) { return { key: s.key, label: s.label, col: s.col }; });
      }
      return [];
    }
    if (XA.model === "rf") {
      const tr = xaForestTrace(state.sensors);
      return tr ? tr.rows.map(function (r) { return { key: r.key, label: r.label, col: r.col, share: r.share, trees: r.trees, traceTotal: tr.total }; }) : [];
    }
    /* SVM: no per-feature explanation exists, so no features are claimed. */
    return [];
  }
  /* ---- probability for the selected model (all from real inference) ----- */
  function xaSelectedProb() {
    const per = (state.prediction && state.prediction.perModelPct) || null;
    if (XA.model === "dt") {
      if (per && per.dt != null) return Number(per.dt);
      const p = xaDtPath(state.sensors);
      return p ? p.prob * 100 : null;
    }
    if (XA.model === "svm") {
      if (per && per.svm != null) return Number(per.svm);
      const e = xaSvmExplain(state.sensors);
      return e && e.prob != null ? e.prob * 100 : null;
    }
    return Number(state.risk);
  }
  function xaRenderHero() {
    const prob = xaSelectedProb();
    const lvl = xaLevelOf(prob == null ? 0 : prob);
    setText("xaMachine", (state.machine && state.machine.id) || "CNC-MILL-08");
    setText("xaRiskVal", prob == null ? "--" : fmt(prob, 1));
    setText("xaRiskLevelTxt", prob == null ? "NOT AVAILABLE" : lvl.label);
    setText("xaModelName", xaModelLabel(XA.model));
    setText("xaWhyModelTag", xaModelLabel(XA.model));
    const box = $("xaRiskLevelBox");
    if (box) box.className = "xa-hero-level " + xaBandClass(lvl.key);
    setText("xaRiskIco", prob == null ? "?" : lvl.icon);
    setText("xaRiskSub", prob == null
      ? "this model did not return a probability for the current input"
      : "failure probability reported by the " + xaModelLabel(XA.model));
    /* ---- WHY? strongest REAL signals for the selected model ---- */
    const host = $("xaWhyList");
    const foot = $("xaWhyFoot");
    if (!host) return;
    if (XA.model === "svm") {
      host.innerHTML = '<div class="xa-na">SVM EXPLANATION</div>' +
        '<div class="xa-na-text">Detailed feature-level explanation is <strong>not available</strong> from the current SVM artifact. ' +
        'An RBF kernel computes a distance-weighted sum over ' + (xaSvmArt() ? fmtInt(xaSvmArt().supportVectors || 1900) : "many") +
        ' support vectors and has no per-feature weight vector, so feature importance cannot be derived. ' +
        'No importance values are shown for this model rather than inventing them.</div>';
      if (foot) foot.innerHTML = '<span class="xa-tag xa-tag-na">PREDICTION AVAILABLE</span><span class="xa-tag xa-tag-na">DETAILED FEATURE EXPLANATION NOT AVAILABLE</span>';
      return;
    }
    const feats = xaActiveFeatures();
    if (!feats.length) {
      host.innerHTML = '<div class="xa-na-text">No explanation data available for this model on the current input.</div>';
      if (foot) foot.innerHTML = "";
      return;
    }
    /* For the DT the same feature can be tested more than once on one path.
       Collapse repeats and state the real multiplicity instead of showing
       duplicate-looking rows, and replace the (meaningless) bar with the
       order the path first reached that feature. */
    const seen = new Map();
    feats.forEach(function (f, idx) {
      if (!seen.has(f.key)) seen.set(f.key, { f: f, first: idx + 1, hits: 0 });
      seen.get(f.key).hits += 1;
    });
    const top = Array.from(seen.values()).slice(0, 5);
    const maxShare = Math.max.apply(null, top.map(function (r) { return r.f.share || 0; }).concat([0.0001]));
    host.innerHTML = top.map(function (r, i) {
      const f = r.f;
      const s = f.share ? xaStrengthWord(f.share) : null;
      const pct = f.share ? clamp((f.share / maxShare) * 100, 3, 100) : 0;
      const glyph = s ? s.cls : "s-med";
      const word = s ? s.word : "ON PATH";
      const tail = f.share
        ? fmtInt(f.trees || 0) + " of " + fmtInt((f.traceTotal || 300)) + " trees"
        : "tested " + r.hits + (r.hits > 1 ? " times" : " time") + " on the decision path";
      return '<div class="xa-why-row' + (XA.feature === f.key ? " sel" : "") + '" data-xa-feature="' + esc(f.key || "") + '">' +
        '<span class="xa-why-rank">' + (i + 1) + '</span>' +
        '<span class="xa-why-name">' + esc(f.label) + '</span>' +
        '<span class="xa-why-bar">' + (f.share
          ? '<i class="' + glyph + '" style="width:' + pct.toFixed(1) + '%"></i>'
          : '<i class="s-path" style="width:' + clamp(((top.length - i) / top.length) * 100, 8, 100).toFixed(0) + '%"></i>') + '</span>' +
        '<span class="xa-why-tag ' + glyph + '">' + esc(word) + '</span>' +
        '<span class="xa-why-num">' + esc(tail) + '</span>' +
        '</div>';
    }).join("");
    if (foot) {
      foot.innerHTML = XA.model === "dt"
        ? '<span class="xa-tag">REAL DECISION PATH \u00b7 ' + feats.length + ' SPLITS</span><span class="xa-tag xa-tag-na">NOT CAUSAL</span>'
        : '<span class="xa-tag">RELATIVE MODEL SPLIT USAGE</span><span class="xa-tag xa-tag-na">NOT CAUSAL \u00b7 NOT SHAP</span>';
    }
  }
  /* ---- main explanation visualization --------------------------------- */
  function xaRenderImpact() {
    const host = $("xaImpactBody");
    if (!host) return;
    const note = $("xaImpactNote");
    const method = $("xaMethodNote");
    const title = $("xaImpactTitle");
    const pill = $("xaImpactPill");
    if (XA.model === "svm") {
      if (title) title.textContent = "SVM Feature Importance";
      if (pill) { pill.textContent = "NOT AVAILABLE"; pill.className = "dataset-pill pill-offline"; }
      if (method) method.innerHTML = '<strong>EXPLANATION NOT AVAILABLE FOR THIS MODEL.</strong> An RBF-kernel SVM has no per-feature coefficient vector, so a feature-importance chart cannot be produced honestly from this artifact.';
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">EXPLANATION NOT AVAILABLE FOR THIS MODEL</div>' +
        '<div class="xa-na-text">The fitted artifact provides ' + fmtInt((xaSvmVectors() || []).length || 0) +
        ' support vectors, dual coefficients, a fitted scaler and the Platt calibration &mdash; but no feature weights. ' +
        'Rather than fabricate a ranking, no bars are drawn.</div></div>';
      if (note) note.innerHTML = "";
      return;
    }
    const usage = xaSplitUsage(XA.model);
    if (!usage.ok) {
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">NOT AVAILABLE</div><div class="xa-na-text">The fitted ' + esc(xaModelLabel(XA.model)) + ' structure could not be read, so no importance is shown.</div></div>';
      if (pill) { pill.textContent = "NOT AVAILABLE"; pill.className = "dataset-pill pill-offline"; }
      return;
    }
    if (title) title.textContent = XA.model === "dt" ? "Decision Tree Split Usage" : "Random Forest Split Usage";
    if (pill) { pill.textContent = "RELATIVE MODEL IMPORTANCE"; pill.className = "dataset-pill"; }
    if (method) {
      method.innerHTML = XA.model === "dt"
        ? 'Counts how many times the fitted CART tree splits on each feature. Read from <code>DT_TREE</code> in <code>model-artifacts.js</code> &mdash; ' + fmtInt(usage.splits) + ' internal splits across ' + fmtInt(usage.nodes) + ' nodes. Bars are relative, normalized to the most-used feature; they are not causal.'
        : 'Counts how many times the 300 fitted trees split on each feature, read from <code>FOREST</code> in <code>model-data.js</code> &mdash; ' + fmtInt(usage.splits) + ' internal splits across ' + fmtInt(usage.nodes) + ' nodes. This is a structural statistic, <strong>not</strong> SHAP and not impurity-weighted importance. Relative model importance only.';
    }
    const max = usage.rows.length ? usage.rows[0].count : 1;
    host.innerHTML = usage.rows.map(function (r) {
      const s = xaStrengthWord(r.share);
      const pct = clamp((r.count / (max || 1)) * 100, 1.5, 100);
      const cells = Math.round((r.count / (usage.splits || 1)) * 20);
      return '<div class="xa-bar-row' + (XA.feature === r.key ? " sel" : "") + '" data-xa-feature="' + esc(r.key || "") + '" role="button" tabindex="0">' +
        '<span class="xa-bar-name">' + esc(r.label) + '</span>' +
        '<span class="xa-bar-track"><i class="' + s.cls + '" style="width:' + pct.toFixed(1) + '%"></i>' +
        '<em class="xa-cells" aria-hidden="true">' + "\u2588".repeat(Math.max(1, Math.min(20, cells))) + '</em></span>' +
        '<span class="xa-bar-tag ' + s.cls + '">' + esc(s.word) + '</span>' +
        '<span class="xa-bar-num">' + fmtInt(r.count) + '</span>' +
        '<span class="xa-bar-pct">' + fmt(r.share * 100, 1) + '%</span>' +
        '</div>';
    }).join("");
    if (note) {
      note.innerHTML = 'Source: ' + (XA.model === "dt" ? "<code>DT_TREE</code> (model-artifacts.js)" : "<code>FOREST</code> (model-data.js)") +
        ' \u00b7 ' + fmtInt(usage.trees) + (XA.model === "dt" ? " tree" : " trees") + ' \u00b7 ' + fmtInt(usage.splits) + ' internal splits. ' +
        'Impurity-weighted importance is <strong>not available</strong> (per-node sample counts were never serialized), so none is displayed. ' +
        'These values show how much the model <em>relies</em> on each feature, not what causes a failure.';
    }
  }
  /* ---- REAL Decision Tree path visual ---------------------------------- */
  function xaRenderDtPath() {
    const host = $("xaExplainBody");
    if (!host) return;
    if (XA.model === "svm") { xaRenderSvm(); return; }
    if (XA.model === "rf") { xaRenderForestTrace(); return; }
    const pill = $("xaExplainPill");
    const path = xaDtPath(state.sensors);
    if (!path) {
      if (pill) { pill.textContent = "NOT AVAILABLE"; pill.className = "dataset-pill pill-offline"; }
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">DECISION PATH NOT AVAILABLE</div><div class="xa-na-text">The fitted tree could not be read from the artifact, so no path is shown.</div></div>';
      return;
    }
    if (pill) { pill.textContent = path.depth + " SPLITS \u00b7 REAL"; pill.className = "dataset-pill"; }
    const lvl = xaLevelOf(path.prob * 100);
    let html = '<div class="xa-tree-root"><span class="xa-tree-rootk">MACHINE INPUT</span>' +
      '<span class="xa-tree-rootv">' + esc(typeLabel(state.sensors.productType)) + ' \u00b7 ' +
      fmt(state.sensors.airTemperature, 1) + 'K \u00b7 ' + fmt(state.sensors.processTemperature, 1) + 'K \u00b7 ' +
      Math.round(state.sensors.rotationalSpeed) + 'rpm \u00b7 ' + fmt(state.sensors.torque, 1) + 'Nm \u00b7 ' +
      Math.round(state.sensors.toolWear) + 'min</span></div>';
    html += path.steps.map(function (s) {
      const cmp = s.categorical
        ? (s.col === "Type_H" ? "input Type_H = 1" : s.col === "Type_L" ? "input Type_L = 1" : "input Type_M = 1")
        : "input " + fmt(s.value, 2) + (s.left ? " \u2264 " : " > ") + fmt(s.threshold, 2);
      return '<div class="xa-tree-node' + (XA.feature === s.key ? " sel" : "") + '" data-xa-feature="' + esc(s.key || "") + '">' +
        '<div class="xa-tree-line"></div>' +
        '<span class="xa-tree-feat">' + esc(s.label) + '</span>' +
        '<span class="xa-tree-test">' + esc(s.col) + ' ' + (s.categorical ? "\u2260 0" : "\u2264 " + fmt(s.threshold, 2)) + '</span>' +
        '<span class="xa-tree-cmp">' + esc(cmp) + '</span>' +
        '<span class="xa-tree-branch ' + (s.left ? "b-left" : "b-right") + '">' + (s.left ? "\u25B6 TRUE \u2192 LEFT" : "\u25B6 FALSE \u2192 RIGHT") + '</span>' +
        '</div>';
    }).join("");
    html += '<div class="xa-tree-leaf ' + xaBandClass(lvl.key) + '">' +
      '<span class="xa-tree-leafk">LEAF \u00b7 PREDICTION</span>' +
      '<span class="xa-tree-leafv">' + esc(lvl.label) + '</span>' +
      '<span class="xa-tree-leafn">p = ' + fmt(path.prob * 100, 1) + '% (leaf value ' + fmt(path.leaf, 4) + ')</span>' +
      '</div>';
    const dtArt = window.EXTRA_MODEL_DATA ? window.EXTRA_MODEL_DATA.dt : null;
    html += '<div class="muted-note">Every threshold above is read from the fitted tree in <code>model-artifacts.js</code> &mdash; nothing is hard-coded. ' +
      'Grown with <code>gini</code>, <code>max_depth=' + (dtArt && dtArt.params ? dtArt.params.max_depth : "n/a") +
      '</code>, <code>min_samples_leaf=' + (dtArt && dtArt.params ? dtArt.params.min_samples_leaf : "n/a") + '</code>, ' +
      'traversed ' + path.depth + ' nodes for this input.</div>';
    host.innerHTML = html;
  }
  /* ---- REAL per-input forest trace ------------------------------------- */
  function xaRenderForestTrace() {
    const host = $("xaExplainBody");
    if (!host) return;
    const pill = $("xaExplainPill");
    const tr = xaForestTrace(state.sensors);
    if (!tr) {
      if (pill) { pill.textContent = "NOT AVAILABLE"; pill.className = "dataset-pill pill-offline"; }
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">NOT AVAILABLE</div><div class="xa-na-text">The forest artifact could not be read.</div></div>';
      return;
    }
    if (pill) { pill.textContent = "REAL TRACE \u00b7 " + fmtInt(tr.total) + " TREES"; pill.className = "dataset-pill"; }
    const maxT = tr.rows.length ? tr.rows[0].trees : 1;
    const votePct = clamp((tr.highRiskTrees / (tr.total || 1)) * 100, 0, 100);
    host.innerHTML =
      '<div class="xa-vote">' +
      '<span class="xa-vote-k">TREES VOTING FAILURE (leaf p &gt; 0.5)</span>' +
      '<span class="xa-vote-v"><b>' + fmtInt(tr.highRiskTrees) + '</b> / ' + fmtInt(tr.total) + '</span>' +
      '<span class="xa-vote-track"><i class="' + (votePct > 50 ? "b-right" : "b-left") + '" style="width:' + votePct.toFixed(1) + '%"></i></span>' +
      '<span class="xa-vote-n">' + fmt(votePct, 1) + '% of the forest voted for the failure class. Averaging those leaf probabilities produces the ' +
      fmt(state.risk, 1) + '% headline risk.</span></div>' +
      '<div class="xa-sub-h">WHICH FEATURE GATED THIS INPUT</div>' +
      '<div class="xa-sub-n">For the current reading every one of the ' + fmtInt(tr.total) +
      ' fitted trees was walked node by node. The counts below are the real number of trees whose decision hinged on each feature.</div>' +
      '<div class="xa-trace-rows">' + tr.rows.map(function (r) {
        const s = xaStrengthWord(r.share);
        const pct = clamp((r.trees / (maxT || 1)) * 100, 1.5, 100);
        return '<div class="xa-bar-row' + (XA.feature === r.key ? " sel" : "") + '" data-xa-feature="' + esc(r.key || "") + '" role="button" tabindex="0">' +
          '<span class="xa-bar-name">' + esc(r.label) + '</span>' +
          '<span class="xa-bar-track"><i class="' + s.cls + '" style="width:' + pct.toFixed(1) + '%"></i></span>' +
          '<span class="xa-bar-tag ' + s.cls + '">' + esc(s.word) + '</span>' +
          '<span class="xa-bar-num">' + fmtInt(r.trees) + '/' + fmtInt(tr.total) + '</span>' +
          '<span class="xa-bar-pct">' + fmtInt(r.nodes) + ' nodes \u00b7 depth ' + fmt(r.avgDepth, 1) + '</span>' +
          '</div>';
      }).join("") + '</div>' +
      '<div class="muted-note">Left figure counts the trees whose decision hinged on that feature (of ' + fmtInt(tr.total) + '); node count is how often it was evaluated. ' +
      'Depth is the average position at which the feature was evaluated (0 = root). A feature evaluated near the root influenced more of the final decision. ' +
      'This is a genuine per-input trace of the fitted forest, not an attribution estimate.</div>';
  }
  /* ---- SVM: honest "no feature explanation" panel ----------------------- */
  function xaRenderSvm() {
    const host = $("xaExplainBody");
    if (!host) return;
    const pill = $("xaExplainPill");
    const e = xaSvmExplain(state.sensors);
    if (!e) {
      if (pill) { pill.textContent = "NOT AVAILABLE"; pill.className = "dataset-pill pill-offline"; }
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">SVM EXPLANATION</div><div class="xa-na-text">The SVM artifact could not be read, so no decision score is shown.</div></div>';
      return;
    }
    if (pill) { pill.textContent = "REAL INTERNALS"; pill.className = "dataset-pill"; }
    const lvl = xaLevelOf(e.prob != null ? e.prob * 100 : 0);
    const domPct = e.totalAbs > 0 ? clamp((e.dominance / e.totalAbs) * 100, 0, 100) : 0;
    const topAbs = e.top.length ? (Math.abs(e.top[0].term) || 1) : 1;
    host.innerHTML =
      '<div class="xa-na-box compact"><div class="xa-na">SVM EXPLANATION</div>' +
      '<div class="xa-na-text">Detailed feature-level explanation is <strong>not available</strong> from the current SVM artifact. ' +
      'No feature importance is shown, because none can be honestly derived.</div></div>' +
      '<div class="xa-svm-grid">' +
      '<div class="xa-svm-cell"><span class="xa-svm-k">PROBABILITY</span><span class="xa-svm-v ' + xaBandClass(lvl.key) + '">' + (e.prob != null ? fmt(e.prob * 100, 1) + "%" : "N/A") + '</span><span class="xa-svm-n">Platt-calibrated failure probability</span></div>' +
      '<div class="xa-svm-cell"><span class="xa-svm-k">DECISION SCORE</span><span class="xa-svm-v">' + fmt(e.margin, 4) + '</span><span class="xa-svm-n">kernel sum + bias, pre-calibration</span></div>' +
      '<div class="xa-svm-cell"><span class="xa-svm-k">SUPPORT VECTORS</span><span class="xa-svm-v">' + fmtInt(e.supportVectors) + '</span><span class="xa-svm-n">fitted on the training split</span></div>' +
      '<div class="xa-svm-cell"><span class="xa-svm-k">GAMMA</span><span class="xa-svm-v">' + fmt(e.gamma, 4) + '</span><span class="xa-svm-n">RBF width in standardized space</span></div>' +
      '</div>' +
      '<div class="xa-sub-h">MOST INFLUENTIAL SUPPORT VECTORS</div>' +
      '<div class="xa-sub-n">Each row is a real term &alpha;<sub>i</sub>&middot;K(x, x<sub>i</sub>) from the decision function. ' +
      'These are support-vector weights, <strong>not</strong> feature weights.</div>' +
      '<div class="xa-svm-rows">' + e.top.map(function (t, i) {
        const sh = t.term < 0 ? "b-left" : "b-right";
        const mag = clamp((Math.abs(t.term) / topAbs) * 100, 4, 100);
        return '<div class="xa-svm-row"><span class="xa-svm-rank">' + (i + 1) + '</span>' +
          '<span class="xa-svm-sv">SV #' + fmtInt(t.k) + '</span>' +
          '<span class="xa-svm-track"><i class="' + sh + '" style="width:' + mag.toFixed(1) + '%"></i></span>' +
          '<span class="xa-svm-term">' + (t.term >= 0 ? "+" : "") + fmt(t.term, 4) + '</span>' +
          '<span class="xa-svm-d">dist ' + fmt(t.dist, 2) + '&sigma;</span></div>';
      }).join("") + '</div>' +
      '<div class="muted-note">The top ' + e.top.length + ' support vectors account for ' + fmt(domPct, 1) +
      '% of the total absolute kernel sum. This is genuine model state, but it is <strong>not</strong> a per-feature explanation.</div>';
  }
  /* ---- feature relevance for the CURRENT model only -------------------- */
  function xaFeatureContext() {
    const usage = XA.model === "svm" ? null : xaSplitUsage(XA.model);
    const path = XA.model === "dt" ? xaDtPath(state.sensors) : null;
    const trace = XA.model === "rf" ? xaForestTrace(state.sensors) : null;
    return { usage: usage, path: path, trace: trace };
  }
  function xaFeatureRelevance(fkey, ctx) {
    if (XA.model === "svm") return { word: "NOT AVAILABLE", cls: "s-none", source: "No feature explanation exists for an RBF kernel." };
    const f = xaFeat(fkey);
    if (!f) return { word: "N/A", cls: "s-none", source: "" };
    if (XA.model === "dt") {
      if (ctx.path && ctx.path.steps.some(function (s) { return s.key === fkey; })) {
        return { word: "ON PATH", cls: "s-high", source: "Real Decision Tree decision path" };
      }
      if (ctx.usage && ctx.usage.rows.length) {
        const row = ctx.usage.rows.filter(function (r) { return r.key === fkey; })[0];
        if (row) {
          const s = xaStrengthWord(row.share);
          return { word: s.word, cls: s.cls, source: "Decision Tree split usage (" + fmtInt(row.count) + " of " + fmtInt(ctx.usage.splits) + " splits)", detail: row };
        }
      }
      return { word: "NOT SPLIT", cls: "s-none", source: "The fitted tree never splits on this feature." };
    }
    if (ctx.trace) {
      const row = ctx.trace.rows.filter(function (r) { return r.key === fkey; })[0];
      if (row) {
        const s = xaStrengthWord(row.share);
        return { word: s.word, cls: s.cls, source: "Gated " + fmtInt(row.trees) + " of " + fmtInt(ctx.trace.total) + " trees on this input", detail: row };
      }
    }
    if (ctx.usage && ctx.usage.rows.length) {
      const row = ctx.usage.rows.filter(function (r) { return r.key === fkey; })[0];
      if (row) {
        const s = xaStrengthWord(row.share);
        return { word: s.word, cls: s.cls, source: "Random Forest split usage (" + fmtInt(row.count) + " splits)", detail: row };
      }
    }
    return { word: "UNUSED", cls: "s-none", source: "No tree in the forest splits on this feature." };
  }
  function xaRenderFeatures() {
    const host = $("xaFeatGrid");
    if (!host) return;
    const ctx = xaFeatureContext();
    const s = state.sensors;
    host.innerHTML = XA_FEATURES.map(function (f) {
      const rel = xaFeatureRelevance(f.key, ctx);
      const val = s[f.key];
      const shown = f.key === "productType" ? typeLabel(val) : fmt(val, f.dec);
      return '<button class="xa-feat' + (XA.feature === f.key ? " sel" : "") + '" type="button" data-xa-feature="' + esc(f.key) + '" aria-pressed="' + (XA.feature === f.key ? "true" : "false") + '">' +
        '<span class="xa-feat-k">' + esc(f.label.toUpperCase()) + '</span>' +
        '<span class="xa-feat-v">' + esc(shown) + (f.unit ? " " + esc(f.unit) : "") + '</span>' +
        '<span class="xa-feat-tag ' + rel.cls + '">' + esc(rel.word) + '</span>' +
        '<span class="xa-feat-zone">&#9679; ' + esc(f.zone) + '</span>' +
        '</button>';
    }).join("");
  }
  /* ---- floating feature HUD (only genuinely-existing fields) ----------- */
  function xaRenderHud() {
    const hud = $("xaHud");
    if (!hud) return;
    if (!XA.feature) { hud.hidden = true; return; }
    const f = xaFeat(XA.feature);
    if (!f) { hud.hidden = true; return; }
    hud.hidden = false;
    setText("xaHudTitle", f.label.toUpperCase());
    const ctx = xaFeatureContext();
    const rel = xaFeatureRelevance(XA.feature, ctx);
    const s = state.sensors;
    const cur = s[XA.feature];
    const base = WI.baseline ? WI.baseline.features[XA.feature] : null;
    const scen = WI.scenario ? WI.scenario.features[XA.feature] : null;
    const isText = XA.feature === "productType";
    const rows = [];
    rows.push(["CURRENT", isText ? typeLabel(s.productType) : fmt(cur, f.dec) + " " + f.unit]);
    rows.push(["BASELINE", base == null ? "NOT CAPTURED" : (isText ? typeLabel(base) : fmt(base, f.dec) + " " + f.unit)]);
    if (scen != null && !isText) {
      const d = Number(scen) - Number(cur);
      rows.push(["SCENARIO", fmt(scen, f.dec) + " " + f.unit + " (" + (d > 0 ? "+" : "") + fmt(d, f.dec) + ")"]);
    }
    rows.push(["MODEL RELEVANCE", '<span class="' + rel.cls + '">' + esc(rel.word) + "</span>", true]);
    if (rel.source) rows.push(["SOURCE", esc(rel.source)]);
    if (isText) {
      rows.push(["MODEL ONE-HOT", "Type_H=" + (s.productType === "H" ? 1 : 0) + " \u00b7 Type_L=" + (s.productType === "L" ? 1 : 0) + " \u00b7 Type_M=" + (s.productType === "M" ? 1 : 0)]);
    } else {
      const kd = kpiDefs().filter(function (d) { return d.key === f.twin; })[0];
      const lvl = indicatorLevel(kd ? kd.pct : 0);
      rows.push(["ADVISORY STATUS", esc(lvl.label)]);
      rows.push(["TREND", esc(ccTrendWord(f.twin))]);
      const lim = SENSOR_LIMITS[XA.feature];
      rows.push(["INPUT RANGE", lim ? fmt(lim.min, 1) + " \u2013 " + fmt(lim.max, 1) + " " + f.unit : "N/A"]);
    }
    const body = $("xaHudBody");
    if (body) {
      body.innerHTML = '<table class="xa-hud-table"><tbody>' + rows.map(function (r) {
        return "<tr><th>" + esc(r[0]) + "</th><td>" + (r[2] ? r[1] : esc(String(r[1]))) + "</td></tr>";
      }).join("") + "</tbody></table>" +
        '<div class="xa-hud-note">Advisory status and trend come from the rule engine and the telemetry buffer, not from the classifier. ' +
        'Input ranges are this prototype&rsquo;s visualisation limits, not industrial safety limits.</div>';
    }
  }
  /* ---- baseline vs scenario (real V8 integration) ---------------------- */
  function xaRenderCompare() {
    const host = $("xaCompare");
    if (!host) return;
    const pill = $("xaComparePill");
    if (!WI.baseline) {
      if (pill) { pill.textContent = "NO BASELINE"; pill.className = "dataset-pill pill-offline"; }
      host.innerHTML = '<div class="xa-na-text">No baseline captured yet. Open the Scenario Lab and run an analysis, or load a scenario, to compare baseline against a scenario here.</div>';
      return;
    }
    if (!WI.scenario) {
      if (pill) { pill.textContent = "BASELINE ONLY"; pill.className = "dataset-pill"; }
      host.innerHTML = '<div class="xa-cmp-single"><span class="xa-cmp-k">BASELINE RISK</span>' +
        '<span class="xa-cmp-v">' + fmt(WI.baseline.risk, 1) + '%</span>' +
        '<span class="xa-cmp-n">Random Forest at the captured baseline operating point.</span></div>' +
        '<div class="xa-na-text">Run a what-if analysis in the Scenario Lab to compare a scenario against this baseline.</div>';
      return;
    }
    if (pill) { pill.textContent = "BASELINE vs SCENARIO"; pill.className = "dataset-pill"; }
    const b = WI.baseline, sc = WI.scenario;
    const delta = sc.risk - b.risk;
    const dir = Math.abs(delta) < 0.05 ? "flat" : delta > 0 ? "up" : "down";
    const lvlB = xaLevelOf(b.risk), lvlS = xaLevelOf(sc.risk);
    const ctx = xaFeatureContext();
    const changed = XA_FEATURES.filter(function (f) { return String(b.features[f.key]) !== String(sc.features[f.key]); });
    host.innerHTML =
      '<div class="xa-cmp-row">' +
      '<div class="xa-cmp-box"><span class="xa-cmp-k">BASELINE</span><span class="xa-cmp-v">' + fmt(b.risk, 1) + '%</span><span class="xa-cmp-l ' + xaBandClass(lvlB.key) + '">' + esc(lvlB.label) + '</span></div>' +
      '<div class="xa-cmp-arrow"><span>\u2192</span></div>' +
      '<div class="xa-cmp-box"><span class="xa-cmp-k">SCENARIO</span><span class="xa-cmp-v">' + fmt(sc.risk, 1) + '%</span><span class="xa-cmp-l ' + xaBandClass(lvlS.key) + '">' + esc(lvlS.label) + '</span></div>' +
      '<div class="xa-cmp-delta ' + dir + '"><span class="xa-cmp-k">CHANGE</span><span class="xa-cmp-dv">' + (delta > 0 ? "+" : "") + fmt(delta, 1) + ' pp</span><span class="xa-cmp-n">percentage points</span></div>' +
      '</div>' +
      '<div class="xa-sub-h">CHANGED INPUTS</div>' +
      (changed.length ? '<div class="xa-changed">' + changed.map(function (f) {
        const bv = b.features[f.key], sv = sc.features[f.key];
        const isText = f.key === "productType";
        const rel = xaFeatureRelevance(f.key, ctx);
        const d = isText ? null : Number(sv) - Number(bv);
        return '<div class="xa-chg' + (XA.feature === f.key ? " sel" : "") + '" data-xa-feature="' + esc(f.key) + '" role="button" tabindex="0">' +
          '<span class="xa-chg-k">' + esc(f.label) + '</span>' +
          '<span class="xa-chg-v">' + esc(isText ? typeLabel(bv) : fmt(bv, f.dec)) + ' \u2192 ' + esc(isText ? typeLabel(sv) : fmt(sv, f.dec)) + (f.unit ? " " + esc(f.unit) : "") + '</span>' +
          (d == null ? "" : '<span class="xa-chg-d ' + (d > 0 ? "up" : "down") + '">' + (d > 0 ? "+" : "") + fmt(d, f.dec) + "</span>") +
          '<span class="xa-chg-r ' + rel.cls + '">' + esc(rel.word) + "</span>" +
          "</div>";
      }).join("") + "</div>" : '<div class="xa-na-text">No input feature differs between baseline and this scenario.</div>') +
      '<div class="xa-obs"><span class="xa-obs-k">MODEL-SUPPORTED OBSERVATION</span>' +
      '<p>' + (changed.length
        ? "The scenario changed " + changed.length + " input feature" + (changed.length === 1 ? "" : "s") + " and the Random Forest moved from " +
          fmt(b.risk, 1) + "% to " + fmt(sc.risk, 1) + "% (" + (delta > 0 ? "+" : "") + fmt(delta, 1) + " percentage points). " +
          (XA.model === "svm"
            ? "For the SVM no per-feature importance exists, so this model cannot attribute the change to individual features."
            : "In the currently selected " + esc(xaModelLabel(XA.model)) + ", the changed features listed above carry the relevance weights shown beside them.")
        : "The scenario matches the baseline input, so the model output is unchanged.") +
      '</p><p class="xa-obs-n">This reports a measured model response to a changed input. It is <strong>not</strong> a causal claim, and this prototype does not compute per-feature causal effects on the risk change.</p></div>';
  }
  /* ---- model availability panel ---------------------------------------- */
  function xaRenderCompareModels() {
    const host = $("xaCompareModels");
    if (!host) return;
    const yes = '<span class="xa-avail yes">&#10003; AVAILABLE</span>';
    const no = '<span class="xa-avail no">&#10007; NOT AVAILABLE</span>';
    const rows = ["dt", "rf", "svm"].map(function (k) {
      const cap = xaCapability(k);
      let detail = "";
      if (k === "dt") { const u = xaSplitUsage("dt"); detail = u.ok ? fmtInt(u.splits) + " real splits, depth read from the artifact" : ""; }
      else if (k === "rf") { const u = xaSplitUsage("rf"); detail = u.ok ? fmtInt(u.trees) + " trees, " + fmtInt(u.splits) + " real splits" : ""; }
      else { const a = xaSvmArt(); detail = a ? fmtInt(a.supportVectors || 0) + " support vectors, gamma " + fmt(a.params ? a.params.gamma : 0, 4) : ""; }
      return '<div class="xa-av-row' + (XA.model === k ? " sel" : "") + '" data-xa-model="' + k + '" role="button" tabindex="0">' +
        '<span class="xa-av-name">' + esc(xaModelLabel(k)) + (XA.model === k ? ' <em class="xa-av-cur">SELECTED</em>' : "") + "</span>" +
        '<span class="xa-av-i"><b>Prediction</b>' + (cap.prob ? yes : no) + "</span>" +
        '<span class="xa-av-i"><b>Decision path</b>' + (cap.path ? yes : no) + "</span>" +
        '<span class="xa-av-i"><b>Feature importance</b>' + (cap.importance ? yes : no) + "</span>" +
        '<span class="xa-av-i"><b>Decision score</b>' + (cap.score ? yes : no) + "</span>" +
        '<span class="xa-av-d">' + esc(detail) + "</span></div>";
    }).join("");
    host.innerHTML = rows + '<div class="muted-note">Availability is detected at runtime from the loaded artifacts &mdash; it is not assumed. ' +
      'The three models do <strong>not</strong> support the same explanation types, and this page never substitutes one model&rsquo;s explanation for another&rsquo;s.</div>';
  }
  /* ---- animated pipeline (visual only) -------------------------------- */
  const XA_FLOW = [
    { k: "TELEMETRY", d: "5 sensor readings" },
    { k: "EXTRACTION", d: "5 features + 3 one-hot" },
    { k: "MODEL ANALYSIS", d: "tree / 300 trees / kernel" },
    { k: "SPLIT USAGE", d: "relative model reliance" },
    { k: "DECISION PATH", d: "real node traversal" },
    { k: "RISK ENGINE", d: "probability" },
    { k: "EXPLANATION", d: "model-supported only" }
  ];
  function xaRenderFlow() {
    const host = $("xaFlow");
    if (!host) return;
    host.innerHTML = XA_FLOW.map(function (s, i) {
      const na = XA.model === "svm" && s.k === "SPLIT USAGE";
      const on = s.k === "DECISION PATH" || (s.k === "SPLIT USAGE" && XA.model !== "svm");
      return '<div class="xa-flow-step' + (on ? " on" : "") + (na ? " na" : "") + '" style="--d:' + i + '">' +
        '<span class="xa-flow-k">' + esc(s.k) + "</span>" +
        '<span class="xa-flow-d">' + esc(na ? "NOT AVAILABLE" : s.d) + "</span>" +
        '<span class="xa-flow-pip"></span></div>';
    }).join("");
  }
  /* ---- AI decision summary + honest natural language ------------------- */
  function xaSummaryLines() {
    const prob = xaSelectedProb();
    const out = [];
    if (prob == null) {
      out.push({ k: "Current risk", v: "NOT AVAILABLE" });
      out.push({ k: "Model", v: xaModelLabel(XA.model) + " returned no probability" });
      out.push({ k: "Decision", v: "CANNOT BE STATED" });
      return out;
    }
    const lvl = xaLevelOf(prob);
    out.push({ k: "Current risk", v: fmt(prob, 1) + "% (" + lvl.label + ")" });
    out.push({ k: "Model", v: xaModelLabel(XA.model) });
    const feats = xaActiveFeatures();
    if (XA.model === "svm") {
      out.push({ k: "Decision", v: lvl.label });
      out.push({ k: "Most important signals", v: "NOT AVAILABLE \u2014 an RBF kernel exposes no per-feature importance" });
      return out;
    }
    if (XA.model === "dt") {
      out.push({ k: "Decision", v: lvl.label + " \u00b7 leaf p = " + fmt(prob, 1) + "%" });
      out.push({ k: "Decision path", v: feats.length ? feats.map(function (f) { return f.label; }).join(" \u2192 ") : "NOT AVAILABLE" });
    } else {
      const tr = xaForestTrace(state.sensors);
      out.push({ k: "Decision", v: lvl.label + " \u00b7 " + fmtInt(tr ? tr.highRiskTrees : 0) + " of 300 trees voted failure" });
      const top = feats.slice(0, 3).map(function (f) { return f.label; });
      out.push({ k: "Most important model signals", v: top.length ? top.join(", ") + " (relative model split usage)" : "NOT AVAILABLE" });
    }
    return out;
  }
  function xaNarrative() {
    const prob = xaSelectedProb();
    if (prob == null) return "The selected model did not return a probability for the current input, so no explanation can be stated.";
    if (XA.model === "svm") {
      return "The RBF SVM reports a calibrated failure probability of " + fmt(prob, 1) + "% for the current reading. " +
        "This model does not expose a per-feature breakdown: an RBF kernel sums distance-weighted terms over " +
        fmtInt((xaSvmVectors() || []).length || 0) + " support vectors and has no feature weight vector, so no feature importance can be stated for it.";
    }
    const parts = [];
    const lvl = xaLevelOf(prob);
    parts.push("The " + xaModelLabel(XA.model) + " assigns a failure probability of " + fmt(prob, 1) + "% (" + lvl.label + ") to the current reading.");
    const feats = xaActiveFeatures();
    if (XA.model === "dt") {
      if (feats.length) parts.push("The input reaches its prediction by " + feats.length + " consecutive splits on " +
        feats.map(function (f) { return f.label; }).join(", ") + ". Every threshold is read from the fitted tree.");
    } else {
      const tr = xaForestTrace(state.sensors);
      if (tr && tr.rows.length) {
        parts.push("Across the 300 fitted trees, " +
          tr.rows.slice(0, 3).map(function (r) { return r.label + " gated " + fmtInt(r.trees) + " trees"; }).join(", ") +
          ". These are relative model usage statistics.");
      }
    }
    if (WI.scenario && WI.baseline) {
      const d = WI.scenario.risk - WI.baseline.risk;
      parts.push("The last scenario run moved the Random Forest risk from " + fmt(WI.baseline.risk, 1) + "% to " +
        fmt(WI.scenario.risk, 1) + "% (" + (d > 0 ? "+" : "") + fmt(d, 1) + " percentage points).");
    }
    parts.push("These statements describe model behaviour on this input. They do not establish physical causation, and no Shapley value or remaining-useful-life estimate is produced.");
    return parts.join(" ");
  }
  function xaRenderSummary() {
    const host = $("xaSummary");
    if (!host) return;
    const lines = xaSummaryLines();
    host.innerHTML = '<div class="xa-sum-grid">' + lines.map(function (l) {
      return '<div class="xa-sum-item"><span class="xa-sum-k">' + esc(l.k) + '</span><span class="xa-sum-v">' + esc(l.v) + "</span></div>";
    }).join("") + "</div>" +
      '<div class="xa-narr"><span class="xa-narr-k">EXPLANATION</span><p>' + esc(xaNarrative()) + "</p></div>" +
      '<div class="xa-rul"><span class="xa-rul-k">RUL</span><span class="xa-rul-v">NOT AVAILABLE</span>' +
      '<span class="xa-rul-n">Binary classification model &mdash; predicts failure / no failure only. No remaining-useful-life model exists.</span></div>';
  }
  /* ---- maintenance advisory (reuses the EXISTING rule engine) ---------- */
  function xaRenderAdvisory() {
    const host = $("xaAdvisory");
    if (!host) return;
    const p = state.prediction;
    const recs = (p && p.recommendations) || [];
    if (!recs.length) {
      host.innerHTML = '<div class="xa-na-box"><div class="xa-na">NO SUPPORTED MAINTENANCE ADVISORY AVAILABLE</div>' +
        '<div class="xa-na-text">The advisory rule engine produced no recommendation for this reading. Nothing is invented to fill the gap.</div></div>';
      return;
    }
    const lvl = xaLevelOf(state.risk);
    host.innerHTML = '<div class="xa-adv-top"><span class="xa-adv-k">RISK</span>' +
      '<span class="xa-adv-v ' + xaBandClass(lvl.key) + '">' + fmt(state.risk, 1) + "% \u00b7 " + esc(lvl.label) + "</span></div>" +
      '<div class="xa-adv-list">' + recs.slice(0, 4).map(function (r) {
        const pr = priorityToDisplay(r.priority);
        return '<div class="xa-adv-item ' + esc(pr.cls || "") + '">' +
          '<span class="xa-adv-p">' + esc(pr.label) + "</span>" +
          '<span class="xa-adv-a">' + esc(r.action) + "</span>" +
          '<span class="xa-adv-r">' + esc(r.reason) + "</span></div>";
      }).join("") + "</div>" +
      '<div class="muted-note">Source: the rule-based advisory engine in <code>model-data.js</code> using documented AI4I failure definitions. ' +
      'This is <strong>not</strong> a model explanation and <strong>not</strong> a predicted failure mode &mdash; the classifiers only predict failure / no failure. ' +
      'STATUS: ADVISORY.</div>';
  }
  /* ---- explanation history (session-only, real events only) ------------ */
  function xaPushHistory(label) {
    XA.history.unshift({ time: nowStamp(), model: XA.model, label: label, risk: xaSelectedProb() });
    if (XA.history.length > 12) XA.history.length = 12;
  }
  function xaRenderHistory() {
    const host = $("xaHistory");
    if (!host) return;
    setText("xaHistPill", XA.history.length ? XA.history.length + " EVENTS" : "SESSION ONLY");
    if (!XA.history.length) {
      host.innerHTML = '<div class="xa-na-text">No explanation events yet in this session. Entries appear only when you inspect a model, select a feature or switch explanation mode.</div>';
      return;
    }
    host.innerHTML = '<div class="xa-hist">' + XA.history.map(function (h) {
      return '<div class="xa-hist-row"><span class="xa-hist-t">' + esc(h.time) + "</span>" +
        '<span class="xa-hist-m">' + esc(xaModelLabel(h.model)) + "</span>" +
        '<span class="xa-hist-l">' + esc(h.label) + "</span>" +
        '<span class="xa-hist-r">' + (h.risk == null ? "--" : fmt(h.risk, 1) + "%") + "</span></div>";
    }).join("") + "</div>" +
      '<div class="muted-note">Session-only log of real explanation events. Nothing is pre-seeded or back-dated.</div>';
  }
  /* ---- digital-twin conceptual highlight ------------------------------- */
  /* Purely visual: it changes nothing about the machine geometry or physics. */
  function xaTwinFocus() {
    if (!XA.feature) return null;
    const f = xaFeat(XA.feature);
    return f ? f.twin : null;
  }
  /* ---- main dispatcher (called from renderAll) ------------------------- */
  function updateXai() {
    if (state.view !== "xai") return;
    xaRenderHero();
    xaRenderImpact();
    xaRenderDtPath();
    xaRenderFeatures();
    xaRenderHud();
    xaRenderCompare();
    xaRenderCompareModels();
    xaRenderFlow();
    xaRenderSummary();
    xaRenderAdvisory();
    xaRenderHistory();
    const sel = $("xaSelNote");
    if (sel) {
      const cap = xaCapability(XA.model);
      const bits = [];
      bits.push(cap.prob ? "Prediction available" : "Prediction NOT AVAILABLE");
      bits.push(cap.path ? "Decision path available" : "Decision path NOT AVAILABLE");
      bits.push(cap.importance ? "Feature importance available" : "Feature importance NOT AVAILABLE");
      if (cap.score) bits.push("Decision score available");
      sel.innerHTML = "<b>" + esc(xaModelLabel(XA.model)) + ":</b> " + esc(bits.join(" \u00b7 ")) +
        ' <span class="xa-sel-why">' + esc(cap.importanceWhy || "") + "</span>";
    }
    document.body.classList.toggle("xa-mode", !!XA.mode);
  }
  /* ---- interactions ---------------------------------------------------- */
  function xaSelectFeature(key) {
    const f = xaFeat(key);
    if (!f) return;
    XA.feature = (XA.feature === key) ? null : key;
    /* The twin mirrors the selection so the operator sees the conceptual part.
       state.sensors is never touched by any of this. */
    if (CC && CC.selectedSensor !== undefined) CC.selectedSensor = xaTwinFocus() || null;
    xaPushHistory(XA.feature ? "Feature inspected: " + f.label : "Feature cleared");
    updateXai();
    ccEvent("FEATURE SELECTED \u00b7 " + (XA.feature ? f.label.toUpperCase() : "CLEARED"));
  }
  function xaSelectModel(model) {
    if (["dt", "rf", "svm"].indexOf(model) < 0) return;
    XA.model = model;
    XA.feature = null;
    if (CC && CC.selectedSensor !== undefined) CC.selectedSensor = null;
    const btns = document.querySelectorAll("#xaSelBtns .xa-sel-btn");
    Array.prototype.forEach.call(btns, function (b) {
      const on = b.getAttribute("data-model") === model;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    xaPushHistory("Explanation inspected");
    updateXai();
    ccEvent("EXPLAIN MODEL \u2192 " + xaModelLabel(model).toUpperCase());
  }
  function xaToggleMode() {
    XA.mode = !XA.mode;
    const btn = $("xaModeBtn");
    if (btn) btn.innerHTML = XA.mode ? "&#10005; EXIT EXPLANATION MODE" : "EXPLANATION MODE";
    document.body.classList.toggle("xa-mode", XA.mode);
    xaPushHistory(XA.mode ? "Explanation mode enabled" : "Explanation mode disabled");
    updateXai();
    ccEvent(XA.mode ? "EXPLANATION MODE ENABLED" : "EXPLANATION MODE DISABLED");
  }
  /* ---- Explainable AI report export (no fabricated rows) --------------- */
  function xaExportReport() {
    const prob = xaSelectedProb();
    const lines = [];
    const push = (a, b) => lines.push([a, b == null ? "NOT AVAILABLE" : String(b)]);
    push("REPORT", "Explainable AI \u2014 Machine Intelligence V9.0");
    push("Generated", nowFull());
    push("Machine", (state.machine && state.machine.id) || "CNC-MILL-08");
    push("Selected model", xaModelLabel(XA.model));
    push("Failure probability (%)", prob == null ? null : fmt(prob, 1));
    push("Risk level", prob == null ? null : xaLevelOf(prob).label);
    push("RUL", "NOT AVAILABLE \u2014 binary classification model");
    push("SHAP values", "NOT COMPUTED \u2014 none exist in this system");
    push("Causality", "NOT ESTABLISHED \u2014 model importance is not causation");
    lines.push(["", ""]);
    push("MACHINE INPUT", "");
    push("Product type", typeLabel(state.sensors.productType));
    push("Air temperature (K)", fmt(state.sensors.airTemperature, 1));
    push("Process temperature (K)", fmt(state.sensors.processTemperature, 1));
    push("Rotational speed (rpm)", String(Math.round(state.sensors.rotationalSpeed)));
    push("Torque (Nm)", fmt(state.sensors.torque, 1));
    push("Tool wear (min)", String(Math.round(state.sensors.toolWear)));
    lines.push(["", ""]);
    push("FEATURE IMPORTANCE", "");
    if (XA.model === "svm") {
      push("Feature importance", "NOT AVAILABLE \u2014 RBF kernel has no feature weight vector");
    } else {
      const u = xaSplitUsage(XA.model);
      if (u.ok) {
        push("Metric", "Relative model split usage (NOT SHAP, NOT impurity-weighted)");
        push("Total internal splits", fmtInt(u.splits));
        u.rows.forEach(function (r) { push("  " + r.label, fmtInt(r.count) + " splits (" + fmt(r.share * 100, 2) + "%)"); });
      } else push("Feature importance", "NOT AVAILABLE");
    }
    lines.push(["", ""]);
    push("DECISION PATH / SCORE", "");
    if (XA.model === "dt") {
      const path = xaDtPath(state.sensors);
      if (path) {
        push("Splits traversed", path.depth);
        path.steps.forEach(function (s, i) {
          push("  Step " + (i + 1), s.label + ": " + (s.categorical ? s.col + " != 0" : s.col + " <= " + fmt(s.threshold, 2)) +
            " | input " + fmt(s.value, 2) + " -> " + (s.left ? "TRUE (left)" : "FALSE (right)"));
        });
        push("Leaf probability", fmt(path.prob, 4));
      } else push("Decision path", "NOT AVAILABLE");
    } else if (XA.model === "svm") {
      const e = xaSvmExplain(state.sensors);
      if (e) {
        push("Decision score", fmt(e.margin, 6));
        push("Support vectors", String(e.supportVectors));
        push("Gamma", fmt(e.gamma, 6));
        e.top.forEach(function (t) { push("  SV #" + t.k, "alpha*K = " + fmt(t.term, 6) + ", dist " + fmt(t.dist, 4)); });
      } else push("Decision score", "NOT AVAILABLE");
    } else {
      const tr = xaForestTrace(state.sensors);
      if (tr) {
        push("Trees voting failure", fmtInt(tr.highRiskTrees) + " of " + fmtInt(tr.total));
        tr.rows.forEach(function (r) { push("  " + r.label, fmtInt(r.trees) + " trees gated (" + fmt(r.share * 100, 1) + "%)"); });
      }
    }
    lines.push(["", ""]);
    push("BASELINE vs SCENARIO", "");
    if (WI.baseline && WI.scenario) {
      push("Baseline risk (%)", fmt(WI.baseline.risk, 1));
      push("Scenario risk (%)", fmt(WI.scenario.risk, 1));
      push("Change (pp)", (WI.scenario.risk - WI.baseline.risk > 0 ? "+" : "") + fmt(WI.scenario.risk - WI.baseline.risk, 1));
      push("Scenario name", WI.scenario.name || "unnamed");
    } else push("Comparison", "NOT AVAILABLE \u2014 run a Scenario Lab analysis");
    lines.push(["", ""]);
    push("LIMITATIONS", "");
    push("Attribution method", XA.model === "svm" ? "none available for this model" : "relative split usage \u2014 not SHAP");
    push("Physical causation", "not established");
    push("Failure mode prediction", "NOT AVAILABLE \u2014 binary models only");
    const csv = lines.map(function (r) {
      return r.map(function (cell) { return '"' + String(cell).replace(/"/g, '""') + '"'; }).join(",");
    }).join("\r\n");
    try {
      const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "explainable-ai-report-" + exportStamp() + ".csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
      toast("Explainable AI report exported.", "ok");
      ccEvent("XAI REPORT EXPORTED \u00b7 " + xaModelLabel(XA.model).toUpperCase());
    } catch (err) {
      toast("Export failed: " + err.message, "error");
    }
  }
  function bindXai() {
    const host = $("view-xai");
    if (!host) return;
    /* one delegated listener covers every [data-xa-feature] surface */
    host.addEventListener("click", function (e) {
      const mv = e.target.closest("[data-xa-model], .xa-sel-btn");
      if (mv) { xaSelectModel(mv.getAttribute("data-xa-model") || mv.getAttribute("data-model")); return; }
      const f = e.target.closest("[data-xa-feature]");
      if (f) {
        const k = f.getAttribute("data-xa-feature");
        if (k) xaSelectFeature(k);
      }
    });
    host.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") return;
      const el = e.target.closest("[data-xa-feature],[data-xa-model]");
      if (!el) return;
      e.preventDefault();
      el.click();
    });
    const close = $("xaHudClose");
    if (close) close.addEventListener("click", function () {
      XA.feature = null;
      if (CC && CC.selectedSensor !== undefined) CC.selectedSensor = null;
      updateXai();
    });
    const mode = $("xaModeBtn");
    if (mode) mode.addEventListener("click", xaToggleMode);
    const why = $("xaWhyBtn");
    if (why) why.addEventListener("click", function () {
      switchView("xai");
      xaPushHistory("WHY panel opened");
      toast("Showing the model-supported explanation.", "info");
    });
    const exp = $("xaExportBtn");
    if (exp) exp.addEventListener("click", xaExportReport);
  }
  /* ============================ ANALYTICS VIEW ============================ */
  function drawMultiLine(canvas, series, opts) {
    const options = opts || {};
    const maxLen = Math.max.apply(null, series.map((s) => s.values.length).concat([0]));
    if (!maxLen) { drawEmpty(canvas, options.emptyText || "No data."); return; }
    const c = canvasCtx(canvas);
    if (!c) return;
    const ctx = c.ctx, w = c.w, h = c.h;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    const pad = 6;
    const stepX = maxLen > 1 ? (w - pad * 2) / (maxLen - 1) : 0;
    series.forEach((s) => {
      if (!s.values.length) return;
      ctx.beginPath();
      s.values.forEach((v, i) => {
        const x = pad + i * stepX;
        const y = h - pad - (clamp(v, 0, 100) / 100) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = hex(s.color);
      ctx.lineWidth = 1.8;
      ctx.stroke();
    });
  }
  function updateAnalyticsView() {
    if (state.view !== "analytics") return;
    const rows = state.history;
    const risks = rows.map((r) => Number(r.risk));
    const counts = { NORMAL: 0, WARNING: 0, CRITICAL: 0 };
    rows.forEach((r) => { if (counts[r.status] != null) counts[r.status] += 1; });
    setText("anTotal", fmtInt(rows.length));
    setText("anNormal", fmtInt(counts.NORMAL));
    setText("anWarning", fmtInt(counts.WARNING));
    setText("anCritical", fmtInt(counts.CRITICAL));
    setText("anAvgRisk", risks.length ? fmt(risks.reduce((a, b) => a + b, 0) / risks.length, 1) + "%" : "--");
    setText("anMaxRisk", risks.length ? fmt(Math.max.apply(null, risks), 1) + "%" : "--");
    setText("anHealth", fmt(state.health, 0) + " / 100");
    setText("anThresholdNote", "Thresholds: warning ≥ " + state.settings.warnThreshold + "% · critical ≥ " +
      state.settings.critThreshold + "% (configurable in Settings).");
    drawHistogram($("chartRiskDist"), risks, { bins: 12, color: "#f43f5e", xLabel: "risk %", emptyText: "No recorded events yet — run a scenario, a preset or the telemetry simulation." });
    const maxCount = Math.max(1, counts.NORMAL, counts.WARNING, counts.CRITICAL);
    drawBarChart($("chartStatusDist"), [
      { label: "Normal", value: counts.NORMAL, display: fmtInt(counts.NORMAL), color: "#10b981" },
      { label: "Warning", value: counts.WARNING, display: fmtInt(counts.WARNING), color: "#f59e0b" },
      { label: "Critical", value: counts.CRITICAL, display: fmtInt(counts.CRITICAL), color: "#f43f5e" }
    ], { max: maxCount, padL: 76, padR: 70 });
    drawMultiLine($("chartSensorTrend"), [
      { values: rows.map((r) => Number(r.torque) * 1.25), color: "#06b6d4" },
      { values: rows.map((r) => Number(r.toolWear) / 2.6), color: "#f43f5e" },
      { values: risks, color: "#f59e0b" }
    ], { emptyText: "No recorded events yet." });
    drawLineChart($("chartPredTime"), { values: risks, color: "#38bdf8", yMax: 100, emptyText: "No recorded events yet." });
    setText("anSpanPill", rows.length ? rows.length + " EVENTS" : "EMPTY");
    setText("anEmptyNote", rows.length
      ? "Session window: " + rows[0].time + " → " + rows[rows.length - 1].time + " (" + rows.length + " events, capped at 500). Zero-risk flat lines mean the model reported no failure signature at all — that is a real prediction, not missing data."
      : "Startup state: no recorded events yet. Run a scenario, the telemetry simulation or a preset — each event is recorded honestly.");
  }
  /* ============================= HISTORY VIEW ============================= */
  function historyRowsFiltered() {
    const searchNode = $("histSearch");
    const filterNode = $("histFilter");
    const sortNode = $("histSort");
    const q = (searchNode && searchNode.value ? searchNode.value : "").trim().toLowerCase();
    const f = (filterNode && filterNode.value) || "all";
    const rows = state.history.filter((r) => {
      if (f !== "all" && r.status !== f) return false;
      if (!q) return true;
      const hay = (r.machine + " " + r.time + " " + r.source + " " + r.status + " " + r.productType).toLowerCase();
      return hay.indexOf(q) >= 0;
    });
    // Sort control (§13): default order stays chronological ascending; the view reverses for "newest first".
    const sort = (sortNode && sortNode.value) || "newest";
    if (sort === "riskDesc") rows.sort((a, b) => Number(b.risk) - Number(a.risk));
    else if (sort === "riskAsc") rows.sort((a, b) => Number(a.risk) - Number(b.risk));
    else if (sort === "machine") rows.sort((a, b) => String(a.machine).localeCompare(String(b.machine)) || Number(a.ts) - Number(b.ts));
    else if (sort === "oldest") rows.sort((a, b) => Number(a.ts) - Number(b.ts));
    return { rows: rows, sort: sort };
  }
  function historyStatusClass(word) {
    if (word === "CRITICAL") return "table-status-critical";
    if (word === "WARNING") return "table-status-warning";
    return "table-status-normal";
  }
  function updateHistoryView() {
    const body = $("historyTableBody");
    if (!body) return;
    setText("storagePill", Storage.available ? "LOCAL STORAGE" : "MEMORY ONLY");
    if (state.view !== "history") {
      setText("histCount", state.history.length + " events recorded (capped at 500). Open this view to browse them.");
      return;
    }
    const filtered = historyRowsFiltered();
    const sort = filtered.sort;
    let rows;
    if (sort === "newest") rows = filtered.rows.slice(-200).reverse();          // newest first (default)
    else if (sort === "oldest") rows = filtered.rows.slice(0, 200);             // oldest first
    else rows = filtered.rows.slice(0, 200);                                    // risk / machine order as sorted
    body.innerHTML = rows.length ? rows.map((r, i) => `
      <tr>
        <td style="font-family:var(--font-mono);">${sort === "newest" ? rows.length - i : i + 1}</td>
        <td style="font-family:var(--font-mono);">${esc(r.time)}</td>
        <td>${esc(r.machine)}</td>
        <td>${esc(r.productType)}</td>
        <td>${fmt(r.airTemperature, 1)}</td>
        <td>${fmt(r.processTemperature, 1)}</td>
        <td>${Math.round(Number(r.rotationalSpeed))}</td>
        <td>${fmt(r.torque, 1)}</td>
        <td>${Math.round(Number(r.toolWear))}</td>
        <td style="font-family:var(--font-mono);font-weight:700;">${fmt(r.risk, 1)}%</td>
        <td><span class="table-status-pill ${historyStatusClass(r.status)}">${esc(r.status)}</span></td>
        <td><span class="chip-tag" style="background:rgba(255,255,255,0.06);color:#cbd5e1;">${esc(r.source)}</span></td>
        <td style="font-size:0.75rem;color:var(--text-dim);">${esc(r.model)}</td>
      </tr>`).join("")
      : '<tr><td colspan="13" style="color:var(--text-dim);">No matching events. Run a prediction, scenario or the telemetry simulation to record one.</td></tr>';
    const sortLabel = { newest: "newest first", oldest: "oldest first", riskDesc: "highest risk first", riskAsc: "lowest risk first", machine: "machine A→Z" }[sort] || "newest first";
    setText("histCount", rows.length + " of " + state.history.length + " recorded events shown (" + sortLabel + " · history capped at 500).");
    updateExportNote();
  }
  function bindHistoryControls() {
    const search = $("histSearch");
    if (search) search.addEventListener("input", updateHistoryView);
    const filter = $("histFilter");
    if (filter) filter.addEventListener("change", updateHistoryView);
    const sortSel = $("histSort");
    if (sortSel) sortSel.addEventListener("change", updateHistoryView);
    const clear = $("btnHistClear");
    if (clear) {
      clear.addEventListener("click", () => {
        if (!state.history.length) { toast("History is already empty.", "info"); return; }
        clearHistory();
        toast("Prediction history cleared.", "ok");
      });
    }
  }
  /* ============================= EXPORT CENTER ============================ */
  function csvEscape(value) {
    const s = String(value == null ? "" : value);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function toCsv(headers, rows) {
    return [headers.join(",")].concat(rows.map((r) => r.map(csvEscape).join(","))).join("\r\n");
  }
  function downloadFile(filename, text, mime) {
    try {
      const blob = new Blob(["\ufeff" + text], { type: (mime || "text/csv") + ";charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 1500);
      return true;
    } catch (err) {
      console.error(err);
      toast("Export failed: " + err.message, "error");
      return false;
    }
  }
  function exportStamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "-" + pad(d.getHours()) + pad(d.getMinutes());
  }
  function exportPredictionCsv() {
    if (!state.prediction) { toast("Run a prediction first — nothing to export yet.", "warn"); return; }
    const s = state.sensors;
    const headers = ["Machine", "Timestamp", "ProductType", "AirTemperature_K", "ProcessTemperature_K",
      "RotationalSpeed_rpm", "Torque_Nm", "ToolWear_min", "RandomForestRisk_pct", "HealthIndex", "Status",
      "WarningThreshold_pct", "CriticalThreshold_pct", "AdvisoryIndicators"];
    const advisory = (indicators() || []).map((x) => x.code + "=" + fmt(x.pct, 1) + "%/" + indicatorLevel(x.pct).label).join(" | ");
    const rows = [[state.machine.id, nowFull(), s.productType, fmt(s.airTemperature, 1), fmt(s.processTemperature, 1),
      Math.round(s.rotationalSpeed), fmt(s.torque, 1), Math.round(s.toolWear), fmt(state.risk, 1), Math.round(state.health),
      statusWord(state.status.key), state.settings.warnThreshold, state.settings.critThreshold, advisory]];
    if (downloadFile("prediction-" + exportStamp() + ".csv", toCsv(headers, rows))) {
      toast("Prediction CSV exported (current reading).", "ok");
    }
  }
  function exportTelemetryCsv() {
    const t = state.telemetry;
    if (!t.risk.length) { toast("No telemetry samples yet — run the simulation or a prediction first.", "warn"); return; }
    const headers = ["Sample", "AirTemperature_K", "ProcessTemperature_K", "RotationalSpeed_rpm", "Torque_Nm", "ToolWear_min", "Risk_pct"];
    const n = t.risk.length;
    const offset = Math.max(0, t.air.length - n);
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push([i + 1, fmt(t.air[offset + i], 1), fmt(t.proc[offset + i], 1), Math.round(t.rpm[offset + i]),
        fmt(t.torque[offset + i], 1), Math.round(t.wear[offset + i]), fmt(t.risk[i], 1)]);
    }
    if (downloadFile("telemetry-buffer-" + exportStamp() + ".csv", toCsv(headers, rows))) {
      toast("Telemetry CSV exported (" + n + " samples).", "ok");
    }
  }
  function exportHistoryCsv() {
    if (!state.history.length) { toast("Prediction history is empty — nothing to export.", "warn"); return; }
    const headers = ["Timestamp", "Time", "Machine", "ProductType", "AirTemperature_K", "ProcessTemperature_K",
      "RotationalSpeed_rpm", "Torque_Nm", "ToolWear_min", "Risk_pct", "Health", "Status", "Source", "Model"];
    const rows = state.history.map((r) => [r.ts, r.time, r.machine, r.productType, r.airTemperature, r.processTemperature,
      r.rotationalSpeed, r.torque, r.toolWear, r.risk, r.health, r.status, r.source, r.model]);
    if (downloadFile("prediction-history-" + exportStamp() + ".csv", toCsv(headers, rows))) {
      toast("History CSV exported (" + rows.length + " events).", "ok");
    }
  }
  function exportMetricsCsv() {
    const data = metricsData();
    if (!data) { toast("Model metrics unavailable (model-data.js not loaded).", "error"); return; }
    const headers = ["Model", "F1", "Precision", "Recall", "ROC_AUC", "LiveInference", "Artifact", "Status"];
    const rows = (data.models || []).map((m) => [m.name,
      Number.isFinite(m.metrics.f1) ? m.metrics.f1 : "N/A",
      Number.isFinite(m.metrics.precision) ? m.metrics.precision : "N/A",
      Number.isFinite(m.metrics.recall) ? m.metrics.recall : "N/A",
      Number.isFinite(m.metrics.rocAuc) ? m.metrics.rocAuc : "N/A",
      m.live ? "yes" : "no", m.artifact, m.status]);
    const metaHeaders = ["Field", "Value"];
    const metaRows = [
      ["Dataset", data.dataset.name], ["Records", data.dataset.records], ["Failures", data.dataset.failures],
      ["FailureRate_pct", data.dataset.failureRate], ["MissingValues", data.dataset.missingValues],
      ["Split", data.pipeline.split], ["CrossValidation", data.pipeline.cv], ["Scoring", data.pipeline.scoring],
      ["ImbalanceHandling", data.pipeline.imbalance], ["Scaling", data.pipeline.scaling], ["Encoding", data.pipeline.encoding]
    ];
    const csv = "# MODEL METRICS (documented values; N/A = not saved in project outputs)\r\n" +
      toCsv(headers, rows) + "\r\n\r\n# STUDY PIPELINE\r\n" + toCsv(metaHeaders, metaRows) + "\r\n";
    if (downloadFile("model-metrics-" + exportStamp() + ".csv", csv)) {
      toast("Model metrics exported (documented values only).", "ok");
    }
  }
  function updateExportNote() {
    const n = state.history.length;
    if (!n) { setText("exportNote", "Startup state: nothing to export yet — run a prediction first."); return; }
    setText("exportNote", n + " recorded events available. Exports contain only data actually produced in this session — no fabricated values, and N/A stays N/A in the metrics export.");
  }
  function bindExportControls() {
    const map = [
      ["btnExpPredictions", exportPredictionCsv],
      ["btnExpTelemetry", exportTelemetryCsv],
      ["btnExpHistory", exportHistoryCsv],
      ["btnExpMetrics", exportMetricsCsv],
      ["btnGenReport", generateReport],
      ["btnPrintReport", printReport]
    ];
    map.forEach((pair) => {
      const node = $(pair[0]);
      if (node) node.addEventListener("click", pair[1]);
    });
  }
  /* ============================ DATASET VIEW ============================== */
  function dsSchemaRows() {
    return [
      ["AirTemp", "Input (numeric)", "Ambient air temperature in Kelvin"],
      ["ProcTemp", "Input (numeric)", "Process temperature in Kelvin"],
      ["RPM", "Input (numeric)", "Spindle rotational speed (rpm)"],
      ["Torque", "Input (numeric)", "Spindle torque (Nm)"],
      ["ToolWear", "Input (numeric)", "Cumulative tool wear (min)"],
      ["Type_H / Type_L / Type_M", "Input (one-hot)", "Product quality variant — exactly one column is 1"],
      ["Failure", "Target (binary)", "0 = normal operation · 1 = recorded failure"],
      ["FailureType", "Excluded (leakage)", "TWF / HDF / PWF / OSF / RNF / None — never fed to the model"]
    ];
  }
  function updateDatasetView() {
    if (!$("dsRecords")) return;
    const data = metricsData();
    if (!data) return;
    const ds = data.dataset || {};
    setText("dsRecords", fmtInt(ds.records));
    setText("dsFailures", fmtInt(ds.failures));
    setText("dsRate", fmt(ds.failureRate, 2) + "%");
    setText("dsMissing", fmtInt(ds.missingValues));
    setText("dsFeatures", ds.features ? fmtInt(ds.features.length) : "--");
    setText("dsTarget", ds.target || "--");
    setText("dsBalancePill", fmtInt(ds.records) + " records");
    const failPct = ds.records ? (ds.failures / ds.records) * 100 : 0;
    const barNormal = $("dsBarNormal");
    const barFail = $("dsBarFail");
    if (barNormal) barNormal.style.width = (100 - failPct).toFixed(2) + "%";
    if (barFail) barFail.style.width = Math.max(0.6, failPct).toFixed(2) + "%";
    setText("dsImbalanceNote", "Only " + fmt(ds.failureRate, 2) + "% of records are failures (" + fmtInt(ds.failures) +
      " of " + fmtInt(ds.records) + "). This imbalance is why F1 was the tuning metric and class_weight=\"balanced\" was applied to all three models. " +
      (ds.leakageNote || ""));
    const body = $("dsSchemaBody");
    if (body && !body.children.length) {
      body.innerHTML = dsSchemaRows().map((r) =>
        `<tr><td style="font-family:var(--font-mono);color:#38bdf8;">${esc(r[0])}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td></tr>`
      ).join("");
    }
  }
  /* ========================== SCENARIO LAB ================================ */
  function scenarioList() { return (EX && EX.SCENARIO_LAB) ? EX.SCENARIO_LAB : []; }
  function scenarioById(id) { return scenarioList().filter((x) => x.id === id)[0] || null; }
  function buildScenarioCards() {
    const host = $("scenarioCards");
    if (!host) return;
    const list = scenarioList();
    host.innerHTML = list.map((sc) => `
      <button type="button" class="scenario-btn tone-${esc(sc.tone)}" data-scenario="${esc(sc.id)}">
        <span class="scenario-head">
          <span class="scenario-name">${esc(sc.name)}</span>
          <span class="scenario-ref">${sc.referenceRisk == null ? "custom" : "ref ≈ " + fmt(sc.referenceRisk, 1) + "%"}</span>
        </span>
        <span class="scenario-desc">${esc(sc.description)}</span>
      </button>`).join("");
    setText("scenarioCount", list.length + " scenarios");
  }
  function markScenarioSelected(id) {
    document.querySelectorAll(".scenario-btn").forEach((b) => {
      b.classList.toggle("selected", b.getAttribute("data-scenario") === id);
    });
  }
  function runScenarioById(id) {
    const sc = scenarioById(id);
    if (!sc) return;
    if (!sc.values) { applyCustomScenario(); return; }
    state.scenario.selected = id;
    Object.assign(state.sensors, sc.values);
    state.machine.id = "CNC-MILL-08";
    markScenarioSelected(id);
    syncUIFromState();
    runPipeline("scenario");
    toast("Scenario \"" + sc.name + "\" applied — " + fmt(state.risk, 1) + "% (" + statusWord(state.status.key) + ").", "ok");
  }
  function updateScenarioIndicator() {
    const host = $("scenarioCurrent");
    if (!host) return;
    const sc = scenarioById(state.scenario.selected);
    const badge = $("scenarioStatePill");
    if (!sc) {
      host.innerHTML = "No scenario has been run yet. Pick a card above, or build a custom scenario, and the result will appear here — scored live by the Random Forest.";
      if (badge) badge.textContent = "IDLE";
      return;
    }
    if (badge) badge.textContent = sc.id === "custom" ? "CUSTOM" : "RUNNING";
    const s = state.sensors;
    host.innerHTML = `
      <div class="sc-head">
        <span class="sc-name">${esc(sc.name)}</span>
        <span class="sc-risk ${state.status.cls}">${fmt(state.risk, 1)}% · ${esc(statusWord(state.status.key))}</span>
      </div>
      <div class="sc-desc">${esc(sc.description)}</div>
      <div class="sc-values">${esc(typeLabel(s.productType))} · ${fmt(s.airTemperature, 1)} K · ${fmt(s.processTemperature, 1)} K · ${Math.round(s.rotationalSpeed)} rpm · ${fmt(s.torque, 1)} Nm · ${Math.round(s.toolWear)} min</div>
      <div class="muted-note">Reference value: ${sc.referenceRisk == null ? "—" : "≈ " + fmt(sc.referenceRisk, 1) + "%"} · Live value: ${fmt(state.risk, 1)}%. The live value is the real forest output and always wins.</div>`;
  }
  function readCustomGrid() {
    // NOTE: signature must stay (id, min, max, def) — an extra parameter here
    // shifted min/max and silently collapsed every custom value to its default.
    const read = (id, min, max, def) => {
      const node = $(id);
      const raw = node ? String(node.value).trim() : "";
      const num = Number(raw);
      if (raw === "" || !Number.isFinite(num)) return { value: def, issue: true };
      return { value: clamp(num, min, max), issue: num < min || num > max };
    };
    const typeNode = $("cType");
    const out = {
      productType: typeNode ? typeNode.value : "M",
      airTemperature: read("cAir", 295, 305, 298.5).value,
      processTemperature: read("cProc", 305, 315, 308.8).value,
      rotationalSpeed: read("cRpm", 1100, 2900, 1540).value,
      torque: read("cTorque", 3.5, 80, 39.5).value,
      toolWear: read("cWear", 0, 260, 85).value
    };
    return out;
  }
  function applyCustomScenario() {
    const values = readCustomGrid();
    state.scenario.selected = "custom";
    Object.assign(state.sensors, values);
    state.machine.id = "CNC-MILL-08";
    markScenarioSelected("custom");
    syncUIFromState();
    runPipeline("scenario");
    toast("Custom scenario applied — " + fmt(state.risk, 1) + "% (" + statusWord(state.status.key) + ").", "ok");
  }
  function fillCustomGridFromCurrent() {
    const s = state.sensors;
    const set = (id, v) => { const n = $(id); if (n) n.value = v; };
    set("cAir", s.airTemperature.toFixed(1));
    set("cProc", s.processTemperature.toFixed(1));
    set("cRpm", Math.round(s.rotationalSpeed));
    set("cTorque", s.torque.toFixed(1));
    set("cWear", Math.round(s.toolWear));
    const t = $("cType");
    if (t) t.value = s.productType;
    toast("Custom grid now mirrors the current reading.", "info");
  }
  /* ====================== WHAT-IF SIMULATOR (§10) ========================= */
  /* Scores BOTH operating points with the REAL Random Forest: the live state
     and the modified grid. No interpolation — two independent model runs. */
  function whatIfCompare() {
    const host = $("whatIfResult");
    if (!host) return;
    if (!modelReady()) { modelUnavailable(); return; }
    const mod = readCustomGrid();
    let a, b;
    try {
      a = MD.predictFailure(sanitizeSensors(state.sensors));
      b = MD.predictFailure(sanitizeSensors(Object.assign({}, state.sensors, mod)));
    } catch (err) {
      host.innerHTML = '<div class="whatif-empty">Inference failed: ' + esc(err.message) + '</div>';
      return;
    }
    const riskA = clamp(Number(a.probabilityPercentage) || 0, 0, 100);
    const riskB = clamp(Number(b.probabilityPercentage) || 0, 0, 100);
    const stA = statusFromRisk(riskA), stB = statusFromRisk(riskB);
    const cur = state.sensors;
    const fmtDelta = (x, y, d) => {
      const delta = Number(y) - Number(x);
      if (Math.abs(delta) < 1e-9) return '<span class="whatif-same">unchanged</span>';
      return '<span class="' + (delta > 0 ? "whatif-up" : "whatif-down") + '">' +
        (delta > 0 ? "+" : "") + fmt(delta, d) + "</span>";
    };
    const row = (label, x, y, d, unit) => `
      <tr><td>${label}</td>
        <td>${typeof x === "number" ? fmt(x, d) : esc(String(x))}${unit || ""}</td>
        <td>${typeof y === "number" ? fmt(y, d) : esc(String(y))}${unit || ""}</td>
        <td>${fmtDelta(x, y, d)}</td></tr>`;
    const delta = riskB - riskA;
    host.innerHTML = `
      <div class="whatif-head"><span>CURRENT SCENARIO</span> vs <span>MODIFIED SCENARIO</span> — both scored by the Random Forest</div>
      <table class="whatif-table">
        <thead><tr><th>Feature</th><th>Current</th><th>Modified</th><th>Δ</th></tr></thead>
        <tbody>
          <tr><td>Product type</td><td>${esc(typeLabel(cur.productType))}</td><td>${esc(typeLabel(mod.productType))}</td><td>${cur.productType === mod.productType ? '<span class="whatif-same">unchanged</span>' : '<span class="whatif-up">changed</span>'}</td></tr>
          ${row("Air temperature", cur.airTemperature, mod.airTemperature, 1, " K")}
          ${row("Process temperature", cur.processTemperature, mod.processTemperature, 1, " K")}
          ${row("Rotational speed", cur.rotationalSpeed, mod.rotationalSpeed, 0, " rpm")}
          ${row("Torque", cur.torque, mod.torque, 1, " Nm")}
          ${row("Tool wear", cur.toolWear, mod.toolWear, 0, " min")}
          <tr class="whatif-risk-row"><td>Failure probability</td><td>${fmt(riskA, 1)}%</td><td>${fmt(riskB, 1)}%</td><td>${fmtDelta(riskA, riskB, 1)}</td></tr>
          <tr class="whatif-risk-row"><td>Risk level</td><td>${esc(stA.tag)}</td><td>${esc(stB.tag)}</td><td>${stA.key === stB.key ? '<span class="whatif-same">unchanged</span>' : '<span class="whatif-up">escalated</span>'}</td></tr>
        </tbody>
      </table>
      <div class="muted-note">Both probabilities are live Random Forest outputs computed for the two operating points in your browser — nothing is interpolated or fabricated. Δ risk = ${delta >= 0 ? "+" : ""}${fmt(delta, 1)} percentage points. Use <strong>Apply Custom Scenario</strong> to move the app to the modified state.</div>`;
    toast("What-if: " + fmt(riskA, 1) + "% → " + fmt(riskB, 1) + "% (real forest, both runs).", "info");
  }
  /* ==================================================================
     WHAT-IF ENGINE (V8) :: SCENARIO LAB
     ------------------------------------------------------------------
     Every probability shown here comes from ONE call to the EXISTING
     inference entry point MD.predictFailure(features) - the same pure
     function the central pipeline uses. There is no second model, no
     interpolation and no scripted result anywhere in this block.
     ================================================================== */
  const WI = {
    baseline: null,
    scenario: null,
    history: [],
    slotA: null,
    slotB: null,
    running: false,
    autoTimer: 0,
    previewTimer: 0,
    scanTimer: 0
  };
  const WI_SPECS = [
    { key: "Air", field: "airTemperature", label: "AIR TEMPERATURE", unit: "K", dec: 1 },
    { key: "Proc", field: "processTemperature", label: "PROCESS TEMPERATURE", unit: "K", dec: 1 },
    { key: "Rpm", field: "rotationalSpeed", label: "ROTATIONAL SPEED", unit: "rpm", dec: 0 },
    { key: "Torque", field: "torque", label: "TORQUE", unit: "Nm", dec: 1 },
    { key: "Wear", field: "toolWear", label: "TOOL WEAR", unit: "min", dec: 0 }
  ];
  function wiSpec(key) { return WI_SPECS.filter(function (x) { return x.key === key; })[0]; }
  function wiRange(key) {
    var id = key === "Air" ? "airTemperature" : key === "Proc" ? "processTemperature"
      : key === "Rpm" ? "rotationalSpeed" : key === "Torque" ? "torque" : "toolWear";
    return SENSOR_LIMITS[id];
  }
  /* ---------------- validation ----------------
     Never lets NaN / Infinity / "" or an out-of-range number reach
     inference: MD.predictFailure is only called with values that pass. */
  function wiReadFeatures() {
    const out = { productType: (state.sensors && state.sensors.productType) || "M" };
    const errors = [];
    WI_SPECS.forEach(function (spec) {
      const lim = wiRange(spec.key);
      const range = $("wi" + spec.key + "Range");
      const num = $("wi" + spec.key + "Num");
      const errNode = $("wi" + spec.key + "Err");
      const raw = num ? String(num.value).trim() : "";
      const parsed = Number(raw);
      let value = lim.def;
      let msg = "";
      if (raw === "") {
        msg = "Enter a number (" + lim.min + "-" + lim.max + " " + spec.unit + ").";
      } else if (!Number.isFinite(parsed)) {
        msg = "Value must be a finite number.";
      } else if (parsed < lim.min || parsed > lim.max) {
        value = clamp(parsed, lim.min, lim.max);
        msg = "Outside visualization range - clamped to " + fmt(value, spec.dec) + " " + spec.unit + ".";
      } else {
        value = parsed;
      }
      if (errNode) { errNode.textContent = msg; errNode.classList.toggle("on", !!msg); }
      if (raw === "" || !Number.isFinite(parsed)) {
        errors.push(spec.label + ": " + (msg || "invalid value"));
      }
      out[spec.field] = value;
      if (range) {
        const rv = Number(range.value);
        if (Number.isFinite(rv) && Math.abs(rv - value) > 1e-9) range.value = String(value);
      }
      if (num && document.activeElement !== num) {
        const want = String(spec.dec ? Number(value.toFixed(spec.dec)) : Math.round(value));
        if (num.value !== want) num.value = want;
      }
    });
    return { features: out, errors: errors };
  }
  function wiCaptureBaseline() {
    if (!modelReady()) return null;
    const features = sanitizeSensors(state.sensors);
    let prediction = null;
    try {
      prediction = MD.predictFailure(features);
    } catch (err) {
      console.error("Baseline inference failed:", err);
      return null;
    }
    WI.baseline = {
      features: features,
      prediction: prediction,
      risk: clamp(Number(prediction.probabilityPercentage) || 0, 0, 100),
      ts: Date.now()
    };
    return WI.baseline;
  }
  function wiPushTwinPreview(features) {
    TWIN.scenarioPreview = features;
    if (state.view === "whatif") twFrame(performance.now());
  }
  /* Uses the app's EXISTING statusFromRisk() - no new classification. */
  function wiLevelOf(risk) {
    const st = statusFromRisk(clamp(Number(risk) || 0, 0, 100));
    const icons = { normal: "\u25CF", warning: "\u25B2", critical: "\u25C6" };
    const names = { normal: "LOW", warning: "MEDIUM", critical: "CRITICAL" };
    return { key: st.key, tag: st.tag, label: names[st.key], icon: icons[st.key], cls: st.cls };
  }
  function wiVerdict(pct) {
    return (Number(pct) >= 50) ? "FAILURE" : "NO FAILURE";
  }

  function wiRenderModels(res) {
    const host = $("wiModels");
    if (!host) return;
    const defs = [
      { key: "dt", name: "DECISION TREE", sub: "CART classifier" },
      { key: "rf", name: "RANDOM FOREST", sub: "300 trees \u00b7 headline" },
      { key: "svm", name: "SVM (RBF)", sub: "support vectors" }
    ];
    const pm = res.prediction.perModelPct || {};
    let html = "";
    defs.forEach(function (d) {
      const has = pm[d.key] != null && Number.isFinite(Number(pm[d.key]));
      const pct = has ? Number(pm[d.key]) : null;
      const lv = has ? wiLevelOf(pct) : { key: "normal", label: "N/A", cls: "", icon: "\u2014" };
      html += '<div class="wi-model">';
      html += '<div class="wi-model-head"><span class="wi-model-name">' + esc(d.name) +
        '</span><span class="wi-model-sub">' + esc(d.sub) + '</span></div>';
      if (has) {
        html += '<div class="wi-model-pct">' + fmt(pct, 1) + '<small>%</small></div>';
        html += '<div class="wi-model-bar"><span class="wi-model-fill ' + lv.cls +
          '" style="width:' + clamp(pct, 0, 100) + '%"></span></div>';
        html += '<div class="wi-model-foot"><span class="wi-model-verdict ' + lv.cls + '">' +
          esc(lv.icon) + " " + esc(wiVerdict(pct)) + '</span><span class="wi-model-lv ' + lv.cls +
          '">' + esc(lv.label) + "</span></div>";
      } else {
        html += '<div class="wi-model-pct wi-na">\u2014</div>';
        html += '<div class="wi-model-bar"><span class="wi-model-fill"></span></div>';
        html += '<div class="wi-model-foot"><span class="wi-model-verdict">Model artifact unavailable</span></div>';
      }
      html += "</div>";
    });
    host.innerHTML = html;
  }
  function wiRenderRisk(res) {
    const base = WI.baseline;
    setText("wiBaseRisk", base ? fmt(base.risk, 1) : "--");
    setText("wiScenRisk", fmt(res.risk, 1));
    const delta = res.risk - (base ? base.risk : 0);
    const dEl = $("wiDelta");
    if (dEl) {
      if (Math.abs(delta) <= 0.05) { dEl.textContent = fmt(delta, 1); dEl.className = "wi-risk-v flat"; }
      else { dEl.textContent = (delta > 0 ? "+" : "") + fmt(delta, 1); dEl.className = "wi-risk-v " + (delta > 0 ? "up" : "down"); }
    }
    const lv = wiLevelOf(res.risk);
    const wrap = $("wiLevelWrap");
    if (wrap) wrap.className = "wi-level " + lv.cls;
    setText("wiLevelIcon", lv.icon);
    setText("wiLevelText", "RISK LEVEL: " + lv.label);
    setText("wiBasePct", base ? fmt(base.risk, 1) + "%" : "--");
    setText("wiScenPct", fmt(res.risk, 1) + "%");
    const bb = $("wiBaseBar"), sb = $("wiScenBar");
    if (bb) bb.style.width = (base ? clamp(base.risk, 0, 100) : 0) + "%";
    if (sb) { sb.style.width = clamp(res.risk, 0, 100) + "%"; sb.className = "wi-bar-fill wi-bar-scen " + lv.cls; }
    const rel = (base && base.risk > 0) ? ((res.risk - base.risk) / base.risk) * 100 : null;
    setText("wiMeta", "Baseline " + (base ? fmt(base.risk, 1) + "%" : "n/a") + " \u2192 scenario " +
      fmt(res.risk, 1) + "% \u00b7 change " + (delta >= 0 ? "+" : "") + fmt(delta, 1) +
      " percentage points" + (rel == null ? "." : " (relative " + (rel >= 0 ? "+" : "") + fmt(rel, 1) + "%)."));
    setText("wiStatePill", "RESULT \u00b7 " + lv.label);
  }
  function wiRenderAdvisory(res) {
    const host = $("wiAdvisory");
    if (!host) return;
    const recs = (res.prediction && res.prediction.recommendations) || [];
    if (!recs.length) {
      host.innerHTML = '<div class="muted-note">MAINTENANCE ADVISORY \u2014 No supported advisory available.</div>';
      return;
    }
    let html = '<div class="wi-advisory-head">AI MAINTENANCE ADVISORY ' +
      '<span class="wi-advisory-tag">ADVISORY \u00b7 NOT AUTOMATED CONTROL</span></div><ul class="wi-advisory-list">';
    recs.forEach(function (r) {
      html += '<li><span class="wi-advisory-pri">' + esc(r.priority) + '</span><span class="wi-advisory-action">' +
        esc(r.action) + '</span><span class="wi-advisory-reason">' + esc(r.reason) + "</span></li>";
    });
    html += '</ul><div class="muted-note">Advisory text comes from the existing documented AI4I-style rule engine, not from a model.</div>';
    host.innerHTML = html;
  }
  function wiRenderTrend() {
    const canvas = $("wiTrendChart");
    if (!canvas) return;
    const runs = WI.history.slice().reverse();
    setText("wiTrendPill", runs.length ? runs.length + " RUNS" : "NO DATA");
    drawLineChart(canvas, {
      // Single-series API: one value per real run, oldest -> newest.
      values: runs.map(function (r) { return Number(r.risk); }),
      color: "#00E5FF",
      yMax: 100,
      thresholds: WI.history.length ? [
        { value: state.settings.warnThreshold, color: "var(--status-warning)" },
        { value: state.settings.critThreshold, color: "var(--status-critical)" }
      ] : [],
      emptyText: "Run a what-if analysis to plot real predictions here."
    });
  }

  function wiRenderHistory() {
    const host = $("wiHistory");
    if (!host) return;
    setText("wiHistoryCount", WI.history.length + " RUN" + (WI.history.length === 1 ? "" : "S"));
    if (!WI.history.length) {
      host.innerHTML = '<div class="wi-empty">NO SCENARIOS ANALYZED</div>';
      return;
    }
    let html = "";
    WI.history.forEach(function (r, i) {
      const lv = wiLevelOf(r.risk);
      const delta = r.risk - (WI.baseline ? WI.baseline.risk : 0);
      const cls = Math.abs(delta) <= 0.05 ? "flat" : delta > 0 ? "up" : "down";
      html += '<div class="wi-hrow">';
      html += '<span class="wi-hidx">' + (WI.history.length - i) + "</span>";
      html += '<span class="wi-hname">' + esc(r.label) + "</span>";
      html += '<span class="wi-hrisk">' + (WI.baseline ? fmt(WI.baseline.risk, 1) + "%" : "--") +
        ' <span class="wi-harow">\u2192</span> <b>' + fmt(r.risk, 1) + "%</b></span>";
      html += '<span class="wi-hdelta ' + cls + '">' + (delta > 0 ? "+" : "") + fmt(delta, 1) + " pp</span>";
      html += '<span class="wi-hlv ' + lv.cls + '">' + esc(lv.icon) + " " + esc(lv.label) + "</span>";
      html += '<span class="wi-hmodel">DT ' + fmt(r.models.dt, 1) + "% \u00b7 RF " + fmt(r.models.rf, 1) +
        "% \u00b7 SVM " + fmt(r.models.svm, 1) + "%</span>";
      html += "</div>";
    });
    host.innerHTML = html;
  }
  function wiRenderCompare() {
    const host = $("wiCompare");
    if (!host) return;
    const cols = [{ tag: "BASELINE", run: WI.baseline }, { tag: "SCENARIO A", run: WI.slotA }, { tag: "SCENARIO B", run: WI.slotB }];
    if (!WI.slotA && !WI.slotB) {
      host.innerHTML = '<div class="wi-empty">Run at least one analysis to enable Baseline / Scenario A / Scenario B comparison.</div>';
      return;
    }
    let html = '<table class="wi-table"><thead><tr><th>Feature</th>';
    cols.forEach(function (c) { html += "<th>" + esc(c.tag) + "</th>"; });
    html += "</tr></thead><tbody>";
    WI_SPECS.forEach(function (spec) {
      html += "<tr><td>" + esc(spec.label) + "</td>";
      cols.forEach(function (c) {
        html += "<td>" + (c.run ? fmt(c.run.features[spec.field], spec.dec) : '<span class="wi-na">--</span>') + "</td>";
      });
      html += "</tr>";
    });
    const riskRow = function (label, fn) {
      html += '<tr class="wi-table-risk"><td>' + label + "</td>";
      cols.forEach(function (c) { html += "<td>" + (c.run ? fn(c.run) : '<span class="wi-na">--</span>') + "</td>"; });
      html += "</tr>";
    };
    riskRow("Failure probability", function (r) { return fmt(r.risk, 1) + "%"; });
    riskRow("Risk level", function (r) {
      const lv = wiLevelOf(r.risk);
      return '<span class="' + lv.cls + '">' + esc(lv.icon) + " " + esc(lv.label) + "</span>";
    });
    const modelOf = function (r, k) {
      if (r.models && r.models[k] != null) return Number(r.models[k]);
      const p = r.prediction && r.prediction.perModelPct ? r.prediction.perModelPct : null;
      return p && p[k] != null ? Number(p[k]) : null;
    };
    const modelCell = function (r, k) {
      const v = modelOf(r, k);
      return v == null || !Number.isFinite(v) ? '<span class="wi-na">N/A</span>' : fmt(v, 1) + "%";
    };
    riskRow("Decision Tree", function (r) { return modelCell(r, "dt"); });
    riskRow("SVM (RBF)", function (r) { return modelCell(r, "svm"); });
    html += "</tbody></table>";
    host.innerHTML = html;
  }
  /* Genuine ablation-based importance from the existing computeContributions. */
  function wiRenderImpact(features) {
    const host = $("wiImpact");
    if (!host) return;
    let contributions = [];
    if (EX && typeof EX.computeContributions === "function") {
      try { contributions = EX.computeContributions(features) || []; }
      catch (err) { console.error("Contribution computation failed:", err); contributions = []; }
    }
    if (!contributions.length) {
      setText("wiImpactPill", "NOT AVAILABLE");
      host.innerHTML = '<div class="wi-empty">FEATURE IMPACT \u2014 MODEL-SUPPORTED FEATURE IMPORTANCE NOT AVAILABLE</div>';
      setText("wiImpactNote", "This build exposes no feature-importance data, so none is shown. Nothing is estimated or invented.");
      return;
    }
    setText("wiImpactPill", "ABLATION vs NOMINAL");
    const max = Math.max.apply(null, contributions.map(function (c) { return Math.abs(Number(c.deltaPp) || 0); }).concat([1]));
    let html = "";
    contributions.forEach(function (c) {
      const d = Number(c.deltaPp) || 0;
      const wpc = (Math.abs(d) / max) * 100;
      html += '<div class="wi-improw"><span class="wi-imp-label">' + esc(c.label) + '</span>';
      html += '<span class="wi-imp-bar"><span class="wi-imp-fill ' + (d >= 0 ? "up" : "down") +
        '" style="width:' + wpc.toFixed(1) + '%"></span></span>';
      html += '<span class="wi-imp-val ' + (d >= 0 ? "up" : "down") + '">' + (d >= 0 ? "+" : "") + fmt(d, 1) + " pp</span>";
      html += "</div>";
    });
    host.innerHTML = html;
    setText("wiImpactNote", "Method: each feature is held at the documented nominal baseline while the rest of the SCENARIO reading stays as-is; the bar shows how many failure-probability percentage points that feature adds (+) or removes (\u2212). Produced by the real 300-tree forest in your browser. This is ablation, not SHAP, and not a probability decomposition.");
  }
  function wiScan(on) {
    const overlay = $("wiScan");
    if (!overlay) return;
    const items = [].slice.call(overlay.querySelectorAll(".scan-steps li"));
    if (WI.scanTimer) { window.clearInterval(WI.scanTimer); WI.scanTimer = 0; }
    if (!on) { overlay.hidden = true; items.forEach(function (li) { li.classList.remove("on", "done"); }); return; }
    overlay.hidden = false;
    items.forEach(function (li) { li.classList.remove("on", "done"); });
    let i = 0;
    WI.scanTimer = window.setInterval(function () {
      if (i > 0) { items[i - 1].classList.remove("on"); items[i - 1].classList.add("done"); }
      if (i >= items.length) {
        window.clearInterval(WI.scanTimer); WI.scanTimer = 0;
        window.setTimeout(function () { wiScan(false); }, 200);
        return;
      }
      items[i].classList.add("on");
      i++;
    }, 175);
  }

  /* ---------------- THE REAL RUN ----------------
     Reads validated scenario features, calls the EXISTING inference
     function once, then renders whatever that call returned. */
  function wiRun(opts) {
    const options = opts || {};
    if (WI.running) return;
    if (!modelReady()) { modelUnavailable(); return; }
    const read = wiReadFeatures();
    const vbox = $("wiValidation");
    if (read.errors.length) {
      if (vbox) { vbox.textContent = "Fix the highlighted fields before running the models: " + read.errors.join(" \u00b7 "); vbox.className = "wi-validation on"; }
      toast("Scenario not run \u2014 " + read.errors.length + " invalid value(s).", "error");
      ccEvent("WHAT-IF BLOCKED \u00b7 INVALID FEATURES");
      return;
    }
    if (vbox) { vbox.textContent = ""; vbox.className = "wi-validation"; }
    if (!WI.baseline) wiCaptureBaseline();
    WI.running = true;
    const runBtn = $("btnWiRun");
    if (runBtn) runBtn.disabled = true;
    wiScan(true);
    let prediction = null;
    try {
      prediction = MD.predictFailure(read.features);
    } catch (err) {
      console.error("What-if inference failed:", err);
      toast("Inference failed: " + err.message, "error");
      ccEvent("WHAT-IF ERROR \u00b7 INFERENCE FAILED");
      WI.running = false;
      if (runBtn) runBtn.disabled = false;
      wiScan(false);
      return;
    }
    const risk = clamp(Number(prediction.probabilityPercentage) || 0, 0, 100);
    const pm = prediction.perModelPct || {};
    const nameEl = $("wiName");
    const label = (nameEl && String(nameEl.value).trim()) || "Custom scenario";
    const result = {
      features: read.features,
      prediction: prediction,
      risk: risk,
      label: label,
      ts: Date.now(),
      time: nowStamp(),
      models: {
        dt: Number.isFinite(Number(pm.dt)) ? Number(pm.dt) : null,
        rf: Number.isFinite(Number(pm.rf)) ? Number(pm.rf) : null,
        svm: Number.isFinite(Number(pm.svm)) ? Number(pm.svm) : null
      }
    };
    WI.scenario = result;
    WI.history.unshift(result);
    if (WI.history.length > 40) WI.history.length = 40;
    WI.slotB = WI.slotA;
    WI.slotA = result;
    wiRenderRisk(result);
    wiRenderModels(result);
    wiRenderAdvisory(result);
    wiRenderHistory();
    wiRenderCompare();
    wiRenderTrend();
    wiRenderImpact(read.features);
    wiPushTwinPreview(read.features);
    const band = statusFromRisk(risk).key;
    ccEvent("SCENARIO RECEIVED \u00b7 " + label.toUpperCase());
    ccEvent("FEATURES VALIDATED \u00b7 5/5 IN RANGE");
    ccEvent("BASELINE INFERENCE COMPLETE \u00b7 RF " + (WI.baseline ? fmt(WI.baseline.risk, 1) : "--") + "%");
    ccEvent("SCENARIO INFERENCE COMPLETE \u00b7 RF " + fmt(risk, 1) + "% \u00b7 DT " +
      (result.models.dt != null ? fmt(result.models.dt, 1) : "N/A") + "% \u00b7 SVM " +
      (result.models.svm != null ? fmt(result.models.svm, 1) : "N/A") + "%", band === "normal" ? null : band);
    ccEvent("MODEL COMPARISON UPDATED \u00b7 3 MODELS SCORED");
    ccEvent("RISK UPDATED \u00b7 " + wiLevelOf(risk).label);
    if (!options.silent) {
      toast("What-if complete \u2014 baseline " + (WI.baseline ? fmt(WI.baseline.risk, 1) : "--") +
        "% \u2192 scenario " + fmt(risk, 1) + "% (real models).", "ok");
    }
    WI.running = false;
    if (runBtn) runBtn.disabled = false;
  }
  /* Slider / numeric moves update values + twin only. Models run on button. */
  function wiPreview() {
    const read = wiReadFeatures();
    if (read.errors.length) return;
    wiPushTwinPreview(read.features);
  }
  /* Presets set INPUT FEATURES ONLY - never a stored risk value. */
  const WI_PRESETS = [
    { id: "normal", name: "NORMAL OPERATION", desc: "Balanced thermal + low wear.",
      values: { airTemperature: 298.2, processTemperature: 308.6, rotationalSpeed: 1500, torque: 38, toolWear: 40 } },
    { id: "torque", name: "HIGH TORQUE", desc: "Torque toward the top of the input range.",
      values: { airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 1540, torque: 68, toolWear: 85 } },
    { id: "wear", name: "HIGH TOOL WEAR", desc: "Cutting time past the 200 min advisory mark.",
      values: { airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 1540, torque: 39.5, toolWear: 248 } },
    { id: "temp", name: "HIGH TEMPERATURE", desc: "Air + process temperature driven high.",
      values: { airTemperature: 303.5, processTemperature: 313.5, rotationalSpeed: 1540, torque: 39.5, toolWear: 85 } },
    { id: "speed", name: "HIGH SPEED", desc: "Spindle speed near the top of the range.",
      values: { airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 2800, torque: 39.5, toolWear: 85 } },
    { id: "stress", name: "COMBINED STRESS", desc: "Torque, wear, heat and speed together.",
      values: { airTemperature: 302, processTemperature: 312, rotationalSpeed: 2400, torque: 66, toolWear: 232 } }
  ];
  function wiRenderPresets() {
    const host = $("wiPresets");
    if (!host) return;
    let html = "";
    WI_PRESETS.forEach(function (p) {
      html += '<button type="button" class="wi-preset" data-wi-preset="' + esc(p.id) + '" title="' + esc(p.desc) + '">';
      html += '<span class="wi-preset-name">' + esc(p.name) + "</span>";
      html += '<span class="wi-preset-desc">' + esc(p.desc) + "</span></button>";
    });
    host.innerHTML = html;
  }
  function wiApplyPreset(id) {
    const preset = WI_PRESETS.filter(function (x) { return x.id === id; })[0];
    if (!preset) return;
    WI_SPECS.forEach(function (spec) {
      const v = preset.values[spec.field];
      if (v == null) return;
      const range = $("wi" + spec.key + "Range");
      const num = $("wi" + spec.key + "Num");
      if (range) range.value = String(v);
      if (num) num.value = String(spec.dec ? Number(Number(v).toFixed(spec.dec)) : Math.round(v));
    });
    const nameEl = $("wiName");
    if (nameEl && !String(nameEl.value).trim()) nameEl.value = preset.name;
    ccEvent("SCENARIO CREATED \u00b7 PRESET " + preset.name.toUpperCase());
    const auto = $("wiAuto");
    if (auto && auto.checked) wiRun({ silent: true });
    else { wiPreview(); toast("Preset \"" + preset.name + "\" loaded \u2014 press RUN WHAT-IF ANALYSIS to score it.", "info"); }
  }

  /* RESET restores the baseline in place - no page reload.
     History is deliberately preserved. */
  function wiReset() {
    if (!WI.baseline) wiCaptureBaseline();
    const base = WI.baseline ? WI.baseline.features : sanitizeSensors(state.sensors);
    WI_SPECS.forEach(function (spec) {
      const v = base[spec.field];
      const range = $("wi" + spec.key + "Range");
      const num = $("wi" + spec.key + "Num");
      if (range) range.value = String(v);
      if (num) num.value = String(spec.dec ? Number(Number(v).toFixed(spec.dec)) : Math.round(v));
      const errNode = $("wi" + spec.key + "Err");
      if (errNode) { errNode.textContent = ""; errNode.classList.remove("on"); }
    });
    const nameEl = $("wiName");
    if (nameEl) nameEl.value = "";
    WI.scenario = null;
    WI.slotA = null;
    WI.slotB = null;
    setText("wiBaseRisk", WI.baseline ? fmt(WI.baseline.risk, 1) : "--");
    setText("wiScenRisk", "--");
    setText("wiDelta", "--");
    const dEl = $("wiDelta");
    if (dEl) dEl.className = "wi-risk-v flat";
    setText("wiBasePct", WI.baseline ? fmt(WI.baseline.risk, 1) + "%" : "--");
    setText("wiScenPct", "--");
    const bb = $("wiBaseBar"), sb = $("wiScenBar");
    if (bb) bb.style.width = (WI.baseline ? clamp(WI.baseline.risk, 0, 100) : 0) + "%";
    if (sb) { sb.style.width = "0%"; sb.className = "wi-bar-fill wi-bar-scen"; }
    const wrap = $("wiLevelWrap");
    if (wrap) wrap.className = "wi-level";
    setText("wiLevelIcon", "\u25CB");
    setText("wiLevelText", "NO SCENARIO RUN");
    setText("wiMeta", "Baseline is the live machine reading at the moment the lab was opened. Run an analysis to compare.");
    setText("wiStatePill", "IDLE");
    const host = $("wiModels");
    if (host) host.innerHTML = '<div class="wi-empty wi-empty-wide">Run a what-if analysis to see the real response of all three fitted models.</div>';
    const adv = $("wiAdvisory");
    if (adv) adv.innerHTML = "";
    const vbox = $("wiValidation");
    if (vbox) { vbox.textContent = ""; vbox.className = "wi-validation"; }
    const imp = $("wiImpact");
    if (imp) imp.innerHTML = '<div class="wi-empty">FEATURE IMPACT \u2014 MODEL-SUPPORTED FEATURE IMPORTANCE NOT AVAILABLE</div>';
    setText("wiImpactPill", "NOT AVAILABLE");
    setText("wiImpactNote", "This build exposes no feature-importance data, so none is shown. Nothing is estimated or invented.");
    wiRenderHistory();
    wiRenderCompare();
    wiRenderTrend();
    wiPushTwinPreview(base);
    ccEvent("SCENARIO RESET \u00b7 BASELINE RESTORED");
    toast("Scenario reset to baseline. History preserved.", "ok");
  }
  /* Export contains ONLY real recorded values. */
  function wiExport() {
    if (!WI.scenario) { toast("Run a what-if analysis first \u2014 there is nothing to export yet.", "warn"); return; }
    const b = WI.baseline, r = WI.scenario;
    const headers = ["Scenario", "Timestamp", "Field", "Baseline", "Scenario", "Unit"];
    const rows = [];
    const bpm = b && b.prediction.perModelPct ? b.prediction.perModelPct : null;
    const nf = function (v) { return Number.isFinite(Number(v)) ? fmt(Number(v), 1) : "N/A"; };
    WI_SPECS.forEach(function (spec) {
      rows.push([r.label, r.time, spec.label, b ? fmt(b.features[spec.field], spec.dec) : "", fmt(r.features[spec.field], spec.dec), spec.unit]);
    });
    rows.push([r.label, r.time, "Random Forest risk", b ? fmt(b.risk, 1) : "", fmt(r.risk, 1), "%"]);
    rows.push([r.label, r.time, "Decision Tree risk", bpm ? nf(bpm.dt) : "", nf(r.models.dt), "%"]);
    rows.push([r.label, r.time, "SVM risk", bpm ? nf(bpm.svm) : "", nf(r.models.svm), "%"]);
    rows.push([r.label, r.time, "Risk change", b ? fmt(b.risk, 1) : "",
      (r.risk - (b ? b.risk : 0) >= 0 ? "+" : "") + fmt(r.risk - (b ? b.risk : 0), 1), "percentage points"]);
    rows.push([r.label, r.time, "Risk level", b ? wiLevelOf(b.risk).label : "", wiLevelOf(r.risk).label, ""]);
    rows.push([r.label, r.time, "RUL", "Not available", "Not available", "binary classification model"]);
    if (downloadFile("scenario-whatif-" + exportStamp() + ".csv", toCsv(headers, rows))) {
      ccEvent("SCENARIO EXPORTED \u00b7 CSV");
      toast("Scenario CSV exported (real model outputs only).", "ok");
    }
  }
  function bindWhatIf() {
    if (!$("btnWiRun")) return;
    wiRenderPresets();
    WI_SPECS.forEach(function (spec) {
      const range = $("wi" + spec.key + "Range");
      const num = $("wi" + spec.key + "Num");
      if (range) range.addEventListener("input", function () { if (num) num.value = range.value; wiPreview(); });
      if (num) {
        num.addEventListener("input", function () {
          const raw = String(num.value).trim();
          const parsed = Number(raw);
          const lim = wiRange(spec.key);
          const errNode = $("wi" + spec.key + "Err");
          if (raw === "" || !Number.isFinite(parsed)) {
            if (errNode) { errNode.textContent = "Enter a finite number."; errNode.classList.add("on"); }
            return;
          }
          if (errNode) { errNode.textContent = ""; errNode.classList.remove("on"); }
          if (range) range.value = String(clamp(parsed, lim.min, lim.max));
          wiPreview();
        });
        num.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); wiRun(); } });
      }
    });
    const presets = $("wiPresets");
    if (presets) presets.addEventListener("click", function (e) {
      const btn = e.target.closest("[data-wi-preset]");
      if (!btn) return;
      wiApplyPreset(btn.getAttribute("data-wi-preset"));
    });
    const run = $("btnWiRun");
    if (run) run.addEventListener("click", function () { wiRun(); });
    const reset = $("btnWiReset");
    if (reset) reset.addEventListener("click", wiReset);
    const exp = $("btnWiExport");
    if (exp) exp.addEventListener("click", wiExport);
    const auto = $("wiAuto");
    if (auto) auto.addEventListener("change", function () {
      ccEvent(auto.checked ? "AUTO ANALYZE ENABLED" : "AUTO ANALYZE DISABLED");
      if (auto.checked) wiRun({ silent: true });
    });
    wiCaptureBaseline();
    if (WI.baseline) setText("wiBaseRisk", fmt(WI.baseline.risk, 1));
    setText("wiBasePct", WI.baseline ? fmt(WI.baseline.risk, 1) + "%" : "--");
    const bb = $("wiBaseBar");
    if (bb && WI.baseline) bb.style.width = clamp(WI.baseline.risk, 0, 100) + "%";
    const mh = $("wiModels");
    if (mh) mh.innerHTML = '<div class="wi-empty wi-empty-wide">Run a what-if analysis to see the real response of all three fitted models.</div>';
    wiRenderHistory();
    wiRenderCompare();
    wiRenderTrend();
  }


  function bindScenarioControls() {
    buildScenarioCards();
    const grid = $("scenarioCards");
    if (grid) {
      grid.addEventListener("click", (e) => {
        const btn = e.target.closest(".scenario-btn");
        if (!btn) return;
        runScenarioById(btn.getAttribute("data-scenario"));
      });
    }
    const run = $("btnScenarioRun");
    if (run) run.addEventListener("click", () => {
      if (state.scenario.selected) runScenarioById(state.scenario.selected);
      else applyCustomScenario();
    });
    const reset = $("btnScenarioReset");
    if (reset) reset.addEventListener("click", () => runScenarioById("healthy_baseline"));
    const applyBtn = $("btnCustomApply");
    if (applyBtn) applyBtn.addEventListener("click", applyCustomScenario);
    const fill = $("btnCustomFillCurrent");
    if (fill) fill.addEventListener("click", fillCustomGridFromCurrent);
    const whatIf = $("btnWhatIf");
    if (whatIf) whatIf.addEventListener("click", whatIfCompare);
  }
  /* ============================== DEMO MODE =============================== */
  /* Scripted 7-step presentation. Every step is a real simulation event that
     runs the REAL forest; the step values were measured against the bundled
     model when this demo was designed (0.0% → 3.4% → 20.7% → 36.2% → 52.3%
     → 87.0% → 84.5% when this script was authored). */
  const DEMO_STEPS = [
    { label: "Step 1 of 7 · Healthy machine",
      note: "Nominal operating point — the forest reports no failure signature.",
      values: { productType: "M", airTemperature: 298.2, processTemperature: 308.7, rotationalSpeed: 1520, torque: 38.5, toolWear: 42 }, focus: "dashboard" },
    { label: "Step 2 of 7 · Tool wear increases",
      note: "Tool wear approaches the advisory 200 min limit.",
      values: { productType: "M", airTemperature: 298.6, processTemperature: 309.0, rotationalSpeed: 1470, torque: 52, toolWear: 195 }, focus: "live" },
    { label: "Step 3 of 7 · Torque increases",
      note: "Extra spindle load pushes the predicted risk upward.",
      values: { productType: "M", airTemperature: 298.8, processTemperature: 309.1, rotationalSpeed: 1450, torque: 56, toolWear: 205 }, focus: "live" },
    { label: "Step 4 of 7 · Risk rises",
      note: "Risk climbs through the warning zone — watch the trend and KPIs.",
      values: { productType: "M", airTemperature: 299.0, processTemperature: 309.2, rotationalSpeed: 1430, torque: 58, toolWear: 212 }, focus: "dashboard" },
    { label: "Step 5 of 7 · Warning state",
      note: "Warning-level risk is now raising alerts in the alert feed.",
      values: { productType: "M", airTemperature: 299.2, processTemperature: 309.3, rotationalSpeed: 1440, torque: 60, toolWear: 220 }, focus: "prediction" },
    { label: "Step 6 of 7 · Critical risk",
      note: "Risk crosses the critical threshold — see which features drive it.",
      values: { productType: "L", airTemperature: 299.6, processTemperature: 309.3, rotationalSpeed: 1400, torque: 62, toolWear: 230 }, focus: "xai" },
    { label: "Step 7 of 7 · Maintenance recommendation",
      note: "The advisory engine raises the urgent tool change — no RUL is invented.",
      values: { productType: "L", airTemperature: 300.0, processTemperature: 309.3, rotationalSpeed: 1380, torque: 64, toolWear: 238 }, focus: "maintenance" }
  ];
  function startDemo() {
    state.demo.active = true;
    state.demo.step = -1;
    const banner = $("demoBanner");
    if (banner) banner.classList.remove("hidden");
    toast("Demo mode started — 7 scripted simulation steps, each scored by the real forest.", "info");
    setDemoPlaying(true);
  }
  function setDemoPlaying(playing) {
    if (state.demo.timer) { window.clearInterval(state.demo.timer); state.demo.timer = null; }
    state.demo.playing = !!playing;
    if (state.demo.playing) {
      state.demo.timer = window.setInterval(() => stepDemo(1), 3600);
      if (state.demo.step < 0) stepDemo(1);
      else applyDemoStep(state.demo.step);
    }
    const play = $("btnDemoPlay");
    const pause = $("btnDemoPause");
    if (play) play.disabled = state.demo.playing;
    if (pause) pause.disabled = !state.demo.playing;
  }
  function stepDemo(direction) {
    let next = state.demo.step + direction;
    if (next >= DEMO_STEPS.length) {
      stopDemo("Demo complete — leaving the last state loaded for inspection.");
      return;
    }
    if (next < 0) next = 0;
    applyDemoStep(next);
  }
  function applyDemoStep(index) {
    state.demo.step = index;
    const step = DEMO_STEPS[index];
    Object.assign(state.sensors, step.values);
    state.machine.id = "CNC-MILL-08";
    state.scenario.selected = null;
    markScenarioSelected(null);
    syncUIFromState();
    runPipeline("demo");
    setText("demoStepLabel", step.label + " — " + step.note);
    const progress = $("demoProgress");
    if (progress) progress.style.width = Math.round(((index + 1) / DEMO_STEPS.length) * 100) + "%";
    if (step.focus && state.view !== step.focus) switchView(step.focus);
  }
  function demoReset() {
    setDemoPlaying(false);
    applyDemoStep(0);
    toast("Demo reset to step 1.", "info");
  }
  function stopDemo(msg) {
    if (state.demo.timer) { window.clearInterval(state.demo.timer); state.demo.timer = null; }
    state.demo.active = false;
    state.demo.playing = false;
    const banner = $("demoBanner");
    if (banner) banner.classList.add("hidden");
    if (msg) toast(msg, "info");
  }
  function bindDemoControls() {
    const start = $("btnDemoMode");
    if (start) start.addEventListener("click", () => { if (!state.demo.active) startDemo(); else setDemoPlaying(true); });
    const qa = $("qaDemo");
    if (qa) qa.addEventListener("click", () => { if (!state.demo.active) startDemo(); else setDemoPlaying(true); });
    const play = $("btnDemoPlay");
    if (play) play.addEventListener("click", () => setDemoPlaying(true));
    const pause = $("btnDemoPause");
    if (pause) pause.addEventListener("click", () => setDemoPlaying(false));
    const reset = $("btnDemoReset");
    if (reset) reset.addEventListener("click", demoReset);
    const close = $("btnDemoClose");
    if (close) close.addEventListener("click", () => stopDemo("Demo mode closed."));
  }
  /* ============================ MACHINE FLEET ============================= */
  function fleetList() { return (EX && EX.SIMULATED_FLEET) ? EX.SIMULATED_FLEET : []; }
  function machineValues(m) { return m.monitored ? Object.assign({}, state.sensors) : m.values; }
  function fleetRisk(values) {
    if (!modelReady()) return null;
    try {
      const r = MD.predictFailure(values);
      return { risk: Number(r.probabilityPercentage), health: Number(r.healthIndex) };
    } catch (err) { return null; }
  }
  function buildFleetGrid() {
    const host = $("fleetGrid");
    if (!host) return;
    host.innerHTML = fleetList().map((m) => `
      <button type="button" class="fleet-card" data-fleet="${esc(m.id)}" id="fleetCard-${esc(m.id)}">
        <span class="fleet-top">
          <span class="fleet-id">${esc(m.name)}</span>
          <span class="fleet-badge" id="fleetBadge-${esc(m.id)}">--</span>
        </span>
        <span class="fleet-job">${esc(m.job)}${m.monitored ? " · live state" : " · simulated profile"}</span>
        <span class="fleet-metrics">
          <span class="fleet-metric"><span class="fleet-metric-label">Risk</span><span class="fleet-metric-value" id="fleetRisk-${esc(m.id)}">--</span></span>
          <span class="fleet-metric"><span class="fleet-metric-label">Health</span><span class="fleet-metric-value" id="fleetHealth-${esc(m.id)}">--</span></span>
          <span class="fleet-metric"><span class="fleet-metric-label">Type</span><span class="fleet-metric-value" id="fleetType-${esc(m.id)}">--</span></span>
        </span>
      </button>`).join("");
  }
  function updateFleetMetrics() {
    if (state.view !== "fleet") return;
    if (!$("fleetGrid")) return;
    if (!$("fleetCard-CNC-01")) buildFleetGrid();
    const counts = { normal: 0, warning: 0, critical: 0 };
    fleetList().forEach((m) => {
      const values = machineValues(m);
      const r = fleetRisk(values);
      const status = r ? statusFromRisk(r.risk) : null;
      setText("fleetRisk-" + m.id, r ? fmt(r.risk, 1) + "%" : "--");
      setText("fleetHealth-" + m.id, r ? String(Math.round(r.health)) : "--");
      setText("fleetType-" + m.id, values.productType ? "Type " + values.productType : "--");
      const badge = $("fleetBadge-" + m.id);
      if (badge && status) {
        badge.textContent = statusWord(status.key);
        badge.className = "fleet-badge fleet-" + status.key;
        counts[status.key] += 1;
      }
      const card = $("fleetCard-" + m.id);
      if (card) card.classList.toggle("fleet-selected", !!m.monitored);
    });
    setText("fleetSummary", counts.normal + " normal · " + counts.warning + " warning · " + counts.critical + " critical — all live RF outputs");
  }
  function selectFleetMachine(id) {
    const m = fleetList().filter((x) => x.id === id)[0];
    if (!m) return;
    if (!m.monitored && m.values) Object.assign(state.sensors, m.values);
    state.machine.id = m.name;
    state.scenario.selected = null;
    markScenarioSelected(null);
    syncUIFromState();
    runPipeline("fleet");
    toast(m.monitored ? "Back on the monitored asset (live state)." : "Loaded simulated machine " + m.id + " into the pipeline.", "ok");
    switchView("viz3d");
  }
  function bindFleetControls() {
    const grid = $("fleetGrid");
    if (grid) {
      grid.addEventListener("click", (e) => {
        const btn = e.target.closest(".fleet-card");
        if (btn) selectFleetMachine(btn.getAttribute("data-fleet"));
      });
    }
  }
  /* =========================== MAINTENANCE VIEW =========================== */
  function priorityToDisplay(priority) {
    if (priority === "CRITICAL") return { label: "CRITICAL", cls: "maint-critical" };
    if (priority === "URGENT") return { label: "HIGH", cls: "maint-urgent" };
    if (priority === "WARNING" || priority === "SCHEDULED") return { label: "MEDIUM", cls: "maint-warning" };
    return { label: "LOW", cls: "maint-normal" };
  }
  function updateMaintenance() {
    const list = $("maintList");
    const p = state.prediction;
    const recs = (p && p.recommendations) || [];
    if (list) {
      list.innerHTML = recs.length ? recs.map((r) => `
        <div class="prescription-card priority-${esc(r.priority)}">
          <div class="presc-icon" aria-hidden="true">${getPrescriptionIcon(r.priority)}</div>
          <div class="presc-details">
            <div class="presc-action-row">
              <span class="presc-badge">${esc(priorityToDisplay(r.priority).label)}</span>
              <span class="presc-action">${esc(r.action)}</span>
            </div>
            <div class="presc-reason">${esc(r.reason)}</div>
          </div>
        </div>`).join("")
        : '<div class="presc-reason">Awaiting the first inference…</div>';
      setText("maintCountPill", recs.length + " item(s)");
    }
    if (!p) return;
    const top = recs[0];
    if (top) {
      const disp = priorityToDisplay(top.priority);
      const node = $("maintPriority");
      if (node) {
        node.textContent = disp.label;
        node.className = "maint-priority " + disp.cls;
      }
      setText("maintPriorityPill", "PRIORITY: " + disp.label);
      setText("maintReason", top.reason);
      setText("maintAction", top.action);
    }
    setText("maintUpdated", "Updated " + nowStamp() + " · status " + statusWord(state.status.key) +
      " at " + fmt(state.risk, 1) + "% (real Random Forest output)");
    const host = $("maintIndicators");
    if (host) {
      host.innerHTML = (indicators() || []).map((x) => {
        const lv = indicatorLevel(x.pct);
        return `<div class="maint-ind-row">
          <span class="maint-ind-code">${esc(x.code)}</span>
          <span class="maint-ind-name">${esc(x.label)}</span>
          <span class="maint-ind-bar"><span class="maint-ind-fill maint-${lv.key}" style="width:${clamp(x.pct, 1, 100)}%"></span></span>
          <span class="maint-ind-pct">${fmt(x.pct, 1)}%</span>
          <span class="maint-ind-level maint-${lv.key}">${esc(lv.label)}</span>
        </div>`;
      }).join("");
    }
  }
  /* ======================= OPTIONAL CSV DATASET LOADER ==================== */
  /* The AI4I CSV is not bundled with this project. If the user supplies it,
     this loader parses it locally (nothing is uploaded) and builds honest
     distributions. If not supplied, the panel stays empty by design. */
  function parseCsvLine(line) {
    const out = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line.charAt(i);
      if (inQuotes) {
        if (ch === '"') {
          if (line.charAt(i + 1) === '"') { cur += '"'; i += 1; }
          else inQuotes = false;
        } else cur += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === ",") { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }
  function handleDatasetFile(file) {
    if (!file) return;
    setText("datasetFileName", file.name + " (" + Math.max(1, Math.round(file.size / 1024)) + " KB)");
    const reader = new FileReader();
    reader.onload = () => {
      try { ingestDataset(String(reader.result), file.name); }
      catch (err) { toast("Could not parse the file: " + err.message, "error"); }
    };
    reader.onerror = () => toast("Could not read the selected file.", "error");
    reader.readAsText(file);
  }
  function ingestDataset(text, filename) {
    const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim().length);
    if (lines.length < 3) throw new Error("file appears to be empty");
    const header = parseCsvLine(lines[0].replace(/^\ufeff/, ""));
    const findCol = (names) => {
      for (let i = 0; i < names.length; i++) {
        const idx = header.indexOf(names[i]);
        if (idx >= 0) return idx;
      }
      return -1;
    };
    const col = {
      type: findCol(["Type", "type"]),
      air: findCol(["Air temperature [K]", "AirTemp", "Air temperature"]),
      proc: findCol(["Process temperature [K]", "ProcTemp", "Process temperature"]),
      rpm: findCol(["Rotational speed [rpm]", "RPM", "Rotational speed"]),
      torque: findCol(["Torque [Nm]", "Torque"]),
      wear: findCol(["Tool wear [min]", "ToolWear", "Tool wear"]),
      failure: findCol(["Machine failure", "Failure", "failure"]),
      id: findCol(["Product ID", "ProductID", "UDI"])
    };
    const missing = ["type", "air", "proc", "rpm", "torque", "wear", "failure"].filter((k) => col[k] < 0);
    if (missing.length) throw new Error("header is missing expected columns (" + missing.join(", ") + ") — is this the AI4I 2020 CSV?");
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const cells = parseCsvLine(lines[i]);
      rows.push({
        id: col.id >= 0 ? cells[col.id] : String(i),
        type: String(cells[col.type] || "").trim(),
        air: Number(cells[col.air]),
        proc: Number(cells[col.proc]),
        rpm: Number(cells[col.rpm]),
        torque: Number(cells[col.torque]),
        wear: Number(cells[col.wear]),
        failure: Number(cells[col.failure])
      });
    }
    if (!rows.length) throw new Error("no data rows found");
    renderDatasetPreview(rows, filename);
    toast("Loaded " + fmtInt(rows.length) + " rows from " + filename + " — parsed locally.", "ok");
  }
  function renderDatasetPreview(rows, filename) {
    const failures = rows.filter((r) => r.failure === 1).length;
    const rate = (failures / rows.length) * 100;
    const complete = rows.filter((r) => [r.air, r.proc, r.rpm, r.torque, r.wear].every((v) => Number.isFinite(v))).length;
    const loaded = $("datasetLoaded");
    if (loaded) loaded.classList.remove("hidden");
    const stats = $("dsLoadedStats");
    if (stats) {
      const tiles = [
        ["Rows loaded", fmtInt(rows.length), ""],
        ["Failures", fmtInt(failures), "st-crit"],
        ["Failure rate", fmt(rate, 2) + "%", "st-warn"],
        ["Complete rows", fmtInt(complete), ""],
        ["Types present", Array.from(new Set(rows.map((r) => r.type))).join(" / "), ""],
        ["Source", filename, ""]
      ];
      stats.innerHTML = tiles.map((t) =>
        `<div class="stat-tile ${t[2]}"><span class="stat-label">${esc(t[0])}</span><span class="stat-value">${esc(t[1])}</span></div>`
      ).join("");
    }
    const series = {
      dsChartAir: rows.map((r) => r.air), dsChartProc: rows.map((r) => r.proc),
      dsChartRpm: rows.map((r) => r.rpm), dsChartTorque: rows.map((r) => r.torque),
      dsChartWear: rows.map((r) => r.wear)
    };
    Object.keys(series).forEach((id) => {
      const canvas = $(id);
      if (canvas) drawHistogram(canvas, series[id], { bins: 14, color: "#06b6d4", emptyText: "No numeric data." });
    });
    drawBarChart($("dsChartFail"), [
      { label: "Normal", value: rows.length - failures, display: fmtInt(rows.length - failures), color: "#10b981" },
      { label: "Failure", value: failures, display: fmtInt(failures), color: "#f43f5e" }
    ], { max: Math.max(1, rows.length - failures, failures), padL: 64, padR: 70 });
    const head = $("dsPreviewHead");
    if (head) head.innerHTML = ["ID", "Type", "Air (K)", "Proc (K)", "RPM", "Torque (Nm)", "Wear (min)", "Failure"].map((h) => "<th>" + h + "</th>").join("");
    const body = $("dsPreviewBody");
    if (body) {
      body.innerHTML = rows.slice(0, 12).map((r) => `
        <tr>
          <td style="font-family:var(--font-mono);">${esc(r.id)}</td>
          <td>${esc(r.type)}</td>
          <td>${fmt(r.air, 1)}</td>
          <td>${fmt(r.proc, 1)}</td>
          <td>${Math.round(r.rpm)}</td>
          <td>${fmt(r.torque, 1)}</td>
          <td>${Math.round(r.wear)}</td>
          <td><span class="table-status-pill ${r.failure === 1 ? "table-status-critical" : "table-status-normal"}">${r.failure === 1 ? "Failure" : "Normal"}</span></td>
        </tr>`).join("");
    }
    setText("dsPreviewNote", "Showing the first 12 of " + fmtInt(rows.length) + " rows from your local file. Distributions above are computed from the full file. The file was not uploaded anywhere and is not stored.");
    setText("datasetFallbackNote", "Local file in use: " + filename + " — parsed in your browser only.");
    state.dataset.rows = rows.length;
    state.dataset.name = filename;
    state.dataset.stats = { failures: failures, rate: rate };
  }
  function bindDatasetLoader() {
    const input = $("datasetFileInput");
    if (input) input.addEventListener("change", (e) => handleDatasetFile(e.target.files && e.target.files[0]));
  }
  /* ======================= SETTINGS & SYSTEM STATUS ======================= */
  function applySettingsToUI() {
    const s = state.settings;
    const setVal = (id, v) => { const n = $(id); if (n) n.value = v; };
    setVal("setInterval", s.simInterval);
    setVal("setSpeed", s.simSpeed);
    setVal("setNoise", s.noise);
    setVal("setWarning", s.warnThreshold);
    setVal("setCritical", s.critThreshold);
    setText("setNoiseVal", fmt(s.noise, 2) + "×");
    const auto = $("setAutoRun");
    if (auto) auto.checked = s.autoRun;
    const compact = $("setCompact");
    if (compact) compact.checked = s.compact;
    document.body.classList.toggle("compact", s.compact);
    setText("thresholdPill", "WARN ≥ " + s.warnThreshold + "% · CRIT ≥ " + s.critThreshold + "%");
    setText("settingsSavedPill", Storage.available ? "PERSISTED LOCALLY" : "MEMORY ONLY");
    setText("storagePill2", Storage.available ? "LOCAL STORAGE ACTIVE" : "MEMORY ONLY (STORAGE BLOCKED)");
  }
  function updateSystemStatus() {
    const data = metricsData();
    setText("sysDataset", data ? data.dataset.name : "--");
    setText("sysModels", "Decision Tree · Random Forest · SVM (RBF)");
    setText("sysValidation", data ? data.pipeline.cv : "--");
    setText("sysOptimization", "F1-score via GridSearchCV");
    setText("sysDeployment", "Offline browser prototype (no server, no build step)");
    setText("sysData", state.dataset.rows ? "User-loaded CSV + simulated telemetry" : "Simulated telemetry / bundled dataset stats");
    setText("sysInference", modelReady() ? "Random Forest live · DT & SVM not connected" : "UNAVAILABLE");
    const pill = $("sysStatusPill");
    if (pill) {
      pill.textContent = modelReady() ? "SYSTEM OK" : "DEGRADED";
      pill.className = "dataset-pill" + (modelReady() ? "" : " chip-bad");
    }
  }
  function bindSettingsControls() {
    const interval = $("setInterval");
    if (interval) interval.addEventListener("change", () => {
      state.settings.simInterval = clamp(Number(interval.value) || 700, 200, 5000);
      saveSettings();
      toast("Stream interval set to " + state.settings.simInterval + " ms.", "info");
      if (state.sim.running) { stopSimulation(true); startSimulation(); }
    });
    const speed = $("setSpeed");
    if (speed) speed.addEventListener("change", () => {
      const v = Number(speed.value);
      state.settings.simSpeed = [0.5, 1, 2].indexOf(v) >= 0 ? v : 1;
      saveSettings();
    });
    const noise = $("setNoise");
    if (noise) noise.addEventListener("input", () => {
      state.settings.noise = clamp(Number(noise.value) || 0, 0, 3);
      setText("setNoiseVal", fmt(state.settings.noise, 2) + "×");
      saveSettings();
    });
    const auto = $("setAutoRun");
    if (auto) auto.addEventListener("change", () => {
      state.settings.autoRun = !!auto.checked;
      saveSettings();
      toast(auto.checked ? "Telemetry will auto-run on the next load." : "Auto-run disabled.", "info");
    });
    const warn = $("setWarning");
    if (warn) warn.addEventListener("change", () => {
      state.settings.warnThreshold = clamp(Number(warn.value) || 28, 1, 99);
      if (state.settings.critThreshold <= state.settings.warnThreshold) {
        state.settings.critThreshold = Math.min(100, state.settings.warnThreshold + 1);
      }
      saveSettings();
      applySettingsToUI();
      runPipeline("manual", { record: false });
      toast("Warning threshold set to " + state.settings.warnThreshold + "%.", "info");
    });
    const crit = $("setCritical");
    if (crit) crit.addEventListener("change", () => {
      state.settings.critThreshold = clamp(Number(crit.value) || 65, 2, 100);
      if (state.settings.critThreshold <= state.settings.warnThreshold) {
        state.settings.warnThreshold = Math.max(1, state.settings.critThreshold - 1);
      }
      saveSettings();
      applySettingsToUI();
      runPipeline("manual", { record: false });
      toast("Critical threshold set to " + state.settings.critThreshold + "%.", "info");
    });
    const compact = $("setCompact");
    if (compact) compact.addEventListener("change", () => {
      state.settings.compact = !!compact.checked;
      saveSettings();
      applySettingsToUI();
    });
    const fullscreen = $("btnFullscreen");
    if (fullscreen) fullscreen.addEventListener("click", () => {
      try {
        if (!document.fullscreenElement) {
          if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
        } else if (document.exitFullscreen) document.exitFullscreen();
      } catch (err) { toast("Fullscreen is not available in this browser.", "warn"); }
    });
    const clearHistory2 = $("btnClearHistory2");
    if (clearHistory2) clearHistory2.addEventListener("click", () => {
      if (!state.history.length) { toast("History is already empty.", "info"); return; }
      clearHistory();
      toast("Prediction history cleared.", "ok");
    });
    const resetSim = $("btnResetSim");
    if (resetSim) resetSim.addEventListener("click", () => {
      loadPreset("nominal");
      toast("Simulation reset to the nominal baseline.", "ok");
    });
    const clearAll = $("btnClearStored");
    if (clearAll) clearAll.addEventListener("click", () => {
      Storage.remove(LS_SETTINGS);
      Storage.remove(LS_HISTORY);
      state.settings = loadSettings();
      state.history = [];
      state.lastRecordTs = 0;
      applySettingsToUI();
      updateHistoryView();
      updateAnalyticsView();
      runPipeline("manual", { record: false });
      toast("All stored app data cleared; settings restored to defaults.", "ok");
    });
  }
  /* ======================= NAVIGATION & GLOBAL BINDINGS =================== */
  const VIEWS = ["dashboard", "live", "viz3d", "prediction", "models", "xai", "analytics", "whatif", "history", "dataset", "scenarios", "fleet", "maintenance", "settings"];
  function switchView(name) {
    if (VIEWS.indexOf(name) < 0) name = "dashboard";
    state.view = name;
    VIEWS.forEach((v) => {
      const node = $("view-" + v);
      if (node) node.classList.toggle("hidden", v !== name);
    });
    document.querySelectorAll(".nav-btn").forEach((b) => {
      const active = b.getAttribute("data-view") === name;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });
    if (name !== "whatif" && TWIN.scenarioPreview) TWIN.scenarioPreview = null;
    ccMoveTwin(name);
    ccEvent("VIEW → " + ccViewLabel(name).toUpperCase());
    renderAll("view");
    try { window.scrollTo({ top: 0, behavior: "smooth" }); } catch (e) { window.scrollTo(0, 0); }
  }
  function bindNavigation() {
    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.addEventListener("click", () => switchView(b.getAttribute("data-view")));
    });
  }
  function bindGlobalControls() {
    bindNavigation();
    bindPredictionControls();
    bindHistoryControls();
    bindExportControls();
    bindScenarioControls();
    bindWhatIf();
    bindDemoControls();
    bindFleetControls();
    bindDatasetLoader();
    bindSettingsControls();
    bindReportControls();
    const simToggle = $("btnToggleSim");
    if (simToggle) simToggle.addEventListener("click", toggleSimulation);
    const simStart = $("btnSimStart");
    if (simStart) simStart.addEventListener("click", startSimulation);
    const simStop = $("btnSimStop");
    if (simStop) simStop.addEventListener("click", () => stopSimulation());
    const simTickBtn = $("btnSimTick");
    if (simTickBtn) simTickBtn.addEventListener("click", simTick);
    const bell = $("btnAlerts");
    if (bell) bell.addEventListener("click", () => toggleAlertPanel());
    const bellClose = $("btnAlertClose");
    if (bellClose) bellClose.addEventListener("click", () => toggleAlertPanel(false));
    const viewAll = $("btnViewAllAlerts");
    if (viewAll) viewAll.addEventListener("click", () => toggleAlertPanel(true));
    const alertList = $("alertList");
    if (alertList) {
      alertList.addEventListener("click", (e) => {
        const item = e.target.closest(".alert-item");
        if (!item) return;
        toggleAlertPanel(false);
        switchView(item.getAttribute("data-alert-view") || "dashboard");
      });
    }
    const quickActions = [
      ["qaHealthy", () => runScenarioById("healthy_baseline")],
      ["qaCritical", () => runScenarioById("combined_risk")],
      ["qaPredict", () => switchView("prediction")],
      ["qaScenario", () => switchView("scenarios")],
      ["qaExport", () => switchView("history")]
    ];
    quickActions.forEach((pair) => {
      const node = $(pair[0]);
      if (node) node.addEventListener("click", pair[1]);
    });
    const cmpSelector = $("cmSelector");
    if (cmpSelector) {
      cmpSelector.addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-btn");
        if (!btn) return;
        state.cmpModel = btn.getAttribute("data-model");
        cmpSelector.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
        renderConfusionMatrix(state.cmpModel);
      });
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { toggleAlertPanel(false); closeReport(); }
    });
    window.addEventListener("resize", debounce(() => { updateWaveform(); renderAll("resize"); }, 200));
    window.addEventListener("beforeunload", () => { if (state.sim.running) stopSimulation(true); });
  }
  /* =========================== TOASTS & SHARED UI ========================= */
  function debounce(fn, wait) {
    let timer = null;
    return function () {
      const args = arguments;
      const self = this;
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => { timer = null; fn.apply(self, args); }, wait);
    };
  }
  function toast(message, type) {
    const host = $("toastHost");
    if (!host) return;
    const el = document.createElement("div");
    el.className = "toast toast-" + (type || "info");
    el.textContent = message;
    host.appendChild(el);
    window.setTimeout(() => el.classList.add("toast-out"), 3200);
    window.setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, 3800);
  }
  function sessionStats() {
    const rows = state.history;
    const risks = rows.map((r) => Number(r.risk)).filter((v) => Number.isFinite(v));
    const counts = { NORMAL: 0, WARNING: 0, CRITICAL: 0 };
    rows.forEach((r) => { if (counts[r.status] != null) counts[r.status] += 1; });
    return {
      total: rows.length,
      counts: counts,
      avg: risks.length ? risks.reduce((a, b) => a + b, 0) / risks.length : null,
      max: risks.length ? Math.max.apply(null, risks) : null,
      first: rows.length ? rows[0].time : null,
      last: rows.length ? rows[rows.length - 1].time : null
    };
  }
  /* ======================= 3D OPERATING-SPACE MAP ========================= */
  /* Background points are REAL Random Forest evaluations over a sampled
     operating space (ToolWear × Torque × ProcessTemp; Type, AirTemp and RPM
     held at the current reading). The current machine point comes from its
     actual sensors. No external 3D library — custom projection on canvas. */
  const VIZ3D = {
    canvas: null, rotX: -0.42, rotY: 0.72, zoom: 1,
    dragging: false, lastX: 0, lastY: 0, moved: false,
    grid: null, gridKey: "", computing: false,
    showNormal: true, showCritical: true, showCurrent: true
  };
  const VIZ_AXES = {
    wear: { min: 0, max: 260 },
    torque: { min: 5, max: 80 },
    proc: { min: 305, max: 315 }
  };
  function vizGridKey() {
    const s = state.sensors;
    return s.productType + "|" + s.airTemperature.toFixed(1) + "|" + Math.round(s.rotationalSpeed / 50);
  }
  function viz3dEnsureGrid() {
    if (!modelReady()) return;
    if (VIZ3D.grid && VIZ3D.gridKey === vizGridKey()) return;
    VIZ3D.computing = true;
    const wearSteps = [0, 40, 80, 120, 160, 200, 240, 260];
    const torqueSteps = [5, 20, 35, 50, 65, 80];
    const procSteps = [305, 307.5, 310, 312.5, 315];
    const pts = [];
    for (let a = 0; a < wearSteps.length; a++) {
      for (let b = 0; b < torqueSteps.length; b++) {
        for (let c = 0; c < procSteps.length; c++) {
          let risk = 0;
          try {
            risk = Number(MD.predictFailure({
              productType: state.sensors.productType,
              airTemperature: state.sensors.airTemperature,
              processTemperature: procSteps[c],
              rotationalSpeed: state.sensors.rotationalSpeed,
              torque: torqueSteps[b],
              toolWear: wearSteps[a]
            }).probabilityPercentage) || 0;
          } catch (err) { risk = 0; }
          pts.push({ w: wearSteps[a], t: torqueSteps[b], p: procSteps[c], risk: risk });
        }
      }
    }
    VIZ3D.grid = pts;
    VIZ3D.gridKey = vizGridKey();
    VIZ3D.computing = false;
  }
  function viz3dProject(x, y, z) {
    const nx = ((x - VIZ_AXES.wear.min) / (VIZ_AXES.wear.max - VIZ_AXES.wear.min)) * 2 - 1;
    const ny = ((y - VIZ_AXES.torque.min) / (VIZ_AXES.torque.max - VIZ_AXES.torque.min)) * 2 - 1;
    const nz = ((z - VIZ_AXES.proc.min) / (VIZ_AXES.proc.max - VIZ_AXES.proc.min)) * 2 - 1;
    const cosY = Math.cos(VIZ3D.rotY), sinY = Math.sin(VIZ3D.rotY);
    const x1 = nx * cosY - nz * sinY;
    const z1 = nx * sinY + nz * cosY;
    const cosX = Math.cos(VIZ3D.rotX), sinX = Math.sin(VIZ3D.rotX);
    const y1 = ny * cosX - z1 * sinX;
    const z2 = ny * sinX + z1 * cosX;
    const dist = 3.4;
    const scale = VIZ3D.zoom * Math.min(VIZ3D.w, VIZ3D.h) * 0.40;
    const persp = dist / (dist + z2);
    return {
      x: VIZ3D.w / 2 + x1 * scale * persp,
      y: VIZ3D.h / 2 - y1 * scale * persp,
      z: z2,
      s: persp
    };
  }
  function viz3dRiskColor(risk) {
    if (risk >= state.settings.critThreshold) return "#f43f5e";
    if (risk >= state.settings.warnThreshold) return "#f59e0b";
    return "#10b981";
  }
  function reportHtml() {
    const s = state.sensors;
    const p = state.prediction || {};
    const pm = p.physicsMetrics || {};
    const ind = indicators() || [];
    const contrib = state.contributions || [];
    const recs = p.recommendations || [];
    const stats = sessionStats();
    const row = (k, v) => "<tr><td>" + esc(k) + "</td><td>" + esc(v) + "</td></tr>";
    const sensorTable = "<table class=\"report-table\">" + [
      row("Product variant", typeLabel(s.productType)),
      row("Air temperature", fmt(s.airTemperature, 1) + " K (" + celsius(s.airTemperature) + " °C)"),
      row("Process temperature", fmt(s.processTemperature, 1) + " K (" + celsius(s.processTemperature) + " °C)"),
      row("Rotational speed", Math.round(s.rotationalSpeed) + " rpm"),
      row("Torque", fmt(s.torque, 1) + " Nm"),
      row("Tool wear", Math.round(s.toolWear) + " min"),
      row("Calculated power", fmtInt(pm.powerWatts) + " W"),
      row("Thermal delta", fmt(pm.tempDelta, 1) + " K"),
      row("Overstrain product", fmtInt(pm.overstrain) + " min·Nm")
    ].join("") + "</table>";
    const predTable = "<table class=\"report-table\">" + [
      row("Model", "Random Forest · 300 trees (real live inference)"),
      row("Failure risk", fmt(state.risk, 1) + "%"),
      row("Machine health", Math.round(state.health) + " / 100"),
      row("Classification", statusWord(state.status.key) + " (warn ≥ " + state.settings.warnThreshold + "%, crit ≥ " + state.settings.critThreshold + "%)"),
      row("Generated", nowFull())
    ].join("") + "</table>";
    const indTable = "<table class=\"report-table\">" + ind.map((x) =>
      row(x.code + " — " + x.label, fmt(x.pct, 1) + "% (" + indicatorLevel(x.pct).label + ")")).join("") + "</table>";
    const contribTable = (contrib.length ? contrib.map((c) =>
      row(c.label, "current " + String(c.currentText) + (c.unit ? " " + c.unit : "") +
        " · baseline " + String(c.baselineText) + (c.unit ? " " + c.unit : "") +
        " · Δrisk " + (c.deltaPp > 0 ? "+" : "") + fmt(c.deltaPp, 1) + " pp")).join("")
      : "<tr><td colspan=\"2\">Feature contribution not available.</td></tr>");
    const recList = recs.length ? "<ul class=\"report-list\">" + recs.map((r) =>
      "<li><strong>" + esc(priorityToDisplay(r.priority).label) + "</strong> — " + esc(r.action) + ": " + esc(r.reason) + "</li>").join("") + "</ul>"
      : "<p class=\"report-muted\">No advisory recommendations are currently active.</p>";
    const statsTable = "<table class=\"report-table\">" + [
      row("Recorded events (session)", fmtInt(stats.total)),
      row("Normal / Warning / Critical", stats.counts.NORMAL + " / " + stats.counts.WARNING + " / " + stats.counts.CRITICAL),
      row("Average risk", stats.avg == null ? "—" : fmt(stats.avg, 1) + "%"),
      row("Peak risk", stats.max == null ? "—" : fmt(stats.max, 1) + "%"),
      row("Session window", stats.total ? stats.first + " → " + stats.last : "—")
    ].join("") + "</table>";
    return `
      <div class="report">
        <div class="report-head">
          <h4>Machine Failure Prediction V4.0 &mdash; Machine Report</h4>
          <div class="report-meta">Asset ${esc(state.machine.id)} · generated locally in this browser · ${esc(nowFull())}</div>
        </div>
        <section><h5>Current Reading (simulated inputs)</h5>${sensorTable}</section>
        <section><h5>Random Forest Prediction (real model output)</h5>${predTable}</section>
        <section><h5>Advisory Indicators (rule-based &mdash; NOT model outputs)</h5>${indTable}</section>
        <section><h5>Feature Contribution (ablation vs nominal baseline)</h5><table class="report-table">${contribTable}</table></section>
        <section><h5>Advisory Recommendations</h5>${recList}</section>
        <section><h5>Session Statistics</h5>${statsTable}</section>
        <section class="report-notes"><h5>Honesty &amp; Scope Notes</h5>
          <ul>
            <li>Risk values are real inference from the bundled 300-tree Random Forest; they are not fabricated or estimated.</li>
            <li>Decision Tree and SVM were benchmark-only in the training study; apart from the documented RF F1 (0.691) and ROC-AUC (0.969), their metrics are N/A.</li>
            <li>Telemetry, scenarios, demo steps and fleet profiles are simulated &mdash; there is no live factory connection.</li>
            <li>No remaining-useful-life model exists in this project; no remaining-time estimates are produced.</li>
          </ul>
        </section>
      </div>`;
  }
  function generateReport() {
    const modal = $("reportModal");
    const body = $("reportBody");
    if (!modal || !body) return;
    body.innerHTML = reportHtml();
    modal.classList.remove("hidden");
    toast("Machine report generated from real session values.", "ok");
  }
  function closeReport() {
    const modal = $("reportModal");
    if (modal) modal.classList.add("hidden");
  }
  function printReport() {
    if (!state.prediction) { toast("Run a prediction first — nothing to report yet.", "warn"); return; }
    const modal = $("reportModal");
    if (modal && modal.classList.contains("hidden")) generateReport();
    window.setTimeout(() => {
      try { window.print(); } catch (err) { toast("Printing is not available in this browser.", "warn"); }
    }, 160);
  }
  function bindReportControls() {
    const close = $("btnReportClose");
    if (close) close.addEventListener("click", closeReport);
    const print = $("btnReportPrint");
    if (print) print.addEventListener("click", () => {
      try { window.print(); } catch (err) { toast("Printing is not available in this browser.", "warn"); }
    });
    const overlay = $("reportModal");
    if (overlay) overlay.addEventListener("click", (e) => { if (e.target === overlay) closeReport(); });
  }
  function viz3dRender() {
    const canvas = $("viz3dCanvas");
    if (!canvas) return;
    const c = canvasCtx(canvas);
    if (!c) return;
    VIZ3D.canvas = canvas;
    VIZ3D.w = c.w;
    VIZ3D.h = c.h;
    if (VIZ3D.showRisk === undefined) VIZ3D.showRisk = true;
    const ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    ctx.fillStyle = "rgba(10,15,28,0.6)";
    ctx.fillRect(0, 0, c.w, c.h);
    if (!modelReady()) {
      ctx.fillStyle = "rgba(148,163,184,0.85)";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("3D map requires the bundled Random Forest (model-data.js).", c.w / 2, c.h / 2);
      return;
    }
    viz3dEnsureGrid();
    const grid = VIZ3D.grid || [];
    const warn = state.settings.warnThreshold;
    const crit = state.settings.critThreshold;
    const axes = [
      { from: viz3dProject(VIZ_AXES.wear.min, VIZ_AXES.torque.min, VIZ_AXES.proc.min), to: viz3dProject(VIZ_AXES.wear.max, VIZ_AXES.torque.min, VIZ_AXES.proc.min), label: "TOOL WEAR →" },
      { from: viz3dProject(VIZ_AXES.wear.min, VIZ_AXES.torque.min, VIZ_AXES.proc.min), to: viz3dProject(VIZ_AXES.wear.min, VIZ_AXES.torque.max, VIZ_AXES.proc.min), label: "TORQUE ↑" },
      { from: viz3dProject(VIZ_AXES.wear.min, VIZ_AXES.torque.min, VIZ_AXES.proc.min), to: viz3dProject(VIZ_AXES.wear.min, VIZ_AXES.torque.min, VIZ_AXES.proc.max), label: "PROC TEMP →" }
    ];
    ctx.save();
    ctx.strokeStyle = "rgba(148,163,184,0.35)";
    ctx.lineWidth = 1.2;
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillStyle = "rgba(148,163,184,0.75)";
    axes.forEach((a) => {
      ctx.beginPath();
      ctx.moveTo(a.from.x, a.from.y);
      ctx.lineTo(a.to.x, a.to.y);
      ctx.stroke();
      ctx.fillText(a.label, a.to.x + 6, a.to.y + 4);
    });
    ctx.restore();
    const projected = grid.map((p) => {
      const pr = viz3dProject(p.w, p.t, p.p);
      return { x: pr.x, y: pr.y, z: pr.z, s: pr.s, risk: p.risk };
    }).sort((a, b) => b.z - a.z);
    projected.forEach((pt) => {
      const isCrit = pt.risk >= crit;
      const isWarn = !isCrit && pt.risk >= warn;
      if ((isCrit && !VIZ3D.showCritical) || (isWarn && !VIZ3D.showRisk) || (!isCrit && !isWarn && !VIZ3D.showNormal)) return;
      ctx.globalAlpha = clamp(0.22 + pt.s * 0.55, 0.2, 0.85);
      ctx.fillStyle = isCrit ? "#f43f5e" : isWarn ? "#f59e0b" : "#10b981";
      const size = 2.2 + pt.s * 3.4;
      ctx.fillRect(pt.x - size / 2, pt.y - size / 2, size, size);
    });
    ctx.globalAlpha = 1;
    if (VIZ3D.showCurrent) {
      const s = state.sensors;
      const cur = viz3dProject(s.toolWear, s.torque, s.processTemperature);
      ctx.save();
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 16;
      ctx.beginPath();
      ctx.arc(cur.x, cur.y, 6.4, 0, Math.PI * 2);
      ctx.fillStyle = "#38bdf8";
      ctx.fill();
      ctx.restore();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cur.x, cur.y, 10.6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px Outfit, sans-serif";
      ctx.fillText("CURRENT · " + fmt(state.risk, 1) + "%", cur.x + 15, cur.y - 9);
    }
    ctx.fillStyle = "rgba(148,163,184,0.85)";
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("green < " + warn + "%   ·   amber " + warn + "–" + crit + "%   ·   red ≥ " + crit + "%", 10, 16);
    ctx.fillText("Background: " + grid.length + " simulated Random Forest evaluations · drag to rotate · scroll to zoom", 10, c.h - 8);
  }
  /* ================================ BOOT ================================= */
  function onReady() {
    if (state.booted) return;
    state.booted = true;
    renderPresets();
    renderHistoricalTable();
    bindGlobalControls();
    bindXai();
    bindViz3D();
    loadHistory();
    applySettingsToUI();
    updateSystemStatus();
    if (!modelReady()) { modelUnavailable(); return; }
    runPipeline("init");
    v6Init();
    if (state.settings.autoRun) {
      startSimulation();
      if (state.view === "live") switchView("dashboard");
    }
    console.log("Machine Failure Prediction V4.0 ready. Random Forest trees active:", modelReady() ? "yes" : "no");
  }
  function boot() {
    try {
      onReady();
    } catch (err) {
      console.error("Boot failed:", err);
      const chip = $("sysStatusChip");
      if (chip) {
        chip.classList.add("chip-bad");
        chip.innerHTML = '<span class="sys-dot"></span> SYSTEM ERROR';
      }
      toast("Startup error — see the browser console for details.", "error");
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
  function updateViz3D() {
    if (state.view !== "viz3d") return;
    viz3dRender();
    const s = state.sensors;
    const box = $("viz3dReading");
    if (box) {
      box.innerHTML = `
        <div class="xai-sum-grid">
          <div class="xai-sum-item"><span class="xai-sum-label">Marker source</span>
            <span class="xai-sum-value">${esc(typeLabel(s.productType))} · wear ${Math.round(s.toolWear)} min · torque ${fmt(s.torque, 1)} Nm · proc ${fmt(s.processTemperature, 1)} K</span></div>
          <div class="xai-sum-item"><span class="xai-sum-label">Random Forest risk</span>
            <span class="xai-sum-value">${fmt(state.risk, 1)}% (${esc(statusWord(state.status.key))})</span></div>
          <div class="xai-sum-item"><span class="xai-sum-label">Grid conditioning</span>
            <span class="xai-sum-value">AirTemp ${fmt(s.airTemperature, 1)} K · RPM ${Math.round(s.rotationalSpeed)} held fixed</span></div>
        </div>`;
    }
    setText("viz3dPill", VIZ3D.computing ? "COMPUTING GRID…" : "SIMULATED GRID · LIVE RF");
  }
  function bindViz3D() {
    const canvas = $("viz3dCanvas");
    if (!canvas) return;
    canvas.addEventListener("mousedown", (e) => {
      VIZ3D.dragging = true;
      VIZ3D.lastX = e.clientX;
      VIZ3D.lastY = e.clientY;
    });
    window.addEventListener("mouseup", () => { VIZ3D.dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!VIZ3D.dragging) return;
      const dx = e.clientX - VIZ3D.lastX;
      const dy = e.clientY - VIZ3D.lastY;
      VIZ3D.lastX = e.clientX;
      VIZ3D.lastY = e.clientY;
      VIZ3D.rotY += dx * 0.008;
      VIZ3D.rotX = clamp(VIZ3D.rotX - dy * 0.006, -1.3, 1.3);
      viz3dRender();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      VIZ3D.zoom = clamp(VIZ3D.zoom * (e.deltaY > 0 ? 0.92 : 1.08), 0.4, 2.6);
      viz3dRender();
    }, { passive: false });
    canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        VIZ3D.dragging = true;
        VIZ3D.lastX = e.touches[0].clientX;
        VIZ3D.lastY = e.touches[0].clientY;
      }
    }, { passive: true });
    canvas.addEventListener("touchmove", (e) => {
      if (!VIZ3D.dragging || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - VIZ3D.lastX;
      const dy = e.touches[0].clientY - VIZ3D.lastY;
      VIZ3D.lastX = e.touches[0].clientX;
      VIZ3D.lastY = e.touches[0].clientY;
      VIZ3D.rotY += dx * 0.01;
      VIZ3D.rotX = clamp(VIZ3D.rotX - dy * 0.008, -1.3, 1.3);
      viz3dRender();
    }, { passive: true });
    canvas.addEventListener("touchend", () => { VIZ3D.dragging = false; }, { passive: true });
    canvas.addEventListener("dblclick", () => {
      VIZ3D.rotX = -0.42;
      VIZ3D.rotY = 0.72;
      VIZ3D.zoom = 1;
      viz3dRender();
    });
    const zoomIn = $("btnVizZoomIn");
    if (zoomIn) zoomIn.addEventListener("click", () => { VIZ3D.zoom = clamp(VIZ3D.zoom * 1.15, 0.4, 2.6); viz3dRender(); });
    const zoomOut = $("btnVizZoomOut");
    if (zoomOut) zoomOut.addEventListener("click", () => { VIZ3D.zoom = clamp(VIZ3D.zoom * 0.88, 0.4, 2.6); viz3dRender(); });
    const reset = $("btnVizReset");
    if (reset) reset.addEventListener("click", () => {
      VIZ3D.rotX = -0.42;
      VIZ3D.rotY = 0.72;
      VIZ3D.zoom = 1;
      viz3dRender();
      toast("3D camera reset.", "info");
    });
    const refresh = $("btnVizRefresh");
    if (refresh) refresh.addEventListener("click", () => {
      VIZ3D.grid = null;
      VIZ3D.gridKey = "";
      viz3dRender();
      toast("Operating-space grid re-evaluated with the current holding values.", "ok");
    });
    const toggles = [
      ["vizShowNormal", "showNormal"],
      ["vizShowRisk", "showRisk"],
      ["vizShowCritical", "showCritical"],
      ["vizShowCurrent", "showCurrent"]
    ];
    toggles.forEach((pair) => {
      const node = $(pair[0]);
      if (node) node.addEventListener("change", () => { VIZ3D[pair[1]] = !!node.checked; viz3dRender(); });
    });
  }
/* ==========================================================================
     MACHINE INTELLIGENCE V8.0 — DIGITAL TWIN + COMMAND-CENTER PRESENTATION
     ------------------------------------------------------------------------
     This block only renders the interface. It never invents data: every risk
     value, status band, sensor reading and model output drawn here comes from
     state.* / MODEL_DATA through the existing central pipeline.
     ========================================================================== */
  const CC = {
    band: "", lastEventAt: 0, scanTimer: 0, scanAt: 0,
    replayTimer: null, replayIdx: -1, replayRisks: {}, presenting: false,
    clockTimer: 0, selectedSensor: ""
  };
  function ccText(id, value) {
    const node = $(id);
    if (node && value != null && node.textContent !== String(value)) node.textContent = String(value);
  }
  /* Smooth number easing — only re-renders the REAL value, never a fake one. */
  function ccNumber(id, target, digits, suffix) {
    const node = $(id);
    if (!node) return;
    const d = digits == null ? 1 : digits;
    const to = Number.isFinite(Number(target)) ? Number(target) : 0;
    const key = "__ccNum";
    const from = typeof node[key] === "number" ? node[key] : to;
    node[key] = to;
    if (Math.abs(to - from) < Math.pow(10, -d) / 2) { node.textContent = fmt(to, d) + (suffix || ""); return; }
    const start = performance.now();
    const dur = 420;
    const tick = (now) => {
      const k = clamp((now - start) / dur, 0, 1);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt(from + (to - from) * e, d) + (suffix || "");
      if (k < 1 && node[key] === to) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  const CC_VIEW_LABELS = {
    dashboard: "Overview", live: "Live AI", prediction: "Predict", viz3d: "Digital Twin",
    models: "Models", xai: "Explain", analytics: "Analytics", history: "History",
    dataset: "Dataset", scenarios: "Scenarios", whatif: "Scenario Lab", fleet: "Fleet", maintenance: "Maintenance", settings: "System"
  };
  function ccViewLabel(v) { return CC_VIEW_LABELS[v] || String(v); }
  /* The twin stage is shared: Overview hero <-> Digital Twin view. */
  function ccMoveTwin(viewName) {
    const stage = $("twinStage");
    if (!stage) return;
    const target = viewName === "viz3d" ? $("twinVizHolder") : viewName === "whatif" ? $("twinWhatIfHolder") : $("twinHeroHolder");
    if (target && stage.parentElement !== target) target.appendChild(stage);
  }
  /* ---- AI event stream (real user actions only) ---- */
  function ccEvent(message, band) {
    const host = $("evList");
    if (!host || !message) return;
    const now = Date.now();
    if (now - CC.lastEventAt < 240) return;
    CC.lastEventAt = now;
    const first = host.querySelector(".ev-item");
    if (first && (first.querySelector(".ev-time") || {}).textContent === "--:--:--") host.innerHTML = "";
    const item = document.createElement("div");
    item.className = "ev-item" + (band === "warning" ? " ev-warn" : band === "critical" ? " ev-crit" : "");
    const t = document.createElement("span");
    t.className = "ev-time";
    t.textContent = nowStamp();
    const pip = document.createElement("span");
    pip.className = "ev-pip";
    const msg = document.createElement("span");
    msg.className = "ev-msg";
    msg.textContent = String(message);
    item.appendChild(t); item.appendChild(pip); item.appendChild(msg);
    host.insertBefore(item, host.firstChild);
    while (host.children.length > 9) host.removeChild(host.lastChild);
  }
  /* ---- prediction scan sequence (presentation only; result is the real one) ---- */
  function ccScan() {
    const overlay = $("scanOverlay");
    if (!overlay) return;
    const items = [].slice.call(overlay.querySelectorAll(".scan-steps li"));
    if (!items.length) return;
    if (Date.now() - CC.scanAt < 900) return;
    CC.scanAt = Date.now();
    overlay.classList.remove("hidden");
    items.forEach((li) => { li.classList.remove("on"); li.classList.remove("done"); });
    if (CC.scanTimer) window.clearInterval(CC.scanTimer);
    let i = 0;
    CC.scanTimer = window.setInterval(() => {
      if (i > 0) { items[i - 1].classList.remove("on"); items[i - 1].classList.add("done"); }
      if (i >= items.length) {
        window.clearInterval(CC.scanTimer);
        CC.scanTimer = 0;
        window.setTimeout(() => overlay.classList.add("hidden"), 240);
        return;
      }
      items[i].classList.add("on");
      i++;
    }, 170);
  }
  /* ---- presentation mode ---- */
  function ccSetPresentation(on) {
    CC.presenting = on == null ? !CC.presenting : !!on;
    document.body.classList.toggle("presentation", CC.presenting);
    const btn = $("btnPresentation");
    if (btn) btn.innerHTML = CC.presenting ? "&#9673; EXIT PRESENTATION" : "&#9974; PRESENTATION MODE";
    ccEvent(CC.presenting ? "PRESENTATION MODE ENABLED" : "PRESENTATION MODE DISABLED");
  }
  function ccClock() {
    const node = $("ccClock");
    if (node) node.textContent = nowStamp();
  }
/* ============================== DIGITAL TWIN ============================= */
  const TWIN = {
    rotY: 0.62, rotX: -0.34, zoom: 1, autoRotate: true, interactive: true,
    dragging: false, moved: false, lastX: 0, lastY: 0, idleAt: 0,
    shaft: 0, fan: 0, gear: 0, vib: 0, t: 0,
    faces: [], spinners: [], screen: {}, raf: 0, last: 0, started: false
  };
  const TWIN_COLORS = {
    steel: "#18233A", steel2: "#212E4B", dark: "#0B1220", bright: "#31456F",
    metal: "#8FA6C8", glass: "#0E5A7A", warning: "#FFB020", critical: "#FF3D71", cyan: "#00E5FF"
  };
  function twFace(points, color, opts) {
    const o = opts || {};
    return { p: points, c: color, e: !!o.emissive, a: o.alpha == null ? 1 : o.alpha, n: o.normal || [0, 0, 1] };
  }
  function twBox(x0, y0, z0, x1, y1, z1, color, opts) {
    const o = opts || {};
    const v = [
      [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ];
    const defs = [
      [[v[0], v[1], v[2], v[3]], [0, 0, -1]], [[v[5], v[4], v[7], v[6]], [0, 0, 1]],
      [[v[4], v[0], v[3], v[7]], [-1, 0, 0]], [[v[1], v[5], v[6], v[2]], [1, 0, 0]],
      [[v[3], v[2], v[6], v[7]], [0, 1, 0]], [[v[4], v[5], v[1], v[0]], [0, -1, 0]]
    ];
    return defs.map((d) => twFace(d[0], color, { emissive: o.emissive, alpha: o.alpha, normal: d[1] }));
  }
  function twCylinderFaces(cx, cz, y0, y1, r, color, seg, emissive) {
    const out = [];
    const n = seg || 14;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const b = ((i + 1) / n) * Math.PI * 2;
      const x0 = cx + Math.cos(a) * r, z0 = cz + Math.sin(a) * r;
      const x1 = cx + Math.cos(b) * r, z1 = cz + Math.sin(b) * r;
      out.push(twFace([[x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]], color,
        { emissive: emissive, normal: [Math.cos((a + b) / 2), 0, Math.sin((a + b) / 2)] }));
    }
    return out;
  }
  function twGear(cx, cy, z, r, teeth, color, phase) {
    const pts = [];
    for (let i = 0; i < teeth * 2; i++) {
      const a = (i / (teeth * 2)) * Math.PI * 2 + phase;
      const rad = i % 2 === 0 ? r : r * 0.74;
      pts.push([cx + Math.cos(a) * rad, cy + Math.sin(a) * rad, z]);
    }
    return twFace(pts, color, { normal: [0, 0, 1] });
  }
  function twBlade(cx, cy, z, a0, len, wide, color) {
    const p = [];
    const pts = [[0, 0], [len, -wide], [len * 1.15, 0], [len, wide]];
    pts.forEach((pt) => {
      const a = a0;
      p.push([cx + Math.cos(a) * pt[0] - Math.sin(a) * pt[1], cy + Math.sin(a) * pt[0] + Math.cos(a) * pt[1], z]);
    });
    return twFace(p, color, { normal: [0, 0, 1] });
  }
  function twBuildMachine() {
    const F = [];
    const C = TWIN_COLORS;
    // bed / base
    F.push.apply(F, twBox(-1.75, -1.34, -1.0, 1.75, -1.04, 1.0, C.steel));
    F.push.apply(F, twBox(-1.75, -1.05, 0.9, 1.75, -0.99, 1.0, C.cyan, { emissive: true })); // front status light bar
    // column
    F.push.apply(F, twBox(-1.58, -1.0, -0.52, -1.0, 1.34, 0.46, C.steel2));
    F.push.apply(F, twBox(-1.6, -0.5, -0.28, -1.56, 0.9, 0.18, C.bright, { emissive: true, alpha: 0.8 }));
    F.push.apply(F, twBox(-1.34, 0.16, 0.46, -1.06, 0.9, 0.52, C.dark, { alpha: 0.9 })); // control panel
    F.push.apply(F, twBox(-1.32, 0.24, 0.52, -1.2, 0.5, 0.54, C.cyan, { emissive: true, alpha: 0.75 })); // panel readout
    // ram / head
    F.push.apply(F, twBox(-1.02, 0.62, -0.44, 0.44, 1.24, 0.44, C.steel2));
    F.push.apply(F, twBox(-0.98, 0.54, -0.4, 0.4, 0.62, 0.4, C.bright, { emissive: true, alpha: 0.5 }));
    // spindle housing
    F.push.apply(F, twBox(-0.1, 0.18, -0.28, 0.42, 0.6, 0.3, C.bright));
    // table + workpiece
    F.push.apply(F, twBox(-0.66, -1.02, -0.82, 1.5, -0.88, 0.82, C.steel2));
    F.push.apply(F, twBox(-0.34, -0.88, -0.32, 0.34, -0.74, 0.32, C.glass, { emissive: true, alpha: 0.75 }));
    // coolant nozzle + chip guard
    F.push.apply(F, twBox(0.62, -0.86, -0.3, 0.7, -0.2, -0.22, C.metal));
    F.push.apply(F, twBox(0.86, -1.0, -0.6, 0.92, -0.34, 0.6, C.bright, { emissive: true, alpha: 0.35 }));
    TWIN.faces = F;
    TWIN.spinners = [
      { kind: "cyl", cx: 0.16, cz: 0.02, y0: 0.1, y1: -0.22, r: 0.12, color: C.metal, speed: 5.2 },
      { kind: "cyl", cx: 0.16, cz: 0.02, y0: -0.22, y1: -0.38, r: 0.17, color: C.bright, speed: 5.2 },
      { kind: "cyl", cx: 0.16, cz: 0.02, y0: -0.38, y1: -0.74, r: 0.075, color: C.metal, speed: 8.4 },
      { kind: "disc", cx: -1.29, cy: 0.34, z: 0.48, r: 0.28, color: "#0D1A2E", blades: 5, speed: -6.5 },
      { kind: "gear", cx: -1.18, cy: -1.17, z: 1.01, r: 0.3, teeth: 10, color: "#7E93B8", speed: 3.1 },
      { kind: "gear", cx: -0.66, cy: -1.17, z: 1.01, r: 0.21, teeth: 7, color: "#4C6B9E", speed: -4.6 }
    ];
    TWIN.sensorDefs = [
      { key: "air", label: "AIR TEMP", unit: "K", p: [-1.52, -1.16, 1.02] },
      { key: "proc", label: "PROCESS TEMP", unit: "K", p: [0.62, -0.9, 0.7] },
      { key: "rpm", label: "SPINDLE RPM", unit: "rpm", p: [0.18, 1.28, 0.08] },
      { key: "torque", label: "TORQUE", unit: "Nm", p: [-0.62, 0.3, 0.52] },
      { key: "wear", label: "TOOL WEAR", unit: "min", p: [0.16, -0.8, 0.16] }
    ];
    TWIN.particles = [];
    for (let i = 0; i < 26; i++) {
      TWIN.particles.push({ t: Math.random(), sp: 0.18 + Math.random() * 0.22, j: Math.random() * 6.28 });
    }
  }
function twBand() { return (state && state.status) ? state.status.key : "normal"; }
  function twBandColors(band) {
    if (band === "critical") return { main: "#FF3D71", alt: "#FF7A3D", glow: "rgba(255,61,113,0.55)" };
    if (band === "warning") return { main: "#FFB020", alt: "#7C4DFF", glow: "rgba(255,176,32,0.5)" };
    return { main: "#00E5FF", alt: "#2979FF", glow: "rgba(0,229,255,0.45)" };
  }
  function twHex(c) {
    if (typeof c !== "string" || c.charAt(0) !== "#") return [128, 128, 128];
    const h = c.length === 4 ? "#" + c[1] + c[1] + c[2] + c[2] + c[3] + c[3] : c.slice(0, 7);
    const n = parseInt(h.slice(1), 16);
    if (!Number.isFinite(n)) return [128, 128, 128];
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function twProject(p, w, h) {
    const cy = Math.cos(TWIN.rotY), sy = Math.sin(TWIN.rotY);
    const cx = Math.cos(TWIN.rotX), sx = Math.sin(TWIN.rotX);
    const x1 = p[0] * cy - p[2] * sy;
    const z1 = p[0] * sy + p[2] * cy;
    const y1 = p[1] * cx - z1 * sx;
    const z2 = p[1] * sx + z1 * cx;
    const dist = 4.6;
    const persp = dist / (dist + z2);
    const s = TWIN.zoom * Math.min(w, h) * 0.31 * persp;
    return { x: w / 2 + x1 * s, y: h * 0.46 - y1 * s, z: z2 };
  }
  function twRotNormal(n) {
    const cy = Math.cos(TWIN.rotY), sy = Math.sin(TWIN.rotY);
    const cx = Math.cos(TWIN.rotX), sx = Math.sin(TWIN.rotX);
    const x1 = n[0] * cy - n[2] * sy;
    const z1 = n[0] * sy + n[2] * cy;
    return [x1, n[1] * cx - z1 * sx, n[1] * sx + z1 * cx];
  }
  function twDynamicFaces(band) {
    const out = [];
    const cols = twBandColors(band);
    TWIN.spinners.forEach((sp) => {
      if (sp.kind === "cyl") {
        const seg = 14, step = (Math.PI * 2) / seg;
        for (let i = 0; i < seg; i++) {
          const a = i * step + TWIN.shaft, b = (i + 1) * step + TWIN.shaft;
          const marked = (i % 7 === 0);
          out.push(twFace([
            [sp.cx + Math.cos(a) * sp.r, sp.y0, sp.cz + Math.sin(a) * sp.r],
            [sp.cx + Math.cos(b) * sp.r, sp.y0, sp.cz + Math.sin(b) * sp.r],
            [sp.cx + Math.cos(b) * sp.r, sp.y1, sp.cz + Math.sin(b) * sp.r],
            [sp.cx + Math.cos(a) * sp.r, sp.y1, sp.cz + Math.sin(a) * sp.r]
          ], marked ? cols.main : sp.color, { emissive: marked, alpha: marked ? 0.95 : 1, normal: [Math.cos((a + b) / 2), 0, Math.sin((a + b) / 2)] }));
        }
      } else if (sp.kind === "disc") {
        const ring = [];
        for (let i = 0; i < 14; i++) { const a = (i / 14) * Math.PI * 2; ring.push([sp.cx + Math.cos(a) * sp.r, sp.cy + Math.sin(a) * sp.r, sp.z]); }
        out.push(twFace(ring, sp.color, { normal: [0, 0, 1] }));
        for (let i = 0; i < sp.blades; i++) {
          out.push(twBlade(sp.cx, sp.cy, sp.z + 0.01, TWIN.fan + (i / sp.blades) * Math.PI * 2, sp.r * 0.92, sp.r * 0.22, i === 0 ? cols.main : "#6C81A6"));
        }
        const hub = [];
        for (let i = 0; i < 8; i++) { const a = (i / 8) * Math.PI * 2; hub.push([sp.cx + Math.cos(a) * sp.r * 0.18, sp.cy + Math.sin(a) * sp.r * 0.18, sp.z + 0.02]); }
        out.push(twFace(hub, cols.main, { emissive: true, normal: [0, 0, 1] }));
      } else if (sp.kind === "gear") {
        out.push(twGear(sp.cx, sp.cy, sp.z, sp.r, sp.teeth, sp.color, TWIN.gear * (sp.speed > 0 ? 0.42 : -0.62)));
      }
    });
    return out;
  }
function twDrawFloor(ctx, w, h, cols) {
    ctx.lineWidth = 1;
    for (let i = -6; i <= 6; i++) {
      const k = i * 0.5;
      const a = twProject([k, -1.36, -3.2], w, h);
      const b = twProject([k, -1.36, 3.2], w, h);
      const d = twProject([-3.2, -1.36, k], w, h);
      const e = twProject([3.2, -1.36, k], w, h);
      ctx.strokeStyle = "rgba(0,229,255,0.07)";
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(e.x, e.y); ctx.stroke();
    }
    const c0 = twProject([0, -1.36, 0], w, h);
    const rad = Math.min(w, h) * 0.2 * TWIN.zoom;
    const pulse = 0.5 + 0.5 * Math.sin(TWIN.t * (twBand() === "critical" ? 5 : 1.8));
    const g = ctx.createRadialGradient(c0.x, c0.y, rad * 0.2, c0.x, c0.y, rad * 1.5);
    g.addColorStop(0, cols.glow.replace(/[\d.]+\)$/, (0.16 + pulse * 0.14).toFixed(3) + ")"));
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(c0.x, c0.y, rad * 1.6, rad * 0.6, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  function twGlow(ctx, w, h, p, radius, color, alpha) {
    const q = twProject(p, w, h);
    const g = ctx.createRadialGradient(q.x, q.y, 1, q.x, q.y, radius);
    g.addColorStop(0, color.replace(/[\d.]+\)$/, alpha.toFixed(3) + ")"));
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
    ctx.fill();
    return q;
  }
  function twDrawMachine(ctx, w, h, cols, vx, vy) {
    const list = TWIN.faces.concat(twDynamicFaces(twBand()));
    const draw = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      const pts = f.p.map((p) => {
        const q = twProject(p, w, h);
        return { x: q.x + vx, y: q.y + vy, z: q.z };
      });
      let zsum = 0;
      for (let j = 0; j < pts.length; j++) zsum += pts[j].z;
      draw.push({ pts: pts, f: f, z: zsum / pts.length });
    }
    draw.sort((a, b) => b.z - a.z);
    const L = [0.42, 0.72, 0.55];
    draw.forEach((item) => {
      const f = item.f;
      const n = twRotNormal(f.n);
      const shade = clamp(0.38 + 0.62 * Math.max(0, n[0] * L[0] + n[1] * L[1] + n[2] * L[2]), 0, 1.3);
      const rgb = twHex(f.c);
      const g0 = f.e ? 0.45 : 0;
      const r = Math.round(clamp(rgb[0] * shade * (1 - g0) + 10 * g0, 0, 255));
      const g = Math.round(clamp(rgb[1] * shade * (1 - g0) + 14 * g0, 0, 255));
      const b = Math.round(clamp(rgb[2] * shade * (1 - g0) + 20 * g0, 0, 255));
      ctx.globalAlpha = clamp(f.a, 0, 1);
      ctx.beginPath();
      ctx.moveTo(item.pts[0].x, item.pts[0].y);
      for (let j = 1; j < item.pts.length; j++) ctx.lineTo(item.pts[j].x, item.pts[j].y);
      ctx.closePath();
      ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
      ctx.fill();
      if (f.e) {
        ctx.strokeStyle = cols.glow;
        ctx.lineWidth = 1.1;
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }
function twDrawFx(ctx, w, h, cols, band, dt) {
    const s = state.sensors || {};
    const stress = clamp((Number(state.risk) || 0) / 100, 0, 1);
    const heat = clamp(((Number(s.processTemperature) || 305) - 310) / 5, 0, 1);
    if (heat > 0.02) twGlow(ctx, w, h, [0, -0.76, 0], Math.min(w, h) * 0.2 * (0.6 + heat), "rgba(255,120,60,0.95)", 0.1 + heat * 0.3);
    const torque = clamp(((Number(s.torque) || 0) - 45) / 35, 0, 1);
    if (torque > 0.02) twGlow(ctx, w, h, [-1.24, 0.34, 0.42], Math.min(w, h) * 0.16 * (0.6 + torque), "rgba(255,176,32,0.95)", 0.08 + torque * 0.26);
    const wear = clamp(((Number(s.toolWear) || 0) - 150) / 110, 0, 1);
    if (wear > 0.02) twGlow(ctx, w, h, [0.16, -0.78, 0.16], Math.min(w, h) * 0.12 * (0.6 + wear), "rgba(255,61,113,0.95)", 0.1 + wear * 0.3);
    const from = twProject([0.16, -0.68, 0.2], w, h);
    const to = twProject([1.42, 0.92, -0.3], w, h);
    const mid = twProject([0.78, 0.28, 0.1], w, h);
    ctx.save();
    ctx.setLineDash([7, 9]);
    ctx.lineDashOffset = -TWIN.t * 44;
    ctx.strokeStyle = cols.glow;
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.quadraticCurveTo(mid.x, mid.y, to.x, to.y); ctx.stroke();
    ctx.setLineDash([]);
    TWIN.particles.forEach((p) => {
      p.t += dt * p.sp * (1 + stress * 1.2);
      if (p.t > 1) p.t -= 1;
      const a = p.t, b = 1 - p.t;
      const x = b * b * from.x + 2 * b * a * mid.x + a * a * to.x;
      const y = b * b * from.y + 2 * b * a * mid.y + a * a * to.y;
      const alpha = Math.sin(p.t * Math.PI);
      ctx.fillStyle = "rgba(" + (p.t > 0.5 ? "124,77,255" : "0,229,255") + "," + alpha.toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(x, y, 1.2 + alpha * 1.8, 0, Math.PI * 2);
      ctx.fill();
    });
    // AI engine node at the end of the data flow
    const nodePulse = 1 + 0.18 * Math.sin(TWIN.t * 3.4);
    ctx.strokeStyle = cols.main;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const ang = (i / 6) * Math.PI * 2 + TWIN.t * 0.5;
      const px = to.x + Math.cos(ang) * 13 * nodePulse;
      const py = to.y + Math.sin(ang) * 13 * nodePulse;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "rgba(11,18,32,0.85)";
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#F5F7FF";
    ctx.font = "600 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("RF 300T", to.x, to.y + 3);
    ctx.restore();
    // sensor nodes
    (TWIN.sensorDefs || []).forEach((sd) => {
      const q = twProject(sd.p, w, h);
      TWIN.screen[sd.key] = { x: q.x, y: q.y };
      const lvl = ccSensorLevel(sd.key);
      const rate = 1.4 + (lvl.key === "critical" ? 4.6 : lvl.key === "warning" ? 2.4 : 0);
      const pulse = (Math.sin(TWIN.t * rate) + 1) / 2;
      const col = lvl.key === "critical" ? "#FF3D71" : lvl.key === "warning" ? "#FFB020" : "#00E5FF";
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.22 + pulse * 0.6;
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(q.x, q.y, 5 + pulse * 6, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(q.x, q.y, 2.6, 0, Math.PI * 2); ctx.fill();
      if (CC.selectedSensor === sd.key) {
        ctx.setLineDash([4, 4]);
        ctx.lineDashOffset = -TWIN.t * 22;
        ctx.strokeStyle = "#F5F7FF";
        ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.arc(q.x, q.y, 15, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#F5F7FF";
        ctx.font = "600 9px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(sd.label, q.x + 12, q.y - 8);
      }
    });
  }
  function twFrame(ts) {
    TWIN.raf = window.requestAnimationFrame(twFrame);
    const canvas = $("twinCanvas");
    const stage = canvas ? canvas.parentElement : null;
    if (!canvas || !stage || !stage.offsetParent || document.hidden) { TWIN.last = ts; return; }
    const c = canvasCtx(canvas);
    if (!c) { TWIN.last = ts; return; }
    const ctx = c.ctx, w = c.w, h = c.h;
    const dt = clamp((ts - (TWIN.last || ts)) / 1000, 0, 0.05);
    TWIN.last = ts;
    TWIN.t += dt;
    // Scenario Lab: while the lab is open the twin renders the SCENARIO values so it
    // reacts to what-if conditions. state.sensors is never modified, so the live
    // pipeline, prediction history and alerts stay exactly as they were.
    const scenActive = !!(TWIN.scenarioPreview && state.view === "whatif");
    const s = scenActive ? TWIN.scenarioPreview : (state.sensors || {});
    const previewRisk = scenActive
      ? (WI.scenario ? WI.scenario.risk : (WI.baseline ? WI.baseline.risk : state.risk))
      : state.risk;
    const band = scenActive ? wiLevelOf(previewRisk).key : twBand();
    const cols = twBandColors(band);
    const speedFactor = clamp(0.5 + (Number(s.rotationalSpeed) || 0) / 1900, 0.5, 2.1);
    TWIN.shaft += dt * 5.4 * speedFactor;
    TWIN.fan += dt * (5.4 + 4 * clamp((Number(previewRisk) || 0) / 100, 0, 1));
    TWIN.gear += dt * 3.2;
    if (TWIN.autoRotate && !TWIN.dragging && (Date.now() - TWIN.idleAt) > 2600) TWIN.rotY += dt * 0.16;
    const vib = band === "critical" ? 1.7 : band === "warning" ? 0.85 : 0.12;
    const vx = Math.sin(TWIN.t * 37) * vib;
    const vy = Math.cos(TWIN.t * 29) * vib;
    ctx.clearRect(0, 0, w, h);
    twDrawFloor(ctx, w, h, cols);
    twDrawMachine(ctx, w, h, cols, vx, vy);
    /* V9: conceptual feature highlight. Purely visual - it draws a glow, a
       connection line and a HUD label over the existing sensor node. It does
       not move, resize, rotate or otherwise alter the machine. */
    if (state.view === "xai") xaDrawTwinHighlight(ctx, w, h);
    twDrawFx(ctx, w, h, cols, band, dt);
  }
  /* ---- V9 conceptual component highlight (visual only) ----------------- */
  const XA_ZONE_P = {
    wear: [0.16, -0.8, 0.16], torque: [-0.62, 0.3, 0.52], rpm: [0.18, 1.28, 0.08],
    proc: [0.62, -0.9, 0.7], air: [-1.52, -1.16, 1.02]
  };
  function xaDrawTwinHighlight(ctx, w, h) {
    const key = XA.feature ? (xaFeat(XA.feature) || {}).twin : null;
    const p = key ? XA_ZONE_P[key] : null;
    if (!p) return;
    const q = twProject(p, w, h);
    if (!q) return;
    const pulse = (Math.sin(TWIN.t * 2.4) + 1) / 2;
    const col = "0,229,255";
    /* soft glow */
    const g = ctx.createRadialGradient(q.x, q.y, 2, q.x, q.y, 46 + pulse * 16);
    g.addColorStop(0, "rgba(" + col + "," + (0.3 + pulse * 0.16).toFixed(3) + ")");
    g.addColorStop(1, "rgba(" + col + ",0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(q.x, q.y, 46 + pulse * 16, 0, Math.PI * 2);
    ctx.fill();
    /* connection line to the machine centre */
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.lineDashOffset = -TWIN.t * 26;
    ctx.strokeStyle = "rgba(" + col + ",0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(q.x, q.y);
    ctx.lineTo(w * 0.5, h * 0.55);
    ctx.stroke();
    ctx.restore();
    /* pulsing ring */
    ctx.strokeStyle = "rgba(" + col + "," + (0.5 + pulse * 0.4).toFixed(3) + ")";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(q.x, q.y, 13 + pulse * 7, 0, Math.PI * 2);
    ctx.stroke();
    /* HUD label with the CONCEPTUAL zone name */
    const f = xaFeat(XA.feature);
    const label = f ? f.label.toUpperCase() : "";
    const zone = f ? f.zone : "";
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    const tw = Math.max(ctx.measureText(label).width, ctx.measureText(zone).width) + 18;
    const bx = Math.min(Math.max(q.x + 18, 6), w - tw - 6);
    const by = Math.max(q.y - 40, 6);
    ctx.fillStyle = "rgba(11,18,32,0.86)";
    ctx.strokeStyle = "rgba(" + col + ",0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(bx, by, tw, 32);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#E6F6FF";
    ctx.fillText(label, bx + 8, by + 13);
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(" + col + ",0.9)";
    ctx.fillText("~ " + zone, bx + 8, by + 25);
  }
const CC_SENSORS = {
    air: { label: "AIR TEMP", unit: "K", hud: "hudAir", strip: "stripValAir", spark: "stripSparkAir", digits: 1 },
    proc: { label: "PROCESS TEMP", unit: "K", hud: "hudProc", strip: "stripValProc", spark: "stripSparkProc", digits: 1 },
    rpm: { label: "SPINDLE RPM", unit: "rpm", hud: "hudRpm", strip: "stripValRpm", spark: "stripSparkRpm", digits: 0 },
    torque: { label: "TORQUE", unit: "Nm", hud: "hudTorque", strip: "stripValTorque", spark: "stripSparkTorque", digits: 1 },
    wear: { label: "TOOL WEAR", unit: "min", hud: "hudWear", strip: "stripValWear", spark: "stripSparkWear", digits: 0 }
  };
  /* Advisory rule level (TWF / HDF / PWF / OSF) — not a model output. */
  function ccSensorLevel(key) {
    const def = kpiDefs().filter((d) => d.key === key)[0];
    return def ? { key: def.level, label: def.levelLabel } : { key: "normal", label: "NORMAL" };
  }
  function ccSensorValue(key) {
    const s = state.sensors || {};
    const map = { air: s.airTemperature, proc: s.processTemperature, rpm: s.rotationalSpeed, torque: s.torque, wear: s.toolWear };
    return Number(map[key]);
  }
  function ccTrendWord(key) {
    const t = trendOf(state.sparks[key] || [], false);
    if (t.delta == null || Math.abs(t.delta) < 1e-9) return "STABLE";
    return t.delta > 0 ? "RISING" : "FALLING";
  }
  function ccSpark(id, series) {
    const node = $(id);
    if (!node) return;
    const data = (series || []).slice(-26);
    if (data.length < 2) { node.setAttribute("points", ""); return; }
    let min = Infinity, max = -Infinity;
    data.forEach((v) => { const n = Number(v); if (Number.isFinite(n)) { if (n < min) min = n; if (n > max) max = n; } });
    const span = (max - min) || 1;
    const pts = data.map((v, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 22 - ((Number(v) - min) / span) * 20;
      return x.toFixed(1) + "," + y.toFixed(1);
    });
    node.setAttribute("points", pts.join(" "));
  }
  function ccShowSensor(key) {
    const cfg = CC_SENSORS[key];
    const pop = $("twinHud");
    if (!cfg || !pop) return;
    CC.selectedSensor = key;
    const lvl = ccSensorLevel(key);
    const value = ccSensorValue(key);
    ccText("twinHudName", cfg.label);
    ccNumber("twinHudValue", value, cfg.digits);
    ccText("twinHudUnit", cfg.unit);
    const statusNode = $("twinHudStatus");
    if (statusNode) {
      statusNode.textContent = String(lvl.label).toUpperCase();
      statusNode.className = lvl.key === "critical" ? "crit" : lvl.key === "warning" ? "warn" : "ok";
    }
    ccText("twinHudTrend", ccTrendWord(key));
    pop.classList.remove("hidden");
  }
  function ccHideSensor() {
    CC.selectedSensor = "";
    const pop = $("twinHud");
    if (pop) pop.classList.add("hidden");
  }
function ccSyncSignals() {
    const host = $("aiSignals");
    if (!host) return;
    const contribs = (state.contributions || []).slice();
    if (!contribs.length) {
      host.innerHTML = '<div class="muted-note" style="margin-top:0">Awaiting the first inference&hellip;</div>';
      return;
    }
    contribs.sort((a, b) => Math.abs(b.deltaPp) - Math.abs(a.deltaPp));
    const top = contribs.slice(0, 4);
    const maxAbs = Math.max.apply(null, top.map((c) => Math.abs(c.deltaPp)).concat([0.5]));
    host.innerHTML = top.map((c) => {
      const pct = Math.max(3, (Math.abs(c.deltaPp) / maxAbs) * 100);
      const sign = c.deltaPp >= 0 ? "+" : "\u2212";
      return '<div class="sig-row' + (c.deltaPp >= maxAbs * 0.6 ? " sig-high" : "") + '">' +
        '<span class="sig-name">' + esc(c.label) + '</span>' +
        '<span class="sig-track"><span class="sig-fill" style="width:' + pct.toFixed(1) + '%"></span></span>' +
        '<span class="sig-val">' + sign + fmt(Math.abs(c.deltaPp), 1) + " pp</span></div>";
    }).join("");
  }
  function ccSync(reason) {
    if (!state.prediction) return;
    const risk = clamp(Number(state.risk) || 0, 0, 100);
    const band = state.status.key;
    const cols = twBandColors(band);
    const arc = $("arcFill");
    if (arc) arc.style.strokeDashoffset = String((596 * (1 - risk / 100)).toFixed(1));
    const arcBox = $("riskArc");
    if (arcBox) arcBox.className = "risk-arc band-" + band;
    ccNumber("twinRiskVal", risk, 1);
    ccText("twinRiskLabel", state.status.tag || statusWord(band));
    ccNumber("nnValRf", risk, 1, "%");
    ccNumber("twinHealthVal", Math.round(state.health), 0, " / 100");
    ccText("twinStatusVal", statusWord(band));
    ccText("twinModelVal", modelReady() ? "RANDOM FOREST" : "UNAVAILABLE");
    ccText("twinAssetId", state.machine.id);
    const tag = $("twinBandTag");
    if (tag) {
      tag.textContent = state.status.tag || statusWord(band);
      tag.className = "twin-band band-" + band;
    }
    if (CC.band !== band) document.body.setAttribute("data-band", band);
    // display bands derived from the configured warning / critical thresholds
    const warn = Number(state.settings.warnThreshold) || 25;
    const crit = Number(state.settings.critThreshold) || 65;
    const mid = (warn + crit) / 2;
    const region = risk < warn ? 1 : risk < mid ? 2 : risk < crit ? 3 : 4;
    if (arcBox) {
      const chips = arcBox.querySelectorAll(".arc-regions span");
      for (let i = 0; i < chips.length; i++) chips[i].classList.toggle("rg-on", i + 1 === region);
    }
    // HUD labels, strip and sparklines
    Object.keys(CC_SENSORS).forEach((key) => {
      const cfg = CC_SENSORS[key];
      const value = ccSensorValue(key);
      const lvl = ccSensorLevel(key);
      ccNumber(cfg.hud, value, cfg.digits);
      ccNumber(cfg.strip, value, cfg.digits);
      ccSpark(cfg.spark, state.sparks[key]);
      const cls = lvl.key === "critical" ? " hud-crit" : lvl.key === "warning" ? " hud-warn" : "";
      document.querySelectorAll('.twin-hud[data-sensor="' + key + '"]').forEach((n) => {
        n.className = "twin-hud" + (n.className.indexOf("hud-") >= 0 ? "" : "") + cls + " " +
          (n.classList.contains("hud-t") ? "hud-t" : n.classList.contains("hud-tl") ? "hud-tl" : n.classList.contains("hud-tr") ? "hud-tr" : n.classList.contains("hud-bl") ? "hud-bl" : "hud-br");
      });
      const stripItem = document.querySelector('.strip-item[data-k="' + key + '"]');
      if (stripItem) stripItem.className = "strip-item" + (lvl.key === "critical" ? " s-crit" : lvl.key === "warning" ? " s-warn" : "");
    });
    const riskItem = document.querySelector('.strip-item[data-k="risk"]');
    ccNumber("stripValRisk", risk, 1);
    ccSpark("stripSparkRisk", state.sparks.risk);
    if (riskItem) riskItem.className = "strip-item s-risk" + (band === "critical" ? " s-crit" : band === "warning" ? " s-warn" : "");
    if (CC.selectedSensor) ccShowSensor(CC.selectedSensor);
    ccSyncSignals();
    // top-bar mode + telemetry indicators
    const mode = $("ccMode");
    if (mode) {
      const demo = state.demo && state.demo.active;
      const sim = state.sim && state.sim.running;
      mode.textContent = demo ? "DEMO" : sim ? "LIVE SIM" : "MANUAL";
      mode.className = "live-tag " + (demo ? "mode-demo" : sim ? "" : "mode-manual");
    }
    const tele = $("teleChip");
    if (tele) {
      const sim = !!(state.sim && state.sim.running);
      tele.innerHTML = '<span class="sys-dot' + (sim ? "" : " sys-dot-grey") + '"></span> TELEMETRY ' + (sim ? "ACTIVE" : "IDLE");
      tele.classList.toggle("chip-warn", false);
    }
    // ---- events (only from real actions) ----
    if (reason === "init") {
      ccEvent("MODEL READY \u00b7 RANDOM FOREST 300 TREES");
      ccEvent("PIPELINE RUN \u00b7 RF RISK " + fmt(risk, 1) + "%", band);
    } else if (reason === "sim") {
      ccEvent("TELEMETRY SIM TICK \u00b7 RF INFERENCE " + fmt(risk, 1) + "%", band);
    } else if (reason === "scenario") {
      ccEvent("SCENARIO APPLIED \u00b7 RF RISK " + fmt(risk, 1) + "%", band);
    } else if (reason === "demo") {
      ccEvent("DEMO STEP \u00b7 RF RISK " + fmt(risk, 1) + "%", band);
    } else if (reason !== "view" && reason !== "resize" && reason != null) {
      ccEvent("RANDOM FOREST INFERENCE \u00b7 RISK " + fmt(risk, 1) + "%", band);
    }
    if (CC.band && CC.band !== band) ccEvent("RISK BAND \u2192 " + statusWord(band), band);
    CC.band = band;
  }
const CC_REPLAY = [
      { name: "NORMAL", values: { productType: "M", airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 1540, torque: 39.5, toolWear: 85 } },
      { name: "TORQUE RISE", values: { productType: "M", airTemperature: 298.5, processTemperature: 308.8, rotationalSpeed: 1540, torque: 57, toolWear: 85 } },
      { name: "TOOL WEAR RISE", values: { productType: "M", airTemperature: 298.5, processTemperature: 309.4, rotationalSpeed: 1540, torque: 57, toolWear: 165 } },
      { name: "TEMPERATURE RISE", values: { productType: "M", airTemperature: 302.4, processTemperature: 313.2, rotationalSpeed: 1480, torque: 60, toolWear: 180 } },
      { name: "RISK INCREASE", values: { productType: "M", airTemperature: 303.6, processTemperature: 314.4, rotationalSpeed: 1250, torque: 73, toolWear: 235 } }
    ];
  function ccReplayPaint() {
    const track = $("replayTrack");
    if (!track) return;
    const steps = track.querySelectorAll(".rp-step");
    for (let i = 0; i < steps.length; i++) {
      steps[i].classList.toggle("rp-on", i === CC.replayIdx);
      steps[i].classList.toggle("rp-done", i < CC.replayIdx);
      const risk = CC.replayRisks[i];
      ccText("rpRisk" + i, risk == null ? "--" : fmt(risk, 1) + "%");
    }
  }
  function ccReplayStep(i) {
    const idx = clamp(i, 0, CC_REPLAY.length - 1);
    const step = CC_REPLAY[idx];
    CC.replayIdx = idx;
    Object.assign(state.sensors, step.values);
    syncUIFromState();
    runPipeline("scenario");
    CC.replayRisks[idx] = state.risk;
    ccText("replayNote", "STEP " + (idx + 1) + " / " + CC_REPLAY.length + " \u00b7 " + step.name);
    ccReplayPaint();
  }
  function ccReplayStop() {
    if (CC.replayTimer) { window.clearInterval(CC.replayTimer); CC.replayTimer = null; }
  }
  function ccReplayPlay() {
    ccReplayStop();
    if (CC.replayIdx < 0) { CC.replayRisks = {}; CC.replayIdx = -1; ccReplayStep(0); }
    else ccReplayStep(CC.replayIdx);
    ccScan();
    CC.replayTimer = window.setInterval(() => {
      const next = CC.replayIdx + 1;
      if (next >= CC_REPLAY.length) {
        ccReplayStop();
        ccText("replayNote", "SEQUENCE COMPLETE \u00b7 RESET TO RUN AGAIN");
        return;
      }
      ccReplayStep(next);
    }, 950);
  }
  function ccReplayReset() {
    ccReplayStop();
    CC.replayRisks = {};
    CC.replayIdx = -1;
    ccText("replayNote", "Idle \u2014 press PLAY to run the sequence.");
    ccReplayPaint();
  }
  function twResetCam() {
    TWIN.rotX = -0.34; TWIN.rotY = 0.62; TWIN.zoom = 1; TWIN.idleAt = 0;
    ccEvent("CAMERA RESET");
  }
  function twPick(clientX, clientY) {
    const canvas = $("twinCanvas");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left, py = clientY - rect.top;
    let best = null, bestD = 24;
    Object.keys(TWIN.screen).forEach((k) => {
      const p = TWIN.screen[k];
      const d = Math.sqrt((p.x - px) * (p.x - px) + (p.y - py) * (p.y - py));
      if (d < bestD) { bestD = d; best = k; }
    });
    if (best) ccShowSensor(best); else ccHideSensor();
  }
  function twBind() {
    const canvas = $("twinCanvas");
    if (!canvas) return;
    const down = (x, y) => { TWIN.dragging = true; TWIN.moved = false; TWIN.lastX = x; TWIN.lastY = y; TWIN.idleAt = Date.now(); };
    const move = (x, y) => {
      if (!TWIN.dragging || !TWIN.interactive) return;
      const dx = x - TWIN.lastX, dy = y - TWIN.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) TWIN.moved = true;
      TWIN.lastX = x; TWIN.lastY = y; TWIN.idleAt = Date.now();
      TWIN.rotY += dx * 0.008;
      TWIN.rotX = clamp(TWIN.rotX - dy * 0.006, -1.25, 1.25);
    };
    canvas.addEventListener("mousedown", (e) => down(e.clientX, e.clientY));
    window.addEventListener("mousemove", (e) => move(e.clientX, e.clientY));
    window.addEventListener("mouseup", (e) => {
      if (TWIN.dragging && !TWIN.moved) twPick(e.clientX, e.clientY);
      TWIN.dragging = false;
      TWIN.idleAt = Date.now();
    });
    canvas.addEventListener("touchstart", (e) => { if (e.touches[0]) down(e.touches[0].clientX, e.touches[0].clientY); }, { passive: true });
    canvas.addEventListener("touchmove", (e) => { if (e.touches[0]) move(e.touches[0].clientX, e.touches[0].clientY); }, { passive: true });
    canvas.addEventListener("touchend", () => { TWIN.dragging = false; TWIN.idleAt = Date.now(); }, { passive: true });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      TWIN.zoom = clamp(TWIN.zoom * (e.deltaY > 0 ? 0.92 : 1.08), 0.65, 1.75);
      TWIN.idleAt = Date.now();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => twResetCam());
  }
function v6Init() {
    if (CC.started) return;
    CC.started = true;
    twBuildMachine();
    twBind();
    ccClock();
    if (CC.clockTimer) window.clearInterval(CC.clockTimer);
    CC.clockTimer = window.setInterval(ccClock, 1000);
    // camera + mode controls
    const auto = $("twinAutoRotate");
    if (auto) auto.addEventListener("click", () => {
      TWIN.autoRotate = !TWIN.autoRotate;
      auto.classList.toggle("active", TWIN.autoRotate);
      auto.innerHTML = TWIN.autoRotate ? "&#8635; AUTO ROTATE" : "&#8635; AUTO ROTATE OFF";
      ccEvent("AUTO ROTATE " + (TWIN.autoRotate ? "ENABLED" : "PAUSED"));
    });
    const reset = $("twinResetCam");
    if (reset) reset.addEventListener("click", twResetCam);
    const inter = $("twinInteractive");
    if (inter) inter.addEventListener("click", () => {
      TWIN.interactive = !TWIN.interactive;
      inter.classList.toggle("active", TWIN.interactive);
      ccEvent("INTERACTIVE MODE " + (TWIN.interactive ? "ENABLED" : "LOCKED"));
    });
    // HUD label -> sensor popup
    document.querySelectorAll(".twin-hud").forEach((node) => {
      node.addEventListener("click", () => ccShowSensor(node.getAttribute("data-sensor")));
    });
    const close = $("twinHudClose");
    if (close) close.addEventListener("click", ccHideSensor);
    // presentation mode
    const present = $("btnPresentation");
    if (present) present.addEventListener("click", () => ccSetPresentation());
    const exit = $("btnExitPresentation");
    if (exit) exit.addEventListener("click", () => ccSetPresentation(false));
    const goto = $("btnGotoPredict");
    if (goto) goto.addEventListener("click", () => switchView("prediction"));
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && CC.presenting) ccSetPresentation(false);
    });
    // scenario replay
    const play = $("btnReplayPlay");
    if (play) play.addEventListener("click", ccReplayPlay);
    const pause = $("btnReplayPause");
    if (pause) pause.addEventListener("click", () => { ccReplayStop(); ccText("replayNote", "PAUSED \u00b7 step " + (CC.replayIdx + 1)); ccEvent("REPLAY PAUSED"); });
    const rst = $("btnReplayReset");
    if (rst) rst.addEventListener("click", () => { ccReplayReset(); ccEvent("REPLAY RESET"); });
    // scan sequence on the real prediction actions (result stays the real one)
    ["btnApplySensors", "btnCustomApply", "btnWhatIf"].forEach((id) => {
      const node = $(id);
      if (node) node.addEventListener("click", ccScan);
    });
    if (!TWIN.raf) TWIN.raf = window.requestAnimationFrame(twFrame);
    ccReplayPaint();
    ccEvent("DIGITAL TWIN ONLINE \u00b7 V9.0 SCENARIO & WHAT-IF INTELLIGENCE");
  }
})();
