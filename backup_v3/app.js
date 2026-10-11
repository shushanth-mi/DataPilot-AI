document.addEventListener("DOMContentLoaded", () => {
  const state = {
    productType: "M",
    airTemperature: 298.5,
    processTemperature: 308.8,
    rotationalSpeed: 1540,
    torque: 39.5,
    toolWear: 85,
    isSimulating: false,
    simTimer: null,
    historyBuffer: []
  };

  const MAX_HISTORY = 45;

  const el = {
    sliderAirTemp: document.getElementById("sliderAirTemp"),
    sliderProcTemp: document.getElementById("sliderProcTemp"),
    sliderSpeed: document.getElementById("sliderSpeed"),
    sliderTorque: document.getElementById("sliderTorque"),
    sliderWear: document.getElementById("sliderWear"),
    valAirTemp: document.getElementById("valAirTemp"),
    valAirTempC: document.getElementById("valAirTempC"),
    valProcTemp: document.getElementById("valProcTemp"),
    valProcTempC: document.getElementById("valProcTempC"),
    valSpeed: document.getElementById("valSpeed"),
    valTorque: document.getElementById("valTorque"),
    valWear: document.getElementById("valWear"),
    badgeProductType: document.getElementById("badgeProductType"),
    typeButtons: document.querySelectorAll(".type-btn"),
    presetsList: document.getElementById("presetsList"),
    btnResetDefaults: document.getElementById("btnResetDefaults"),
    btnToggleSim: document.getElementById("btnToggleSim"),
    simBtnLabel: document.getElementById("simBtnLabel"),
    activeAssetId: document.getElementById("activeAssetId"),
    predictionCard: document.getElementById("predictionOverviewCard"),
    riskPercentage: document.getElementById("riskPercentage"),
    gaugeFill: document.getElementById("gaugeFill"),
    machineStatusBadge: document.getElementById("machineStatusBadge"),
    machineStatusText: document.getElementById("machineStatusText"),
    riskLevelTag: document.getElementById("riskLevelTag"),
    healthIndexVal: document.getElementById("healthIndexVal"),
    metricPower: document.getElementById("metricPower"),
    metricTempDelta: document.getElementById("metricTempDelta"),
    metricOverstrain: document.getElementById("metricOverstrain"),
    pctTWF: document.getElementById("pctTWF"), barTWF: document.getElementById("barTWF"),
    pctHDF: document.getElementById("pctHDF"), barHDF: document.getElementById("barHDF"),
    pctPWF: document.getElementById("pctPWF"), barPWF: document.getElementById("barPWF"),
    pctOSF: document.getElementById("pctOSF"), barOSF: document.getElementById("barOSF"),
    pctRNF: document.getElementById("pctRNF"), barRNF: document.getElementById("barRNF"),
    prescriptionsList: document.getElementById("prescriptionsList"),
    canvas: document.getElementById("telemetryChart"),
    tableBody: document.getElementById("telemetryTableBody")
  };

  function init() {
    renderPresets();
    renderHistoricalTable();
    initHistoryBuffer();
    bindEvents();
    syncUIFromState();
    runPrediction();
    startCanvasLoop();
  }

  function renderPresets() {
    if (!MODEL_DATA || !MODEL_DATA.presetScenarios) return;
    el.presetsList.innerHTML = MODEL_DATA.presetScenarios.map((preset) => `
      <button type="button" class="preset-chip" data-preset-id="${preset.id}" title="${preset.description}">
        <span class="chip-tag ${preset.badgeClass}">${preset.badge}</span>
        <span>${preset.name}</span>
      </button>
    `).join("");
  }

  function renderHistoricalTable() {
    if (!MODEL_DATA || !MODEL_DATA.historicalLogs) return;
    el.tableBody.innerHTML = MODEL_DATA.historicalLogs.map((log) => {
      let statusClass = "table-status-normal";
      let rowClass = "";
      if (log.status === "Warning") statusClass = "table-status-warning";
      if (log.status === "Critical") { statusClass = "table-status-critical"; rowClass = "row-critical"; }
      return `
        <tr class="${rowClass}" data-log-id="${log.id}">
          <td style="font-family:var(--font-mono);font-weight:600;color:#38bdf8;">${log.id}</td>
          <td><span class="chip-tag" style="background:rgba(255,255,255,0.06);color:#f1f5f9;">Type ${log.type}</span></td>
          <td>${log.airTemp.toFixed(1)} K</td>
          <td>${log.procTemp.toFixed(1)} K</td>
          <td>${log.speed} rpm</td>
          <td>${log.torque.toFixed(1)} Nm</td>
          <td>${log.wear} min</td>
          <td style="font-family:var(--font-mono);font-weight:700;">${log.risk.toFixed(1)}%</td>
          <td><span class="table-status-pill ${statusClass}">${log.status}</span></td>
          <td style="font-family:var(--font-mono);font-size:0.75rem;color:${log.failureType === 'None' ? 'var(--text-dim)' : '#fb7185'};">${log.failureType}</td>
        </tr>
      `;
    }).join("");
  }

  function bindEvents() {
    const sliders = [
      { input: el.sliderAirTemp, key: "airTemperature", isFloat: true },
      { input: el.sliderProcTemp, key: "processTemperature", isFloat: true },
      { input: el.sliderSpeed, key: "rotationalSpeed", isFloat: false },
      { input: el.sliderTorque, key: "torque", isFloat: true },
      { input: el.sliderWear, key: "toolWear", isFloat: false }
    ];
    sliders.forEach(({ input, key, isFloat }) => {
      input.addEventListener("input", (e) => {
        state[key] = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
        clearActivePresetHighlight();
        syncUIFromState();
        runPrediction();
      });
    });

    el.typeButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const type = btn.getAttribute("data-type");
        state.productType = type;
        el.typeButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        el.badgeProductType.textContent = `Type ${type}`;
        clearActivePresetHighlight();
        runPrediction();
      });
    });

    el.presetsList.addEventListener("click", (e) => {
      const chip = e.target.closest(".preset-chip");
      if (!chip) return;
      loadPreset(chip.getAttribute("data-preset-id"));
    });

    el.btnResetDefaults.addEventListener("click", () => loadPreset("nominal"));
    el.btnToggleSim.addEventListener("click", toggleSimulation);

    el.tableBody.addEventListener("click", (e) => {
      const tr = e.target.closest("tr");
      if (!tr) return;
      const logId = tr.getAttribute("data-log-id");
      const record = MODEL_DATA.historicalLogs.find((r) => r.id === logId);
      if (record) {
        state.productType = record.type;
        state.airTemperature = record.airTemp;
        state.processTemperature = record.procTemp;
        state.rotationalSpeed = record.speed;
        state.torque = record.torque;
        state.toolWear = record.wear;
        el.activeAssetId.textContent = record.id;
        clearActivePresetHighlight();
        syncUIFromState();
        runPrediction();
      }
    });
  }

  function loadPreset(presetId) {
    const preset = MODEL_DATA.presetScenarios.find((p) => p.id === presetId);
    if (!preset) return;
    Object.assign(state, preset.values);
    document.querySelectorAll(".preset-chip").forEach((c) => {
      c.classList.toggle("active", c.getAttribute("data-preset-id") === presetId);
    });
    syncUIFromState();
    runPrediction();
  }

  function clearActivePresetHighlight() {
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
  }

  function syncUIFromState() {
    el.sliderAirTemp.value = state.airTemperature;
    el.valAirTemp.textContent = state.airTemperature.toFixed(1);
    el.valAirTempC.textContent = `(${MODEL_DATA.features.airTemperature.toCelsius(state.airTemperature)}°C)`;
    el.sliderProcTemp.value = state.processTemperature;
    el.valProcTemp.textContent = state.processTemperature.toFixed(1);
    el.valProcTempC.textContent = `(${MODEL_DATA.features.processTemperature.toCelsius(state.processTemperature)}°C)`;
    el.sliderSpeed.value = state.rotationalSpeed;
    el.valSpeed.textContent = state.rotationalSpeed;
    el.sliderTorque.value = state.torque;
    el.valTorque.textContent = state.torque.toFixed(1);
    el.sliderWear.value = state.toolWear;
    el.valWear.textContent = state.toolWear;
    el.typeButtons.forEach((b) => b.classList.toggle("active", b.getAttribute("data-type") === state.productType));
    el.badgeProductType.textContent = `Type ${state.productType}`;
  }

  function runPrediction() {
    const result = MODEL_DATA.predictFailure(state);

    el.riskPercentage.textContent = result.probabilityPercentage;
    el.riskLevelTag.textContent = `${result.riskLevel} RISK`;
    el.healthIndexVal.textContent = `${result.healthIndex} / 100`;

    const circumference = 440;
    const offset = circumference - (circumference * result.probability);
    el.gaugeFill.style.strokeDashoffset = Math.max(0, offset);

    el.predictionCard.className = `card prediction-overview-card ${result.statusClass}`;
    if (result.probability >= 0.65) {
      el.gaugeFill.style.stroke = "var(--status-critical)";
      el.machineStatusBadge.style.color = "#fb7185";
    } else if (result.probability >= 0.28) {
      el.gaugeFill.style.stroke = "var(--status-warning)";
      el.machineStatusBadge.style.color = "#fbbf24";
    } else {
      el.gaugeFill.style.stroke = "var(--status-optimal)";
      el.machineStatusBadge.style.color = "#34d399";
    }
    el.machineStatusText.textContent = result.status;

    el.metricPower.textContent = `${result.physicsMetrics.powerWatts.toLocaleString()} W`;
    el.metricTempDelta.textContent = `${result.physicsMetrics.tempDelta} K`;
    el.metricOverstrain.textContent = `${result.physicsMetrics.overstrain.toLocaleString()} min·Nm`;

    updateProgressBar(el.barTWF, el.pctTWF, result.failureModeProbabilities.twf);
    updateProgressBar(el.barHDF, el.pctHDF, result.failureModeProbabilities.hdf);
    updateProgressBar(el.barPWF, el.pctPWF, result.failureModeProbabilities.pwf);
    updateProgressBar(el.barOSF, el.pctOSF, result.failureModeProbabilities.osf);
    updateProgressBar(el.barRNF, el.pctRNF, result.failureModeProbabilities.rnf);

    renderPrescriptions(result.recommendations);

    pushHistoryPoint({
      speed: state.rotationalSpeed / 30,
      torque: state.torque,
      wear: state.toolWear / 3,
      risk: result.probability * 100
    });
  }

  function updateProgressBar(barEl, pctEl, value) {
    pctEl.textContent = `${value}%`;
    barEl.style.width = `${Math.min(100, Math.max(1, value))}%`;
    barEl.className = "mode-progress-bar";
    if (value >= 50) barEl.classList.add("bar-critical");
    else if (value >= 25) barEl.classList.add("bar-warning");
  }

  function renderPrescriptions(recommendations) {
    if (!recommendations || recommendations.length === 0) {
      el.prescriptionsList.innerHTML = `<div class="presc-reason">No active prescriptive alerts.</div>`;
      return;
    }
    el.prescriptionsList.innerHTML = recommendations.map((r) => `
      <div class="prescription-card priority-${r.priority}">
        <div class="presc-icon" aria-hidden="true">${getPrescriptionIcon(r.priority)}</div>
        <div class="presc-details">
          <div class="presc-action-row">
            <span class="presc-badge">${r.priority}</span>
            <span class="presc-action">${r.action}</span>
          </div>
          <div class="presc-reason">${r.reason}</div>
        </div>
      </div>
    `).join("");
  }

  function getPrescriptionIcon(priority) {
    if (priority === "CRITICAL" || priority === "URGENT") {
      return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    }
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
  }

  function toggleSimulation() {
    state.isSimulating = !state.isSimulating;
    if (state.isSimulating) {
      el.btnToggleSim.classList.add("active");
      el.simBtnLabel.textContent = "Live Stream: RUNNING";
      el.activeAssetId.textContent = "STREAM-EDGE-01";
      let cycle = 0;
      state.simTimer = setInterval(() => {
        cycle++;
        const speedNoise = (Math.random() - 0.5) * 22;
        const torqueNoise = (Math.random() - 0.5) * 1.8;
        const tempNoise = (Math.random() - 0.5) * 0.15;
        state.rotationalSpeed = Math.round(Math.min(2850, Math.max(1150, state.rotationalSpeed + speedNoise)));
        state.torque = Number((Math.min(78, Math.max(5, state.torque + torqueNoise))).toFixed(1));
        state.airTemperature = Number((Math.min(304, Math.max(296, state.airTemperature + tempNoise))).toFixed(1));
        if (cycle % 4 === 0 && state.toolWear < 255) state.toolWear += 1;
        syncUIFromState();
        runPrediction();
      }, 700);
    } else {
      el.btnToggleSim.classList.remove("active");
      el.simBtnLabel.textContent = "Live Stream: OFF";
      clearInterval(state.simTimer);
      state.simTimer = null;
    }
  }

  function initHistoryBuffer() {
    for (let i = 0; i < MAX_HISTORY; i++) {
      state.historyBuffer.push({ speed: state.rotationalSpeed / 30, torque: state.torque, wear: state.toolWear / 3, risk: 2.5 });
    }
  }

  function pushHistoryPoint(point) {
    state.historyBuffer.push(point);
    if (state.historyBuffer.length > MAX_HISTORY) state.historyBuffer.shift();
  }

  function startCanvasLoop() {
    function draw() { drawWaveform(); requestAnimationFrame(draw); }
    requestAnimationFrame(draw);
  }

  function drawWaveform() {
    const canvas = el.canvas;
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
    if (state.historyBuffer.length < 2) return;
    const traces = [
      { key: "speed", color: "#06b6d4", max: 100 },
      { key: "torque", color: "#f59e0b", max: 80 },
      { key: "wear", color: "#10b981", max: 90 },
      { key: "risk", color: "#f43f5e", max: 100, glow: true }
    ];
    const stepX = width / (MAX_HISTORY - 1);
    traces.forEach((trace) => {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = trace.color;
      ctx.lineWidth = trace.glow ? 2.5 : 1.7;
      if (trace.glow) { ctx.shadowColor = trace.color; ctx.shadowBlur = 10; }
      state.historyBuffer.forEach((pt, index) => {
        const x = index * stepX;
        const normalizedVal = Math.min(trace.max, Math.max(0, pt[trace.key])) / trace.max;
        const y = height - (normalizedVal * (height - 24)) - 12;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    });
  }

  init();
});
