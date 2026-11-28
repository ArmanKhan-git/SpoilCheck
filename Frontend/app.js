// SpoilCheck tester frontend. Served by the backend at http://127.0.0.1:8000
const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);

const state = { srgb: null, ir: null, trueLabel: null, hasResult: false, busy: false };

// ---------- helpers ----------
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const prettyLabel = (o) => `${o.variety} · ${o.state}`;

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

function setImage(kind, file) {
  state[kind] = file;
  const img = $(`${kind}-preview`);
  const drop = $(`${kind}-drop`);
  const text = $(`${kind}-text`);
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  if (file) {
    img.dataset.url = URL.createObjectURL(file);
    img.src = img.dataset.url;
    img.hidden = false;
    drop.classList.add("has-image");
    text.textContent = file.name;
  } else {
    img.hidden = true;
    drop.classList.remove("has-image");
    text.textContent = kind === "srgb" ? "Click or drop a fruit photo" : "Click or drop a thermal image";
  }
  if (kind === "ir" && file) {
    const m = file.name.match(/_(\d+(?:\.\d+)?)C_(\d+(?:\.\d+)?)C\./i);
    if (m) { $("t-min").value = m[1]; $("t-max").value = m[2]; }
  }
  $("predict-btn").disabled = !state.srgb;
}

function setupDrop(kind) {
  const drop = $(`${kind}-drop`);
  const input = $(`${kind}-input`);
  input.addEventListener("change", () => {
    if (input.files[0]) { state.trueLabel = null; setImage(kind, input.files[0]); }
  });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) { state.trueLabel = null; setImage(kind, f); }
  });
}

function setupToggle(kind) {
  const toggle = $(`${kind}-toggle`);
  toggle.addEventListener("change", () => {
    $(`${kind}-fields`).hidden = !toggle.checked;
    $(`${kind}-state`).textContent = toggle.checked ? "On" : "Off";
    if (state.hasResult) predict();   // re-run so the effect of the sensor is visible immediately
  });
}

function setToggle(kind, on) {
  const toggle = $(`${kind}-toggle`);
  toggle.checked = on;
  $(`${kind}-fields`).hidden = !on;
  $(`${kind}-state`).textContent = on ? "On" : "Off";
}

// ---------- backend calls ----------
async function checkHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    if (!r.ok) throw new Error();
    const h = await r.json();
    $("status").textContent = `Models loaded · ${h.classes.length} classes`;
    $("status").className = "status status-ok";
    (h.fruits || []).forEach((f) => {
      const o = document.createElement("option");
      o.value = f;
      o.textContent = f;
      $("fruit-select").appendChild(o);
    });
    const sel = $("sample-label");
    (h.sample_labels || []).forEach((l) => {
      const o = document.createElement("option");
      o.value = l;
      o.textContent = l.replace(/_Not_spoiled$/, " · Fresh").replace(/_Spoiled$/, " · Spoiled").replace(/_/g, " ");
      sel.appendChild(o);
    });
    if (!h.background_removal) {
      $("bg-toggle").disabled = true;
      $("bg-toggle").closest(".check").classList.add("is-disabled");
      $("bg-hint").textContent = "Background removal is not available on this server.";
    }
    if (!h.samples_available) {
      $("sample-btn").disabled = true;
      $("sample-label").disabled = true;
    }
  } catch {
    $("status").textContent = "Backend not reachable";
    $("status").className = "status status-bad";
    showError("Cannot reach the backend. Start it with Backend\\run.bat, then reload this page.");
  }
}

async function fetchFile(url, name) {
  const r = await fetch(`${API}${url}`);
  if (!r.ok) throw new Error(`Could not load ${name}`);
  return new File([await r.blob()], name, { type: "image/jpeg" });
}

async function loadSample() {
  showError("");
  const btn = $("sample-btn");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const label = $("sample-label").value;
    const r = await fetch(`${API}/api/sample${label ? `?label=${encodeURIComponent(label)}` : ""}`);
    const s = await r.json();
    if (!r.ok) throw new Error(s.detail || "Could not load a sample");

    setImage("srgb", await fetchFile(s.srgb.url, s.srgb.name));
    if (s.ir) {
      setImage("ir", await fetchFile(s.ir.url, s.ir.name));
      $("t-min").value = s.ir.t_min;
      $("t-max").value = s.ir.t_max;
      setToggle("ir", true);
    } else {
      setImage("ir", null);
      setToggle("ir", false);
    }
    if (s.gas) {
      $("adc").value = s.gas.adc; $("rs").value = s.gas.rs; $("rs-r0").value = s.gas.rs_r0; $("ppm").value = s.gas.methane_ppm;
      setToggle("gas", true);
    } else {
      setToggle("gas", false);
    }
    state.trueLabel = s.label;
    $("fruit-select").value = "";   // samples test automatic detection
    await predict();
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Load sample";
  }
}

async function predict() {
  if (!state.srgb || state.busy) return;
  showError("");
  const useIr = $("ir-toggle").checked;
  const useGas = $("gas-toggle").checked;

  const fd = new FormData();
  fd.append("srgb", state.srgb);
  fd.append("use_ir", useIr);
  fd.append("use_gas", useGas);
  if ($("fruit-select").value) fd.append("fruit", $("fruit-select").value);
  fd.append("clean_bg", $("bg-toggle").checked);
  if (useIr) {
    if (!state.ir) return showError("IR sensor is on: add a thermal image, or switch it off.");
    if ($("t-min").value === "" || $("t-max").value === "") return showError("IR sensor is on: enter t_min and t_max.");
    fd.append("ir", state.ir);
    fd.append("t_min", $("t-min").value);
    fd.append("t_max", $("t-max").value);
  }
  if (useGas) {
    const vals = { adc: $("adc").value, rs: $("rs").value, rs_r0: $("rs-r0").value, methane_ppm: $("ppm").value };
    if (Object.values(vals).some((v) => v === "")) return showError("Methane sensor is on: fill in all four readings, or switch it off.");
    Object.entries(vals).forEach(([k, v]) => fd.append(k, v));
  }

  state.busy = true;
  const btn = $("predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const r = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Prediction failed");
    render(data);
  } catch (e) {
    showError(e.message === "Failed to fetch" ? "Cannot reach the backend. Is Backend\\run.bat running?" : e.message);
  } finally {
    state.busy = false;
    btn.disabled = !state.srgb;
    btn.textContent = "Predict";
  }
}

// ---------- rendering ----------
function renderUnsure(f, fruitGiven) {
  const box = $("unsure");
  box.innerHTML = "";
  const lines = [];
  if (fruitGiven) {
    lines.push(["Fruit set to ", fruitGiven, ": the model only decided fresh or spoiled."]);
  } else if (f.unsure_fruit) {
    const c = f.fruit_candidates.map((x) => `${x.variety} ${pct(x.probability)}`).join(", ");
    lines.push(["Not sure which fruit this is ", `(${c})`, ". Choose the fruit under the photo for a reliable answer."]);
  }
  if (f.unsure_state) {
    lines.push(["Not sure if it is fresh or spoiled: ", `spoilage probability ${pct(f.spoiled_probability)}`, " is close to 50%."]);
  } else if (f.unsure_fruit) {
    lines.push(["The fresh/spoiled call is clearer: ", `${f.state} (${pct(f.spoiled_probability)} spoiled)`, "."]);
  }
  lines.forEach(([a, b, c]) => {
    const p = document.createElement("p");
    const strong = document.createElement("b");
    strong.textContent = b;
    p.append(a, strong, c);
    box.appendChild(p);
  });
  box.hidden = lines.length === 0;
  $("verdict").classList.toggle("is-unsure", f.unsure_fruit || f.unsure_state);
}

function render(d) {
  state.hasResult = true;
  $("empty").hidden = true;
  $("results").hidden = false;

  const f = d.fusion;
  const fresh = f.state === "Fresh";
  $("verdict").className = `verdict ${fresh ? "is-fresh" : "is-spoiled"}`;
  $("v-variety").textContent = f.variety;
  $("v-state").textContent = f.state;
  $("v-state").className = `badge ${fresh ? "fresh" : "spoiled"}`;
  $("v-conf").textContent = pct(f.confidence);
  $("v-spoil").textContent = pct(f.spoiled_probability);
  $("v-meter").style.left = `${(f.spoiled_probability * 100).toFixed(1)}%`;
  renderUnsure(f, d.fruit_given);

  const truth = $("truth");
  if (state.trueLabel) {
    const ok = state.trueLabel === f.label;
    const t = state.trueLabel.replace(/_Not_spoiled$/, " · Fresh").replace(/_Spoiled$/, " · Spoiled").replace(/_/g, " ");
    truth.textContent = `${ok ? "✓ Correct" : "✗ Wrong"}: dataset label is ${t}`;
    truth.className = `truth ${ok ? "ok" : "bad"}`;
    truth.hidden = false;
  } else {
    truth.hidden = true;
  }

  const chips = $("sensor-chips");
  chips.innerHTML = "";
  [["Camera", d.sensors_used.camera], ["IR thermal", d.sensors_used.ir], ["Methane", d.sensors_used.gas]].forEach(([n, on]) => {
    const c = document.createElement("span");
    c.className = `chip ${on ? "on" : "off"}`;
    c.textContent = `${n}: ${on ? "used" : "off"}`;
    chips.appendChild(c);
  });

  renderBackground(d.background);

  const list = $("top-list");
  list.innerHTML = "";
  f.top.forEach((t) => {
    const li = document.createElement("li");
    li.className = "bar-row";
    li.innerHTML = `<div><div class="bar-label"><span></span></div><div class="bar-track"><div class="bar-fill"></div></div></div><span class="bar-pct"></span>`;
    li.querySelector(".bar-label span").textContent = prettyLabel(t);
    li.querySelector(".bar-fill").style.width = `${(t.probability * 100).toFixed(1)}%`;
    li.querySelector(".bar-pct").textContent = pct(t.probability);
    list.appendChild(li);
  });

  const body = $("models-body");
  body.innerHTML = "";
  [["Fusion (final)", d.fusion], ["Camera only (ConvNeXt)", d.camera], ["IR only (ResNet18)", d.ir], ["Gas only (MLP)", d.gas]].forEach(([name, m]) => {
    const tr = document.createElement("tr");
    const cells = m ? [name, prettyLabel(m), pct(m.confidence)] : [name, "sensor off", "–"];
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = v;
      if (!m && i > 0) td.className = "off";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  $("timing").textContent = `Inference took ${d.inference_ms} ms on the server. The gas-only model is close to guessing (see the ablation results).`;
}

function renderBackground(bg) {
  const fig = $("bg-figure"), note = $("bg-note");
  fig.hidden = true;
  note.hidden = true;
  if (!bg || !bg.requested) return;
  if (bg.applied) {
    $("bg-preview").src = bg.preview;
    $("bg-caption").textContent = `What the camera model saw: background removed (fruit covers ${pct(bg.mask_coverage)} of the photo).`;
    fig.hidden = false;
  } else {
    note.textContent = `Background not removed (${bg.reason || "cut-out failed"}): the original photo was used.`;
    note.hidden = false;
  }
}

// ---------- init ----------
setupDrop("srgb");
setupDrop("ir");
setupToggle("ir");
setupToggle("gas");
$("predict-btn").addEventListener("click", predict);
$("fruit-select").addEventListener("change", () => { if (state.hasResult) predict(); });
$("bg-toggle").addEventListener("change", () => { if (state.hasResult) predict(); });
$("sample-btn").addEventListener("click", loadSample);
checkHealth();
