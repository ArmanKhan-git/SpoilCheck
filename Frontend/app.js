// SpoilCheck tester frontend. Served by the backend (http://127.0.0.1:8000, or the PC's LAN address from a phone)
const API = location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);

// photos: [{file, url}] = several photos of ONE fruit (different sides)
const state = { photos: [], ir: null, trueLabel: null, hasResult: false, busy: false, maxViews: 6 };
const MAX_SIDE = 1600;   // phone photos are shrunk before upload (the model works at 224 px anyway)

// ---------- helpers ----------
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const prettyLabel = (o) => `${o.variety} · ${o.state}`;
const isNarrow = () => window.matchMedia("(max-width: 860px)").matches;

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

// Shrink big photos in the browser: uploads from a phone over Wi-Fi go from ~5 MB to ~0.4 MB.
async function shrink(file) {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = MAX_SIDE / Math.max(bmp.width, bmp.height);
    if (scale >= 1 && file.size < 2.5e6) { bmp.close(); return file; }
    const w = Math.round(bmp.width * Math.min(scale, 1)), h = Math.round(bmp.height * Math.min(scale, 1));
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    canvas.getContext("2d").drawImage(bmp, 0, 0, w, h);
    bmp.close();
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch {
    return file;   // browser can't decode it: send the original, the server will say if it's unreadable
  }
}

// ---------- camera photos (several) ----------
function updatePhotoUi() {
  const n = state.photos.length;
  $("srgb-drop").classList.toggle("has-image", n > 0);
  $("srgb-clear").hidden = n === 0;
  $("srgb-pick-text").textContent = n ? "Add more" : "Choose photos";
  $("srgb-text").hidden = n > 0;
  const full = n >= state.maxViews;
  $("srgb-input").disabled = full;
  $("srgb-camera").disabled = full;
  document.querySelectorAll(".btn-file").forEach((b) => b.classList.toggle("is-disabled", full));
  $("predict-btn").disabled = n === 0 || state.busy;

  const list = $("srgb-thumbs");
  list.innerHTML = "";
  state.photos.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "thumb";
    const img = document.createElement("img");
    img.src = p.url;
    img.alt = `Photo ${i + 1}`;
    const tag = document.createElement("span");
    tag.className = "thumb-tag";
    tag.textContent = i + 1;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "thumb-remove";
    rm.setAttribute("aria-label", `Remove photo ${i + 1}`);
    rm.textContent = "×";
    rm.addEventListener("click", () => removePhoto(i));
    li.append(img, tag, rm);
    list.appendChild(li);
  });
}

async function addPhotos(files, replace = false) {
  const imgs = [...files].filter((f) => f.type.startsWith("image/") || f.type === "");
  if (!imgs.length) return;
  showError("");
  if (replace) clearPhotos(false);
  const room = state.maxViews - state.photos.length;
  if (imgs.length > room) showError(`Only ${state.maxViews} photos per fruit: the extra ones were skipped.`);
  for (const f of imgs.slice(0, room)) {
    const file = await shrink(f);
    state.photos.push({ file, url: URL.createObjectURL(file) });
  }
  updatePhotoUi();
}

function removePhoto(i) {
  URL.revokeObjectURL(state.photos[i].url);
  state.photos.splice(i, 1);
  state.trueLabel = null;
  updatePhotoUi();
}

function clearPhotos(update = true) {
  state.photos.forEach((p) => URL.revokeObjectURL(p.url));
  state.photos = [];
  if (update) updatePhotoUi();
}

function setupPhotos() {
  ["srgb-input", "srgb-camera"].forEach((id) => {
    const input = $(id);
    input.addEventListener("change", async () => {
      state.trueLabel = null;
      await addPhotos(input.files);
      input.value = "";   // allow picking the same file again
    });
  });
  $("srgb-clear").addEventListener("click", () => { state.trueLabel = null; clearPhotos(); });
  const drop = $("srgb-drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (e) => { state.trueLabel = null; addPhotos(e.dataTransfer.files); });
}

// ---------- thermal image (one) ----------
function setIr(file) {
  state.ir = file;
  const img = $("ir-preview"), drop = $("ir-drop"), text = $("ir-text");
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  if (file) {
    img.dataset.url = URL.createObjectURL(file);
    img.src = img.dataset.url;
    img.hidden = false;
    drop.classList.add("has-image");
    text.textContent = file.name;
    const m = file.name.match(/_(\d+(?:\.\d+)?)C_(\d+(?:\.\d+)?)C\./i);
    if (m) { $("t-min").value = m[1]; $("t-max").value = m[2]; }
  } else {
    img.hidden = true;
    drop.classList.remove("has-image");
    text.textContent = "Tap or drop a thermal image";
  }
}

function setupIrDrop() {
  const drop = $("ir-drop"), input = $("ir-input");
  input.addEventListener("change", () => { if (input.files[0]) { state.trueLabel = null; setIr(input.files[0]); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) { state.trueLabel = null; setIr(f); }
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
    if (h.max_views) state.maxViews = h.max_views;
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
      $("sample-box").hidden = true;   // no dataset on this server: hide the sample tools
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

    await addPhotos([await fetchFile(s.srgb.url, s.srgb.name)], true);
    if (s.ir) {
      setIr(await fetchFile(s.ir.url, s.ir.name));
      $("t-min").value = s.ir.t_min;
      $("t-max").value = s.ir.t_max;
      setToggle("ir", true);
    } else {
      setIr(null);
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

async function predict({ scroll = false } = {}) {
  if (!state.photos.length || state.busy) return;
  showError("");
  const useIr = $("ir-toggle").checked;
  const useGas = $("gas-toggle").checked;

  const fd = new FormData();
  state.photos.forEach((p) => fd.append("srgb", p.file));
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
  const n = state.photos.length;
  btn.textContent = n > 1 ? `Checking ${n} photos…` : "Predicting…";
  try {
    const r = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Prediction failed");
    render(data);
    // on a phone the result is below the inputs: bring it into view
    if (scroll && isNarrow()) $("results-title").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showError(e.message === "Failed to fetch" ? "Cannot reach the backend. Is the server still running?" : e.message);
  } finally {
    state.busy = false;
    btn.disabled = !state.photos.length;
    btn.textContent = "Predict";
  }
}

// ---------- rendering ----------
function renderUnsure(f, d) {
  const box = $("unsure");
  box.innerHTML = "";
  const lines = [];
  const views = d.views || [];
  if (d.fruit_given) {
    lines.push(["Fruit set to ", d.fruit_given, ": the model only decided fresh or spoiled."]);
  } else if (f.unsure_fruit) {
    const c = f.fruit_candidates.map((x) => `${x.variety} ${pct(x.probability)}`).join(", ");
    lines.push(["Not sure which fruit this is ", `(${c})`, ". Choose the fruit under the photos for a reliable answer."]);
  }
  if (f.unsure_state) {
    lines.push(["Not sure if it is fresh or spoiled: ", `spoilage probability ${pct(f.spoiled_probability)}`, " is close to 50%."]);
  } else if (f.unsure_fruit) {
    lines.push(["The fresh/spoiled call is clearer: ", `${f.state} (${pct(f.spoiled_probability)} spoiled)`, "."]);
  }
  if (views.length > 1 && d.views_spoiled > 0 && f.state === "Fresh") {
    lines.push(["", `${d.views_spoiled} of ${views.length} photos look spoiled on their own`,
      ". Rot is often on one side only: check that side before trusting the overall answer."]);
  }
  lines.forEach(([a, b, c]) => {
    const p = document.createElement("p");
    const strong = document.createElement("b");
    strong.textContent = b;
    p.append(a, strong, c);
    box.appendChild(p);
  });
  box.hidden = lines.length === 0;
  const sideWarning = views.length > 1 && d.views_spoiled > 0 && f.state === "Fresh";
  $("verdict").classList.toggle("is-unsure", f.unsure_fruit || f.unsure_state || sideWarning);
}

function render(d) {
  state.hasResult = true;
  $("empty").hidden = true;
  $("results").hidden = false;

  const f = d.fusion;
  const fresh = f.state === "Fresh";
  const n = (d.views || []).length || 1;
  $("verdict").className = `verdict ${fresh ? "is-fresh" : "is-spoiled"}`;
  $("v-eyebrow").textContent = n > 1 ? `Fusion model · final answer from ${n} photos` : "Fusion model · final answer";
  $("v-variety").textContent = f.variety;
  $("v-state").textContent = f.state;
  $("v-state").className = `badge ${fresh ? "fresh" : "spoiled"}`;
  $("v-conf").textContent = pct(f.confidence);
  $("v-spoil").textContent = pct(f.spoiled_probability);
  $("v-meter").style.left = `${(f.spoiled_probability * 100).toFixed(1)}%`;
  renderUnsure(f, d);

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
  [[n > 1 ? `Camera (${n} photos)` : "Camera", d.sensors_used.camera], ["IR thermal", d.sensors_used.ir], ["Methane", d.sensors_used.gas]].forEach(([name, on]) => {
    const c = document.createElement("span");
    c.className = `chip ${on ? "on" : "off"}`;
    c.textContent = `${name}: ${on ? "used" : "off"}`;
    chips.appendChild(c);
  });

  renderBackground(d);
  renderViews(d);

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

// single photo: show the cut-out large; several photos: the cut-outs appear in "Each photo"
function renderBackground(d) {
  const fig = $("bg-figure"), note = $("bg-note");
  fig.hidden = true;
  note.hidden = true;
  const views = d.views || [{ background: d.background }];
  const bg = views[0].background;
  if (!bg || !bg.requested) return;
  if (views.length === 1) {
    if (bg.applied) {
      $("bg-preview").src = bg.preview;
      $("bg-caption").textContent = `What the camera model saw: background removed (fruit covers ${pct(bg.mask_coverage)} of the photo).`;
      fig.hidden = false;
    } else {
      note.textContent = `Background not removed (${bg.reason || "cut-out failed"}): the original photo was used.`;
      note.hidden = false;
    }
    return;
  }
  const failed = views.filter((v) => !v.background.applied).length;
  note.textContent = failed
    ? `Background removed on ${views.length - failed} of ${views.length} photos; the others were used as taken.`
    : "Background removed on every photo (the pictures below are what the camera model saw).";
  note.hidden = false;
}

function renderViews(d) {
  const views = d.views || [];
  $("views-block").hidden = views.length < 2;
  if (views.length < 2) return;
  $("views-hint").textContent = `The final answer averages the ${views.length} photos. ${d.views_spoiled} of them look spoiled on their own.`;
  const list = $("views-list");
  list.innerHTML = "";
  views.forEach((v, i) => {
    const li = document.createElement("li");
    li.className = `view ${v.state === "Fresh" ? "is-fresh" : "is-spoiled"}`;
    const img = document.createElement("img");
    img.src = (v.background.applied && v.background.preview) || (state.photos[i] && state.photos[i].url) || "";
    img.alt = `Photo ${i + 1}`;
    const info = document.createElement("div");
    info.className = "view-info";
    const t = document.createElement("p");
    t.className = "view-title";
    t.textContent = `Photo ${i + 1} · ${v.variety}`;
    const s = document.createElement("p");
    s.className = "view-state";
    const badge = document.createElement("span");
    badge.className = `badge small ${v.state === "Fresh" ? "fresh" : "spoiled"}`;
    badge.textContent = v.state;
    s.append(badge, ` ${pct(v.spoiled_probability)} spoiled`);
    info.append(t, s);
    li.append(img, info);
    list.appendChild(li);
  });
}

// ---------- init ----------
setupPhotos();
setupIrDrop();
setupToggle("ir");
setupToggle("gas");
$("predict-btn").addEventListener("click", () => predict({ scroll: true }));
$("fruit-select").addEventListener("change", () => { if (state.hasResult) predict(); });
$("bg-toggle").addEventListener("change", () => { if (state.hasResult) predict(); });
$("sample-btn").addEventListener("click", loadSample);
updatePhotoUi();
checkHealth();
