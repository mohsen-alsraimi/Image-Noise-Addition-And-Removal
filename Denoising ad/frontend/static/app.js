// Image Noise Lab - page logic (plain JavaScript, no framework)

const $ = (id) => document.getElementById(id);
let CONFIG = null;
const charts = {};
const FILTER_COLORS = {
  "Noisy (Baseline)": "#9ca3af", "Gaussian Filter": "#3b82f6", "Median Filter": "#f59e0b",
  "Bilateral Filter": "#ec4899", "NLM Filter": "#8b5cf6", "Lee Filter": "#14b8a6", "TV Filter": "#22c55e",
};
const LEVELS = ["Low", "Medium", "High"];
const short = (name) => name.replace(" Filter", "");

// Plain-language help for every noise and filter (used in chips, tooltips and the "How it works" tab)
const NOISE_INFO = {
  "Gaussian": { emoji: "🎞️", tip: "Fine grain on every pixel, like a photo taken in low light.",
    text: "A small random number is added to every pixel. It looks like fine grain all over the photo.",
    formula: "noisy = clean + n   (n is random, average 0)", where: "Low-light photos, cheap sensors" },
  "Salt-and-Pepper": { emoji: "🧂", tip: "Random pixels become pure black or pure white, like dust or broken pixels.",
    text: "Some random pixels are switched to pure black (0) or pure white (1). All other pixels stay the same.",
    formula: "pixel → 0 or 1   for a share of the pixels", where: "Errors when sending images, dusty scans" },
  "Speckle": { emoji: "✨", tip: "Grain that is stronger in bright areas, like ultrasound or radar images.",
    text: "The random change is multiplied by the pixel itself, so bright areas get more noise than dark areas.",
    formula: "noisy = clean + clean × n", where: "Ultrasound, radar and laser images" },
  "Poisson": { emoji: "💡", tip: "Counting noise: when a camera catches very little light, the count of light is random.",
    text: "A camera counts tiny units of light (photons). With little light the count is random, so the photo gets grainy. A smaller 'peak' means less light and more noise.",
    formula: "noisy = Poisson(clean × peak) / peak", where: "Night photos, astronomy, X-ray" },
};
const FILTER_INFO = {
  "Gaussian Filter": { tip: "Blurs the photo a little by taking a weighted average of nearby pixels.",
    text: "Each pixel becomes a weighted average of its neighbours. Close neighbours count more.",
    formula: "new = Σ weight × neighbour", good: "Very fast, removes grain", bad: "Blurs edges" },
  "Median Filter": { tip: "Sorts the nearby pixels and keeps the middle one. Great at removing black and white dots.",
    text: "It sorts the pixels in a small window and keeps the middle value. Black or white dots are never in the middle, so they disappear.",
    formula: "new = middle value of the sorted window", good: "Best for salt-and-pepper", bad: "Can erase thin lines" },
  "Bilateral Filter": { tip: "Averages only neighbours that look similar, so edges stay sharp.",
    text: "Like a blur, but a neighbour only counts if it is close AND has a similar brightness. Pixels across an edge are ignored.",
    formula: "weight = closeness × similarity", good: "Keeps edges sharp", bad: "Weak on strong noise" },
  "NLM Filter": { tip: "Finds small patches that look alike anywhere nearby and averages them together.",
    text: "Non-Local Means compares the small patch around each pixel with other patches nearby and averages the ones that look the same.",
    formula: "new = average of pixels with similar patches", good: "High quality, keeps texture", bad: "Slowest filter" },
  "Lee Filter": { tip: "Smooths flat areas but leaves edges alone. Made for speckle noise.",
    text: "It checks how much the pixels change in each window. Flat area → smooth it. Busy area (edge) → keep the pixel.",
    formula: "new = average + W × (pixel − average)", good: "Best for speckle", bad: "Less useful for dots" },
  "TV Filter": { tip: "Flattens small ups and downs but keeps big jumps (edges).",
    text: "Total Variation looks for a smoother photo that is still close to the noisy one. Small wiggles are flattened, big edges stay.",
    formula: "make the photo smooth, but stay close to the noisy one", good: "Great for grain, keeps edges", bad: "Can look a bit cartoon-like" },
};

// ------------------------------------------------------------------ tooltip after 1.5 s
(function setupTooltips() {
  const tip = $("tooltip");
  let timer = null, current = null;
  const hide = () => { clearTimeout(timer); tip.classList.remove("show"); current = null; };
  document.addEventListener("mouseover", (event) => {
    const el = event.target.closest("[data-tip]");
    if (el === current) return;
    hide();
    if (!el) return;
    current = el;
    timer = setTimeout(() => {
      tip.textContent = el.dataset.tip;
      const box = el.getBoundingClientRect();
      tip.style.left = "0px"; tip.style.top = "0px"; tip.classList.add("show");
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let left = Math.min(Math.max(8, box.left + box.width / 2 - w / 2), window.innerWidth - w - 8);
      let top = box.bottom + 10;
      if (top + h > window.innerHeight - 8) top = box.top - h - 10;
      tip.style.left = left + "px"; tip.style.top = top + "px";
    }, 1500);
  });
  document.addEventListener("mousedown", hide);
  window.addEventListener("scroll", hide, true);
})();

function toast(message) {
  const t = $("toast"); t.textContent = message; t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 4500);
}

// ------------------------------------------------------------------ number and download helpers
// PSNR is "∞" when the result is exactly the clean photo (no error at all).
const psnrText = (v) => (v === null || v === undefined ? "∞" : v);
const psnrSort = (r) => (r.psnr === null ? Infinity : r.psnr);
function download(name, content, type) {
  const url = content.startsWith("data:") ? content : URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a"); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
const csvLine = (cells) => cells.map((c) => (typeof c === "string" && /[",\n]/.test(c) ? `"${c.replaceAll('"', '""')}"` : c)).join(",");

// ------------------------------------------------------------------ small helpers
function segmented(id, onChange) {
  const box = $(id);
  box.addEventListener("click", (event) => {
    const button = event.target.closest(".seg");
    if (!button) return;
    box.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b === button));
    if (onChange) onChange(button.dataset.value);
  });
  return () => box.querySelector(".seg.active").dataset.value;
}

function chipGroup(id, items, { multi = true, on = items, tips = {}, labels = {}, onChange } = {}) {
  const box = $(id);
  box.innerHTML = "";
  items.forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (on.includes(item) ? " on" : "");
    chip.textContent = labels[item] || item;
    chip.dataset.value = item;
    if (tips[item]) chip.dataset.tip = tips[item];
    chip.addEventListener("click", () => {
      if (!multi) box.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
      chip.classList.toggle("on", multi ? !chip.classList.contains("on") : true);
      if (onChange) onChange();
    });
    box.appendChild(chip);
  });
  return {
    get: () => [...box.querySelectorAll(".chip.on")].map((c) => c.dataset.value),
    set: (values) => box.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", values.includes(c.dataset.value))),
  };
}

function loading(container, text) {
  container.innerHTML = `<div class="empty"><div class="spinner"></div><h2>${text}</h2></div>`;
}

function destroyCharts(prefix) {
  Object.keys(charts).filter((k) => k.startsWith(prefix)).forEach((k) => { charts[k].destroy(); delete charts[k]; });
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab.dataset.tab));
}));

// ------------------------------------------------------------------ TAB 1: one photo
const single = { dataset: "BSD68", image: null, file: null };
let noiseChips, filterChips;
const getSource = segmented("source-toggle", (v) => {
  $("source-dataset").classList.toggle("hidden", v !== "dataset");
  $("source-upload").classList.toggle("hidden", v !== "upload");
});
segmented("dataset-toggle", (v) => { single.dataset = v; showThumbs(); });
const getScope = segmented("scope-toggle", (v) => $("one-noise").classList.toggle("hidden", v === "all"));
const getLevel = segmented("level-toggle", () => updateLevelNote());
const getMode = segmented("mode-toggle", (v) => { $("custom-params").classList.toggle("hidden", v !== "custom"); buildParamSliders(); });

function showThumbs() {
  const box = $("thumbs");
  const names = CONFIG.datasets[single.dataset];
  if (!names.includes(single.image)) single.image = names[0];
  box.innerHTML = "";
  names.forEach((name) => {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/dataset/${single.dataset}/${name}`;
    img.alt = name;
    img.dataset.tip = `Use ${name} from ${single.dataset}`;
    if (name === single.image) img.classList.add("on");
    img.addEventListener("click", () => {
      single.image = name;
      box.querySelectorAll("img").forEach((i) => i.classList.toggle("on", i === img));
    });
    box.appendChild(img);
  });
}

function updateLevelNote() {
  const noise = noiseChips.get()[0];
  const level = getLevel();
  const custom = CONFIG.noise_custom[noise];
  $("custom-level").classList.toggle("hidden", level !== "Custom");
  const slider = $("custom-value");
  slider.min = custom.min; slider.max = custom.max; slider.step = custom.step;
  if (level === "Custom") {
    if (!slider.dataset.noise || slider.dataset.noise !== noise) { slider.value = CONFIG.noise_levels[noise].Medium; slider.dataset.noise = noise; }
    $("custom-value-label").textContent = slider.value;
    $("level-note").textContent = `Custom ${custom.unit}.`;
  } else {
    $("level-note").textContent = `${level} = ${CONFIG.noise_levels[noise][level]} (${custom.unit}), the same value main.py uses.`;
  }
}
$("custom-value").addEventListener("input", (e) => { $("custom-value-label").textContent = e.target.value; });

// upload box
$("upload").addEventListener("change", (e) => {
  single.file = e.target.files[0] || null;
  $("upload-label").textContent = single.file ? `✔ ${single.file.name}` : "Click or drop a photo here";
});
const drop = document.querySelector(".dropzone");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("drag")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) { single.file = e.dataTransfer.files[0]; $("upload-label").textContent = `✔ ${single.file.name}`; }
});

$("filters-all").addEventListener("click", () => { filterChips.set(CONFIG.filters); buildParamSliders(); });
$("filters-none").addEventListener("click", () => { filterChips.set([]); buildParamSliders(); });

function buildParamSliders() {
  const box = $("custom-params");
  if (getMode() !== "custom") return;
  box.innerHTML = "";
  filterChips.get().forEach((name) => {
    const group = document.createElement("div");
    group.className = "param-group";
    group.innerHTML = `<h4>${name}</h4>`;
    CONFIG.filter_params[name].forEach((p) => {
      let value = CONFIG.defaults[name][p.name];
      if (value === null || value === undefined) value = 0.1; // NLM "automatic" h starts at 0.1
      const row = document.createElement("div");
      row.className = "param";
      row.innerHTML = `<div class="param-label"><span>${p.label}</span><b>${value}</b></div>
        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${value}" data-filter="${name}" data-param="${p.name}" data-tip="${p.tip}">`;
      row.querySelector("input").addEventListener("input", (e) => { row.querySelector("b").textContent = e.target.value; });
      group.appendChild(row);
    });
    box.appendChild(group);
  });
  if (!filterChips.get().length) box.innerHTML = `<p class="small-note">Select a filter to change its settings.</p>`;
}

$("run-single").addEventListener("click", async () => {
  const filters = filterChips.get();
  if (!filters.length) return toast("Please choose at least one filter.");
  const form = new FormData();
  if (getSource() === "upload") {
    if (!single.file) return toast("Please choose a photo to upload first.");
    if (!/\.(png|jpe?g|bmp)$/i.test(single.file.name)) return toast("Please upload a PNG, JPG or BMP photo.");
    if (single.file.size > CONFIG.max_file_mb * 1024 * 1024) return toast(`This file is too big. Please use a photo smaller than ${CONFIG.max_file_mb} MB.`);
    form.append("upload", single.file);
  } else {
    form.append("dataset", single.dataset);
    form.append("image", single.image);
  }
  form.append("scope", getScope());
  form.append("noise", noiseChips.get()[0]);
  form.append("level", getLevel());
  form.append("custom_value", $("custom-value").value);
  form.append("seed", $("seed").value || 42);
  form.append("mode", getMode());
  filters.forEach((f) => form.append("filters", f));
  document.querySelectorAll("#custom-params input[type=range]").forEach((s) => form.append(`${s.dataset.filter}::${s.dataset.param}`, s.value));

  const button = $("run-single");
  button.disabled = true; button.textContent = "Working…";
  loading($("single-results"), getScope() === "all" ? "Trying every noise and level… (a few seconds)" : "Adding noise and cleaning it…");
  try {
    const response = await fetch("/api/run-single", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    data.scope === "all" ? renderAll(data) : renderOne(data);
  } catch (error) {
    $("single-results").innerHTML = `<div class="empty"><h2>Could not run</h2><p>${error.message}</p></div>`;
  } finally {
    button.disabled = false; button.textContent = "Run";
  }
});

function renderOne(data) {
  const box = $("single-results");
  const ranked = [...data.filters].sort((a, b) => psnrSort(b) - psnrSort(a));
  const best = ranked[0];
  const fastest = [...data.filters].sort((a, b) => a.time_ms - b.time_ms)[0];
  const gain = best.psnr === null ? "∞" : (best.psnr - data.noisy_psnr).toFixed(2);
  const levelText = data.level === "Custom" ? `custom ${data.value}` : `${data.level.toLowerCase()} (${data.value})`;
  const st = data.stats, info = data.run_info;
  box.innerHTML = `
    <div class="summary">
      <div class="stat" data-tip="How damaged the photo is before cleaning. Lower PSNR = more noise."><div class="k">Noisy photo</div>
        <div class="v">${psnrText(data.noisy_psnr)} dB</div><div class="s">SSIM ${data.noisy_ssim} · ${data.noise} noise, ${levelText}</div></div>
      <div class="stat best" data-tip="The filter whose result is closest to the clean photo (highest PSNR)."><div class="k">Best filter</div>
        <div class="v">${short(best.filter)}</div><div class="s">${psnrText(best.psnr)} dB · <span class="gain up">+${gain} dB</span> better than noisy</div></div>
      <div class="stat" data-tip="The filter that finished the job in the shortest time."><div class="k">Fastest filter</div>
        <div class="v">${short(fastest.filter)}</div><div class="s">${fastest.time_ms} ms</div></div>
      <div class="stat" data-tip="What the noise really did to the photo, measured on every pixel."><div class="k">What the noise did</div>
        <div class="v">${(st.changed_share * 100).toFixed(1)}%</div><div class="s">of pixels changed · measured σ ≈ ${st.measured_sigma} grey levels</div></div>
    </div>
    <div class="section-title">Pictures <span class="small-note">— click any cleaned photo to compare it closely</span></div>
    <div class="grid" id="tiles"></div>
    <div class="section-title">Scores</div>
    <div class="charts">
      <div class="chart-card"><h4 data-tip="PSNR: how close the pixel values are to the clean photo. Higher is better.">PSNR (dB) — higher is better</h4><canvas id="one-psnr"></canvas></div>
      <div class="chart-card"><h4 data-tip="SSIM: how similar the shapes and textures look, from 0 to 1. Higher is better.">SSIM (0–1) — higher is better</h4><canvas id="one-ssim"></canvas></div>
    </div>
    <div class="table-wrap" style="margin-top:14px"><table class="scores" id="one-table"></table></div>
    <div class="note-box" data-tip="Things to keep in mind so the numbers are read correctly.">
      <b>How to read these numbers</b>
      <ul>
        <li>All scores compare with the clean photo at full size (the pictures on screen may be shrunk, the scores are not).</li>
        <li>MSE is the average squared pixel error (0–1 scale); PSNR is the same error on a log scale; SSIM looks at structure.</li>
        <li>${info.image_source === "upload" ? "Your uploaded photo is used as the “clean” reference, so any grain it already had counts as correct." : "Dataset photos are treated as perfectly clean."}${info.image_note ? " It was " + info.image_note + " to stay fast." : ""}</li>
        <li>This is one photo and one random seed. For a fair ranking, use “Whole dataset” with repeats.</li>
      </ul>
    </div>
    <div class="section-title">Reproduce this result</div>
    <div class="repro" id="repro"></div>`;

  const tiles = $("tiles");
  const addTile = (title, image, lines, badge, onClick, tip) => {
    const tile = document.createElement("div");
    tile.className = "tile";
    if (tip) tile.dataset.tip = tip;
    tile.innerHTML = `<img src="${image}" alt="${title}">${badge || ""}<div class="info"><div class="name">${title}</div>${lines}</div>`;
    if (onClick) tile.addEventListener("click", onClick);
    tiles.appendChild(tile);
  };
  addTile("Original (clean)", data.original, `<div class="metrics">The photo before any noise</div><div class="params">fingerprint ${info.image_fingerprint}</div>`, `<span class="badge grey">Reference</span>`);
  addTile(`Noisy (${data.noise})`, data.noisy, `<div class="metrics">PSNR <b>${psnrText(data.noisy_psnr)} dB</b> · SSIM <b>${data.noisy_ssim}</b></div>
    <div class="metrics">MSE <b>${data.noisy_mse}</b></div><div class="params">fingerprint ${info.noisy_fingerprint}</div>`, `<span class="badge grey">Before cleaning</span>`);
  ranked.forEach((row, i) => {
    const diff = row.psnr === null ? "∞" : (row.psnr - data.noisy_psnr).toFixed(2);
    const params = Object.entries(row.params).map(([k, v]) => `${k} ${v === null ? "auto" : v}`).join(", ");
    addTile(row.filter, row.image,
      `<div class="metrics">PSNR <b>${psnrText(row.psnr)} dB</b> <span class="gain ${diff === "∞" || diff >= 0 ? "up" : "down"}">(${diff === "∞" || diff >= 0 ? "+" : ""}${diff})</span></div>
       <div class="metrics">SSIM <b>${row.ssim}</b> · MSE <b>${row.mse}</b> · ${row.time_ms} ms</div>
       <div class="params">${params || "default settings"}</div>`,
      (i === 0 ? `<span class="badge">★ Best</span>` : "") + `<span class="badge rank">#${i + 1}</span>`,
      () => openCompare(data, row), FILTER_INFO[row.filter].tip + " Click to compare closely.");
  });

  // exact numbers table
  $("one-table").innerHTML = `<tr><th>Image</th><th data-tip="Higher is better.">PSNR (dB)</th><th data-tip="Higher is better.">SSIM</th><th data-tip="Average squared pixel error. Lower is better.">MSE</th><th>Time (ms)</th><th>Settings used</th></tr>
    <tr><td>Noisy (before cleaning)</td><td>${psnrText(data.noisy_psnr)}</td><td>${data.noisy_ssim}</td><td>${data.noisy_mse}</td><td>–</td><td>–</td></tr>` +
    ranked.map((r, i) => `<tr><td>${i === 0 ? "★ " : ""}${r.filter}</td><td class="${i === 0 ? "best" : ""}">${psnrText(r.psnr)}</td><td>${r.ssim}</td><td>${r.mse}</td><td>${r.time_ms}</td>
      <td style="text-align:left">${Object.entries(r.params).map(([k, v]) => `${k}=${v === null ? "auto" : v}`).join(", ") || "default"}</td></tr>`).join("");

  // reproducibility panel
  const v = info.versions;
  $("repro").innerHTML = `
    <div class="repro-grid">
      <div><span>Photo</span><b>${info.image}</b> (${info.image_size})</div>
      <div><span>Noise</span><b>${data.noise}</b>, ${data.level} = ${data.value}</div>
      <div><span>Random seed</span><b>${info.seed}</b></div>
      <div><span>Filter settings</span><b>${{ tuned: "Recommended (best_parameters.json)", default: "Default (filter.py)", custom: "Custom" }[info.settings_mode]}</b></div>
      <div><span>Clean / noisy fingerprint</span><b>${info.image_fingerprint}</b> / <b>${info.noisy_fingerprint}</b></div>
      <div><span>Software</span>Python ${v.python} · NumPy ${v.numpy} · OpenCV ${v["opencv-python"]} · scikit-image ${v["scikit-image"]}</div>
    </div>
    <p class="small-note">Same photo + same noise + same seed + same settings = exactly the same pixels, so the same fingerprints and scores. A fingerprint is a short code made from every pixel; it is an ID, not a quality score.</p>
    <div class="row-links" style="gap:10px;flex-wrap:wrap">
      <button class="secondary" id="dl-json" data-tip="Save every setting of this run, so anyone can repeat it exactly.">⬇ Settings (JSON)</button>
      <button class="secondary" id="dl-csv" data-tip="Save all scores of this run as a table you can open in Excel.">⬇ Scores (CSV)</button>
      <button class="secondary" id="dl-noisy" data-tip="Save the noisy photo as a PNG file (lossless).">⬇ Noisy photo</button>
      <button class="secondary" id="dl-best" data-tip="Save the best cleaned photo as a PNG file (lossless).">⬇ Best result</button>
    </div>`;
  const stamp = info.image.replace(/[^a-z0-9]+/gi, "_");
  $("dl-json").onclick = () => download(`run_${stamp}.json`, JSON.stringify(info, null, 2), "application/json");
  $("dl-csv").onclick = () => download(`scores_${stamp}.csv`, [csvLine(["Image", "Noise", "Level", "Noise_Value", "Seed", "Filter", "PSNR", "SSIM", "MSE", "Time_ms", "Settings"]),
    csvLine([info.image, data.noise, data.level, data.value, info.seed, "Noisy (Baseline)", psnrText(data.noisy_psnr), data.noisy_ssim, data.noisy_mse, 0, ""]),
    ...data.filters.map((r) => csvLine([info.image, data.noise, data.level, data.value, info.seed, r.filter, psnrText(r.psnr), r.ssim, r.mse, r.time_ms, JSON.stringify(r.params)]))].join("\n"), "text/csv");
  $("dl-noisy").onclick = () => download(`noisy_${stamp}.png`, data.noisy);
  $("dl-best").onclick = () => download(`best_${short(best.filter)}_${stamp}.png`, best.image);

  destroyCharts("one-");
  const labels = data.filters.map((f) => short(f.filter));
  const colors = data.filters.map((f) => FILTER_COLORS[f.filter]);
  const baseline = (value) => ({ type: "line", label: "Noisy photo", data: labels.map(() => value), borderColor: "#9ca3af", borderDash: [6, 4], pointRadius: 0 });
  charts["one-psnr"] = new Chart($("one-psnr"), { type: "bar",
    data: { labels, datasets: [{ label: "PSNR", data: data.filters.map((f) => f.psnr), backgroundColor: colors, borderRadius: 6 }, baseline(data.noisy_psnr)] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: false } } } });
  charts["one-ssim"] = new Chart($("one-ssim"), { type: "bar",
    data: { labels, datasets: [{ label: "SSIM", data: data.filters.map((f) => f.ssim), backgroundColor: colors, borderRadius: 6 }, baseline(data.noisy_ssim)] },
    options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } } });
}

function renderAll(data) {
  const box = $("single-results");
  const filters = data.table[0].filters.map((f) => f.filter);
  box.innerHTML = `
    <div class="summary">
      <div class="stat"><div class="k">Photo</div><div class="v" style="font-size:16px">${data.label}</div><div class="s">4 noises × 3 strengths</div></div>
      <div class="stat best" id="all-best"></div>
    </div>
    <div class="segmented small" id="all-metric" style="max-width:320px">
      <button class="seg active" data-value="psnr" data-tip="Show PSNR in the table: pixel accuracy, higher is better.">PSNR (dB)</button>
      <button class="seg" data-value="ssim" data-tip="Show SSIM in the table: structure similarity, higher is better.">SSIM</button>
      <button class="seg" data-value="mse" data-tip="Show MSE in the table: average squared pixel error, lower is better.">MSE</button>
    </div>
    <div class="table-wrap"><table class="scores" id="all-table"></table></div>
    <div class="row-links" style="gap:10px;margin-top:8px"><button class="secondary" id="all-csv" data-tip="Save the whole table (PSNR, SSIM and MSE) as a CSV file.">⬇ Scores (CSV)</button>
    <button class="secondary" id="all-json" data-tip="Save the settings of this run, so anyone can repeat it exactly.">⬇ Settings (JSON)</button></div>
    <p class="legend-note">The box marks the best filter in each row. Click a noisy picture to see it bigger.</p>
    <div class="section-title">PSNR at each strength</div>
    <div class="charts" id="all-charts"></div>`;

  const draw = (metric) => {
    const val = (x) => (metric === "psnr" ? psnrSort(x) : x[metric]);
    const shown = (x) => (metric === "psnr" ? psnrText(x.psnr) : x[metric]);
    let html = `<tr><th>Noise · strength</th><th>Noisy photo</th><th>Noisy score</th>${filters.map((f) => `<th data-tip="${FILTER_INFO[f].tip}">${short(f)}</th>`).join("")}</tr>`;
    data.table.forEach((row, index) => {
      const values = row.filters.map(val);
      const best = metric === "mse" ? Math.min(...values) : Math.max(...values);
      const noisy = { psnr: row.noisy_psnr, ssim: row.noisy_ssim, mse: row.noisy_mse };
      html += `<tr><td><b>${row.noise}</b> · ${row.level}</td><td class="thumb"><img src="${row.noisy_image}" data-index="${index}" alt=""></td>
        <td>${shown(noisy)}</td>
        ${row.filters.map((f) => `<td class="${val(f) === best ? "best" : ""}">${shown(f)}</td>`).join("")}</tr>`;
    });
    $("all-table").innerHTML = html;
    $("all-table").querySelectorAll("img").forEach((img) => img.addEventListener("click", () => {
      const row = data.table[img.dataset.index];
      openPicture(`${row.noise} noise · ${row.level}`, data.original, row.noisy_image, "Original", "Noisy");
    }));
  };
  const stamp = data.label.replace(/[^a-z0-9]+/gi, "_");
  $("all-csv").onclick = () => download(`all_noises_${stamp}.csv`, [csvLine(["Image", "Noise", "Level", "Noise_Value", "Seed", "Filter", "PSNR", "SSIM", "MSE", "Time_ms"]),
    ...data.table.flatMap((row) => [csvLine([data.label, row.noise, row.level, row.value, data.run_info.seed, "Noisy (Baseline)", psnrText(row.noisy_psnr), row.noisy_ssim, row.noisy_mse, 0]),
      ...row.filters.map((f) => csvLine([data.label, row.noise, row.level, row.value, data.run_info.seed, f.filter, psnrText(f.psnr), f.ssim, f.mse, f.time_ms]))])].join("\n"), "text/csv");
  $("all-json").onclick = () => download(`run_${stamp}.json`, JSON.stringify(data.run_info, null, 2), "application/json");
  segmented("all-metric", draw);
  draw("psnr");

  // count which filter wins most often
  const wins = {};
  data.table.forEach((row) => { const top = [...row.filters].sort((a, b) => psnrSort(b) - psnrSort(a))[0]; wins[top.filter] = (wins[top.filter] || 0) + 1; });
  const champion = Object.entries(wins).sort((a, b) => b[1] - a[1])[0];
  $("all-best").innerHTML = `<div class="k">Wins most often</div><div class="v">${short(champion[0])}</div><div class="s">best in ${champion[1]} of 12 cases (by PSNR)</div>`;

  destroyCharts("all-");
  CONFIG.noise_types.forEach((noise) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<h4>${noise}</h4><canvas id="all-${noise}"></canvas>`;
    $("all-charts").appendChild(card);
    const rows = data.table.filter((r) => r.noise === noise);
    const datasets = [{ label: "Noisy", data: rows.map((r) => r.noisy_psnr), borderColor: FILTER_COLORS["Noisy (Baseline)"], borderDash: [6, 4] }]
      .concat(filters.map((f) => ({ label: short(f), data: rows.map((r) => r.filters.find((x) => x.filter === f).psnr), borderColor: FILTER_COLORS[f] })));
    charts["all-" + noise] = new Chart($("all-" + noise), { type: "line", data: { labels: LEVELS, datasets },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } } } });
  });
}

// ------------------------------------------------------------------ compare view
function openCompare(data, row) {
  $("modal-title").textContent = `${row.filter} · PSNR ${row.psnr} dB · SSIM ${row.ssim}`;
  const show = (mode) => {
    if (mode === "noisy") setCompare(data.noisy, row.image, "Noisy", "Cleaned");
    if (mode === "original") setCompare(data.original, row.image, "Original", "Cleaned");
    if (mode === "diff") setCompare(data.original, row.diff, "Original", "Difference ×4");
    $("compare").classList.toggle("hidden", mode === "zoom");
    $("zoom-view").classList.toggle("hidden", mode !== "zoom");
    if (mode === "zoom") startZoom(data, row);
    $("modal-note").textContent = mode === "diff" ? "In the difference picture, black = same as the original, bright = still different."
      : mode === "zoom" ? "Each square is one real pixel. Click the small picture to move the zoom area." : "Drag the line to compare.";
  };
  $("compare-toggle").classList.remove("hidden");
  $("compare-toggle").querySelectorAll(".seg").forEach((b, i) => b.classList.toggle("active", i === 0));
  $("compare-toggle").onclick = (e) => {
    const b = e.target.closest(".seg"); if (!b) return;
    $("compare-toggle").querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
    show(b.dataset.value);
  };
  show("noisy");
  $("modal").classList.remove("hidden");
}

function openPicture(title, a, b, tagA, tagB) {
  $("modal-title").textContent = title;
  $("compare").classList.remove("hidden");
  $("zoom-view").classList.add("hidden");
  $("compare-toggle").classList.add("hidden");
  setCompare(a, b, tagA, tagB);
  $("modal-note").textContent = "Drag the line to compare.";
  $("modal").classList.remove("hidden");
}

function setCompare(left, right, tagLeft, tagRight) {
  // The left part shows image A, the right part shows image B.
  $("cmp-a").src = right; $("cmp-b").src = left;
  $("cmp-tag-a").textContent = tagLeft; $("cmp-tag-b").textContent = tagRight;
  const fit = () => {
    // keep the whole picture on screen, then make both pictures exactly the same size
    const img = $("cmp-a"), box = $("compare");
    const ratio = img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 1;
    box.style.width = Math.min(box.parentElement.clientWidth, window.innerHeight * 0.68 * ratio) + "px";
    $("cmp-b").style.width = box.offsetWidth + "px";
    moveHandle(0.5);
  };
  $("cmp-a").onload = fit;
  requestAnimationFrame(fit);
}
function moveHandle(fraction) {
  fraction = Math.min(Math.max(fraction, 0), 1);
  $("cmp-top").style.width = fraction * 100 + "%";
  $("cmp-handle").style.left = fraction * 100 + "%";
}
(function setupCompare() {
  const box = $("compare");
  let dragging = false;
  const move = (x) => { const r = box.getBoundingClientRect(); moveHandle((x - r.left) / r.width); };
  box.addEventListener("pointerdown", (e) => { dragging = true; move(e.clientX); });
  window.addEventListener("pointermove", (e) => { if (dragging) move(e.clientX); });
  window.addEventListener("pointerup", () => { dragging = false; });
  window.addEventListener("resize", () => { if (!$("modal").classList.contains("hidden")) $("cmp-a").onload(); });
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("modal").classList.add("hidden"); });
})();

// ------------------------------------------------------------------ pixel zoom
const zoom = { images: [], cx: 0.5, cy: 0.5 };
function loadImage(src) {
  return new Promise((resolve) => { const img = new Image(); img.onload = () => resolve(img); img.src = src; });
}
async function startZoom(data, row) {
  zoom.images = await Promise.all([data.original, data.noisy, row.image].map(loadImage));
  $("zoom-overview-img").src = row.image;
  $("zoom-cleaned-label").textContent = `Cleaned (${short(row.filter)})`;
  drawZoom();
}
function drawZoom() {
  if (!zoom.images.length) return;
  const [original] = zoom.images;
  const W = original.naturalWidth, H = original.naturalHeight;
  const size = Math.min(+$("zoom-size").value, W, H);
  $("zoom-size-label").textContent = `${size} × ${size} pixels`;
  // top-left corner of the square, kept inside the photo
  const x = Math.round(Math.min(Math.max(zoom.cx * W - size / 2, 0), W - size));
  const y = Math.round(Math.min(Math.max(zoom.cy * H - size / 2, 0), H - size));
  ["zoom-original", "zoom-noisy", "zoom-cleaned"].forEach((id, i) => {
    const ctx = $(id).getContext("2d");
    ctx.imageSmoothingEnabled = false; // show real square pixels, no blur
    ctx.drawImage(zoom.images[i], x, y, size, size, 0, 0, 300, 300);
  });
  const box = $("zoom-box");
  box.style.left = (100 * x / W) + "%"; box.style.top = (100 * y / H) + "%";
  box.style.width = (100 * size / W) + "%"; box.style.height = (100 * size / H) + "%";
}
$("zoom-overview").addEventListener("click", (e) => {
  const r = $("zoom-overview").getBoundingClientRect();
  zoom.cx = (e.clientX - r.left) / r.width; zoom.cy = (e.clientY - r.top) / r.height;
  drawZoom();
});
$("zoom-size").addEventListener("input", drawZoom);

// ------------------------------------------------------------------ dashboard (whole dataset and saved runs)
function renderDashboard(container, summary, { title, images, csvUrl, configUrl, savedTo, repeats = 1, prefix }) {
  const noises = CONFIG.noise_types.filter((n) => summary.some((r) => r.noise === n));
  const filters = CONFIG.filters.filter((f) => summary.some((r) => r.filter === f));
  const levels = LEVELS.filter((l) => summary.some((r) => r.level === l));
  const pick = (noise, level, filter) => summary.find((r) => r.noise === noise && r.level === level && r.filter === filter);

  const bestCards = noises.map((noise) => {
    const totals = filters.map((f) => ({ f, psnr: levels.reduce((s, l) => s + (pick(noise, l, f)?.psnr || 0), 0) / levels.length }));
    const top = totals.sort((a, b) => b.psnr - a.psnr)[0];
    return `<div class="stat best" data-tip="The filter with the highest average PSNR for this noise, over the chosen strengths.">
      <div class="k">${NOISE_INFO[noise].emoji} ${noise}</div><div class="v">${short(top.f)}</div><div class="s">average ${top.psnr.toFixed(2)} dB</div></div>`;
  }).join("");

  container.innerHTML = `
    <div class="summary">
      <div class="stat"><div class="k">Experiment</div><div class="v" style="font-size:17px">${title}</div><div class="s">${images} photo(s)${repeats > 1 ? ` × ${repeats} seeds` : ""} · averages over all runs</div></div>
      ${csvUrl || configUrl ? `<div class="stat"><div class="k">Files</div><div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
        ${csvUrl ? `<a class="secondary" href="${csvUrl}" data-tip="Download every single measurement as a CSV file you can open in Excel.">⬇ Results (CSV)</a>` : ""}
        ${configUrl ? `<a class="secondary" href="${configUrl}" data-tip="Download the exact settings (photos, noise values, seeds, software versions) so the run can be repeated.">⬇ Settings (JSON)</a>` : ""}</div></div>` : ""}
      ${savedTo ? `<div class="stat best"><div class="k">Saved</div><div class="v" style="font-size:18px">results/${savedTo}</div><div class="s">also listed under “Saved results”</div></div>` : ""}
    </div>
    <div class="section-title">Best filter for each noise</div>
    <div class="summary">${bestCards}</div>
    <div class="section-title">How the strength changes the score</div>
    <div class="segmented small" id="${prefix}metric" style="max-width:320px">
      <button class="seg active" data-value="psnr" data-tip="Pixel accuracy in dB. Higher is better.">PSNR</button>
      <button class="seg" data-value="ssim" data-tip="Structure similarity from 0 to 1. Higher is better.">SSIM</button>
      ${summary.some((r) => r.mse !== null && r.mse !== undefined) ? `<button class="seg" data-value="mse" data-tip="Average squared pixel error. Lower is better.">MSE</button>` : ""}
    </div>
    <div class="charts" id="${prefix}charts"></div>
    <div class="section-title">All average scores</div>
    <div class="table-wrap"><table class="scores" id="${prefix}table"></table></div>
    <p class="legend-note">The box marks the best filter in each row. “± x” is the standard deviation: how much the score changes from photo to photo (and seed to seed). If two filters are closer than this, the difference may not be meaningful.</p>
    <div class="section-title">Speed</div>
    <div class="chart-card"><h4 data-tip="Average time one filter needs for one photo. Lower is faster.">Average time per photo (milliseconds, log scale)</h4><canvas id="${prefix}time"></canvas></div>`;

  const draw = (metric) => {
    destroyCharts(prefix + "c-");
    $(prefix + "charts").innerHTML = "";
    noises.forEach((noise) => {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<h4 data-tip="${NOISE_INFO[noise].tip}">${NOISE_INFO[noise].emoji} ${noise}</h4><canvas id="${prefix}c-${noise}"></canvas>`;
      $(prefix + "charts").appendChild(card);
      const datasets = ["Noisy (Baseline)", ...filters].filter((f) => pick(noise, levels[0], f)).map((f) => ({
        label: short(f).replace("Noisy (Baseline)", "Noisy"), data: levels.map((l) => pick(noise, l, f)?.[metric]),
        borderColor: FILTER_COLORS[f], backgroundColor: FILTER_COLORS[f], borderDash: f.startsWith("Noisy") ? [6, 4] : [], tension: .25,
      }));
      charts[prefix + "c-" + noise] = new Chart($(`${prefix}c-${noise}`), { type: "line", data: { labels: levels, datasets },
        options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } } } });
    });
    let html = `<tr><th>Noise · strength</th><th>Noisy</th>${filters.map((f) => `<th data-tip="${FILTER_INFO[f].tip}">${short(f)}</th>`).join("")}</tr>`;
    const cell = (r) => {
      if (!r) return "–";
      const spread = metric === "psnr" ? r.psnr_std : metric === "ssim" ? r.ssim_std : null;
      return `${metric === "psnr" ? psnrText(r.psnr) : r[metric]}${spread ? `<span class="spread"> ±${spread}</span>` : ""}`;
    };
    noises.forEach((noise) => levels.forEach((level) => {
      const rows = filters.map((f) => pick(noise, level, f));
      const values = rows.map((r) => (r ? (metric === "psnr" ? (r.psnr === null ? Infinity : r.psnr) : r[metric]) : NaN));
      const best = metric === "mse" ? Math.min(...values) : Math.max(...values);
      html += `<tr><td><b>${noise}</b> · ${level}</td><td>${cell(pick(noise, level, "Noisy (Baseline)"))}</td>${rows.map((r, i) => `<td class="${values[i] === best ? "best" : ""}">${cell(r)}</td>`).join("")}</tr>`;
    }));
    $(prefix + "table").innerHTML = html;
  };
  segmented(prefix + "metric", draw);
  draw("psnr");

  destroyCharts(prefix + "time");
  const times = filters.map((f) => { const rows = summary.filter((r) => r.filter === f); return rows.reduce((s, r) => s + r.time_ms, 0) / rows.length; });
  charts[prefix + "time"] = new Chart($(prefix + "time"), { type: "bar",
    data: { labels: filters.map(short), datasets: [{ label: "ms per photo", data: times.map((t) => +t.toFixed(2)), backgroundColor: filters.map((f) => FILTER_COLORS[f]), borderRadius: 6 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw} ms per photo` } } },
      scales: { x: { type: "logarithmic", ticks: { maxRotation: 0, callback: (v) => ([0.1, 1, 10, 100, 1000].includes(v) ? v + " ms" : "") } } } } });
}

// ------------------------------------------------------------------ TAB 2: whole dataset
let batchNoises, batchLevels, batchFilters, batchJob = null;
const getBatchDataset = segmented("batch-dataset", () => updateBatchCount(true));
const getBatchMode = segmented("batch-mode");

function updateBatchCount(reset) {
  const total = CONFIG.datasets[getBatchDataset()].length;
  const slider = $("batch-count");
  slider.max = total;
  if (reset) slider.value = Math.min(5, total);
  const n = +slider.value;
  $("batch-count-label").textContent = n === total ? `all ${total}` : `${n} of ${total}`;
  updateEstimate();
}
$("batch-count").addEventListener("input", () => updateBatchCount(false));
$("batch-repeats").addEventListener("input", () => updateEstimate());

function updateEstimate() {
  if (!batchFilters) return;
  const perCase = { "Gaussian Filter": 1, "Median Filter": 2, "Bilateral Filter": 8, "NLM Filter": 150, "Lee Filter": 3, "TV Filter": 35 };
  const repeats = +$("batch-repeats").value;
  $("batch-repeats-label").textContent = repeats === 1 ? "1 (same seeds as main.py)" : `${repeats} different seeds`;
  const cases = +$("batch-count").value * batchNoises.get().length * batchLevels.get().length * repeats;
  const ms = cases * (40 + batchFilters.get().reduce((s, f) => s + perCase[f], 0));
  const sec = Math.round(ms / 1000);
  $("batch-estimate").textContent = `${cases} noisy photos to clean. Rough time: ${sec < 60 ? sec + " seconds" : Math.round(sec / 60) + " minute(s)"}.`;
}

$("run-batch").addEventListener("click", async () => {
  const button = $("run-batch");
  if (batchJob) { // the button works as "Stop" while running
    await fetch(`/api/batch/${batchJob}/cancel`, { method: "POST" });
    return;
  }
  const total = CONFIG.datasets[getBatchDataset()].length;
  const body = { dataset: getBatchDataset(), count: +$("batch-count").value === total ? "all" : +$("batch-count").value,
    noises: batchNoises.get(), levels: batchLevels.get(), filters: batchFilters.get(), mode: getBatchMode(),
    repeats: +$("batch-repeats").value, save: $("batch-save").checked };
  const response = await fetch("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) return toast(data.error);
  batchJob = data.job;
  button.textContent = "Stop"; button.dataset.tip = "Stop the experiment.";
  $("batch-results").innerHTML = `<div class="empty"><div class="spinner"></div><h2>Running the experiment…</h2>
    <div class="progress"><div id="batch-bar"></div></div><p id="batch-status">Starting…</p></div>`;
  pollBatch(body);
});

async function pollBatch(body) {
  const data = await (await fetch(`/api/batch/${batchJob}`)).json();
  if (data.state === "running") {
    $("batch-bar").style.width = (100 * data.done / data.total) + "%";
    $("batch-status").textContent = `${data.done} of ${data.total} done · ${data.seconds} s · now: ${data.current}`;
    return setTimeout(() => pollBatch(body), 800);
  }
  const job = batchJob;
  batchJob = null;
  $("run-batch").textContent = "Start experiment";
  $("run-batch").dataset.tip = "Start the experiment. You can keep using the page while it runs.";
  if (data.state === "cancelled") {
    $("batch-results").innerHTML = `<div class="empty"><h2>Stopped</h2><p>The experiment was stopped before the end.</p></div>`;
    return;
  }
  const images = body.count === "all" ? CONFIG.datasets[body.dataset].length : body.count;
  renderDashboard($("batch-results"), data.summary, { title: `${body.dataset}, ${body.mode === "tuned" ? "recommended" : "default"} settings`,
    images, csvUrl: `/api/batch/${job}/csv`, configUrl: `/api/batch/${job}/config`, savedTo: data.saved_to, repeats: body.repeats, prefix: "b-" });
  if (data.saved_to) refreshSavedList();
}

// ------------------------------------------------------------------ TAB 3: saved results
$("load-saved").addEventListener("click", async () => {
  const name = $("saved-select").value;
  if (!name) return;
  loading($("saved-results"), `Opening ${name}…`);
  const data = await (await fetch(`/api/saved/${name}`)).json();
  if (data.error) return toast(data.error);
  renderDashboard($("saved-results"), data.summary, { title: name, images: data.images,
    csvUrl: `/results-file/${name}/experiment_results.csv`, configUrl: data.has_config ? `/results-file/${name}/run_config.json` : null, prefix: "s-" });
  if (data.pictures.length) {
    const block = document.createElement("div");
    block.innerHTML = `<div class="section-title">Comparison pictures saved by main.py</div><div class="pictures">${data.pictures.map((p) =>
      `<figure><img src="/results-file/${name}/comparisons/${p}" alt="${p}" data-tip="Click to open the full-size picture."><figcaption>${p.replace(".png", "").replaceAll("_", " ")}</figcaption></figure>`).join("")}</div>`;
    block.querySelectorAll("img").forEach((img) => img.addEventListener("click", () => window.open(img.src, "_blank")));
    $("saved-results").appendChild(block);
  }
});

async function refreshSavedList() {
  const cfg = await (await fetch("/api/config")).json();
  CONFIG.results = cfg.results;
  $("saved-select").innerHTML = CONFIG.results.length ? CONFIG.results.slice().reverse().map((r) => `<option>${r}</option>`).join("") : "<option value=''>No saved runs yet</option>";
}

// ------------------------------------------------------------------ TAB: find the best setting
let sweepNoise, sweepFilter;
const getSweepLevel = segmented("sweep-level");
function fillSweepImages() {
  const ds = $("sweep-dataset").value;
  $("sweep-image").innerHTML = CONFIG.datasets[ds].map((n) => `<option>${n}</option>`).join("");
  showSweepPreview();
}
function showSweepPreview() { $("sweep-preview").src = `/dataset/${$("sweep-dataset").value}/${$("sweep-image").value}`; }
function fillSweepParams() {
  const f = sweepFilter.get()[0];
  $("sweep-param").innerHTML = CONFIG.filter_params[f].map((p) => `<option value="${p.name}">${p.label} (${p.min}–${p.max})</option>`).join("");
}
$("sweep-dataset").addEventListener("change", fillSweepImages);
$("sweep-image").addEventListener("change", showSweepPreview);
$("run-sweep").addEventListener("click", async () => {
  const form = new FormData();
  form.append("dataset", $("sweep-dataset").value); form.append("image", $("sweep-image").value);
  form.append("noise", sweepNoise.get()[0]); form.append("level", getSweepLevel());
  form.append("seed", $("sweep-seed").value || 42);
  form.append("filter", sweepFilter.get()[0]); form.append("param", $("sweep-param").value);
  const button = $("run-sweep"); button.disabled = true; button.textContent = "Testing…";
  loading($("sweep-results"), "Trying every value on the same noisy photo…");
  try {
    const response = await fetch("/api/sweep", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    renderSweep(data);
  } catch (error) {
    $("sweep-results").innerHTML = `<div class="empty"><h2>Could not run</h2><p>${error.message}</p></div>`;
  } finally { button.disabled = false; button.textContent = "Find the best value"; }
});
function renderSweep(d) {
  const others = Object.entries(d.other_settings).map(([k, v]) => `${k} = ${v === null ? "auto" : v}`).join(", ") || "none";
  $("sweep-results").innerHTML = `
    <div class="summary">
      <div class="stat best" data-tip="The value that gave the highest PSNR on this photo."><div class="k">Best ${d.param_label}</div>
        <div class="v">${d.best.value}</div><div class="s">PSNR ${psnrText(d.best.psnr)} dB · SSIM ${d.best.ssim}</div></div>
      <div class="stat"><div class="k">Noisy photo</div><div class="v">${psnrText(d.noisy_psnr)} dB</div><div class="s">${d.noise}, ${d.level} = ${d.noise_value} · seed ${d.seed}</div></div>
      <div class="stat"><div class="k">Filter</div><div class="v" style="font-size:18px">${d.filter}</div><div class="s">other settings kept fixed: ${others}</div></div>
    </div>
    <div class="charts">
      <div class="chart-card"><h4 data-tip="Each point is the same noisy photo cleaned with a different value of this setting.">PSNR and SSIM for each value of “${d.param_label}”</h4><canvas id="sweep-chart"></canvas></div>
      <div class="chart-card"><h4>Noisy vs best value</h4><div style="display:flex;gap:8px"><img src="${d.noisy}" style="width:50%;border-radius:8px" alt="Noisy"><img src="${d.best_image}" style="width:50%;border-radius:8px" alt="Best"></div>
        <p class="small-note">Left: noisy. Right: ${d.filter} with ${d.param_label} = ${d.best.value}.</p></div>
    </div>
    <div class="table-wrap" style="margin-top:14px"><table class="scores">
      <tr><th>${d.param_label}</th><th>PSNR (dB)</th><th>SSIM</th><th>MSE</th><th>Time (ms)</th></tr>
      ${d.rows.map((r) => `<tr><td>${r.value}</td><td class="${r.value === d.best.value ? "best" : ""}">${psnrText(r.psnr)}</td><td>${r.ssim}</td><td>${r.mse}</td><td>${r.time_ms}</td></tr>`).join("")}
    </table></div>
    <p class="legend-note">This is a test on one photo. A value that is best here may not be best for every photo – the team chose its recommended settings on all 12 Set12 photos.</p>`;
  destroyCharts("sweep");
  charts["sweep"] = new Chart($("sweep-chart"), { type: "line",
    data: { labels: d.rows.map((r) => r.value), datasets: [
      { label: "PSNR (dB)", data: d.rows.map((r) => r.psnr), borderColor: "#4f46e5", backgroundColor: "#4f46e5", yAxisID: "y", tension: .25 },
      { label: "SSIM", data: d.rows.map((r) => r.ssim), borderColor: "#12946b", backgroundColor: "#12946b", yAxisID: "y1", tension: .25 }] },
    options: { plugins: { legend: { position: "bottom" } }, scales: { y: { position: "left", title: { display: true, text: "PSNR (dB)" } },
      y1: { position: "right", min: 0, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: "SSIM" } } } } });
}

// ------------------------------------------------------------------ TAB 4: how it works
function buildLearn() {
  $("learn-noise").innerHTML = CONFIG.noise_types.map((n) => {
    const i = NOISE_INFO[n], v = CONFIG.noise_levels[n];
    return `<div class="learn-card orange"><h3>${i.emoji} ${n}</h3><p>${i.text}</p><div class="formula">${i.formula}</div>
      <p class="small-note">Seen in: ${i.where}<br>Strengths used: Low ${v.Low} · Medium ${v.Medium} · High ${v.High}</p></div>`;
  }).join("");
  $("learn-filters").innerHTML = CONFIG.filters.map((f) => {
    const i = FILTER_INFO[f];
    return `<div class="learn-card green"><h3>${f}</h3><p>${i.text}</p><div class="formula">${i.formula}</div>
      <div class="proscons"><span class="pro">✔ ${i.good}</span><span class="con">✖ ${i.bad}</span></div></div>`;
  }).join("");
  $("learn-scores").innerHTML = `
    <div class="learn-card purple"><h3>PSNR (dB)</h3><p>Checks how close every pixel is to the clean photo. First we take the average squared difference (MSE), then put it on a log scale.</p>
      <div class="formula">PSNR = 10 · log₁₀(1 / MSE)</div><p class="small-note">Higher is better. +10 dB means 10 times less error.</p></div>
    <div class="learn-card purple"><h3>MSE</h3><p>The plain error: for every pixel take (clean − cleaned), square it, and average over the whole photo. 0 means identical.</p>
      <div class="formula">MSE = average of (clean − result)²</div><p class="small-note">Lower is better. PSNR is made from MSE.</p></div>
    <div class="learn-card purple"><h3>SSIM (0 to 1)</h3><p>Checks if the shapes, edges and textures look the same, a bit like a human would. 1 means identical.</p>
      <div class="formula">compares brightness, contrast and structure</div><p class="small-note">Higher is better.</p></div>
    <div class="learn-card purple"><h3>Time</h3><p>How long the filter needs for one photo. A great filter that is very slow may not be useful for video or big photos.</p>
      <div class="formula">milliseconds per photo</div><p class="small-note">Lower is better.</p></div>`;
}

// ------------------------------------------------------------------ start
(async function start() {
  CONFIG = await (await fetch("/api/config")).json();
  const noiseTips = Object.fromEntries(CONFIG.noise_types.map((n) => [n, NOISE_INFO[n].tip]));
  const noiseLabels = Object.fromEntries(CONFIG.noise_types.map((n) => [n, `${NOISE_INFO[n].emoji} ${n}`]));
  const filterTips = Object.fromEntries(CONFIG.filters.map((f) => [f, FILTER_INFO[f].tip]));
  const filterLabels = Object.fromEntries(CONFIG.filters.map((f) => [f, short(f)]));

  noiseChips = chipGroup("noise-chips", CONFIG.noise_types, { multi: false, on: ["Gaussian"], tips: noiseTips, labels: noiseLabels, onChange: updateLevelNote });
  filterChips = chipGroup("filter-chips", CONFIG.filters, { tips: filterTips, labels: filterLabels, onChange: buildParamSliders });
  batchNoises = chipGroup("batch-noises", CONFIG.noise_types, { tips: noiseTips, labels: noiseLabels, onChange: updateEstimate });
  batchLevels = chipGroup("batch-levels", LEVELS, { tips: { Low: "Include a little noise.", Medium: "Include a medium amount of noise.", High: "Include a lot of noise." }, onChange: updateEstimate });
  batchFilters = chipGroup("batch-filters", CONFIG.filters, { tips: filterTips, labels: filterLabels, onChange: updateEstimate });

  sweepNoise = chipGroup("sweep-noise", CONFIG.noise_types, { multi: false, on: ["Gaussian"], tips: noiseTips, labels: noiseLabels });
  sweepFilter = chipGroup("sweep-filter", CONFIG.filters, { multi: false, on: ["NLM Filter"], tips: filterTips, labels: filterLabels, onChange: fillSweepParams });
  $("batch-repeats").max = CONFIG.max_repeats;
  if (!CONFIG.tuned_available) toast("best_parameters.json was not found, so 'Recommended' uses the default settings.");
  $("saved-select").innerHTML = CONFIG.results.length ? CONFIG.results.slice().reverse().map((r) => `<option>${r}</option>`).join("") : "<option value=''>No saved runs yet</option>";

  showThumbs();
  updateLevelNote();
  fillSweepImages();
  fillSweepParams();
  updateBatchCount(true);
  buildLearn();
})();
