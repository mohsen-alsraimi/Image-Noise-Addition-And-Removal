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
  const ranked = [...data.filters].sort((a, b) => b.psnr - a.psnr);
  const best = ranked[0];
  const fastest = [...data.filters].sort((a, b) => a.time_ms - b.time_ms)[0];
  const gain = (best.psnr - data.noisy_psnr).toFixed(2);
  const levelText = data.level === "Custom" ? `custom ${data.value}` : `${data.level.toLowerCase()} (${data.value})`;
  box.innerHTML = `
    <div class="summary">
      <div class="stat" data-tip="How damaged the photo is before cleaning. Lower PSNR = more noise."><div class="k">Noisy photo</div>
        <div class="v">${data.noisy_psnr} dB</div><div class="s">SSIM ${data.noisy_ssim} · ${data.noise} noise, ${levelText}</div></div>
      <div class="stat best" data-tip="The filter whose result is closest to the clean photo (highest PSNR)."><div class="k">Best filter</div>
        <div class="v">${short(best.filter)}</div><div class="s">${best.psnr} dB · <span class="gain ${gain >= 0 ? "up" : "down"}">${gain >= 0 ? "+" : ""}${gain} dB</span> better than noisy</div></div>
      <div class="stat" data-tip="The filter that finished the job in the shortest time."><div class="k">Fastest filter</div>
        <div class="v">${short(fastest.filter)}</div><div class="s">${fastest.time_ms} ms</div></div>
      <div class="stat" data-tip="The photo that was used."><div class="k">Photo</div><div class="v" style="font-size:16px">${data.label}</div><div class="s">${data.size} pixels</div></div>
    </div>
    <div class="section-title">Pictures <span class="small-note">— click any cleaned photo to compare it closely</span></div>
    <div class="grid" id="tiles"></div>
    <div class="section-title">Scores</div>
    <div class="charts">
      <div class="chart-card"><h4 data-tip="PSNR: how close the pixel values are to the clean photo. Higher is better.">PSNR (dB) — higher is better</h4><canvas id="one-psnr"></canvas></div>
      <div class="chart-card"><h4 data-tip="SSIM: how similar the shapes and textures look, from 0 to 1. Higher is better.">SSIM (0–1) — higher is better</h4><canvas id="one-ssim"></canvas></div>
    </div>`;

  const tiles = $("tiles");
  const addTile = (title, image, lines, badge, onClick, tip) => {
    const tile = document.createElement("div");
    tile.className = "tile";
    if (tip) tile.dataset.tip = tip;
    tile.innerHTML = `<img src="${image}" alt="${title}">${badge || ""}<div class="info"><div class="name">${title}</div>${lines}</div>`;
    if (onClick) tile.addEventListener("click", onClick);
    tiles.appendChild(tile);
  };
  addTile("Original (clean)", data.original, `<div class="metrics">The photo before any noise</div>`, `<span class="badge grey">Reference</span>`);
  addTile(`Noisy (${data.noise})`, data.noisy, `<div class="metrics">PSNR <b>${data.noisy_psnr} dB</b></div><div class="metrics">SSIM <b>${data.noisy_ssim}</b></div>`, `<span class="badge grey">Before cleaning</span>`);
  ranked.forEach((row, i) => {
    const diff = (row.psnr - data.noisy_psnr).toFixed(2);
    const params = Object.entries(row.params).map(([k, v]) => `${k} ${v === null ? "auto" : v}`).join(", ");
    addTile(row.filter, row.image,
      `<div class="metrics">PSNR <b>${row.psnr} dB</b> <span class="gain ${diff >= 0 ? "up" : "down"}">(${diff >= 0 ? "+" : ""}${diff})</span></div>
       <div class="metrics">SSIM <b>${row.ssim}</b> · ${row.time_ms} ms</div>
       <div class="params">${params || "default settings"}</div>`,
      (i === 0 ? `<span class="badge">★ Best</span>` : "") + `<span class="badge rank">#${i + 1}</span>`,
      () => openCompare(data, row), FILTER_INFO[row.filter].tip + " Click to compare closely.");
  });

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
    </div>
    <div class="table-wrap"><table class="scores" id="all-table"></table></div>
    <p class="legend-note">The box marks the best filter in each row. Click a noisy picture to see it bigger.</p>
    <div class="section-title">PSNR at each strength</div>
    <div class="charts" id="all-charts"></div>`;

  const draw = (metric) => {
    let html = `<tr><th>Noise · strength</th><th>Noisy photo</th><th>Noisy score</th>${filters.map((f) => `<th data-tip="${FILTER_INFO[f].tip}">${short(f)}</th>`).join("")}</tr>`;
    data.table.forEach((row, index) => {
      const values = row.filters.map((f) => f[metric]);
      const best = Math.max(...values);
      html += `<tr><td><b>${row.noise}</b> · ${row.level}</td><td class="thumb"><img src="${row.noisy_image}" data-index="${index}" alt=""></td>
        <td>${metric === "psnr" ? row.noisy_psnr : row.noisy_ssim}</td>
        ${row.filters.map((f) => `<td class="${f[metric] === best ? "best" : ""}">${f[metric]}</td>`).join("")}</tr>`;
    });
    $("all-table").innerHTML = html;
    $("all-table").querySelectorAll("img").forEach((img) => img.addEventListener("click", () => {
      const row = data.table[img.dataset.index];
      openPicture(`${row.noise} noise · ${row.level}`, data.original, row.noisy_image, "Original", "Noisy");
    }));
  };
  segmented("all-metric", draw);
  draw("psnr");

  // count which filter wins most often
  const wins = {};
  data.table.forEach((row) => { const top = [...row.filters].sort((a, b) => b.psnr - a.psnr)[0]; wins[top.filter] = (wins[top.filter] || 0) + 1; });
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
function renderDashboard(container, summary, { title, images, csvUrl, prefix }) {
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
      <div class="stat"><div class="k">Experiment</div><div class="v" style="font-size:17px">${title}</div><div class="s">${images} photo(s) · averages over all photos</div></div>
      ${csvUrl ? `<div class="stat"><div class="k">Raw numbers</div><div style="margin-top:10px"><a class="secondary" href="${csvUrl}" data-tip="Download every single measurement as a CSV file you can open in Excel.">⬇ Download CSV</a></div></div>` : ""}
    </div>
    <div class="section-title">Best filter for each noise</div>
    <div class="summary">${bestCards}</div>
    <div class="section-title">How the strength changes the score</div>
    <div class="segmented small" id="${prefix}metric" style="max-width:320px">
      <button class="seg active" data-value="psnr" data-tip="Pixel accuracy in dB. Higher is better.">PSNR</button>
      <button class="seg" data-value="ssim" data-tip="Structure similarity from 0 to 1. Higher is better.">SSIM</button>
    </div>
    <div class="charts" id="${prefix}charts"></div>
    <div class="section-title">All average scores</div>
    <div class="table-wrap"><table class="scores" id="${prefix}table"></table></div>
    <p class="legend-note">The box marks the best filter in each row.</p>
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
    noises.forEach((noise) => levels.forEach((level) => {
      const values = filters.map((f) => pick(noise, level, f)?.[metric]);
      const best = Math.max(...values);
      const base = pick(noise, level, "Noisy (Baseline)");
      html += `<tr><td><b>${noise}</b> · ${level}</td><td>${base ? base[metric] : "–"}</td>${values.map((v) => `<td class="${v === best ? "best" : ""}">${v}</td>`).join("")}</tr>`;
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

function updateEstimate() {
  if (!batchFilters) return;
  const perCase = { "Gaussian Filter": 1, "Median Filter": 2, "Bilateral Filter": 8, "NLM Filter": 150, "Lee Filter": 3, "TV Filter": 35 };
  const cases = +$("batch-count").value * batchNoises.get().length * batchLevels.get().length;
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
    noises: batchNoises.get(), levels: batchLevels.get(), filters: batchFilters.get(), mode: getBatchMode() };
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
    images, csvUrl: `/api/batch/${job}/csv`, prefix: "b-" });
}

// ------------------------------------------------------------------ TAB 3: saved results
$("load-saved").addEventListener("click", async () => {
  const name = $("saved-select").value;
  if (!name) return;
  loading($("saved-results"), `Opening ${name}…`);
  const data = await (await fetch(`/api/saved/${name}`)).json();
  if (data.error) return toast(data.error);
  renderDashboard($("saved-results"), data.summary, { title: name, images: data.images, prefix: "s-" });
  if (data.pictures.length) {
    const block = document.createElement("div");
    block.innerHTML = `<div class="section-title">Comparison pictures saved by main.py</div><div class="pictures">${data.pictures.map((p) =>
      `<figure><img src="/results-file/${name}/comparisons/${p}" alt="${p}" data-tip="Click to open the full-size picture."><figcaption>${p.replace(".png", "").replaceAll("_", " ")}</figcaption></figure>`).join("")}</div>`;
    block.querySelectorAll("img").forEach((img) => img.addEventListener("click", () => window.open(img.src, "_blank")));
    $("saved-results").appendChild(block);
  }
});

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

  if (!CONFIG.tuned_available) toast("best_parameters.json was not found, so 'Recommended' uses the default settings.");
  $("saved-select").innerHTML = CONFIG.results.length ? CONFIG.results.slice().reverse().map((r) => `<option>${r}</option>`).join("") : "<option value=''>No saved runs yet</option>";

  showThumbs();
  updateLevelNote();
  updateBatchCount(true);
  buildLearn();
})();
