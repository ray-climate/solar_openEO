#!/usr/bin/env python
"""Emit a self-contained static HTML viewer for the review gallery.

Writes docs/review_large/index.html. The HTML loads docs/review_large/manifest.json
at runtime and uses localStorage to persist accept/reject decisions. The page is
fully static — no backend needed; works on GitHub Pages or via file://.

Run once after render_review_gallery.py finishes.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
OUT_DIR = REPO / "docs/review_large"
HTML = OUT_DIR / "index.html"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Solar PV Review Gallery</title>
<style>
  :root {
    --keep:   #2c8a3e;
    --reject: #b53030;
    --skip:   #5a6473;
    --bg:     #1f242c;
    --panel:  #2b313a;
    --text:   #e6ebf2;
    --muted:  #9aa3b0;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         font-size: 14px; }
  header { position: sticky; top:0; z-index:10; background:var(--panel);
           padding:10px 16px; border-bottom:1px solid #444;
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header .progress { font-weight:600; font-size:15px; }
  header .progress .keep   { color:var(--keep);   }
  header .progress .reject { color:var(--reject); }
  header .progress .skip   { color:var(--skip);   }
  header button {
    background:#3a4150; color:var(--text); border:1px solid #555;
    padding:6px 12px; border-radius:4px; cursor:pointer; font-size:13px;
  }
  header button:hover { background:#4a5160; }
  header select {
    background:#3a4150; color:var(--text); border:1px solid #555;
    padding:6px 8px; border-radius:4px;
  }
  header label { color:var(--muted); }
  main { padding:14px 18px 80px; max-width:1500px; margin:auto; }
  .meta { background:var(--panel); padding:10px 14px; border-radius:6px;
          display:flex; gap:18px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }
  .meta .chip-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-weight:600; font-size:15px; }
  .meta .badge { padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; }
  .meta .badge.keep   { background:var(--keep);   color:white; }
  .meta .badge.reject { background:var(--reject); color:white; }
  .meta .badge.skip   { background:var(--skip);   color:white; }
  .meta .badge.unrev  { background:#555;          color:white; }
  .meta .kv { color:var(--muted); }
  .meta .kv b { color:var(--text); }
  img.diag { width:100%; height:auto; display:block;
             border-radius:4px; background:#000; }
  .actions { display:flex; gap:12px; margin:14px 0 4px; justify-content:center; }
  .actions button {
    padding:10px 28px; font-size:15px; font-weight:600;
    border:none; border-radius:6px; color:white; cursor:pointer;
    transition: transform 0.05s;
  }
  .actions button:active { transform: scale(0.97); }
  .actions .b-keep   { background:var(--keep);   }
  .actions .b-reject { background:var(--reject); }
  .actions .b-skip   { background:var(--skip);   }
  .actions .b-prev,
  .actions .b-next   { background:#3a4150; border:1px solid #555; color:var(--text); }
  .nav-hint { color:var(--muted); font-size:12px; text-align:center; margin-top:4px; }
  .empty { text-align:center; padding:60px; color:var(--muted); font-size:16px; }
  a, a:visited { color:#7eb8ff; }
</style>
</head>
<body>
<header>
  <span class="progress">
    <span id="prog-cur">0</span> / <span id="prog-total">0</span>
    &nbsp;reviewed:
    <span class="keep">✓ <span id="prog-keep">0</span></span>
    &nbsp;
    <span class="reject">✗ <span id="prog-reject">0</span></span>
    &nbsp;
    <span class="skip">↷ <span id="prog-skip">0</span></span>
  </span>
  <label>Filter:
    <select id="filter">
      <option value="all">all (worst Dice first)</option>
      <option value="unreviewed">unreviewed only</option>
      <option value="keep">keep</option>
      <option value="reject">reject</option>
      <option value="skip">skipped</option>
    </select>
  </label>
  <button id="btn-export">Export CSV</button>
  <button id="btn-import">Import CSV</button>
  <button id="btn-clear">Clear all decisions</button>
  <input type="file" id="import-input" accept=".csv" style="display:none">
</header>

<main>
  <div id="view"></div>
</main>

<script>
"use strict";
const LS_KEY = "solar_pv_review_decisions_v1";

let manifest = null;     // {chips: [...], n_chips, ...}
let order = [];          // indices into manifest.chips, filtered
let cursor = 0;          // index into `order`
let decisions = loadDecisions();

function loadDecisions() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
  catch { return {}; }
}
function saveDecisions() {
  localStorage.setItem(LS_KEY, JSON.stringify(decisions));
}

function applyFilter() {
  const f = document.getElementById("filter").value;
  const chips = manifest.chips;
  order = chips.map((_, i) => i).filter(i => {
    const d = decisions[chips[i].chip_id] || {};
    if (f === "all")        return true;
    if (f === "unreviewed") return !d.decision;
    return d.decision === f;
  });
  cursor = 0;
  render();
}

function updateProgress() {
  const total = manifest.chips.length;
  let keep = 0, reject = 0, skip = 0;
  for (const c of manifest.chips) {
    const d = decisions[c.chip_id]?.decision;
    if (d === "keep")   keep++;
    else if (d === "reject") reject++;
    else if (d === "skip")   skip++;
  }
  document.getElementById("prog-total").textContent  = total;
  document.getElementById("prog-cur").textContent    = keep + reject + skip;
  document.getElementById("prog-keep").textContent   = keep;
  document.getElementById("prog-reject").textContent = reject;
  document.getElementById("prog-skip").textContent   = skip;
}

function fmt(v, digits=3) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(digits);
}

function render() {
  updateProgress();
  const view = document.getElementById("view");
  if (order.length === 0) {
    view.innerHTML = '<div class="empty">No chips match the current filter.</div>';
    return;
  }
  if (cursor < 0) cursor = 0;
  if (cursor >= order.length) cursor = order.length - 1;

  const chip = manifest.chips[order[cursor]];
  const dec  = decisions[chip.chip_id] || {};
  const badgeClass = dec.decision || "unrev";
  const badgeText  = dec.decision === "keep"   ? "✓ Keep"
                   : dec.decision === "reject" ? "✗ Reject"
                   : dec.decision === "skip"   ? "↷ Skip"
                   : "unreviewed";

  view.innerHTML = `
    <div class="meta">
      <span class="chip-id">${chip.chip_id}</span>
      <span class="badge ${badgeClass}">${badgeText}</span>
      <span class="kv">rank <b>${chip.rank}</b> / ${manifest.chips.length} (worst→best Dice)</span>
      <span class="kv">Dice <b>${fmt(chip.dice)}</b></span>
      <span class="kv">P <b>${fmt(chip.precision)}</b></span>
      <span class="kv">R <b>${fmt(chip.recall)}</b></span>
      <span class="kv">panel_frac <b>${fmt(chip.panel_frac, 2)}</b></span>
      <span class="kv">${chip.continent}</span>
      <span class="kv">${chip.lat===null ? "" : `${chip.lat.toFixed(2)}, ${chip.lon.toFixed(2)}`}</span>
      <span class="kv">v2_split <b>${chip.split || "?"}</b></span>
    </div>
    <img class="diag" src="${chip.png}" alt="${chip.chip_id}">
    <div class="actions">
      <button class="b-prev"   onclick="navigate(-1)">← Prev</button>
      <button class="b-keep"   onclick="decide('keep')"  >✓ Keep (1)</button>
      <button class="b-reject" onclick="decide('reject')">✗ Reject (2)</button>
      <button class="b-skip"   onclick="decide('skip')"  >↷ Skip (3)</button>
      <button class="b-next"   onclick="navigate(1)">Next →</button>
    </div>
    <div class="nav-hint">
      Shortcuts: <b>1</b>=Keep, <b>2</b>=Reject, <b>3</b>=Skip, <b>←</b>/<b>→</b>=Prev/Next, <b>U</b>=jump to next unreviewed
    </div>
  `;
}

function decide(d) {
  if (order.length === 0) return;
  const chip = manifest.chips[order[cursor]];
  decisions[chip.chip_id] = { decision: d, ts: new Date().toISOString() };
  saveDecisions();
  navigate(1);
}

function navigate(delta) {
  cursor += delta;
  if (cursor < 0) cursor = 0;
  if (cursor >= order.length) cursor = order.length - 1;
  render();
}

function jumpToNextUnreviewed() {
  for (let i = cursor + 1; i < order.length; i++) {
    const chip = manifest.chips[order[i]];
    if (!decisions[chip.chip_id]?.decision) { cursor = i; render(); return; }
  }
  // wrap-around
  for (let i = 0; i < cursor; i++) {
    const chip = manifest.chips[order[i]];
    if (!decisions[chip.chip_id]?.decision) { cursor = i; render(); return; }
  }
}

function exportCSV() {
  const lines = ["chip_id,decision,timestamp"];
  for (const c of manifest.chips) {
    const d = decisions[c.chip_id];
    if (d?.decision) lines.push(`${c.chip_id},${d.decision},${d.ts}`);
  }
  const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `decisions_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function importCSV(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.split(/\\r?\\n/).filter(Boolean);
    let n = 0;
    for (let i = 0; i < lines.length; i++) {
      const row = lines[i].split(",");
      if (i === 0 && row[0].trim() === "chip_id") continue;
      if (row.length < 2) continue;
      const chip_id  = row[0].trim();
      const decision = row[1].trim();
      const ts       = (row[2] || new Date().toISOString()).trim();
      if (["keep", "reject", "skip"].includes(decision)) {
        decisions[chip_id] = { decision, ts };
        n++;
      }
    }
    saveDecisions();
    alert(`Imported ${n} decisions.`);
    applyFilter();
  };
  reader.readAsText(file);
}

function clearAll() {
  if (!confirm("Clear ALL decisions? This cannot be undone (unless you exported).")) return;
  decisions = {};
  saveDecisions();
  applyFilter();
}

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input,select,textarea")) return;
  if (e.key === "1") { decide("keep");   e.preventDefault(); }
  if (e.key === "2") { decide("reject"); e.preventDefault(); }
  if (e.key === "3") { decide("skip");   e.preventDefault(); }
  if (e.key === "ArrowLeft")  { navigate(-1); e.preventDefault(); }
  if (e.key === "ArrowRight") { navigate( 1); e.preventDefault(); }
  if (e.key === "u" || e.key === "U") { jumpToNextUnreviewed(); e.preventDefault(); }
});

document.getElementById("btn-export").addEventListener("click", exportCSV);
document.getElementById("btn-clear").addEventListener("click", clearAll);
document.getElementById("btn-import").addEventListener("click", () => {
  document.getElementById("import-input").click();
});
document.getElementById("import-input").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) importCSV(f);
  e.target.value = "";  // allow re-importing same file
});
document.getElementById("filter").addEventListener("change", applyFilter);

fetch("manifest.json")
  .then(r => r.json())
  .then(m => {
    manifest = m;
    applyFilter();
  })
  .catch(err => {
    document.getElementById("view").innerHTML =
      `<div class="empty">Could not load manifest.json (${err}).<br>
       Make sure manifest.json is in the same folder as this HTML file.</div>`;
  });
</script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML.write_text(HTML_TEMPLATE)
    print(f"Wrote {HTML.relative_to(REPO)}  ({HTML.stat().st_size/1024:.1f} KB)")
    print(f"Open locally: file://{HTML}")
    print(f"After Pages enabled: https://ray-climate.github.io/solar_openEO/review_large/")


if __name__ == "__main__":
    main()
