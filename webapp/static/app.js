// webapp/static/app.js — DES 验收之眼前端。无框架。
const FC = [[255, 77, 77], [77, 155, 255], [63, 208, 127], [255, 204, 77]];
const $ = (id) => document.getElementById(id);

let CFG = null;            // /config 返回(palette/slots/locked/defaults)
const slotState = {};      // slot index -> chosen primitive
let ws = null, livePath = null;
const shareSeries = [[], [], [], []];   // per-faction share over ticks
const distinctSeries = [], n2Series = [];

// ---- build the 16-position genome grid from /config ----
async function loadConfig() {
  CFG = await (await fetch("/config")).json();
  const g = $("genome");
  const locked = CFG.locked;             // {"1":"F4Nr4",...} string keys
  const slots = new Set(CFG.slots);      // [0,2,3,9,10,13]
  for (let i = 0; i < 16; i++) {
    const d = document.createElement("div");
    if (locked[String(i)] !== undefined) {
      d.className = "pos locked"; d.textContent = locked[String(i)];
    } else if (slots.has(i)) {
      d.className = "pos slot";
      const sel = document.createElement("select");
      CFG.palette.forEach((p) => {
        const o = document.createElement("option"); o.value = p; o.textContent = p; sel.appendChild(o);
      });
      sel.value = "N0"; slotState[i] = "N0";
      sel.onchange = () => { slotState[i] = sel.value; };
      d.appendChild(sel);
    } else {
      d.className = "pos locked"; d.textContent = "N0";   // backbone-fixed
    }
    g.appendChild(d);
  }
}

function readConfig() {
  return {
    slots: { ...slotState },
    grid: +$("cfg-grid").value, K: +$("cfg-K").value, fill: +$("cfg-fill").value,
    T: +$("cfg-T").value, seed: +$("cfg-seed").value, z_max: +$("cfg-zmax").value,
  };
}

// ---- single-layer compositing: faction-count-weighted hue + density alpha ----
let imgW = 128, img = null, ctx = null;
function setupCanvas(n) {
  const cv = $("grid"); cv.width = n; cv.height = n; imgW = n;
  ctx = cv.getContext("2d"); img = ctx.createImageData(n, n);
}
function drawFrame(frame) {
  const n = frame.H;
  if (n !== imgW || !img) setupCanvas(n);
  img.data.fill(0);
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;   // opaque black
  let kCap = +$("cfg-K").value || 64;
  for (const c of frame.cells) {
    const y = c[0], x = c[1], cf = [c[2], c[3], c[4], c[5]];
    const tot = cf[0] + cf[1] + cf[2] + cf[3];
    if (tot <= 0) continue;
    let r = 0, gg = 0, bb = 0;
    for (let f = 0; f < 4; f++) { const w = cf[f] / tot; r += FC[f][0] * w; gg += FC[f][1] * w; bb += FC[f][2] * w; }
    const dens = Math.min(1, tot / kCap), a = 0.15 + 0.85 * dens;   // density -> brightness
    const o = (y * n + x) * 4;
    img.data[o] = r * a; img.data[o + 1] = gg * a; img.data[o + 2] = bb * a; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---- readout panel + hand-drawn charts (no chart lib) ----
function line(id, series, colors, ymax) {
  const cv = $(id), c = cv.getContext("2d");
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  const W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  series.forEach((s, si) => {
    if (!s.length) return;
    c.strokeStyle = colors[si]; c.lineWidth = 1.5; c.beginPath();
    s.forEach((v, i) => {
      const px = s.length > 1 ? i / (s.length - 1) * W : 0;
      const py = H - (v / ymax) * H * 0.92 - 2;
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    });
    c.stroke();
  });
}
function updateReadouts(frame) {
  const r = frame.readouts;
  $("m-tick").textContent = frame.tick;
  $("m-total").textContent = r.total;
  $("m-occ").textContent = r.occupied_cells;
  $("m-distinct").textContent = r.distinct_strains;
  $("m-n2").textContent = r.n2.toFixed(1);
  $("m-dmax").textContent = r.d_max.toFixed(3);
  $("tickLabel").textContent = `tick ${frame.tick} / ${readConfig().T}`;
  for (let f = 0; f < 4; f++) shareSeries[f].push(r.faction_share[f] || 0);
  distinctSeries.push(r.distinct_strains); n2Series.push(r.n2);
  line("shareChart", shareSeries, ["#ff4d4d", "#4d9bff", "#3fd07f", "#ffcc4d"], 0.6);
  const dmax = Math.max(1, ...distinctSeries), nmax = Math.max(1, ...n2Series);
  line("divChart",
    [distinctSeries.map((v) => v / dmax), n2Series.map((v) => v / nmax)],
    ["#a371f7", "#39c5cf"], 1.0);
  renderLeaderboard(frame.leaderboard || []);
}

// ---- 主导株排行榜 (spec §4) ----
function renderLeaderboard(lb) {
  const ul = $("lead"); ul.innerHTML = "";
  lb.forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="seq" title="${e.strain}">${e.strain}</span>` +
      `<span style="color:rgb(${FC[e.faction]})">f${e.faction}</span>` +
      `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${(e.share * 100).toFixed(1)}%</span>`;
    ul.appendChild(li);
  });
}

// ---- WebSocket live loop ----
function start() {
  if (ws) ws.close();
  for (const s of shareSeries) s.length = 0;
  distinctSeries.length = 0; n2Series.length = 0;
  $("gridLabel").textContent = `${readConfig().grid}×${readConfig().grid}`;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $("status").textContent = "运行中"; ws.send(JSON.stringify({ cmd: "start", config: readConfig() })); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "started") { livePath = m.path; $("dataInfo").textContent = `本局 → ${m.path}`; return; }
    if (m.event === "done") { $("status").textContent = "完成"; return; }
    drawFrame(m); updateReadouts(m);
  };
  ws.onclose = () => { if ($("status").textContent === "运行中") $("status").textContent = "已停止"; };
}

// ---- Tier A drilldown: click a cell -> that tick's {strain:count,faction} ----
function onCanvasClick(ev) {
  if (!livePath) return;
  const cv = $("grid"), rect = cv.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * imgW);
  const y = Math.floor((ev.clientY - rect.top) / rect.height * imgW);
  const tick = +$("m-tick").textContent;
  fetch(`/api/cell?path=${encodeURIComponent(livePath)}&tick=${tick}&y=${y}&x=${x}`)
    .then((r) => r.json()).then((d) => {
      const ul = $("cellDetail"); ul.innerHTML = "";
      if (!d.strains.length) { ul.innerHTML = `<li style="color:var(--dim)">格 (${y},${x}) 空</li>`; return; }
      d.strains.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="seq" title="${s.strain}">${s.strain}</span>` +
          `<span style="color:rgb(${FC[s.faction]})">f${s.faction}</span>` +
          `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${s.count}</span>`;
        ul.appendChild(li);
      });
    });
}

window.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  $("startBtn").onclick = start;
  $("grid").addEventListener("click", onCanvasClick);
});
