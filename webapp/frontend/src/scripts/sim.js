// webapp/frontend/src/scripts/sim.js — WS live loop, readout/leaderboard/% updates,
// start/reset state machine (FULL-SPEED, no pause/speed), cell drilldown.
// updateReadouts/renderLeaderboard/onCanvasClick adapted from the old app.js.
import { FC, drawFrame, gridW } from "./render.js";
import { line } from "./charts.js";
import { collectPayload } from "./config.js";

const $ = (id) => document.getElementById(id);
const fmtPct = (v) => (v * 100).toFixed(1) + "%";
let ws = null, livePath = null, started = false;
const shareSeries = [[], [], [], []], distinctSeries = [], n2Series = [];
function resetSeries() { for (const s of shareSeries) s.length = 0; distinctSeries.length = 0; n2Series.length = 0; }

export function start() {
  if (started) { reset(); return; }                  // button doubles as 重置 once running
  resetSeries();
  const cfg = collectPayload();
  $("gridLabel").textContent = `${cfg.grid}×${cfg.grid}`;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    started = true; $("status").textContent = "运行中"; $("status").style.color = "var(--dim)";
    $("startBtn").textContent = "⟳ 重置";
    ws.send(JSON.stringify({ cmd: "start", config: cfg }));
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "error") { $("status").textContent = "配置错误:" + m.msg; $("status").style.color = "#ff6b6b"; return; }
    if (m.event === "started") { livePath = m.path; return; }
    if (m.event === "done") { $("status").textContent = "完成"; return; }
    drawFrame(m, +$("cfg-K").value || 64); updateReadouts(m);
  };
  ws.onclose = () => { if ($("status").textContent === "运行中") $("status").textContent = "已停止"; };
}

export function reset() {
  if (ws) { ws.close(); ws = null; }
  resetSeries(); started = false; livePath = null;
  $("status").textContent = "未开始"; $("status").style.color = "var(--dim)";
  $("startBtn").textContent = "▶ 开始";
  $("tickLabel").textContent = "tick 0 / " + (+$("cfg-T").value || 450);
}

function updateReadouts(frame) {
  const r = frame.readouts;
  $("m-tick").textContent = frame.tick;
  $("m-total").textContent = r.total;
  $("m-occ").textContent = r.occupied_cells;
  $("m-distinct").textContent = r.distinct_strains;
  $("m-n2").textContent = (r.n2 ?? 0).toFixed(1);
  $("m-dmax").textContent = (r.d_max ?? 0).toFixed(3);
  $("tickLabel").textContent = `tick ${frame.tick} / ${+$("cfg-T").value || 450}`;
  const share = r.faction_share || {};
  for (let f = 0; f < 4; f++) shareSeries[f].push(share[f] || 0);
  distinctSeries.push(r.distinct_strains); n2Series.push(r.n2);
  document.querySelectorAll(".player").forEach((p, f) => {
    const pct = p.querySelector(".pct"); if (pct) pct.textContent = fmtPct(share[f] || 0);
  });
  for (let f = 0; f < 4; f++) {
    const el = document.querySelector(`.lpct[data-f="${f}"]`);
    if (el) el.textContent = fmtPct(share[f] || 0);
  }
  line("shareChart", shareSeries, ["#ff4d4d", "#4d9bff", "#3fd07f", "#ffcc4d"], 1.0);
  const dmax = Math.max(1, ...distinctSeries), nmax = Math.max(1, ...n2Series);
  line("divChart", [distinctSeries.map((v) => v / dmax), n2Series.map((v) => v / nmax)],
    ["#a371f7", "#39c5cf"], 1.0);
  renderLeaderboard(frame.leaderboard || []);
}

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

export function onCanvasClick(ev) {
  if (!livePath) return;
  const cv = $("grid"), rect = cv.getBoundingClientRect(), n = gridW();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * n);
  const y = Math.floor((ev.clientY - rect.top) / rect.height * n);
  const tick = +$("m-tick").textContent;
  fetch(`/api/cell?path=${encodeURIComponent(livePath)}&tick=${tick}&y=${y}&x=${x}`)
    .then((r) => { if (!r.ok) throw new Error("cell fetch failed"); return r.json(); })
    .then((d) => {
      if (!d || !d.strains) return;
      const ul = $("cellDetail"); ul.innerHTML = "";
      if (!d.strains.length) { ul.innerHTML = `<li style="color:var(--dim)">格 (${y},${x}) 空</li>`; return; }
      d.strains.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="seq" title="${s.strain}">${s.strain}</span>` +
          `<span style="color:rgb(${FC[s.faction]})">f${s.faction}</span>` +
          `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${s.count}</span>`;
        ul.appendChild(li);
      });
    })
    .catch(() => { $("cellDetail").innerHTML = `<li style="color:var(--dim)">钻取暂不可用,请重试</li>`; });
}
