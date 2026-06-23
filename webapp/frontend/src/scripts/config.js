// webapp/frontend/src/scripts/config.js — /config load, 4-player genome editors,
// legend. Editor structure is built from /config at runtime (palette/slots/locked);
// only primitive letters + run knobs ever leave here (red-line 3).
import { FC } from "./render.js";

const FNAME = ["阵营0", "阵营1", "阵营2", "阵营3"];
export let CFG = null;
export const playerSlots = [0, 1, 2, 3].map(() => ({}));   // faction -> {slotIdx: letter}

export async function loadConfig() {
  CFG = await (await fetch("/config")).json();
  const locked = CFG.locked, slots = new Set(CFG.slots);
  for (let f = 0; f < 4; f++) {
    const host = document.querySelector(`.genome-list[data-faction="${f}"]`);
    host.innerHTML = "";
    for (let i = 0; i < 16; i++) {
      const row = document.createElement("div");
      const gi = `<span class="gi">#${i}</span>`;
      if (locked[String(i)] !== undefined) {
        row.className = "grow fn";
        row.innerHTML = `${gi}<span class="gtype">功能·锁死</span><span class="gtag">${locked[String(i)]}</span>`;
      } else if (slots.has(i)) {
        row.className = "grow slot";
        row.innerHTML = `${gi}<span class="gtype">插槽</span>`;
        const sel = document.createElement("select");
        CFG.palette.forEach((p) => { const o = document.createElement("option"); o.value = p; o.textContent = p; sel.appendChild(o); });
        sel.value = "N0"; playerSlots[f][i] = "N0";
        sel.onchange = () => { playerSlots[f][i] = sel.value; };
        row.appendChild(sel);
      } else {
        row.className = "grow bb";
        row.innerHTML = `${gi}<span class="gtype">骨架</span><span class="gtag">N0</span>`;
      }
      host.appendChild(row);
    }
  }
  const rows = [0, 1, 2, 3].map((f) =>
    `<div class="lrow"><i style="background:rgb(${FC[f]})"></i>${FNAME[f]}<span class="lpct" data-f="${f}">25.0%</span></div>`);
  rows.push(`<div class="lrow" style="color:var(--dim);min-width:0">亮度=密度 · 混色=争夺</div>`);
  document.getElementById("legend").innerHTML = rows.join("");
}

const num = (id) => +document.getElementById(id).value;
export function collectPayload() {
  return {
    players: playerSlots.map((s) => ({ slots: { ...s } })),
    grid: num("cfg-grid"), K: num("cfg-K"), fill: num("cfg-fill"),
    T: num("cfg-T"), seed: num("cfg-seed"), z_max: num("cfg-zmax"),
  };
}
