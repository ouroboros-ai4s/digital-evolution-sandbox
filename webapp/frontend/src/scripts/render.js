// webapp/frontend/src/scripts/render.js — canvas compositing.
// drawFrame copied VERBATIM from the old webapp/static/app.js (red-line 2:
// faithful tensor — faction-count-weighted hue + density alpha, pixelated
// nearest-neighbor, NO smoothing). Only change: kCap is a param, not a DOM read.
export const FC = [[255, 77, 77], [77, 155, 255], [63, 208, 127], [255, 204, 77]];

let imgW = 128, img = null, ctx = null;
export function setupCanvas(n) {
  const cv = document.getElementById("grid"); cv.width = n; cv.height = n; imgW = n;
  ctx = cv.getContext("2d"); img = ctx.createImageData(n, n);
}
export function gridW() { return imgW; }
export function drawFrame(frame, kCap) {
  const n = frame.H;
  if (n !== imgW || !img) setupCanvas(n);
  img.data.fill(0);
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;   // opaque black
  for (const c of frame.cells) {
    const y = c[0], x = c[1], cf = [c[2], c[3], c[4], c[5]];
    const tot = cf[0] + cf[1] + cf[2] + cf[3];
    if (tot <= 0) continue;
    let r = 0, gg = 0, bb = 0;
    for (let f = 0; f < 4; f++) { const w = cf[f] / tot; r += FC[f][0] * w; gg += FC[f][1] * w; bb += FC[f][2] * w; }
    const dens = Math.min(1, tot / kCap), a = 0.15 + 0.85 * dens;
    const o = (y * n + x) * 4;
    img.data[o] = r * a; img.data[o + 1] = gg * a; img.data[o + 2] = bb * a; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
