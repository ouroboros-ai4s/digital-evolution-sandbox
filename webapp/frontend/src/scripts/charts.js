// webapp/frontend/src/scripts/charts.js — hand-drawn line charts,
// copied VERBATIM from the old webapp/static/app.js (no chart lib).
export function line(id, series, colors, ymax) {
  const cv = document.getElementById(id), c = cv.getContext("2d");
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
