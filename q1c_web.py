"""
Interactive Intelligent Scissors in the web browser (Q1(c)).

Use this instead of q1c_gui.py when the matplotlib window does not receive
mouse clicks. A small local server (Python standard library only) runs the
same cost function and Dijkstra code as q1b/q1c; the page in the browser just
shows the image and sends your clicks to it.

Run:
    python3 q1c_web.py                        # Marr-Hildreth fZ, default λ
    python3 q1c_web.py --canny                # Canny edges for fZ (Idea 1)
    python3 q1c_web.py --weights 0.1 0.8 0.1  # custom λZ λG λD (Idea 2)
    python3 q1c_web.py --image other.jpg
Then open the printed URL (it opens automatically). Ctrl+C here to stop.

Controls on the page:
    Click                 add a seed (snapped to the strongest edge nearby)
    Move the pointer      live wire from the last seed to the pointer
    Click near 1st seed,  close the contour and show the mask
      Enter, or "Close"
    u / Backspace / Undo  undo last seed (or reopen a closed contour)
    r / Reset             start over
    w / Save              save results to outputs_q1c/interactive_*

After each click the server runs Dijkstra once from the new seed over the
whole image and sends back the predecessor map; the live wire is then traced
in the browser from that map, so it follows the pointer instantly.
"""

import argparse
import base64
import io
import json
import os
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Agg backend (set by q1a on import)

from q1a import load_gray
from q1b import (feature_maps, link_costs, dijkstra,
                 LAMBDA_Z, LAMBDA_G, LAMBDA_D)
from q1c import canny, snap_seeds, fill_contour, overlay_mask

OUT_DIR = "outputs_q1c"
STATE = {}


# --------------------------------------------------------------------------
# Server-side work
# --------------------------------------------------------------------------
def png_data_url(arr_uint8, mode):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8, mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def handle_seed(req):
    r, c = int(req["r"]), int(req["c"])
    if req.get("snap", True):
        r, c = snap_seeds([(r, c)], STATE["fG"])[0]
    t = time.time()
    _, pred = dijkstra(STATE["costs"], (int(r), int(c)))
    dt = time.time() - t
    print(f"seed ({r}, {c})  Dijkstra {dt:.2f}s", flush=True)
    pred32 = np.ascontiguousarray(pred.astype("<i4"))
    return {"seed": [int(r), int(c)], "time": dt,
            "pred": base64.b64encode(pred32.tobytes()).decode()}


def handle_close(req):
    contour = np.array(req["contour"], dtype=int)
    mask = fill_contour(contour, STATE["img"].shape)
    STATE["mask"], STATE["contour"] = mask, contour
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = (255, 50, 50, 90)
    print(f"closed: {len(contour)} contour px, area {int(mask.sum())} px", flush=True)
    return {"mask": png_data_url(rgba, "RGBA"), "area": int(mask.sum())}


def handle_save(req):
    if "mask" not in STATE:
        return {"error": "close the contour first"}
    os.makedirs(OUT_DIR, exist_ok=True)
    img, mask = STATE["img"], STATE["mask"]
    contour = STATE["contour"]
    c = np.vstack([contour, contour[:1]])
    s = np.array(req["seeds"], dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].imshow(img, cmap="gray")
    axes[0].plot(c[:, 1], c[:, 0], "r-", lw=1.4)
    axes[0].plot(s[:, 1], s[:, 0], "o", mfc="yellow", mec="k", ms=6)
    axes[0].set_title(f"Seeds ({len(s)}) and closed contour\n{STATE['label']}")
    axes[1].imshow(overlay_mask(img, mask))
    axes[1].set_title("Segmentation mask")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/interactive_result.png", dpi=150)
    plt.close(fig)
    plt.imsave(f"{OUT_DIR}/interactive_mask.png", mask, cmap="gray")
    np.savetxt(f"{OUT_DIR}/interactive_seeds.txt", s, fmt="%d", header="row col")
    print("saved; seeds (row, col):", [tuple(map(int, x)) for x in s], flush=True)
    return {"ok": True, "dir": OUT_DIR}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            html = PAGE.replace("__H__", str(STATE["img"].shape[0])) \
                       .replace("__W__", str(STATE["img"].shape[1])) \
                       .replace("__LABEL__", STATE["label"])
            self.send(html.encode(), "text/html; charset=utf-8")
        elif self.path == "/image.png":
            self.send(STATE["png"], "image/png")
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        routes = {"/seed": handle_seed, "/close": handle_close, "/save": handle_save}
        if self.path not in routes:
            self.send(b"not found", "text/plain", 404)
            return
        try:
            out = routes[self.path](req)
        except Exception as e:  # report errors to the page and the terminal
            print("error:", repr(e), flush=True)
            out = {"error": repr(e)}
        self.send(json.dumps(out).encode(), "application/json")


def free_port(start=8000):
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


# --------------------------------------------------------------------------
# Page (HTML + JS)
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Intelligent Scissors</title>
<style>
  :root { --bg:#16181d; --panel:#21252c; --fg:#e8e8e8; --mute:#9aa0a6; --acc:#ffcf33; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px -apple-system, system-ui, sans-serif; }
  .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; }
  canvas { cursor:crosshair; border-radius:6px; box-shadow:0 2px 12px #0008; }
  .side { background:var(--panel); padding:14px 16px; border-radius:8px; width:260px; }
  h1 { font-size:16px; margin:0 0 4px; }
  .mute { color:var(--mute); font-size:12px; }
  button { width:100%; margin:4px 0; padding:8px; border:0; border-radius:6px;
           background:#343a44; color:var(--fg); font-size:14px; cursor:pointer; }
  button:hover { background:#3f4652; }
  button.primary { background:var(--acc); color:#111; font-weight:600; }
  #status { margin:10px 0; padding:8px; background:#0003; border-radius:6px; min-height:36px; }
  ul { padding-left:18px; margin:8px 0 0; color:var(--mute); font-size:12px; line-height:1.6; }
  kbd { background:#0005; padding:1px 5px; border-radius:3px; }
</style></head>
<body><div class="wrap">
  <canvas id="cv"></canvas>
  <div class="side">
    <h1>Intelligent Scissors</h1>
    <div class="mute">__LABEL__</div>
    <div id="status">Click on the image to place the first seed.</div>
    <button class="primary" id="bClose">Close contour</button>
    <button id="bUndo">Undo</button>
    <button id="bReset">Reset</button>
    <button id="bSave">Save results</button>
    <label class="mute"><input type="checkbox" id="snap" checked> Snap seeds to nearest strong edge</label>
    <ul>
      <li>Click: add seed</li>
      <li>Move pointer: live wire (green)</li>
      <li>Click near first seed or <kbd>Enter</kbd>: close</li>
      <li><kbd>u</kbd> / <kbd>⌫</kbd>: undo · <kbd>r</kbd>: reset · <kbd>w</kbd>: save</li>
    </ul>
  </div>
</div>
<script>
const H = __H__, W = __W__, CLOSE_R = 6;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const status = m => document.getElementById('status').textContent = m;
const img = new Image(); img.src = '/image.png';
let S = 1, seeds = [], preds = [], segs = [], closed = false, live = null,
    maskImg = null, busy = false, lastPx = null;

function fit() {
  S = Math.max(1, Math.min((innerHeight - 32) / H, (innerWidth - 320) / W));
  const dpr = devicePixelRatio || 1;
  cv.style.width = (W * S) + 'px'; cv.style.height = (H * S) + 'px';
  cv.width = Math.round(W * S * dpr); cv.height = Math.round(H * S * dpr);
  ctx.setTransform(S * dpr, 0, 0, S * dpr, 0, 0);
  draw();
}
addEventListener('resize', fit);
img.onload = fit;

function trace(pred, r, c) {
  const out = []; let v = r * W + c;
  while (v !== -1) { out.push([Math.floor(v / W), v % W]); v = pred[v]; }
  return out.reverse();
}
function contour() {
  const pts = [];
  for (const s of segs) for (let i = 0; i < s.length - 1; i++) pts.push(s[i]);
  if (!closed && seeds.length) pts.push(seeds[seeds.length - 1]);
  return pts;
}
function poly(pts, color, w, close) {
  if (pts.length < 2) return;
  ctx.beginPath(); ctx.moveTo(pts[0][1] + .5, pts[0][0] + .5);
  for (const p of pts) ctx.lineTo(p[1] + .5, p[0] + .5);
  if (close) ctx.closePath();
  ctx.strokeStyle = color; ctx.lineWidth = w / S; ctx.stroke();
}
function draw() {
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, W, H);
  if (img.complete) ctx.drawImage(img, 0, 0, W, H);
  if (maskImg) ctx.drawImage(maskImg, 0, 0, W, H);
  poly(contour(), '#ff3b3b', 2, closed);
  if (live) poly(live, '#3bff6a', 1.6, false);
  seeds.forEach((s, i) => {
    ctx.beginPath(); ctx.arc(s[1] + .5, s[0] + .5, 4.5 / S, 0, 7);
    ctx.fillStyle = i === 0 ? '#ffffff' : '#ffcf33'; ctx.fill();
    ctx.lineWidth = 1.2 / S; ctx.strokeStyle = '#000'; ctx.stroke();
  });
}
function toPx(e) {
  const b = cv.getBoundingClientRect();
  const r = Math.floor((e.clientY - b.top) / S), c = Math.floor((e.clientX - b.left) / S);
  return (r >= 0 && r < H && c >= 0 && c < W) ? [r, c] : null;
}
async function post(url, body) {
  const res = await fetch(url, {method: 'POST', body: JSON.stringify(body)});
  return res.json();
}

async function addSeed(p) {
  busy = true; cv.style.cursor = 'wait';
  status(`Computing shortest paths from seed ${seeds.length + 1}…`);
  const out = await post('/seed', {r: p[0], c: p[1], snap: document.getElementById('snap').checked});
  busy = false; cv.style.cursor = 'crosshair';
  if (out.error) { status('Error: ' + out.error); return; }
  const s = out.seed;
  if (seeds.length) segs.push(trace(preds[preds.length - 1], s[0], s[1]));
  const bin = atob(out.pred), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  seeds.push(s); preds.push(new Int32Array(buf)); live = null;
  status(`${seeds.length} seed(s). Dijkstra ${out.time.toFixed(2)} s. ` +
         (seeds.length >= 3 ? 'Click near the first (white) seed or press Enter to close.' : ''));
  draw();
}
async function closeContour() {
  if (closed || busy) return;
  if (seeds.length < 3) { status('Need at least 3 seeds to close.'); return; }
  segs.push(trace(preds[preds.length - 1], seeds[0][0], seeds[0][1]));
  closed = true; live = null; draw();
  status('Filling mask…');
  const out = await post('/close', {contour: contour()});
  if (out.error) { status('Error: ' + out.error); return; }
  maskImg = new Image();
  maskImg.onload = draw; maskImg.src = out.mask;
  status(`Closed: ${seeds.length} seeds, area ${out.area} px. Save, Undo or Reset.`);
}
function undo() {
  if (busy) return;
  if (closed) { closed = false; segs.pop(); maskImg = null; }
  else if (seeds.length) { seeds.pop(); preds.pop(); if (segs.length) segs.pop(); }
  live = null; status(`${seeds.length} seed(s).`); draw();
}
function reset() {
  if (busy) return;
  seeds = []; preds = []; segs = []; closed = false; live = null; maskImg = null;
  status('Reset. Click to place the first seed.'); draw();
}
async function save() {
  if (!closed) { status('Close the contour first.'); return; }
  const out = await post('/save', {seeds});
  status(out.error ? 'Error: ' + out.error :
         `Saved to ${out.dir}/interactive_result.png, _mask.png, _seeds.txt`);
}

cv.addEventListener('click', e => {
  if (busy || closed) return;
  const p = toPx(e); if (!p) return;
  if (seeds.length >= 3 && Math.hypot(p[0] - seeds[0][0], p[1] - seeds[0][1]) <= CLOSE_R) {
    closeContour(); return;
  }
  addSeed(p);
});
cv.addEventListener('contextmenu', e => { e.preventDefault(); closeContour(); });
cv.addEventListener('mousemove', e => {
  if (busy || closed || !preds.length) return;
  const p = toPx(e); if (!p) return;
  if (lastPx && p[0] === lastPx[0] && p[1] === lastPx[1]) return;
  lastPx = p; live = trace(preds[preds.length - 1], p[0], p[1]); draw();
});
cv.addEventListener('mouseleave', () => { live = null; draw(); });
addEventListener('keydown', e => {
  if (e.key === 'Enter') closeContour();
  else if (e.key === 'u' || e.key === 'Backspace') { e.preventDefault(); undo(); }
  else if (e.key === 'r') reset();
  else if (e.key === 'w') save();
});
document.getElementById('bClose').onclick = closeContour;
document.getElementById('bUndo').onclick = undo;
document.getElementById('bReset').onclick = reset;
document.getElementById('bSave').onclick = save;
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Intelligent Scissors in the browser")
    ap.add_argument("--image", default="bird.jpg")
    ap.add_argument("--canny", action="store_true",
                    help="use Canny edges instead of Marr-Hildreth for fZ")
    ap.add_argument("--weights", type=float, nargs=3,
                    default=(LAMBDA_Z, LAMBDA_G, LAMBDA_D),
                    metavar=("LZ", "LG", "LD"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    img = load_gray(args.image)
    fZ, fG, Dx, Dy = feature_maps(img)
    if args.canny:
        fZ = np.where(canny(img), 0.0, 1.0)
    lz, lg, ld = args.weights
    STATE.update(
        img=img, fG=fG,
        costs=link_costs(fZ, fG, Dx, Dy, lz, lg, ld),
        label=f"{'Canny' if args.canny else 'Marr-Hildreth'} fZ, "
              f"λ = ({lz}, {lg}, {ld}) · {args.image}",
    )
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    STATE["png"] = buf.getvalue()

    port = args.port or free_port()
    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Intelligent Scissors running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
