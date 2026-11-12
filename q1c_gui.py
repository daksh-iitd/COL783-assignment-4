"""
Interactive Intelligent Scissors (livewire) for Q1(c).

Run:
    python3 q1c_gui.py                  # bird.jpg, Marr-Hildreth fZ
    python3 q1c_gui.py --canny          # Canny edges for fZ (Idea 1)
    python3 q1c_gui.py --weights 0.1 0.8 0.1   # custom λZ λG λD (Idea 2)
    python3 q1c_gui.py --image other.jpg

Controls (in the figure window):
    Left click          add a seed (snapped to the strongest gradient nearby).
                        The live wire from the last seed is committed.
    Move mouse          live wire: shortest path from the last seed to the cursor
    Right click / Enter close the contour (last seed -> first seed) and show mask
    Click near 1st seed also closes the contour
    u / Backspace       undo last seed
    r                   reset
    n                   toggle seed snapping on/off
    w                   save results (overlay, mask, seed list) to outputs_q1c/
    q                   quit

After each click, Dijkstra runs once over the whole image from the new seed
(~0.5 s); moving the mouse then only follows the predecessor map, so the
live wire updates instantly.
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from q1a import load_gray
from q1b import (feature_maps, link_costs, dijkstra, trace_path,
                 LAMBDA_Z, LAMBDA_G, LAMBDA_D)
from q1c import canny, snap_seeds, fill_contour, overlay_mask

# q1a/q1b/q1c select the non-interactive Agg backend on import; switch to a
# GUI backend now (no figures exist yet, so this is allowed).
plt.switch_backend("macosx")

OUT_DIR = "outputs_q1c"
CLOSE_RADIUS = 6  # px: clicking this close to the first seed closes the loop


class LiveWire:
    def __init__(self, img, costs, fG, title):
        self.img, self.costs, self.fG = img, costs, fG
        self.H, self.W = img.shape
        self.snap = True
        self.reset_state()

        # free up keys we use from matplotlib's default shortcuts
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = [v for v in plt.rcParams[k]
                                   if v not in ("u", "r", "n", "w", "q",
                                                "enter", "backspace")]

        self.fig, self.ax = plt.subplots(figsize=(7, 9.5))
        self.fig.canvas.manager.set_window_title("Intelligent Scissors")
        self.ax.imshow(img, cmap="gray")
        self.ax.set_axis_off()
        self.title = title
        (self.committed_line,) = self.ax.plot([], [], "r-", lw=1.6)
        (self.live_line,) = self.ax.plot([], [], "-", color="lime", lw=1.2)
        (self.seed_pts,) = self.ax.plot([], [], "o", mfc="yellow", mec="k", ms=6)
        self.mask_im = None
        self.set_status("Left-click to place the first seed")

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_click)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("key_press_event", self.on_key)
        self.fig.tight_layout()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.seeds = []        # list of (row, col)
        self.segments = []     # segments[i]: path from seeds[i] to seeds[i+1]
        self.preds = []        # preds[i]: predecessor map of Dijkstra from seeds[i]
        self.closed = False
        self.last_cursor = None

    def set_status(self, msg):
        self.ax.set_title(f"{self.title}\n{msg}", fontsize=9)

    # --------------------------------------------------------------- helpers
    def to_pixel(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        r, c = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= r < self.H and 0 <= c < self.W:
            return r, c
        return None

    def toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and str(getattr(tb, "mode", "")) != ""

    def contour(self):
        pts = [s for seg in self.segments for s in seg[:-1]]
        if self.closed:
            return np.array(pts)
        if self.seeds:
            pts.append(self.seeds[-1])
        return np.array(pts)

    def redraw(self):
        c = self.contour()
        if self.closed and len(c):
            c = np.vstack([c, c[:1]])
        if len(c):
            self.committed_line.set_data(c[:, 1], c[:, 0])
        else:
            self.committed_line.set_data([], [])
        s = np.array(self.seeds) if self.seeds else np.empty((0, 2))
        self.seed_pts.set_data(s[:, 1], s[:, 0])
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------------- actions
    def add_seed(self, p):
        if self.snap:
            p = snap_seeds([p], self.fG)[0]
        p = (int(p[0]), int(p[1]))
        if self.seeds:
            self.segments.append(trace_path(self.preds[-1], p))
        self.seeds.append(p)
        self.set_status(f"{len(self.seeds)} seeds — computing distances…")
        self.redraw()
        self.fig.canvas.flush_events()
        t = time.time()
        _, pred = dijkstra(self.costs, p)
        self.preds.append(pred)
        self.set_status(f"{len(self.seeds)} seeds (Dijkstra {time.time() - t:.2f}s). "
                        "Right-click/Enter to close")

    def close_contour(self):
        if len(self.seeds) < 3 or self.closed:
            return
        self.segments.append(trace_path(self.preds[-1], self.seeds[0]))
        self.closed = True
        self.live_line.set_data([], [])
        contour = self.contour()
        self.mask = fill_contour(contour, self.img.shape)
        rgba = np.zeros((self.H, self.W, 4))
        rgba[self.mask] = (1.0, 0.2, 0.2, 0.35)
        if self.mask_im is not None:
            self.mask_im.remove()
        self.mask_im = self.ax.imshow(rgba)
        self.set_status(f"Closed: {len(self.seeds)} seeds, area {self.mask.sum()} px. "
                        "w = save, u = undo, r = reset")
        self.redraw()

    def undo(self):
        if self.closed:
            self.closed = False
            self.segments.pop()
            if self.mask_im is not None:
                self.mask_im.remove()
                self.mask_im = None
        elif self.seeds:
            self.seeds.pop()
            self.preds.pop()
            if self.segments:
                self.segments.pop()
        self.live_line.set_data([], [])
        self.set_status(f"{len(self.seeds)} seeds")
        self.redraw()

    def reset(self):
        self.reset_state()
        if self.mask_im is not None:
            self.mask_im.remove()
            self.mask_im = None
        self.live_line.set_data([], [])
        self.set_status("Reset. Left-click to place the first seed")
        self.redraw()

    def save(self):
        if not self.closed:
            self.set_status("Close the contour first (right-click / Enter)")
            self.fig.canvas.draw_idle()
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        contour = self.contour()
        c = np.vstack([contour, contour[:1]])
        s = np.array(self.seeds)
        fig, axes = plt.subplots(1, 2, figsize=(10, 7))
        axes[0].imshow(self.img, cmap="gray")
        axes[0].plot(c[:, 1], c[:, 0], "r-", lw=1.4)
        axes[0].plot(s[:, 1], s[:, 0], "o", mfc="yellow", mec="k", ms=6)
        axes[0].set_title(f"Seeds ({len(s)}) and closed contour")
        axes[1].imshow(overlay_mask(self.img, self.mask))
        axes[1].set_title("Segmentation mask")
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/interactive_result.png", dpi=150)
        plt.close(fig)
        plt.imsave(f"{OUT_DIR}/interactive_mask.png", self.mask, cmap="gray")
        np.savetxt(f"{OUT_DIR}/interactive_seeds.txt", s, fmt="%d",
                   header="row col")
        print("Seeds (row, col):", [tuple(map(int, x)) for x in s])
        self.set_status(f"Saved to {OUT_DIR}/interactive_*")
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- events
    def on_click(self, event):
        p = self.to_pixel(event)
        print(f"[click] button={event.button} pixel={p} "
              f"toolbar_mode={str(getattr(self.fig.canvas.toolbar, 'mode', ''))!r}",
              flush=True)
        if self.toolbar_busy():
            print("        ignored: zoom/pan tool is active — click it again "
                  "in the toolbar to turn it off", flush=True)
            return
        if event.button == 3:
            self.close_contour()
            return
        if event.button != 1 or p is None or self.closed:
            return
        if len(self.seeds) >= 3:
            d = np.hypot(p[0] - self.seeds[0][0], p[1] - self.seeds[0][1])
            if d <= CLOSE_RADIUS:
                self.close_contour()
                return
        self.add_seed(p)
        self.redraw()

    def on_move(self, event):
        if self.closed or not self.seeds:
            return
        p = self.to_pixel(event)
        if p is None or p == self.last_cursor:
            return
        self.last_cursor = p
        path = trace_path(self.preds[-1], p)
        self.live_line.set_data(path[:, 1], path[:, 0])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key
        print(f"[key] {k!r}", flush=True)
        if k in ("enter", "return"):
            self.close_contour()
        elif k in ("u", "backspace"):
            self.undo()
        elif k == "r":
            self.reset()
        elif k == "n":
            self.snap = not self.snap
            self.set_status(f"Snapping {'on' if self.snap else 'off'}")
            self.fig.canvas.draw_idle()
        elif k == "w":
            self.save()
        elif k == "q":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description="Interactive Intelligent Scissors")
    ap.add_argument("--image", default="bird.jpg")
    ap.add_argument("--canny", action="store_true",
                    help="use Canny edges instead of Marr-Hildreth for fZ")
    ap.add_argument("--weights", type=float, nargs=3,
                    default=(LAMBDA_Z, LAMBDA_G, LAMBDA_D),
                    metavar=("LZ", "LG", "LD"))
    args = ap.parse_args()

    img = load_gray(args.image)
    fZ, fG, Dx, Dy = feature_maps(img)
    if args.canny:
        fZ = np.where(canny(img), 0.0, 1.0)
    lz, lg, ld = args.weights
    costs = link_costs(fZ, fG, Dx, Dy, lz, lg, ld)

    title = (f"{'Canny' if args.canny else 'Marr-Hildreth'} fZ, "
             f"λ=({lz}, {lg}, {ld})")
    LiveWire(img, costs, fG, title)
    print(f"Backend: {plt.get_backend()}. Window open — click on the image "
          "(clicks are logged here).", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
