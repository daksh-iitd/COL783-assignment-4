"""
Q2(d): GrabCut-like iterative segmentation from a single rectangle.

Input: ONE rectangle drawn around the object (no other FG/BG marks).

  * Pixels outside the rectangle are background and stay background.
  * The rectangle's boundary band (BORDER px wide, just inside its edges) is
    treated as known background: the object is assumed not to touch it.
  * Initialisation:  FG Gaussian <- interior pixels of the rectangle
                     BG Gaussian <- boundary-band pixels of the rectangle
  * Then alternate until the labels stop changing:
      1. update memberships of the interior pixels with the current Gaussians
      2. re-fit both Gaussians to the pixels inside the rectangle that now
         carry each label (the boundary band always counts as BG)

Two membership updates are run, so they can be compared:
  ml        : maximum likelihood, a_i = [log p(x_i|F) > log p(x_i|B)]
              (the rule asked for in the assignment)
  graphcut  : the graph cut of part (c) with these Gaussians as the regional
              term (lambda, sigma from q2c.py) - this is what GrabCut does.

Monitored quantities per iteration: fraction of interior pixels relabelled,
and the total log-likelihood  sum_i log p(x_i | class of i)  of the pixels
in the rectangle (plus, for graphcut, the graph-cut energy).

Run:
    python3 q2d.py                    # draw one rectangle, press Enter
    python3 q2d.py --image other.jpg
    python3 q2d.py --lam 0.05 --sigma 30   # graph-cut parameters

Controls in the window:
    drag         draw the rectangle (drawing again replaces it; the box can
                 also be moved / resized by dragging its edges)
    Enter        accept and run (closing the window also does)

Outputs (./outputs_q2d/), for <m> in {ml, graphcut}:
    rectangle.png                 input with the rectangle and boundary band
    <m>_iterations.png            initialisation + first few iterations
    <m>_iter_XX.png               the same, one image per iteration
    <m>_converged_means.png       converged segmentation: class means + boundary
    <m>_converged_overlay.png     converged boundary drawn on the input
    <m>_mask.png                  converged binary FG mask
    convergence.png               relabelled fraction and log-likelihood vs. iteration
    summary.png                   input, converged ML and converged graph cut
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from matplotlib.ticker import MaxNLocator

from q2a import IMG_PATH, load_rgb, fit_gaussian, gaussian_loglik, boundary_mask
from q2c import graph_cut, rms_neighbour_diff, LAMBDA, SIGMA_SCALE


OUT_DIR = "outputs_q2d"
BORDER = 5           # width (px) of the rectangle's boundary band used as BG
MAX_ITERS = 30
TOL = 1e-3           # stop when fewer than 0.1% of interior pixels change
N_SHOW = 5           # iterations shown in the grid (after the initialisation)
BOUNDARY_RGB = (0, 255, 0)


# --------------------------------------------------------------------------
# Single-rectangle selection
# --------------------------------------------------------------------------
def pick_rectangle(img):
    H, W = img.shape[:2]
    state = {"rect": None}
    for k in list(plt.rcParams):            # free Enter from default keymaps
        if k.startswith("keymap."):
            plt.rcParams[k] = [v for v in plt.rcParams[k] if v != "enter"]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img.astype(np.uint8))
    ax.axis("off")
    ax.set_title("Drag a rectangle around the object, then press Enter",
                 fontsize=10)

    def on_select(e0, e1):
        x0, x1 = sorted((e0.xdata, e1.xdata))
        y0, y1 = sorted((e0.ydata, e1.ydata))
        state["rect"] = (max(0, int(round(x0))), max(0, int(round(y0))),
                         min(W, int(round(x1))), min(H, int(round(y1))))
        ax.set_title(f"Rectangle {state['rect']}  -  Enter to run",
                     fontsize=10)
        fig.canvas.draw_idle()

    sel = RectangleSelector(ax, on_select, useblit=True, interactive=True,
                            minspanx=5, minspany=5, spancoords="pixels",
                            props=dict(facecolor="none", edgecolor="lime",
                                       linewidth=1.8))

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    del sel
    return state["rect"]


# --------------------------------------------------------------------------
# GrabCut-like iteration
# --------------------------------------------------------------------------
def total_loglik(X, labels, gf, gb, region):
    """sum_i log p(x_i | class of i) over the pixels in `region`."""
    Lf = gaussian_loglik(X[region], *gf)
    Lb = gaussian_loglik(X[region], *gb)
    return np.where(labels[region], Lf, Lb).sum()


def grabcut_like(img, rect_mask, interior, update, lam, sigma):
    """
    rect_mask : (H, W) bool, pixels inside the rectangle
    interior  : (H, W) bool, rectangle minus its boundary band (unknown pixels)
    update    : "ml" or "graphcut"
    Returns a list of dicts (iteration 0 = initialisation).
    """
    H, W, _ = img.shape
    X = img.reshape(-1, 3)
    rect, inter = rect_mask.ravel(), interior.ravel()
    known_bg = ~inter                        # outside + boundary band

    labels = inter.copy()                    # initial: interior = FG
    gf = fit_gaussian(X[inter])              # FG <- interior
    gb = fit_gaussian(X[rect & ~inter])      # BG <- boundary band
    hist = [dict(labels=labels.copy(), gf=gf, gb=gb, changed=np.nan,
                 loglik=total_loglik(X, labels, gf, gb, rect), energy=np.nan)]

    for it in range(1, MAX_ITERS + 1):
        # ---- 1. membership update with the current Gaussians ----
        Lf = gaussian_loglik(X, *gf)
        Lb = gaussian_loglik(X, *gb)
        if update == "ml":
            new = (Lf > Lb) & inter
            energy = np.nan
        else:
            new, info = graph_cut(img, -Lf, -Lb, np.zeros_like(inter),
                                  known_bg, lam, sigma)
            new = new.ravel()
            energy = info["flow"]
        changed = (new != labels)[inter].mean()
        labels = new

        # ---- 2. re-fit the Gaussians (only pixels inside the rectangle) ----
        fg_px, bg_px = X[labels], X[rect & ~labels]
        if len(fg_px) < 4:
            print(f"  [{update}] iteration {it}: foreground vanished - stop")
            hist.append(dict(labels=labels.copy(), gf=gf, gb=gb,
                             changed=changed, loglik=np.nan, energy=energy))
            break
        gf, gb = fit_gaussian(fg_px), fit_gaussian(bg_px)
        hist.append(dict(labels=labels.copy(), gf=gf, gb=gb, changed=changed,
                         loglik=total_loglik(X, labels, gf, gb, rect),
                         energy=energy))
        if changed < TOL:
            break
    return hist


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def mean_image(img, labels2d, gf, gb):
    """Each pixel -> its class's Gaussian mean, boundaries in green."""
    out = np.where(labels2d[..., None], gf[0], gb[0])
    out[boundary_mask(labels2d)] = BOUNDARY_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_image(img, labels2d):
    out = img.copy()
    out[~labels2d] *= 0.45
    out[boundary_mask(labels2d)] = BOUNDARY_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def save(arr, name):
    plt.imsave(os.path.join(OUT_DIR, name), arr)


def report(img, hist, tag, rect_xy):
    H, W, _ = img.shape
    print(f"\n[{tag}] {len(hist) - 1} iterations")
    for t, h in enumerate(hist):
        lab = h["labels"]
        extra = f"  energy {h['energy']:.4g}" if np.isfinite(h["energy"]) else ""
        ch = "   -  " if t == 0 else f"{h['changed'] * 100:5.2f}%"
        print(f"  iter {t:2d}: FG {lab.mean() * 100:5.1f}%  changed {ch}  "
              f"loglik {h['loglik']:.5g}{extra}  "
              f"FG mean {np.round(h['gf'][0], 1)}  BG mean {np.round(h['gb'][0], 1)}")

    x0, y0, x1, y1 = rect_xy
    n = min(N_SHOW, len(hist) - 1) + 1
    fig, axs = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4.8))
    axs[0].imshow(img.astype(np.uint8))
    axs[0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="lime", linewidth=1.5))
    axs[0].set_title("Input + rectangle")
    for t in range(n):
        h = hist[t]
        img_t = mean_image(img, h["labels"].reshape(H, W), h["gf"], h["gb"])
        save(img_t, f"{tag}_iter_{t:02d}.png")
        axs[t + 1].imshow(img_t)
        sub = ("initialisation" if t == 0 else
               f"iteration {t}  ({h['changed'] * 100:.1f}% changed)")
        axs[t + 1].set_title(f"{sub}\nFG {h['labels'].mean() * 100:.1f}%",
                             fontsize=9)
    for a in axs:
        a.axis("off")
    fig.suptitle(f"GrabCut-like iterations, membership update = {tag}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_iterations.png"), dpi=150)
    plt.close(fig)

    last = hist[-1]
    lab2d = last["labels"].reshape(H, W)
    save(mean_image(img, lab2d, last["gf"], last["gb"]),
         f"{tag}_converged_means.png")
    save(overlay_image(img, lab2d), f"{tag}_converged_overlay.png")
    save((lab2d * 255).astype(np.uint8), f"{tag}_mask.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMG_PATH)
    ap.add_argument("--lam", type=float, default=LAMBDA,
                    help="graph-cut lambda (default from q2c.py)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="graph-cut sigma (default: RMS neighbour difference)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):            # drop stale per-iteration images
        if "_iter_" in f and f.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))

    img, scale = load_rgb(args.image)
    H, W, _ = img.shape
    print(f"Image size: {W} x {H} (scale {scale:.3f})")

    rect = pick_rectangle(img)
    plt.switch_backend("Agg")
    if rect is None:
        raise SystemExit("No rectangle was drawn - nothing to do.")
    x0, y0, x1, y1 = rect
    if min(x1 - x0, y1 - y0) <= 2 * BORDER + 4:
        raise SystemExit("Rectangle too small.")
    print(f"Rectangle (x0, y0, x1, y1) = {rect}, boundary band {BORDER} px")
    with open(os.path.join(OUT_DIR, "rect.txt"), "w") as fh:
        fh.write(f"# (x0, y0, x1, y1) in a {W}x{H} image\n{rect}\n")

    rect_mask = np.zeros((H, W), bool)
    rect_mask[y0:y1, x0:x1] = True
    interior = np.zeros((H, W), bool)
    interior[y0 + BORDER:y1 - BORDER, x0 + BORDER:x1 - BORDER] = True

    # input figure with the boundary band shaded
    shade = img.copy()
    band = rect_mask & ~interior
    shade[band] = 0.4 * shade[band] + 0.6 * np.array([0, 90, 255])
    shade[~rect_mask] *= 0.5
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.imshow(shade.astype(np.uint8))
    ax.set_title(f"Rectangle: interior -> FG init,\n"
                 f"{BORDER}px boundary band (blue) -> BG init", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rectangle.png"), dpi=150)
    plt.close(fig)

    sigma = args.sigma if args.sigma else SIGMA_SCALE * rms_neighbour_diff(img)
    print(f"Graph-cut parameters: lambda = {args.lam:g}, sigma = {sigma:.2f}")

    hists = {}
    for tag in ("ml", "graphcut"):
        hists[tag] = grabcut_like(img, rect_mask, interior, tag, args.lam, sigma)
        report(img, hists[tag], tag, rect)

    # ---- convergence plot ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for tag, style in (("ml", "o-"), ("graphcut", "s-")):
        h = hists[tag]
        it = np.arange(len(h))
        axs[0].plot(it[1:], [100 * x["changed"] for x in h[1:]], style, label=tag)
        axs[1].plot(it, [x["loglik"] for x in h], style, label=tag)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("% of interior pixels relabelled")
    axs[0].set_title("Change in labels")
    axs[1].set_ylabel("sum of log p(x | class)")
    axs[1].set_title("Total log-likelihood inside the rectangle")
    for a in axs:
        a.set_xlabel("Iteration")
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig)

    # ---- summary ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.5))
    axs[0].imshow(shade.astype(np.uint8)); axs[0].set_title("Input + rectangle")
    for a, tag, name in ((axs[1], "ml", "ML update"),
                         (axs[2], "graphcut", "graph-cut update")):
        lab = hists[tag][-1]["labels"].reshape(H, W)
        a.imshow(overlay_image(img, lab))
        a.set_title(f"Converged, {name} ({len(hists[tag]) - 1} iters)")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
