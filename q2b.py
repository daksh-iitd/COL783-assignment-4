"""
Q2(b): k-means (k = 2) segmentation initialised from the marked pixels.

Pipeline:
  1. Load the image and get the known FG / BG pixels from the rectangles
     FG_RECTS / BG_RECTS defined in q2a.py (soldier = FG, rest = BG).
  2. Initial means: mu_F = mean colour of the known FG pixels,
                    mu_B = mean colour of the known BG pixels.
  3. Lloyd's iterations:
       assignment : label each pixel with the nearest mean (Euclidean, RGB)
       clamping   : (optional, ON by default) pixels the user marked keep
                    their marked label regardless of the nearest mean, so the
                    clusters cannot drift too far from the user's intent.
       update     : each mean = average colour of the pixels assigned to it
     until no label changes (or MAX_ITERS).
  4. WCSS (within-cluster sum of squares) after each assignment step:
         WCSS = sum_c sum_{i in c} ||x_i - mu_c||^2
     Lloyd's algorithm never increases it in the unclamped version.
     With clamping, the forced labels are not always the nearest mean, so the
     clamped WCSS is still non-increasing but converges to a higher value.

The script runs BOTH variants (clamped and unclamped) so they can be compared.

Outputs (in ./outputs_q2b/), for <v> in {clamped, free}:
    marked_regions.png            input with the marked rectangles
    <v>_iterations.png            the first few iterations (class means + boundaries)
    <v>_iter_XX.png               the same, one image per iteration
    <v>_converged_means.png       converged segmentation as in (a)
    <v>_converged_overlay.png     converged boundaries drawn on the input
    <v>_mask.png                  converged binary FG mask
    wcss.png                      WCSS vs. iteration for both variants

Run:
    python3 q2b.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from q2a import (IMG_PATH, load_rgb, get_marked_pixels, boundary_mask)


OUT_DIR = "outputs_q2b"
MAX_ITERS = 100
N_SHOW = 5          # number of early iterations shown in the grid figure
BOUNDARY_RGB = (0, 255, 0)


# --------------------------------------------------------------------------
# k-means
# --------------------------------------------------------------------------
def kmeans_2(X, mu_f, mu_b, fg_known, bg_known, clamp, max_iters=MAX_ITERS):
    """
    X        : (N, 3) pixel colours
    mu_f/b   : initial means
    fg_known : (N,) bool, user-marked FG pixels (same for bg_known)
    Returns a list of (labels, means, wcss) per iteration; labels are 1 = FG.
    Iteration t stores the labels from the t-th assignment step, the means
    used for that assignment, and the WCSS of that labelling w.r.t. the
    means recomputed from it.
    """
    means = np.stack([mu_b, mu_f]).astype(np.float64)  # index 0 = BG, 1 = FG
    history = []
    labels = None
    for it in range(max_iters):
        # --- assignment step ---
        d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)  # (N, 2)
        new_labels = np.argmin(d2, axis=1)
        if clamp:
            new_labels[fg_known] = 1
            new_labels[bg_known] = 0

        # --- update step ---
        new_means = means.copy()
        for c in (0, 1):
            sel = new_labels == c
            if sel.any():           # keep the old mean if a cluster empties
                new_means[c] = X[sel].mean(axis=0)

        wcss = sum(((X[new_labels == c] - new_means[c]) ** 2).sum()
                   for c in (0, 1))
        history.append((new_labels.copy(), new_means.copy(), wcss))

        converged = labels is not None and np.array_equal(new_labels, labels)
        labels, means = new_labels, new_means
        if converged:
            break
    return history


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def segmentation_image(labels_2d, means, boundary=True):
    seg = means[labels_2d.astype(int)]          # (H, W, 3): class mean colours
    if boundary:
        seg[boundary_mask(labels_2d)] = BOUNDARY_RGB
    return np.clip(seg, 0, 255).astype(np.uint8)


def save_png(arr, name):
    plt.imsave(os.path.join(OUT_DIR, name), arr)


def run_variant(img, X, fg_mask, bg_mask, clamp, tag):
    H, W, _ = img.shape
    mu_f0 = X[fg_mask.ravel()].mean(axis=0)
    mu_b0 = X[bg_mask.ravel()].mean(axis=0)
    hist = kmeans_2(X, mu_f0, mu_b0, fg_mask.ravel(), bg_mask.ravel(), clamp)

    print(f"\n[{tag}] converged after {len(hist)} iterations")
    print(f"  initial means  FG {np.round(mu_f0, 1)}  BG {np.round(mu_b0, 1)}")
    for t, (lab, mu, w) in enumerate(hist, 1):
        print(f"  iter {t:2d}: WCSS = {w:.4e}  FG {lab.mean() * 100:5.1f}%  "
              f"FG mean {np.round(mu[1], 1)}  BG mean {np.round(mu[0], 1)}")

    # --- first few iterations ---
    n = min(N_SHOW, len(hist))
    fig, axs = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4.6))
    axs[0].imshow(img.astype(np.uint8))
    axs[0].set_title("Input")
    for t in range(n):
        lab, mu, w = hist[t]
        seg = segmentation_image(lab.reshape(H, W), mu)
        save_png(seg, f"{tag}_iter_{t + 1:02d}.png")
        axs[t + 1].imshow(seg)
        axs[t + 1].set_title(f"Iteration {t + 1}\nWCSS = {w:.3e}", fontsize=9)
    for a in axs:
        a.axis("off")
    fig.suptitle(f"k-means iterations ({tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_iterations.png"), dpi=150)
    plt.close(fig)

    # --- converged segmentation ---
    lab, mu, _ = hist[-1]
    lab2d = lab.reshape(H, W)
    save_png(segmentation_image(lab2d, mu), f"{tag}_converged_means.png")
    over = img.copy()
    over[boundary_mask(lab2d)] = BOUNDARY_RGB
    save_png(np.clip(over, 0, 255).astype(np.uint8),
             f"{tag}_converged_overlay.png")
    save_png((lab2d * 255).astype(np.uint8), f"{tag}_mask.png")
    print(f"  marked pixels kept: FG {lab2d[fg_mask].mean() * 100:.1f}%  "
          f"BG {(1 - lab2d[bg_mask]).mean() * 100:.1f}%")
    return hist


def main():
    plt.switch_backend("Agg")  # figures are only saved to files
    os.makedirs(OUT_DIR, exist_ok=True)
    # remove per-iteration images from a previous run (it may have had more
    # iterations, and stale files would be mixed in with the new ones)
    for f in os.listdir(OUT_DIR):
        if "_iter_" in f and f.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, f))
    img, scale = load_rgb(IMG_PATH)
    H, W, _ = img.shape
    X = img.reshape(-1, 3)
    print(f"Image size: {W} x {H} (scale {scale:.3f})")

    fg_mask, bg_mask = get_marked_pixels(img, scale, OUT_DIR)

    hists = {
        "clamped": run_variant(img, X, fg_mask, bg_mask, True, "clamped"),
        "free": run_variant(img, X, fg_mask, bg_mask, False, "free"),
    }

    # --- WCSS plot ---
    # Separate panels: the two curves differ in scale, so a shared y-axis
    # would flatten both of them.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tag, style, col in ((axs[0], "clamped", "o-", "tab:blue"),
                                (axs[1], "free", "s-", "tab:orange")):
        w = [h[2] for h in hists[tag]]
        ax.plot(range(1, len(w) + 1), w, style, color=col)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("WCSS")
        ax.set_title(f"{tag}: marked pixels "
                     f"{'fixed to their class' if tag == 'clamped' else 'free to move'}")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useOffset=False)
    fig.suptitle("k-means: within-cluster sum of squares vs. iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "wcss.png"), dpi=150)
    plt.close(fig)

    # --- side-by-side converged results ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.5))
    axs[0].imshow(img.astype(np.uint8)); axs[0].set_title("Input")
    for a, tag in zip(axs[1:], ("clamped", "free")):
        lab, mu, _ = hists[tag][-1]
        a.imshow(segmentation_image(lab.reshape(H, W), mu))
        a.set_title(f"Converged k-means ({tag}, {len(hists[tag])} iters)")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
