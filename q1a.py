"""
Q1(a): Marr-Hildreth edge detector.

Pipeline (all filtering done by hand on the pixel array with NumPy):
  1. Smooth with a Gaussian G_sigma, applied separably (1-D row pass, then
     1-D column pass). Kernel half-width = ceil(3*sigma), so the full width
     is >= 6*sigma as required.
  2. Apply a 3x3 Laplacian kernel to (G * f). Since convolution commutes with
     differentiation, this equals (laplacian(G)) * f, i.e. the LoG response.
     All arithmetic is in float64 to avoid catastrophic cancellation when the
     Laplacian subtracts nearly-equal neighbouring values.
  3. Mark zero crossings: a pixel is an edge if, along at least one of the
     four directions (horizontal, vertical, two diagonals), its two opposite
     neighbours have LoG values of opposite sign AND the absolute difference
     between them (the "slope" across the crossing) exceeds a threshold.

The slope threshold is given as a fraction of the maximum absolute LoG
response, which keeps it comparable across different sigmas (the LoG
magnitude shrinks roughly like 1/sigma^2 as sigma grows).

Outputs are written to ./outputs_q1a/.
"""

import os
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "outputs_q1a"
MAX_SIDE = 1024


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_gray(path, max_side=MAX_SIDE):
    """Load an image as float64 grayscale in [0, 1]; downscale (library call,
    permitted by the assignment) only if the long side exceeds max_side."""
    img = Image.open(path).convert("L")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def save_gray(arr, path):
    a = np.clip(arr, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------
# Filtering primitives (implemented manually)
# --------------------------------------------------------------------------
def gaussian_kernel_1d(sigma):
    """Normalised 1-D Gaussian with half-width ceil(3*sigma) (width >= 6σ)."""
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def convolve1d(img, kernel, axis):
    """1-D convolution along `axis` with reflect padding.
    Implemented as a weighted sum of shifted copies of the padded image."""
    r = len(kernel) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    n = img.shape[axis]
    kf = kernel[::-1]  # flip for true convolution (symmetric anyway)
    for i, w in enumerate(kf):
        if axis == 0:
            out += w * p[i:i + n, :]
        else:
            out += w * p[:, i:i + n]
    return out


def gaussian_smooth(img, sigma):
    """Separable Gaussian smoothing: rows then columns."""
    k = gaussian_kernel_1d(sigma)
    return convolve1d(convolve1d(img, k, axis=1), k, axis=0)


def convolve2d(img, kernel):
    """Small 2-D convolution with reflect padding (used for the 3x3 Laplacian)."""
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2
    p = np.pad(img, ((rh, rh), (rw, rw)), mode="reflect")
    H, W = img.shape
    kf = kernel[::-1, ::-1]
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if kf[i, j] != 0:
                out += kf[i, j] * p[i:i + H, j:j + W]
    return out


LAPLACIAN_3x3 = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)


def log_response(img, sigma):
    """g = ∇²(G_σ * f), computed in float64."""
    return convolve2d(gaussian_smooth(img, sigma), LAPLACIAN_3x3)


# --------------------------------------------------------------------------
# Zero-crossing detection
# --------------------------------------------------------------------------
def zero_crossings(g, slope_thresh):
    """Return a boolean edge map.

    A zero crossing lies between two adjacent pixels p, q (horizontal,
    vertical or diagonal neighbours) when g(p) and g(q) have opposite signs.
    It is accepted only if the slope across it, |g(p) - g(q)|, exceeds
    T = slope_thresh * max|g|. To keep edges one pixel thick, the crossing is
    assigned to whichever of p, q has the smaller |g| (it is closer to the
    true zero).
    """
    T = slope_thresh * np.abs(g).max()
    H, W = g.shape
    edges = np.zeros((H, W), dtype=bool)
    # forward neighbour offsets: right, down, down-right, down-left
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ys_p = slice(0, H - dy)
        ys_q = slice(dy, H)
        xs_p = slice(max(0, -dx), W - max(0, dx))
        xs_q = slice(max(0, dx), W - max(0, -dx))
        a, b = g[ys_p, xs_p], g[ys_q, xs_q]
        zc = (a * b < 0) & (np.abs(a - b) > T)
        p_closer = np.abs(a) <= np.abs(b)
        edges[ys_p, xs_p] |= zc & p_closer
        edges[ys_q, xs_q] |= zc & ~p_closer
    return edges


def marr_hildreth(img, sigma, slope_thresh):
    g = log_response(img, sigma)
    return zero_crossings(g, slope_thresh), g


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def show_grid(images, titles, path, suptitle, ncols):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray("bird.jpg")
    print("Image size:", img.shape)

    # Chosen "best" parameters (found by experimentation below).
    BEST_SIGMA = 2.5
    BEST_T = 0.07

    # --- best result -----------------------------------------------------
    edges, g = marr_hildreth(img, BEST_SIGMA, BEST_T)
    save_gray(edges.astype(float), f"{OUT_DIR}/edges_best.png")
    g_vis = 0.5 + 0.5 * g / np.abs(g).max()
    save_gray(g_vis, f"{OUT_DIR}/log_response_best.png")
    edges_no_thresh, _ = marr_hildreth(img, BEST_SIGMA, 0.0)
    show_grid([img, g_vis, edges_no_thresh.astype(float), edges.astype(float)],
              ["Input", f"LoG response (σ={BEST_SIGMA})",
               "All zero crossings (T=0)",
               f"Edges (σ={BEST_SIGMA}, T={BEST_T})"],
              f"{OUT_DIR}/best.png", "Marr-Hildreth: best parameters", 4)

    # --- effect of smoothing radius (fixed threshold) ---------------------
    sigmas = [0.5, 1.0, 2.5, 5.0, 8.0]
    ims, tts = [], []
    for s in sigmas:
        e, _ = marr_hildreth(img, s, BEST_T)
        save_gray(e.astype(float), f"{OUT_DIR}/edges_sigma{s}_T{BEST_T}.png")
        ims.append(e.astype(float))
        tts.append(f"σ={s}" + ("  (best)" if s == BEST_SIGMA else ""))
    show_grid(ims, tts, f"{OUT_DIR}/sigma_sweep.png",
              f"Effect of smoothing radius σ (T={BEST_T})", len(sigmas))

    # --- effect of slope threshold (fixed sigma) --------------------------
    thresholds = [0.0, 0.01, 0.07, 0.15, 0.3]
    ims, tts = [], []
    for t in thresholds:
        e, _ = marr_hildreth(img, BEST_SIGMA, t)
        save_gray(e.astype(float), f"{OUT_DIR}/edges_sigma{BEST_SIGMA}_T{t}.png")
        ims.append(e.astype(float))
        tts.append(f"T={t}" + ("  (best)" if t == BEST_T else ""))
    show_grid(ims, tts, f"{OUT_DIR}/threshold_sweep.png",
              f"Effect of slope threshold T (σ={BEST_SIGMA})", len(thresholds))

    # --- joint grid ------------------------------------------------------
    js, jt = [1.0, 2.5, 5.0], [0.02, 0.07, 0.15]
    ims, tts = [], []
    for s in js:
        for t in jt:
            e, _ = marr_hildreth(img, s, t)
            ims.append(e.astype(float))
            tts.append(f"σ={s}, T={t}")
    show_grid(ims, tts, f"{OUT_DIR}/joint_grid.png",
              "Joint sweep: rows = σ, columns = T", len(jt))

    print("Saved results to", OUT_DIR)


if __name__ == "__main__":
    main()
