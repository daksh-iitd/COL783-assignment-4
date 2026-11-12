"""
Q2(c): Graph-cut segmentation (Boykov & Jolly 2001) with the Gaussian colour
models from part (a) as the regional term.

Energy of a labelling a (a_i = 1 foreground, 0 background):

    E(a) = lambda * sum_i R_i(a_i)  +  sum_{i~j} B_ij [a_i != a_j]

  Regional term (from q2a's Gaussians):
        R_i(obj) = -log p(x_i | F),     R_i(bkg) = -log p(x_i | B)
  Boundary term (8-neighbourhood):
        B_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)) / dist(i, j)
     -> cheap to cut between pixels of very different colour, expensive to
        cut between similar ones. sigma is in RGB units.

Graph:
  * one node per pixel, plus source s (= foreground) and sink t (= background)
  * s -> i with capacity lambda * R_i(bkg)   (paid if i ends up background)
    i -> t with capacity lambda * R_i(obj)   (paid if i ends up foreground)
    Adding the same constant to both t-links of a pixel does not change the
    minimum cut, so min(R_obj, R_bkg) is subtracted per pixel; this keeps all
    capacities >= 0 and small.
  * i <-> j with capacity B_ij for neighbouring pixels
  * hard constraints for the marked pixels (FG_RECTS / BG_RECTS in q2a.py):
    known FG gets s -> i = K, known BG gets i -> t = K, with
    K = 1 + max_i sum_j B_ij, so those links are never cut.
  Minimum s-t cut = maximum flow (scipy.sparse.csgraph.maximum_flow).
  After the flow, pixels reachable from s in the residual graph are FG.

SciPy's max-flow works with int32 capacities (larger values silently
overflow), so the float capacities are multiplied by a scale factor chosen so
that every sum of capacities that a flow could reach stays below 2^31.

Run:
    python3 q2c.py                            # tuned default parameters
    python3 q2c.py --lam 0.05 --sigma 30      # choose lambda / sigma

Outputs (./outputs_q2c/):
    marked_regions.png              the known FG / BG rectangles
    best_means.png                  chosen (lambda, sigma): class means + boundary
    best_overlay.png                chosen result: boundary on the input
    best_mask.png                   chosen result: binary FG mask
    ml_vs_graphcut.png              part (a) ML labelling next to the graph cut
    param_comparison.png            best vs. much smaller/larger lambda and sigma
    <name>_overlay.png              the same, one image per setting
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

from q2a import (IMG_PATH, load_rgb, get_marked_pixels, fit_gaussian,
                 gaussian_loglik, boundary_mask)


OUT_DIR = "outputs_q2c"
# Tuned by hand on Q2-img.jpg with a sweep over lambda in [0.01, 0.3] and
# sigma in [0.25, 2] x the RMS neighbour difference (~42 for this image):
# lambda in 0.02-0.05 with sigma ~ RMS gave the cleanest outline of the guard.
LAMBDA = 0.03       # weight of the regional term
SIGMA_SCALE = 1.0   # sigma = SIGMA_SCALE * (RMS colour difference of neighbours)
FACTOR = 30.0       # "much larger / smaller" = multiply / divide by this
INT_LIMIT = 2 ** 31 - 1
BOUNDARY_RGB = (0, 255, 0)

# 8-neighbourhood: each undirected neighbour pair appears once
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def neighbour_pairs(H, W):
    """Yield (idx_a, idx_b, dist, dy, dx) for each neighbour offset."""
    idx = np.arange(H * W).reshape(H, W)
    for dy, dx in OFFSETS:
        y0, y1 = 0, H - dy
        x0, x1 = max(0, -dx), W - max(0, dx)
        a = idx[y0:y1, x0:x1]
        b = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        yield a.ravel(), b.ravel(), np.hypot(dy, dx)


def rms_neighbour_diff(img):
    """sqrt of the mean squared colour difference between 4-neighbours."""
    d = [((img[1:] - img[:-1]) ** 2).sum(-1).ravel(),
         ((img[:, 1:] - img[:, :-1]) ** 2).sum(-1).ravel()]
    return np.sqrt(np.concatenate(d).mean())


def graph_cut(img, Rf, Rb, fg_hard, bg_hard, lam, sigma):
    """
    img      : (H, W, 3) float image
    Rf, Rb   : (H*W,) regional costs -log p(x|F), -log p(x|B)
    fg_hard  : (H*W,) bool, pixels forced to FG (same for bg_hard)
    Returns (labels (H, W) bool, info dict).
    """
    H, W, _ = img.shape
    N = H * W
    S, T = N, N + 1
    X = img.reshape(-1, 3)

    # ---- n-links ----
    rows, cols, caps = [], [], []
    nsum = np.zeros(N)                        # sum of B_ij at each pixel
    for a, b, dist in neighbour_pairs(H, W):
        d2 = ((X[a] - X[b]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * sigma ** 2)) / dist
        rows += [a, b]; cols += [b, a]; caps += [w, w]
        np.add.at(nsum, a, w)
        np.add.at(nsum, b, w)
    K = 1.0 + nsum.max()

    # ---- t-links ----
    m = np.minimum(Rf, Rb)
    cap_s = lam * (Rb - m)                    # s -> i : cost of i being BG
    cap_t = lam * (Rf - m)                    # i -> t : cost of i being FG
    cap_s[fg_hard], cap_t[fg_hard] = K, 0.0
    cap_s[bg_hard], cap_t[bg_hard] = 0.0, K

    # ---- convert to int32 safely ----
    # Any flow is bounded by the total source capacity and by the total sink
    # capacity; every single edge is bounded by its own capacity.
    worst = max(cap_s.sum(), cap_t.sum(), K)
    scale = min(1000.0, 0.9 * INT_LIMIT / worst)
    to_int = lambda c: np.round(c * scale).astype(np.int64)

    pix = np.arange(N)
    rows = np.concatenate(rows + [np.full(N, S), pix])
    cols = np.concatenate(cols + [pix, np.full(N, T)])
    caps = to_int(np.concatenate(caps + [cap_s, cap_t]))
    keep = caps > 0
    G = coo_matrix((caps[keep].astype(np.int32), (rows[keep], cols[keep])),
                   shape=(N + 2, N + 2)).tocsr()
    G.sum_duplicates()
    assert G.data.max() < INT_LIMIT

    # ---- max flow / min cut ----
    t0 = time.time()
    res = maximum_flow(G, S, T, method="dinic")
    t_flow = time.time() - t0

    # residual graph: residual(u, v) = cap(u, v) - flow(u, v); the flow
    # matrix is antisymmetric, so reverse edges get residual = flow.
    R = (G - res.flow).tocsr()
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    reach = breadth_first_order(R, S, directed=True, return_predecessors=False)
    labels = np.zeros(N + 2, bool)
    labels[reach] = True                      # source side = foreground
    labels = labels[:N].reshape(H, W)

    info = dict(flow=res.flow_value / scale, scale=scale, time=t_flow, K=K)
    return labels, info


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def mean_image(img, labels):
    """Each pixel -> mean colour of its class, boundary highlighted."""
    out = np.empty_like(img)
    for v in (False, True):
        sel = labels == v
        out[sel] = img[sel].mean(0) if sel.any() else 0
    out[boundary_mask(labels)] = BOUNDARY_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_image(img, labels):
    out = img.copy()
    out[~labels] *= 0.45                      # darken the background
    out[boundary_mask(labels)] = BOUNDARY_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def save(arr, name):
    plt.imsave(os.path.join(OUT_DIR, name), arr)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    plt.switch_backend("Agg")  # figures are only saved to files
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=LAMBDA,
                    help="weight of the regional term")
    ap.add_argument("--sigma", type=float, default=None,
                    help="sigma of the boundary term in RGB units "
                         "(default: RMS colour difference of neighbours)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    img, scale = load_rgb(IMG_PATH)
    H, W, _ = img.shape
    X = img.reshape(-1, 3)
    print(f"Image size: {W} x {H} (scale {scale:.3f})")

    fg_mask, bg_mask = get_marked_pixels(img, scale, OUT_DIR)

    # ---- regional costs from the part (a) Gaussians ----
    mu_f, cov_f = fit_gaussian(img[fg_mask])
    mu_b, cov_b = fit_gaussian(img[bg_mask])
    Rf = -gaussian_loglik(X, mu_f, cov_f)
    Rb = -gaussian_loglik(X, mu_b, cov_b)
    ml_labels = (Rf < Rb).reshape(H, W)

    # hard constraints: the marked pixels keep their class
    fg_hard = fg_mask.ravel()
    bg_hard = bg_mask.ravel()

    sigma0 = args.sigma if args.sigma else SIGMA_SCALE * rms_neighbour_diff(img)
    lam0 = args.lam
    print(f"Chosen parameters: lambda = {lam0:g}, sigma = {sigma0:.2f}")

    settings = [
        ("best", f"best: lambda={lam0:g}, sigma={sigma0:.1f}", lam0, sigma0),
        ("small_lambda", f"lambda / {FACTOR:g} = {lam0 / FACTOR:.3g}",
         lam0 / FACTOR, sigma0),
        ("large_lambda", f"lambda x {FACTOR:g} = {lam0 * FACTOR:.3g}",
         lam0 * FACTOR, sigma0),
        ("small_sigma", f"sigma / {FACTOR:g} = {sigma0 / FACTOR:.3g}",
         lam0, sigma0 / FACTOR),
        ("large_sigma", f"sigma x {FACTOR:g} = {sigma0 * FACTOR:.3g}",
         lam0, sigma0 * FACTOR),
    ]

    results = {}
    for name, title, lam, sig in settings:
        labels, info = graph_cut(img, Rf, Rb, fg_hard, bg_hard, lam, sig)
        results[name] = (title, labels)
        nb = boundary_mask(labels).sum()
        print(f"  {name:13s} lambda={lam:9.4g} sigma={sig:8.3g}  "
              f"FG {labels.mean() * 100:5.1f}%  boundary px {nb:6d}  "
              f"max-flow {info['flow']:.4g}  ({info['time']:.1f}s)")
        save(overlay_image(img, labels), f"{name}_overlay.png")

    # ---- best result ----
    best = results["best"][1]
    save(mean_image(img, best), "best_means.png")
    save(overlay_image(img, best), "best_overlay.png")
    save((best * 255).astype(np.uint8), "best_mask.png")

    # ---- ML (part a) vs graph cut ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.5))
    axs[0].imshow(img.astype(np.uint8)); axs[0].set_title("Input")
    axs[1].imshow(mean_image(img, ml_labels))
    axs[1].set_title("Part (a): ML, no boundary term")
    axs[2].imshow(mean_image(img, best))
    axs[2].set_title(results["best"][0])
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ml_vs_graphcut.png"), dpi=150)
    plt.close(fig)

    # ---- parameter comparison ----
    fig, axs = plt.subplots(1, 5, figsize=(17, 5.6))
    for a, (name, _, _, _) in zip(axs, settings):
        title, labels = results[name]
        a.imshow(overlay_image(img, labels))
        a.set_title(f"{title}\nFG {labels.mean() * 100:.1f}%", fontsize=9)
        a.axis("off")
    fig.suptitle("Graph cut: chosen parameters vs. much smaller / larger "
                 "lambda and sigma")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "param_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
