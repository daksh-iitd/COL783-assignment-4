"""
Q1(b): Dijkstra's algorithm on the 8-connected pixel graph, with
Intelligent-Scissors edge weights (Mortensen & Barrett 1995):

    w(p, q) = λZ·fZ(q) + λG·fG(q) + λD·fD(p, q)

  fZ(q)    = 0 if q is a Laplacian (LoG) zero crossing, else 1
             (zero crossings come from the Marr-Hildreth detector of q1a)
  fG(q)    = 1 - |∇f(q)| / max|∇f|   (small where the gradient is strong),
             multiplied by 1 for diagonal links and 1/√2 for horizontal/
             vertical links, as in the paper, so diagonal steps cost more
             (they are √2 times longer)
  fD(p, q) = 1/π · [ acos(dp) + acos(dq) ]   (gradient-direction term;
             dp ∈ [0,1], dq ∈ [-1,1], so fD ∈ [0, 1.5]. The SIGGRAPH'95
             version uses 2/(3π) instead, which only rescales fD to [0,1])
             D(p) = unit vector perpendicular to ∇f(p) (the edge direction)
             L(p,q) = unit link vector q-p, flipped if needed so D(p)·L >= 0
             dp = D(p)·L,  dq = L·D(q)
             Cheap when the link runs along the edge (perpendicular to both
             gradients) and the two gradients point the same way.

Dijkstra is run from a seed pixel with each of the
four weight choices, and the resulting distance maps are shown as images.
The shortest path from the seed to a sample "free" point is also traced, as
a check that the weights do follow image edges.

Outputs are written to ./outputs_q1b/.
"""

import os
import heapq
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from q1a import load_gray, gaussian_smooth, marr_hildreth

OUT_DIR = "outputs_q1b"

# Marr-Hildreth parameters (best values found in q1a)
LOG_SIGMA = 2.5
LOG_T = 0.07
# Small amount of smoothing before taking gradients (reduces noise)
GRAD_SIGMA = 1.0
# Weights from Mortensen & Barrett
LAMBDA_Z, LAMBDA_G, LAMBDA_D = 0.43, 0.43, 0.14

# 8 neighbour offsets (dy, dx)
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


# --------------------------------------------------------------------------
# Per-pixel feature maps
# --------------------------------------------------------------------------
def gradient(img, sigma=GRAD_SIGMA):
    """Central-difference gradient of the Gaussian-smoothed image."""
    s = gaussian_smooth(img, sigma)
    p = np.pad(s, 1, mode="reflect")
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / 2.0
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2.0
    return gx, gy


def feature_maps(img):
    """Return fZ (H,W), fG (H,W), and unit edge-direction field (Dx, Dy)."""
    edges, _ = marr_hildreth(img, LOG_SIGMA, LOG_T)
    fZ = np.where(edges, 0.0, 1.0)

    gx, gy = gradient(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    fG = 1.0 - mag / mag.max()

    # D(p) = gradient rotated by 90° and normalised: (gy, -gx)/|∇f|
    safe = np.where(mag > 1e-12, mag, 1.0)
    Dx = np.where(mag > 1e-12, gy / safe, 0.0)
    Dy = np.where(mag > 1e-12, -gx / safe, 0.0)
    return fZ, fG, Dx, Dy


def shift(a, dy, dx, fill):
    """out[y, x] = a[y+dy, x+dx], with `fill` outside the image."""
    H, W = a.shape
    out = np.full_like(a, fill)
    ys_o = slice(max(0, -dy), H - max(0, dy))
    xs_o = slice(max(0, -dx), W - max(0, dx))
    ys_i = slice(max(0, dy), H - max(0, -dy))
    xs_i = slice(max(0, dx), W - max(0, -dx))
    out[ys_o, xs_o] = a[ys_i, xs_i]
    return out


def link_costs(fZ, fG, Dx, Dy, lz, lg, ld):
    """Return a (8, H, W) array: cost[k, y, x] = w(p, q) for p = (y, x) and
    q = p + NEIGHBOURS[k]. Links leaving the image get +inf."""
    H, W = fZ.shape
    costs = np.empty((8, H, W), dtype=np.float64)
    for k, (dy, dx) in enumerate(NEIGHBOURS):
        diag = dy != 0 and dx != 0
        length = np.sqrt(2.0) if diag else 1.0

        fZq = shift(fZ, dy, dx, np.inf)
        fGq = shift(fG, dy, dx, np.inf) * (1.0 if diag else 1.0 / np.sqrt(2.0))

        # Gradient-direction term
        Lx, Ly = dx / length, dy / length
        Dxq, Dyq = shift(Dx, dy, dx, 0.0), shift(Dy, dy, dx, 0.0)
        s = Dx * Lx + Dy * Ly                 # D(p)·(q-p)
        sign = np.where(s >= 0, 1.0, -1.0)    # choose L so that D(p)·L >= 0
        dp = np.clip(sign * s, -1, 1)
        dq = np.clip(sign * (Lx * Dxq + Ly * Dyq), -1, 1)
        fD = (1.0 / np.pi) * (np.arccos(dp) + np.arccos(dq))

        c = np.zeros((H, W))
        if lz:
            c += lz * fZq
        if lg:
            c += lg * fGq
        if ld:
            c += ld * fD
        # mark links that leave the image
        c[~np.isfinite(fZq)] = np.inf
        costs[k] = c
    return costs


# --------------------------------------------------------------------------
# Dijkstra (binary heap, lazy deletion)
# --------------------------------------------------------------------------
def dijkstra(costs, seed, target=None):
    """Single-source shortest paths over the 8-connected pixel grid.

    costs:  (8, H, W) link costs from link_costs().
    seed:   (row, col).
    target: optional (row, col); if given, stop as soon as it is settled
            (its distance and predecessor chain are then final).
    Returns dist (H, W) and pred (H, W) holding the flat index of each
    pixel's predecessor (-1 for the seed).
    """
    _, H, W = costs.shape
    N = H * W
    offs = [dy * W + dx for dy, dx in NEIGHBOURS]
    cost_lists = [costs[k].ravel().tolist() for k in range(8)]

    dist = [float("inf")] * N
    pred = [-1] * N
    done = [False] * N
    s = seed[0] * W + seed[1]
    dist[s] = 0.0
    heap = [(0.0, s)]
    t = -1 if target is None else target[0] * W + target[1]

    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == t:
            break
        for k in range(8):
            w = cost_lists[k][u]
            if w == float("inf"):
                continue
            v = u + offs[k]
            if done[v]:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    return np.array(dist).reshape(H, W), np.array(pred).reshape(H, W)


def trace_path(pred, target):
    """Follow predecessors from target back to the seed."""
    H, W = pred.shape
    v = target[0] * W + target[1]
    path = []
    flat = pred.ravel()
    while v != -1:
        path.append(divmod(v, W))
        v = flat[v]
    return np.array(path[::-1])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray("bird.jpg")
    H, W = img.shape
    print("Image size:", img.shape)

    fZ, fG, Dx, Dy = feature_maps(img)

    # Seed on the bird's outline (left edge of the tail) and a free point
    # further down its back, both snapped to the strongest gradient nearby.
    def snap(pt, r=4):
        y, x = pt
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        win = fG[y0:y1, x0:x1]
        dy, dx = np.unravel_index(np.argmin(win), win.shape)
        return (y0 + dy, x0 + dx)

    seed = snap((95, 105))
    free = snap((265, 120))
    print("Seed:", seed, " Free point:", free)

    # Visualise the per-pixel terms
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
    for ax, im, t in zip(axes,
                         [img, fZ, fG, np.arctan2(Dy, Dx)],
                         ["Input", "fZ (0 at zero crossings)",
                          "fG = 1 - |∇f|/max", "Edge direction angle of D(p)"]):
        ax.imshow(im, cmap="twilight" if "angle" in t else "gray")
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cost_terms.png", dpi=150)
    plt.close(fig)

    configs = [
        ("(i) zero crossings only", "zero_crossing", (1.0, 0.0, 0.0)),
        ("(ii) gradient magnitude only", "gradient_magnitude", (0.0, 1.0, 0.0)),
        ("(iii) gradient direction only", "gradient_direction", (0.0, 0.0, 1.0)),
        (f"(iv) {LAMBDA_Z}·fZ + {LAMBDA_G}·fG + {LAMBDA_D}·fD", "combined",
         (LAMBDA_Z, LAMBDA_G, LAMBDA_D)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 12))
    for col, (title, name, (lz, lg, ld)) in enumerate(configs):
        costs = link_costs(fZ, fG, Dx, Dy, lz, lg, ld)
        dist, pred = dijkstra(costs, seed)
        path = trace_path(pred, free)
        print(f"{title}: max distance = {dist.max():.2f}, "
              f"path length = {len(path)} px, path cost = {dist[free]:.2f}")

        np.save(f"{OUT_DIR}/dist_{name}.npy", dist)
        dn = dist / dist.max()
        plt.imsave(f"{OUT_DIR}/dist_{name}.png", dn, cmap="magma")

        # Row 1: distance map with iso-distance contours
        ax = axes[0, col]
        ax.imshow(dn, cmap="magma")
        ax.contour(dn, levels=15, colors="white", linewidths=0.4, alpha=0.6)
        ax.plot(seed[1], seed[0], "c*", ms=12)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        # Row 2: shortest path from seed to the free point, over the image
        ax = axes[1, col]
        ax.imshow(img, cmap="gray")
        ax.plot(path[:, 1], path[:, 0], "r-", lw=1.5)
        ax.plot(seed[1], seed[0], "c*", ms=12)
        ax.plot(free[1], free[0], "yo", ms=6)
        ax.set_title("Shortest path seed → free point", fontsize=10)
        ax.axis("off")

    fig.suptitle("Dijkstra distance from seed (★) for different edge weights "
                 "(dark = close, bright = far)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/distance_maps.png", dpi=150)
    plt.close(fig)
    print("Saved results to", OUT_DIR)


if __name__ == "__main__":
    main()
