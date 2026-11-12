"""
Q1(c): Intelligent-Scissors segmentation from a sequence of manually chosen
seeds, plus two modifications of the edge weights.

Segmentation
  * Seeds are placed by hand around the bird (listed in SEEDS_DENSE, in
    clockwise order), each snapped to the strongest gradient within 2 px.
  * For each consecutive pair (s_i, s_{i+1}) (and s_last -> s_0 to close the
    loop), Dijkstra from q1b is run from s_i until s_{i+1} is settled, and
    the predecessor chain gives the boundary piece between them.
  * The pieces are joined into one closed 8-connected contour. The object
    mask is everything NOT reachable from the image border by a 4-connected
    flood fill that is blocked by the contour (an 8-connected curve cannot be
    crossed by 4-connected moves).

Idea 1: Canny edges instead of Marr-Hildreth zero crossings in fZ.
  Canny is implemented here: Gaussian gradient -> non-maximum suppression
  along the quantised gradient direction -> double threshold + hysteresis.
  Canny edges sit on gradient maxima (not on sign changes of the LoG), are
  one pixel thick and are far less fragmented on low-contrast boundaries.

Idea 2: changing the relative weights λZ, λG, λD of the three terms.
  Compared: the paper's default (0.43, 0.43, 0.14), a gradient-magnitude-heavy
  setting, a zero-crossing-heavy setting and a direction-heavy setting.

Both ideas are evaluated with a sparse seed set (SEEDS_SPARSE): with fewer
seeds each path segment is longer, so differences between cost functions are
much more visible than with the dense seeds.

Outputs are written to ./outputs_q1c/.
"""

import os
from collections import deque
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from q1a import load_gray, gaussian_smooth
from q1b import (feature_maps, link_costs, dijkstra, trace_path, gradient,
                 LAMBDA_Z, LAMBDA_G, LAMBDA_D)

OUT_DIR = "outputs_q1c"

# Manually chosen seeds (row, col) around the bird, clockwise from the tail.
SEEDS_DENSE = [
    (60, 106),   # tail, top-left corner
    (78, 140),   # tail, top-right
    (110, 151),  # tail, right side
    (190, 133),  # back
    (232, 150),  # top of head
    (262, 160),  # front of head
    (291, 157),  # beak tip
    (271, 136),  # chin / throat
    (270, 112),  # belly
    (262, 92),   # belly where it meets the branch
    (240, 78),   # white wing patch against the branch
    (210, 72),   # wing, lower left
    (186, 80),   # wing, near the leg
    (160, 91),   # back of wing
    (130, 107),  # base of tail
    (100, 104),  # white outer tail feathers
]

# Fewer seeds -> longer path segments -> cost-function differences show up.
SEEDS_SPARSE = [
    (60, 106),
    (110, 151),
    (232, 150),
    (291, 157),
    (262, 92),
    (186, 80),
]


# --------------------------------------------------------------------------
# Canny edge detector (Idea 1)
# --------------------------------------------------------------------------
def canny(img, sigma=1.5, high=0.15, low_ratio=0.4):
    """Canny edges. Thresholds are fractions of the max gradient magnitude."""
    gx, gy = gradient(img, sigma)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Non-maximum suppression: quantise gradient direction to 0/45/90/135°
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = np.zeros(mag.shape, dtype=int)
    q[(ang >= 22.5) & (ang < 67.5)] = 1
    q[(ang >= 67.5) & (ang < 112.5)] = 2
    q[(ang >= 112.5) & (ang < 157.5)] = 3
    # neighbour offsets (dy, dx) along the gradient for each bin
    # (rows grow downward, so +y in gradient terms is +row)
    dirs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}

    p = np.pad(mag, 1, mode="constant")
    H, W = mag.shape
    nms = np.zeros_like(mag)
    for k, (dy, dx) in dirs.items():
        fwd = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        bwd = p[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        keep = (q == k) & (mag >= fwd) & (mag >= bwd)
        nms[keep] = mag[keep]

    # Double threshold + hysteresis (grow strong edges through weak ones)
    hi = high * mag.max()
    lo = low_ratio * hi
    strong = nms >= hi
    weak = nms >= lo
    edges = strong.copy()
    while True:
        e = np.pad(edges, 1)
        grown = np.zeros_like(edges)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= e[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        new = grown & weak
        if (new == edges).all():
            break
        edges = new
    return edges


# --------------------------------------------------------------------------
# Livewire segmentation
# --------------------------------------------------------------------------
def snap_seeds(seeds, fG, r=2):
    """Move each seed to the pixel of strongest gradient within r pixels."""
    H, W = fG.shape
    out = []
    for y, x in seeds:
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        dy, dx = np.unravel_index(np.argmin(fG[y0:y1, x0:x1]),
                                  (y1 - y0, x1 - x0))
        out.append((y0 + dy, x0 + dx))
    return out


def livewire(costs, seeds):
    """Closed contour through the seeds (in order) using shortest paths."""
    pieces, total = [], 0.0
    for i in range(len(seeds)):
        a, b = seeds[i], seeds[(i + 1) % len(seeds)]
        dist, pred = dijkstra(costs, a, target=b)
        total += dist[b]
        path = trace_path(pred, b)
        pieces.append(path[:-1])  # drop endpoint; it starts the next piece
    return np.vstack(pieces), total


def fill_contour(contour, shape):
    """Mask of the region enclosed by a closed 8-connected contour."""
    H, W = shape
    wall = np.zeros(shape, dtype=bool)
    wall[contour[:, 0], contour[:, 1]] = True
    outside = np.zeros(shape, dtype=bool)
    dq = deque()
    for y in range(H):
        for x in (0, W - 1):
            if not wall[y, x] and not outside[y, x]:
                outside[y, x] = True
                dq.append((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if not wall[y, x] and not outside[y, x]:
                outside[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and not wall[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                dq.append((ny, nx))
    return ~outside


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def draw_result(ax, img, contour, seeds, title, crop=None):
    ax.imshow(img, cmap="gray")
    c = np.vstack([contour, contour[:1]])
    ax.plot(c[:, 1], c[:, 0], "r-", lw=1.4)
    s = np.array(seeds)
    ax.plot(s[:, 1], s[:, 0], "o", mfc="yellow", mec="k", ms=6)
    if crop is not None:
        (y0, y1), (x0, x1) = crop
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def overlay_mask(img, mask):
    rgb = np.dstack([img] * 3)
    tint = np.array([1.0, 0.3, 0.3])
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * tint
    rgb[~mask] *= 0.45
    return np.clip(rgb, 0, 1)


BIRD_CROP = ((40, 320), (20, 200))
USER_SEEDS_FILE = f"{OUT_DIR}/interactive_seeds.txt"


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def compare_user_seeds(img, fZ_mh, fG, Dx, Dy, path=USER_SEEDS_FILE):
    """Re-run the default cost and both ideas with seeds chosen by clicking
    in q1c_web.py. The seeds are used exactly as saved (already snapped)."""
    seeds = [tuple(map(int, r)) for r in np.loadtxt(path, dtype=int, ndmin=2)]
    print(f"\nUser seeds from {path}: {len(seeds)} seeds")

    fZ_canny = np.where(canny(img), 0.0, 1.0)
    variants = [
        ("Default: Marr-Hildreth fZ, λ=(0.43, 0.43, 0.14)",
         fZ_mh, (LAMBDA_Z, LAMBDA_G, LAMBDA_D)),
        ("Idea 1: Canny fZ, λ=(0.43, 0.43, 0.14)",
         fZ_canny, (LAMBDA_Z, LAMBDA_G, LAMBDA_D)),
        ("Idea 2: gradient-heavy λ=(0.1, 0.8, 0.1)", fZ_mh, (0.1, 0.8, 0.1)),
        ("Idea 2: zero-crossing-heavy λ=(0.8, 0.1, 0.1)", fZ_mh, (0.8, 0.1, 0.1)),
        ("Idea 2: direction-heavy λ=(0.2, 0.2, 0.6)", fZ_mh, (0.2, 0.2, 0.6)),
    ]
    results = []
    for name, fZ, (lz, lg, ld) in variants:
        costs = link_costs(fZ, fG, Dx, Dy, lz, lg, ld)
        c, tc = livewire(costs, seeds)
        m = fill_contour(c, img.shape)
        results.append((name, c, m))
    base_mask = results[0][2]
    for name, c, m in results:
        print(f"  {name}: contour {len(c)} px, area {m.sum()} px, "
              f"IoU vs default {iou(m, base_mask):.3f}")

    def figure(sel, fname, suptitle):
        fig, axes = plt.subplots(2, len(sel), figsize=(5 * len(sel), 13))
        axes = np.array(axes).reshape(2, len(sel))
        for col, (name, c, m) in enumerate(sel):
            draw_result(axes[0, col], img, c, seeds, name, BIRD_CROP)
            axes[1, col].imshow(overlay_mask(img, m))
            (y0, y1), (x0, x1) = BIRD_CROP
            axes[1, col].set_xlim(x0, x1)
            axes[1, col].set_ylim(y1, y0)
            axes[1, col].set_title(f"Mask: {m.sum()} px, IoU vs default "
                                   f"{iou(m, base_mask):.3f}", fontsize=10)
            axes[1, col].axis("off")
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(f"{OUT_DIR}/{fname}", dpi=150)
        plt.close(fig)

    n = len(seeds)
    figure(results[:2], "user_seeds_idea1_canny.png",
           f"Idea 1 with user-chosen seeds ({n}): Marr-Hildreth vs Canny in fZ")
    figure([results[0]] + results[2:], "user_seeds_idea2_weights.png",
           f"Idea 2 with user-chosen seeds ({n}): changing λZ, λG, λD")
    np.save(f"{OUT_DIR}/user_seeds_mask_default.npy", base_mask)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray("bird.jpg")
    fZ_mh, fG, Dx, Dy = feature_maps(img)

    seeds_dense = snap_seeds(SEEDS_DENSE, fG)
    seeds_sparse = snap_seeds(SEEDS_SPARSE, fG)

    # ---------------- Segmentation with the default cost ------------------
    base_costs = link_costs(fZ_mh, fG, Dx, Dy, LAMBDA_Z, LAMBDA_G, LAMBDA_D)
    contour, cost = livewire(base_costs, seeds_dense)
    mask = fill_contour(contour, img.shape)
    print(f"Default cost, {len(seeds_dense)} seeds: contour {len(contour)} px, "
          f"total cost {cost:.2f}, area {mask.sum()} px")

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    draw_result(axes[0], img, contour, seeds_dense,
                f"Seeds ({len(seeds_dense)}) and closed livewire contour")
    draw_result(axes[1], img, contour, seeds_dense, "Zoom on the bird", BIRD_CROP)
    axes[2].imshow(overlay_mask(img, mask))
    axes[2].set_title("Segmentation mask")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/segmentation.png", dpi=150)
    plt.close(fig)
    plt.imsave(f"{OUT_DIR}/mask.png", mask, cmap="gray")

    # ---------------- Idea 1: Canny vs Marr-Hildreth ----------------------
    canny_edges = canny(img)
    fZ_canny = np.where(canny_edges, 0.0, 1.0)
    canny_costs = link_costs(fZ_canny, fG, Dx, Dy, LAMBDA_Z, LAMBDA_G, LAMBDA_D)

    fig, axes = plt.subplots(2, 3, figsize=(15, 14))
    axes[0, 0].imshow(fZ_mh == 0, cmap="gray")
    axes[0, 0].set_title("Marr-Hildreth zero crossings (σ=2.5, T=0.07)")
    axes[1, 0].imshow(canny_edges, cmap="gray")
    axes[1, 0].set_title("Canny edges (σ=1.5, high=0.15, low=0.4·high)")
    for ax in axes[:, 0]:
        ax.axis("off")
    for row, (name, costs) in enumerate([("Marr-Hildreth fZ", base_costs),
                                         ("Canny fZ", canny_costs)]):
        for col, (seeds, tag) in enumerate([(seeds_dense, "dense"),
                                            (seeds_sparse, "sparse")]):
            c, tc = livewire(costs, seeds)
            print(f"Idea 1, {name}, {tag} seeds: contour {len(c)} px, cost {tc:.2f}")
            draw_result(axes[row, col + 1], img, c, seeds,
                        f"{name}, {len(seeds)} seeds", BIRD_CROP)
    fig.suptitle("Idea 1: Canny edges instead of Marr-Hildreth in the fZ term",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/idea1_canny.png", dpi=150)
    plt.close(fig)

    # ---------------- Idea 2: term weights --------------------------------
    weightings = [
        ("Default (0.43, 0.43, 0.14)", (LAMBDA_Z, LAMBDA_G, LAMBDA_D)),
        ("Gradient-heavy (0.1, 0.8, 0.1)", (0.1, 0.8, 0.1)),
        ("Zero-crossing-heavy (0.8, 0.1, 0.1)", (0.8, 0.1, 0.1)),
        ("Direction-heavy (0.2, 0.2, 0.6)", (0.2, 0.2, 0.6)),
    ]
    fig, axes = plt.subplots(1, len(weightings), figsize=(20, 8))
    for ax, (name, (lz, lg, ld)) in zip(axes, weightings):
        costs = link_costs(fZ_mh, fG, Dx, Dy, lz, lg, ld)
        c, tc = livewire(costs, seeds_sparse)
        print(f"Idea 2, {name}, sparse seeds: contour {len(c)} px, cost {tc:.2f}")
        draw_result(ax, img, c, seeds_sparse, f"{name}\n(λZ, λG, λD)", BIRD_CROP)
    fig.suptitle(f"Idea 2: changing the term weights ({len(seeds_sparse)} seeds)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT_DIR}/idea2_weights.png", dpi=150)
    plt.close(fig)

    # ---------------- Both ideas with seeds clicked in q1c_web.py ---------
    if os.path.exists(USER_SEEDS_FILE):
        compare_user_seeds(img, fZ_mh, fG, Dx, Dy)

    print("Saved results to", OUT_DIR)


if __name__ == "__main__":
    main()
