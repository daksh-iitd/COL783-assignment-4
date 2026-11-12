"""
Q2(a): Maximum-likelihood foreground/background classifier (first step of GrabCut).

Pipeline:
  1. Load Q2-img.jpg (rescaled so the long side is at most 1024 px).
  2. Known foreground pixels (the soldier) and known background pixels
     (wall, windows, sentry box, ground) are given as sub-rectangles of the
     image: FG_RECTS / BG_RECTS below. These are shared by q2b and q2c.
  3. Fit one multivariate Gaussian N(mu, Sigma) in RGB space to each class
     using the marked pixels (sample mean and covariance).
  4. For every pixel x compute the log-likelihood under each class
         log p(x | c) = -1/2 (x-mu_c)^T Sigma_c^{-1} (x-mu_c)
                        -1/2 log|Sigma_c| - 3/2 log(2 pi)
     This is exactly -R_i(c), the region cost from Boykov-Jolly / GrabCut.
  5. Maximum-likelihood labelling: pixel is foreground iff
         log p(x | F) > log p(x | B).
     (No smoothness / boundary term yet - that comes with the graph cut.)
  6. Outputs (in ./outputs_q2a/):
       - marked_regions.png       : input with FG (green) / BG (blue) boxes
       - likelihood_fg.png        : log p(x | F) as a heat map
       - likelihood_bg.png        : log p(x | B) as a heat map
       - likelihoods_side.png     : both maps side by side, shared colour scale
       - segmentation_means.png   : each pixel replaced by its class mean,
                                    class boundaries highlighted
       - segmentation_overlay.png : boundaries drawn on the original image
       - mask.png                 : binary FG mask

Run:
    python3 q2a.py
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


IMG_PATH = "Q2-img.jpg"
OUT_DIR = "outputs_q2a"
MAX_SIDE = 1024

# Known pixels as rectangles (x0, y0, x1, y1), x = column, y = row, in the
# coordinates of the ORIGINAL 321 x 481 image (rescaled automatically if the
# image is resized). Each box lies entirely inside its class.
# Foreground = the soldier.
FG_RECTS = [
    (76, 160, 108, 195),   # bearskin hat
    (88, 200, 100, 215),   # face
    (62, 235, 80, 300),    # red tunic, left side
    (112, 235, 128, 268),  # red sleeve, right side
    (86, 320, 108, 425),   # blue trousers
]
# Background = everything else: building, windows, sentry box, ground.
BG_RECTS = [
    (0, 0, 321, 140),      # top band: facade, windows, top of sentry box
    (0, 140, 50, 365),     # left wall
    (52, 140, 68, 210),    # window left of the hat
    (140, 140, 180, 440),  # wall between soldier and sentry box
    (185, 140, 275, 430),  # red sentry box
    (280, 140, 321, 445),  # right windows / wall
    (0, 395, 75, 445),     # pavement left of the legs
    (0, 445, 321, 481),    # cobblestones at the bottom
]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_rgb(path, max_side=MAX_SIDE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64), scale


def save_rgb(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, name))


def scale_rects(rects, scale):
    return [tuple(int(round(v * scale)) for v in r) for r in rects]


def rects_to_mask(rects, shape):
    """rects are (x0, y0, x1, y1) in the coordinates of the (resized) image."""
    m = np.zeros(shape[:2], dtype=bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return m

# --------------------------------------------------------------------------
# Gaussian model
# --------------------------------------------------------------------------
def fit_gaussian(samples):
    """samples: (N, 3). Returns mean (3,) and covariance (3, 3)."""
    mu = samples.mean(axis=0)
    d = samples - mu
    cov = d.T @ d / (len(samples) - 1)
    cov += 1e-6 * np.eye(3)  # regularise in case of a degenerate region
    return mu, cov


def gaussian_loglik(X, mu, cov):
    """Log density of N(mu, cov) evaluated at each row of X (N, 3)."""
    k = X.shape[1]
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    d = X - mu
    maha = np.einsum("ni,ij,nj->n", d, inv, d)  # squared Mahalanobis distance
    return -0.5 * (maha + logdet + k * np.log(2 * np.pi))


# --------------------------------------------------------------------------
# Visualisation helpers
# --------------------------------------------------------------------------
def boundary_mask(labels):
    """True at pixels that have a 4-neighbour with a different label."""
    b = np.zeros(labels.shape, dtype=bool)
    dv = labels[1:, :] != labels[:-1, :]
    dh = labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= dv
    b[:-1, :] |= dv
    b[:, 1:] |= dh
    b[:, :-1] |= dh
    return b


def save_heatmap(L, name, title, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5, 6.5))
    im = ax.imshow(L, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log-likelihood")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def get_marked_pixels(img, scale, out_dir):
    """Known FG / BG pixels from FG_RECTS / BG_RECTS -> (fg_mask, bg_mask).
    Also saves marked_regions.png showing the boxes."""
    fg_rects = scale_rects(FG_RECTS, scale)
    bg_rects = scale_rects(BG_RECTS, scale)
    fg_mask = rects_to_mask(fg_rects, img.shape)
    bg_mask = rects_to_mask(bg_rects, img.shape)
    assert not (fg_mask & bg_mask).any(), "FG and BG rectangles overlap"

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.imshow(img.astype(np.uint8))
    for rects, col in ((fg_rects, "lime"), (bg_rects, "deepskyblue")):
        for x0, y0, x1, y1 in rects:
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=col, linewidth=1.5))
    ax.set_title("Known FG (green) / BG (blue)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "marked_regions.png"), dpi=150)
    plt.close(fig)
    return fg_mask, bg_mask


def main():
    plt.switch_backend("Agg")  # figures are only saved to files
    os.makedirs(OUT_DIR, exist_ok=True)
    img, scale = load_rgb(IMG_PATH)
    H, W, _ = img.shape
    X = img.reshape(-1, 3)
    print(f"Image size: {W} x {H} (scale {scale:.3f})")

    fg_mask, bg_mask = get_marked_pixels(img, scale, OUT_DIR)

    # ---- fit Gaussians ----
    mu_f, cov_f = fit_gaussian(img[fg_mask])
    mu_b, cov_b = fit_gaussian(img[bg_mask])
    print(f"FG: {fg_mask.sum()} px, mean RGB = {np.round(mu_f, 1)}")
    print(f"BG: {bg_mask.sum()} px, mean RGB = {np.round(mu_b, 1)}")
    print("FG covariance:\n", np.round(cov_f, 1))
    print("BG covariance:\n", np.round(cov_b, 1))

    # ---- per-pixel log-likelihoods ----
    Lf = gaussian_loglik(X, mu_f, cov_f).reshape(H, W)
    Lb = gaussian_loglik(X, mu_b, cov_b).reshape(H, W)

    # Log-likelihoods have a long negative tail (outlier colours); clip the
    # display range to robust percentiles over both maps so they share a scale.
    both = np.concatenate([Lf.ravel(), Lb.ravel()])
    vmin, vmax = np.percentile(both, 1), both.max()
    save_heatmap(Lf, "likelihood_fg.png", "log p(x | foreground)", vmin, vmax)
    save_heatmap(Lb, "likelihood_bg.png", "log p(x | background)", vmin, vmax)

    fig, axs = plt.subplots(1, 2, figsize=(9, 6.5))
    for ax, L, t in ((axs[0], Lf, "log p(x | FG)"), (axs[1], Lb, "log p(x | BG)")):
        im = ax.imshow(L, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label="log-likelihood")
    fig.savefig(os.path.join(OUT_DIR, "likelihoods_side.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- maximum-likelihood classification ----
    labels = (Lf > Lb)  # True = foreground
    acc_f = labels[fg_mask].mean()
    acc_b = (~labels[bg_mask]).mean()
    print(f"Accuracy on marked pixels: FG {acc_f * 100:.1f}%, "
          f"BG {acc_b * 100:.1f}%")
    print(f"Pixels classified FG: {labels.mean() * 100:.1f}%")

    bnd = boundary_mask(labels)

    seg = np.where(labels[..., None], mu_f, mu_b)
    seg[bnd] = (0, 255, 0)  # bright green boundary
    save_rgb(seg, "segmentation_means.png")

    over = img.copy()
    over[bnd] = (0, 255, 0)
    save_rgb(over, "segmentation_overlay.png")
    save_rgb(labels[..., None].repeat(3, axis=2) * 255.0, "mask.png")

    # ---- summary figure ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.5))
    axs[0].imshow(img.astype(np.uint8)); axs[0].set_title("Input")
    axs[1].imshow(np.clip(seg, 0, 255).astype(np.uint8))
    axs[1].set_title("ML segmentation (class means)")
    axs[2].imshow(np.clip(over, 0, 255).astype(np.uint8))
    axs[2].set_title("Boundaries on input")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
    plt.close(fig)
    print(f"Saved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
