"""
Training set ka offline augmentation (validation / test ko haath nahi lagta).
Proposal ke mutabiq: random rotation, horizontal / vertical flip, zoom.

Chalane ka order: python main.py  ->  python augment.py
Output: aug_train.npz
  images      (N, 200, 200) uint8 grayscale
  y_class     (N,)  original image ki class
  y_reg       (N,)  severity augmented image par DOBARA compute hoti hai
                    (rotation / zoom pixels badalte hain, is liye purana score galat hota)
  source_idx  (N,)  kis original image se bani (K-Fold mein leakage rokne ke liye)
Saath mein results/augmentation_preview.png (report ke liye).
"""
import cv2
import numpy as np

from config import (AUG_PATH, AUG_PER_IMAGE, AUG_ROTATION_DEG, AUG_ZOOM_RANGE,
                    CLASS_NAMES, RANDOM_STATE, RESULTS_DIR)
from utils import load_gray_dataset, load_splits, severity_from_gray, tqdm


def augment_one(img, rng):
    h, w = img.shape
    angle = rng.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG)
    zoom = rng.uniform(*AUG_ZOOM_RANGE)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    # BORDER_REFLECT: kone kaale nahi hote, texture hi repeat hota hai
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)   # horizontal
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)   # vertical
    return out


def save_preview(images, aug_images, source_idx, y_class, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    n_show = AUG_PER_IMAGE + 1
    fig, axes = plt.subplots(len(CLASS_NAMES), n_show, figsize=(2.2 * n_show, 2.2 * len(CLASS_NAMES)))
    for c, name in enumerate(CLASS_NAMES):
        src = rng.choice(np.unique(source_idx[y_class[source_idx] == c]))
        row = [images[src]] + list(aug_images[source_idx == src])
        for j in range(n_show):
            ax = axes[c, j]
            ax.imshow(row[j], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title("Original" if j == 0 else f"Aug {j}", fontsize=9)
        axes[c, 0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "augmentation_preview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    train_idx, val_idx, test_idx = load_splits()
    images, y_class, y_reg = load_gray_dataset()

    n_out = len(train_idx) * AUG_PER_IMAGE
    aug_images = np.empty((n_out, *images.shape[1:]), dtype=np.uint8)
    aug_y_class = np.empty(n_out, dtype=np.int64)
    aug_y_reg = np.empty(n_out, dtype=np.float32)
    source_idx = np.empty(n_out, dtype=np.int64)

    k = 0
    for i in tqdm(train_idx, desc="Augmenting"):
        for _ in range(AUG_PER_IMAGE):
            aug = augment_one(images[i], rng)
            aug_images[k] = aug
            aug_y_class[k] = y_class[i]
            aug_y_reg[k] = severity_from_gray(aug)
            source_idx[k] = i
            k += 1

    # Safety: koi augmented image validation / test se nahi bani
    assert not set(source_idx) & (set(val_idx) | set(test_idx))

    np.savez_compressed(AUG_PATH, images=aug_images, y_class=aug_y_class,
                        y_reg=aug_y_reg, source_idx=source_idx)
    print(f"\nTraining originals: {len(train_idx)}, augmented: {n_out}, "
          f"total training: {len(train_idx) + n_out}")
    shift = np.abs(aug_y_reg - y_reg[source_idx])
    print(f"Severity change after augmentation: mean {shift.mean():.3f}, max {shift.max():.3f}")
    print(f"Saved: {AUG_PATH.name}")

    save_preview(images, aug_images, source_idx, y_class, rng)
