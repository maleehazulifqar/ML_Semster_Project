"""
Shared functions: image reading, severity score, dataset loading, splits.
Is file mein sirf functions hain, koi top-level code nahi.
"""
from pathlib import Path

import cv2
import numpy as np

from config import CLASS_NAMES, DATASET_DIR, GRAY_SIZE, IMAGE_EXTS, IMG_SIZE, SPLITS_PATH

try:
    from tqdm import tqdm
except ImportError:  # tqdm na ho tab bhi chale
    def tqdm(iterable, **kwargs):
        return iterable


def read_image(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread jaisa, lekin Windows par non-English folder names ke saath bhi chalta hai."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Image read nahi hui (corrupt ya galat format): {path}")
    return img


def severity_from_gray(gray):
    """
    Severity score (0.0 - 1.0) = pixel variance + Canny edge density.
    Formula teammate ke original utils.py wala hi hai. Ye image array leta hai
    taake augmented images par bhi bilkul yahi formula lag sake.
    NOTE (report ke liye): ye target pixels se nikalta hai (synthetic), asal
    damage measurement nahi hai.
    """
    pixel_var = np.var(gray)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = np.sum(edges > 0) / edges.size
    raw_score = (pixel_var / 5000.0) * 0.4 + (edge_density * 2.0) * 0.6
    return float(np.clip(raw_score, 0.0, 1.0))


def compute_severity_score(image_path):
    return severity_from_gray(read_image(image_path, cv2.IMREAD_GRAYSCALE))


def list_image_files(base_dir=DATASET_DIR):
    """
    Har class folder ki images, naam ke hisaab se sorted (Windows aur Linux par same order).
    Poori pipeline isi order par chalti hai, is liye split indices har script mein
    same images ko point karte hain.
    """
    base_dir = Path(base_dir)
    paths, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder = base_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Folder nahi mila: {folder}\n"
                f"Dataset ka structure dataset/<class_name>/*.jpg hona chahiye."
            )
        files = sorted(
            (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name,
        )
        paths.extend(files)
        labels.extend([class_idx] * len(files))
    return paths, np.array(labels, dtype=np.int64)


def load_dataset_targets(base_dir=DATASET_DIR, img_size=IMG_SIZE):
    """
    CNN ke liye: RGB images (N, 224, 224, 3), float32, [0, 1] mein normalized.
    Returns: X, y_class, y_reg
    """
    paths, y_class = list_image_files(base_dir)
    X = np.empty((len(paths), img_size[1], img_size[0], 3), dtype=np.float32)
    y_reg = np.empty(len(paths), dtype=np.float32)

    for i, path in enumerate(tqdm(paths, desc="Loading images")):
        img = read_image(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        X[i] = img.astype(np.float32) / 255.0
        y_reg[i] = compute_severity_score(path)

    _print_class_counts(y_class)
    return X, y_class, y_reg


def load_gray_dataset(base_dir=DATASET_DIR, size=GRAY_SIZE):
    """
    Classical pipeline aur augmentation ke liye: grayscale uint8 (N, 200, 200).
    Halka hai (~72 MB), is liye laptop par aram se chalta hai.
    Returns: images, y_class, y_reg  (order wahi jo load_dataset_targets ka)
    """
    paths, y_class = list_image_files(base_dir)
    images = np.empty((len(paths), size[1], size[0]), dtype=np.uint8)
    y_reg = np.empty(len(paths), dtype=np.float32)

    for i, path in enumerate(tqdm(paths, desc="Loading grayscale")):
        gray = read_image(path, cv2.IMREAD_GRAYSCALE)
        y_reg[i] = severity_from_gray(gray)          # asal size par (original jaisa)
        if gray.shape != (size[1], size[0]):
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        images[i] = gray

    return images, y_class, y_reg


def load_splits(base_dir=DATASET_DIR):
    """main.py ka banaya hua split parhta hai, aur check karta hai ke dataset badla to nahi."""
    if not Path(SPLITS_PATH).exists():
        raise FileNotFoundError("splits.npz nahi mila. Pehle chalayein: python main.py")
    s = np.load(SPLITS_PATH)
    current = [p.relative_to(base_dir).as_posix() for p in list_image_files(base_dir)[0]]
    if list(s["files"]) != current:
        raise RuntimeError(
            "Dataset ki files split banne ke baad badal gayi hain. "
            "Dobara chalayein: python main.py"
        )
    return s["train_idx"], s["val_idx"], s["test_idx"]


def _print_class_counts(y_class):
    for idx, name in enumerate(CLASS_NAMES):
        print(f"[+] {name:16} {int(np.sum(y_class == idx))} images")
