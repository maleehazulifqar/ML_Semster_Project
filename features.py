"""
Classical baseline ke liye HOG + GLCM features nikalta hai aur cache karta hai.
Original 1800 images + augmented training images, dono ke features ek baar ban kar
features/ folder mein save ho jate hain (baseline script dobara nahi nikalti).

Chalane ka order: python main.py -> python augment.py -> python features.py
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog

from config import AUG_PATH, FEATURES_DIR, GLCM_LEVELS, HOG_SIZE
from utils import load_gray_dataset, tqdm

GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def hog_features(gray):
    """Shape aur edge direction ki information. 128x128 par: 1764 values."""
    small = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)


def glcm_features(gray):
    """
    Texture ki information. Har property aur distance ke liye 4 angles ka
    mean aur range (max - min) lete hain, taake rotated image par bhi features
    milte julte rahen. 6 props x 3 distances x 2 = 36 values.
    """
    q = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    P = graycomatrix(q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        v = graycoprops(P, prop)                 # (n_distances, n_angles)
        feats.append(v.mean(axis=1))
        feats.append(v.max(axis=1) - v.min(axis=1))
    return np.concatenate(feats)


def extract_all(images, desc):
    hog_list, glcm_list = [], []
    for img in tqdm(images, desc=desc):
        hog_list.append(hog_features(img))
        glcm_list.append(glcm_features(img))
    return (np.asarray(hog_list, dtype=np.float32),
            np.asarray(glcm_list, dtype=np.float32))


if __name__ == "__main__":
    FEATURES_DIR.mkdir(exist_ok=True)

    images, y_class, y_reg = load_gray_dataset()
    hog_o, glcm_o = extract_all(images, "Features (original)")
    np.savez(FEATURES_DIR / "features_original.npz",
             hog=hog_o, glcm=glcm_o, y_class=y_class, y_reg=y_reg)
    print(f"Original: HOG {hog_o.shape}, GLCM {glcm_o.shape}")

    if AUG_PATH.exists():
        aug = np.load(AUG_PATH)
        hog_a, glcm_a = extract_all(aug["images"], "Features (augmented)")
        np.savez(FEATURES_DIR / "features_augmented.npz",
                 hog=hog_a, glcm=glcm_a, y_class=aug["y_class"],
                 y_reg=aug["y_reg"], source_idx=aug["source_idx"])
        print(f"Augmented: HOG {hog_a.shape}, GLCM {glcm_a.shape}")
    else:
        print("aug_train.npz nahi mila, sirf original features bane. (python augment.py)")

    print(f"Saved in: {FEATURES_DIR.name}/")
