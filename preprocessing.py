"""
Dataset ko ek baar process karke .npy files mein save karta hai.
Chalane ke liye: python preprocessing.py
"""
import numpy as np
from config import DATASET_DIR, X_PATH, Y_CLASS_PATH, Y_REG_PATH
from utils import load_dataset_targets

if __name__ == "__main__":
    print("Dataset process ho raha hai...")
    X, y_class, y_reg = load_dataset_targets(DATASET_DIR)

    np.save(X_PATH, X)
    np.save(Y_CLASS_PATH, y_class)
    np.save(Y_REG_PATH, y_reg)

    print(f"\nX shape: {X.shape}, min={X.min():.2f}, max={X.max():.2f}")
    print(f"y_class shape: {y_class.shape}")
    print(f"y_reg shape: {y_reg.shape}")
    print("Saved: X_data.npy, y_data.npy, y_reg_data.npy")