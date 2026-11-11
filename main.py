"""
Train / validation / test split (70 / 15 / 15, stratified) banata hai aur
indices splits.npz mein save karta hai. Baaki saari scripts (augment,
baseline, CNN, dashboard) yahi file parhti hain, taake har jagah bilkul
same images train / val / test mein hon.
Chalane ke liye: python main.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from config import (CLASS_NAMES, DATASET_DIR, RANDOM_STATE, SPLITS_PATH,
                    TEST_SIZE, VAL_OF_TEMP, X_PATH)
from utils import list_image_files

paths, y_class = list_image_files(DATASET_DIR)
indices = np.arange(len(paths))

# Step 1: 70% train, 30% temp (stratified)
train_idx, temp_idx = train_test_split(
    indices, test_size=TEST_SIZE, stratify=y_class, random_state=RANDOM_STATE,
)
# Step 2: temp ko aadha aadha validation aur test mein
val_idx, test_idx = train_test_split(
    temp_idx, test_size=VAL_OF_TEMP, stratify=y_class[temp_idx], random_state=RANDOM_STATE,
)

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

files = np.array([p.relative_to(DATASET_DIR).as_posix() for p in paths])
np.savez(SPLITS_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, files=files)

print("--- Dataset Splitting Complete ---")
print(f"Training   : {len(train_idx)} images")
print(f"Validation : {len(val_idx)} images")
print(f"Testing    : {len(test_idx)} images")
print("\nPer class (train / val / test):")
for c, name in enumerate(CLASS_NAMES):
    counts = [int(np.sum(y_class[ix] == c)) for ix in (train_idx, val_idx, test_idx)]
    print(f"  {name:16} {counts[0]:4} / {counts[1]:3} / {counts[2]:3}")
print(f"\nSaved: {SPLITS_PATH.name}")

# Agar preprocessing.py chal chuka hai to normalization check (mmap: poori file RAM mein load nahi hoti)
if X_PATH.exists():
    X = np.load(X_PATH, mmap_mode="r")
    assert X.shape[0] == len(paths), "X_data.npy purana hai. Dobara chalayein: python preprocessing.py"
    assert X.min() >= 0.0 and X.max() <= 1.0, "Images normalize nahi hui!"
    print(f"X_data.npy check OK: {X.shape}, values [0, 1] mein")
