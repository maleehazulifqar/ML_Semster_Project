"""
Dataset check: har class folder mojood hai, 300 images hain, koi file corrupt nahi.
Chalane ke liye: python verify_data.py
"""
from PIL import Image

from config import CLASS_NAMES, DATASET_DIR, IMAGE_EXTS, IMAGES_PER_CLASS


def verify_dataset():
    if not DATASET_DIR.is_dir():
        print(f"Error: '{DATASET_DIR}' folder mojood nahi hai.")
        return

    total_expected = IMAGES_PER_CLASS * len(CLASS_NAMES)
    total_found = 0
    corrupted = []

    print("--- Starting Dataset Verification ---\n")
    for class_name in CLASS_NAMES:
        class_path = DATASET_DIR / class_name
        if not class_path.is_dir():
            print(f"Missing Folder: '{class_name}' (spelling aur capital/small letters check karein)")
            continue

        valid = 0
        for file in class_path.iterdir():
            if file.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(file) as img:
                    img.verify()
                valid += 1
            except Exception as e:
                corrupted.append((file, str(e)))

        total_found += valid
        status = "PASSED" if valid == IMAGES_PER_CLASS else "CHECK COUNT"
        print(f"Class: {class_name:16} | Found: {valid} images | Status: {status}")

    print("\n--- Verification Summary ---")
    print(f"Total Valid Images Found: {total_found} / {total_expected}")
    if corrupted:
        print(f"\nWarning: {len(corrupted)} corrupted files:")
        for path, error in corrupted:
            print(f" - {path}: {error}")
    else:
        print("Success: Zero corrupted files detected!")


if __name__ == "__main__":
    verify_dataset()
