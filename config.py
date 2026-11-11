from pathlib import Path

# Project root (ye file jahan hai wahi)
ROOT = Path(__file__).resolve().parent

# Dataset folder: ROOT/dataset/<class_name>/*.jpg
DATASET_DIR = ROOT / "dataset"

# Class folder names (exact spelling aur casing, Linux/GitHub par case matter karta hai)
CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "scratches",
]

IMAGES_PER_CLASS = 300
IMG_SIZE = (224, 224)          # CNN input size (proposal ke mutabiq)
GRAY_SIZE = (200, 200)         # NEU-DET ka asal size; classical pipeline isi par chalti hai
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}   # Thumbs.db / desktop.ini wagaira skip

# Split settings (proposal ke mutabiq 70 / 15 / 15)
RANDOM_STATE = 42
TEST_SIZE = 0.30       # 30% temp (val + test)
VAL_OF_TEMP = 0.50     # temp ka 50% validation, 50% test

# Preprocessed files (gitignore mein hain)
X_PATH = ROOT / "X_data.npy"
Y_CLASS_PATH = ROOT / "y_data.npy"
Y_REG_PATH = ROOT / "y_reg_data.npy"
SPLITS_PATH = ROOT / "splits.npz"      # main.py banata hai, baaki sab scripts parhti hain

# Augmentation (sirf training set)
AUG_PATH = ROOT / "aug_train.npz"
AUG_PER_IMAGE = 2              # har training image ki kitni augmented copies (1260 -> +2520)
AUG_ROTATION_DEG = 20          # random rotation: -20 se +20 degree
AUG_ZOOM_RANGE = (1.0, 1.2)    # random zoom-in

# Classical baseline (HOG + GLCM)
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
HOG_SIZE = (128, 128)          # HOG se pehle resize (chhota = tez)
GLCM_LEVELS = 64               # 256 gray levels ko 64 mein quantize (tez aur stable)
N_FOLDS = 5
