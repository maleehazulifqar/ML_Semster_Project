"""
Classical baseline: Random Forest (defect class) + Ridge Regression (severity)
HOG / GLCM features par.

Chalane ka order: main.py -> augment.py -> features.py -> baseline_classical.py

Usage:
  python baseline_classical.py                  # HOG, GLCM, HOG+GLCM teeno: 5-fold CV + validation
  python baseline_classical.py --features glcm  # sirf ek feature set
  python baseline_classical.py --no-aug         # augmentation ke baghair (comparison ke liye)
  python baseline_classical.py --no-cv          # sirf validation, jaldi
  python baseline_classical.py --test           # FINAL: test set par bhi (sirf aakhir mein, ek baar)

Validation set model choose karne ke liye hai. Test set sab models final hone ke
baad sirf ek baar use karein, warna test score bharosemand nahi rehta.
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (CLASS_NAMES, FEATURES_DIR, MODELS_DIR, N_FOLDS, RANDOM_STATE,
                    RESULTS_DIR)
from utils import load_splits

FEATURE_SETS = {
    "hog": ["hog"],
    "glcm": ["glcm"],
    "hog+glcm": ["hog", "glcm"],
}


def make_classifier():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)


def make_regressor():
    # RidgeCV training data par khud best alpha choose karta hai
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


def predict_severity(reg, X):
    return np.clip(reg.predict(X), 0.0, 1.0)   # target 0-1 ke darmiyan hai


def stack(data, parts):
    return np.hstack([data[p] for p in parts])


def class_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def reg_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}


def cross_validate(X, yc, yr, groups, is_orig):
    """
    StratifiedGroupKFold: ek image aur us ki augmented copies hamesha ek hi fold mein.
    Score sirf asal (non-augmented) images par.
    """
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, yc, groups), start=1):
        te = te[is_orig[te]]
        clf = make_classifier().fit(X[tr], yc[tr])
        reg = make_regressor().fit(X[tr], yr[tr])
        row = {**class_metrics(yc[te], clf.predict(X[te])),
               **reg_metrics(yr[te], predict_severity(reg, X[te]))}
        rows.append(row)
        print(f"  Fold {fold}: acc {row['accuracy']:.4f}  f1 {row['f1']:.4f}  "
              f"MAE {row['mae']:.4f}  R2 {row['r2']:.4f}")
    df = pd.DataFrame(rows)
    print(f"  CV mean: acc {df.accuracy.mean():.4f} +/- {df.accuracy.std():.4f}  "
          f"f1 {df.f1.mean():.4f}  MAE {df.mae.mean():.4f}  R2 {df.r2.mean():.4f} +/- {df.r2.std():.4f}")
    return df


def evaluate(clf, reg, X, yc, yr, tag, split):
    pred_c = clf.predict(X)
    pred_r = predict_severity(reg, X)
    m = {**class_metrics(yc, pred_c), **reg_metrics(yr, pred_r)}

    print(f"\n  [{split.upper()}] acc {m['accuracy']:.4f}  precision {m['precision']:.4f}  "
          f"recall {m['recall']:.4f}  f1 {m['f1']:.4f}")
    print(f"  [{split.upper()}] MAE {m['mae']:.4f}  MSE {m['mse']:.5f}  R2 {m['r2']:.4f}")
    print(classification_report(yc, pred_c, target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yc, pred_c, display_labels=CLASS_NAMES,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Random Forest ({tag}) - {split}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"cm_{tag}_{split}.png", dpi=120)
    plt.close(fig)

    # Severity: predicted vs true
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yr, pred_r, s=10, alpha=0.6)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    lim = max(yr.max(), pred_r.max()) * 1.05
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True severity"); ax.set_ylabel("Predicted severity")
    ax.set_title(f"Ridge ({tag}) - {split}\nMAE {m['mae']:.3f}, R2 {m['r2']:.3f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"severity_{tag}_{split}.png", dpi=120)
    plt.close(fig)

    report = classification_report(yc, pred_c, target_names=CLASS_NAMES, output_dict=True)
    return m, report


def run(feature_set, use_aug, do_cv, do_test):
    parts = FEATURE_SETS[feature_set]
    tag = feature_set.replace("+", "_") + ("_aug" if use_aug else "_noaug")
    print(f"\n{'=' * 60}\nFeatures: {feature_set} | Augmentation: {use_aug}\n{'=' * 60}")

    orig = np.load(FEATURES_DIR / "features_original.npz")
    X_all, yc_all, yr_all = stack(orig, parts), orig["y_class"], orig["y_reg"]
    train_idx, val_idx, test_idx = load_splits()

    X_tr, yc_tr, yr_tr = X_all[train_idx], yc_all[train_idx], yr_all[train_idx]
    groups = train_idx.copy()
    is_orig = np.ones(len(train_idx), dtype=bool)

    if use_aug:
        aug_file = FEATURES_DIR / "features_augmented.npz"
        if not aug_file.exists():
            raise FileNotFoundError("features_augmented.npz nahi mila. Chalayein: python augment.py, phir python features.py")
        aug = np.load(aug_file)
        assert set(aug["source_idx"]) <= set(train_idx), "Augmented data purane split ka hai, augment.py dobara chalayein"
        X_tr = np.vstack([X_tr, stack(aug, parts)])
        yc_tr = np.concatenate([yc_tr, aug["y_class"]])
        yr_tr = np.concatenate([yr_tr, aug["y_reg"]])
        groups = np.concatenate([groups, aug["source_idx"]])
        is_orig = np.concatenate([is_orig, np.zeros(len(aug["y_class"]), dtype=bool)])

    print(f"Training samples: {len(yc_tr)} ({is_orig.sum()} original), features: {X_tr.shape[1]}")

    result = {"features": feature_set, "augmentation": use_aug}
    if do_cv:
        print(f"\n{N_FOLDS}-Fold Cross-Validation (training set):")
        t = time.time()
        cv_df = cross_validate(X_tr, yc_tr, yr_tr, groups, is_orig)
        result["cv"] = {"per_fold": cv_df.to_dict(orient="records"),
                        "mean": cv_df.mean().to_dict(), "std": cv_df.std().to_dict()}
        print(f"  ({time.time() - t:.0f} sec)")

    print("\nFinal models poore training set par train ho rahe hain...")
    clf = make_classifier().fit(X_tr, yc_tr)
    reg = make_regressor().fit(X_tr, yr_tr)
    print(f"  Ridge best alpha: {reg[-1].alpha_:.3g}")

    rows = []
    splits = [("val", val_idx)] + ([("test", test_idx)] if do_test else [])
    for split, idx in splits:
        m, report = evaluate(clf, reg, X_all[idx], yc_all[idx], yr_all[idx], tag, split)
        result[split] = {"metrics": m, "per_class": report}
        rows.append({"features": feature_set, "augmentation": use_aug, "split": split, **m})
    if do_cv:
        rows.append({"features": feature_set, "augmentation": use_aug, "split": "cv_mean",
                     **result["cv"]["mean"]})

    joblib.dump({"classifier": clf, "regressor": reg, "features": parts,
                 "class_names": CLASS_NAMES}, MODELS_DIR / f"classical_{tag}.joblib", compress=3)
    with open(RESULTS_DIR / f"baseline_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    return rows


def update_summary(rows):
    path = RESULTS_DIR / "baseline_summary.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = ["features", "augmentation", "split"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["split", "features", "augmentation"])
    new.to_csv(path, index=False, float_format="%.4f")
    print(f"\nSummary: {path.relative_to(RESULTS_DIR.parent)}")
    print(new.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HOG/GLCM + Random Forest / Ridge baseline")
    ap.add_argument("--features", default="all", choices=[*FEATURE_SETS, "all"])
    ap.add_argument("--no-aug", action="store_true", help="augmented data use na karein")
    ap.add_argument("--no-cv", action="store_true", help="K-Fold CV skip (jaldi run)")
    ap.add_argument("--test", action="store_true", help="test set par bhi evaluate (sirf final run)")
    args = ap.parse_args()

    if args.test:
        print("NOTE: Test set evaluation. Ye sirf final models par ek baar karein.")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    feature_sets = list(FEATURE_SETS) if args.features == "all" else [args.features]

    all_rows = []
    for fs in feature_sets:
        all_rows += run(fs, use_aug=not args.no_aug, do_cv=not args.no_cv, do_test=args.test)
    update_summary(all_rows)
