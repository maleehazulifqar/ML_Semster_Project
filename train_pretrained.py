import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# 6 NEU defect classes in standard order
CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "scratches",
]


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_rgb_images(images: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Convert grayscale/2D or 3D images into ResNet50-compatible float32 RGB arrays
    and apply ResNet50 preprocessing.
    """
    processed = []
    for img in images:
        arr = np.asarray(img)

        # Normalize to 0..255 uint8 first if float in [0, 1]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Ensure 3 channels (H, W, 3)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Resize to 224x224 if needed
        if (arr.shape[0], arr.shape[1]) != target_size:
            arr = tf.image.resize(arr, target_size, method="bilinear").numpy().astype(np.uint8)

        processed.append(arr)

    rgb_batch = np.stack(processed, axis=0).astype(np.float32)
    return resnet_preprocess(rgb_batch)


def load_dataset_splits(target_size=(224, 224)):
    """
    Load train, val, and test sets cleanly without data leakage.
    Uses aug_train.npz for training if available, else slices X_data.npy via splits.npz.
    """
    X_all = np.load("X_data.npy")
    y_cls_all = np.load("y_data.npy").astype(np.int64)
    y_reg_all = np.load("y_reg_data.npy").astype(np.float32)
    splits = np.load("splits.npz", allow_pickle=True)

    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    if os.path.exists("aug_train.npz"):
        aug = np.load("aug_train.npz", allow_pickle=True)
        if "X_train_full" in aug and "y_train_full" in aug and "y_reg_train_full" in aug:
            X_tr_raw = aug["X_train_full"]
            y_tr_cls = aug["y_train_full"].astype(np.int64)
            y_tr_reg = aug["y_reg_train_full"].astype(np.float32)
            print("Loaded augmented training data from aug_train.npz")
        elif "X_train" in aug and "y_train" in aug and "y_reg_train" in aug:
            X_tr_raw = aug["X_train"]
            y_tr_cls = aug["y_train"].astype(np.int64)
            y_tr_reg = aug["y_reg_train"].astype(np.float32)
            print("Loaded training data from aug_train.npz")
        else:
            train_idx = splits["train_idx"]
            X_tr_raw = X_all[train_idx]
            y_tr_cls = y_cls_all[train_idx]
            y_tr_reg = y_reg_all[train_idx]
            print("Fallback to splits.npz train_idx")
    else:
        train_idx = splits["train_idx"]
        X_tr_raw = X_all[train_idx]
        y_tr_cls = y_cls_all[train_idx]
        y_tr_reg = y_reg_all[train_idx]
        print("aug_train.npz not found; using splits.npz train_idx")

    X_va_raw = X_all[val_idx]
    y_va_cls = y_cls_all[val_idx]
    y_va_reg = y_reg_all[val_idx]

    X_te_raw = X_all[test_idx]
    y_te_cls = y_cls_all[test_idx]
    y_te_reg = y_reg_all[test_idx]

    X_train = prepare_rgb_images(X_tr_raw, target_size=target_size)
    X_val = prepare_rgb_images(X_va_raw, target_size=target_size)
    X_test = prepare_rgb_images(X_te_raw, target_size=target_size)

    return (
        (X_train, y_tr_cls, y_tr_reg),
        (X_val, y_va_cls, y_va_reg),
        (X_test, y_te_cls, y_te_reg),
    )


def build_dual_head_model(num_classes: int = 6, input_shape=(224, 224, 3)):
    """
    Pretrained ResNet50 backbone with two task heads:
      1) 'classification' -> Softmax over 6 defect classes
      2) 'regression'     -> Linear/Sigmoid severity score in [0.0, 1.0]
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = False  # Phase 1: feature extraction with frozen backbone

    x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = layers.Dropout(0.3, name="shared_dropout")(x)

    # Head 1: Classification
    cls_feat = layers.Dense(128, activation="relu", name="cls_dense")(x)
    cls_feat = layers.Dropout(0.2, name="cls_dropout")(cls_feat)
    cls_output = layers.Dense(num_classes, activation="softmax", name="classification")(cls_feat)

    # Head 2: Regression (severity score)
    reg_feat = layers.Dense(64, activation="relu", name="reg_dense")(x)
    reg_output = layers.Dense(1, activation="sigmoid", name="regression")(reg_feat)

    model = models.Model(
        inputs=inputs,
        outputs={"classification": cls_output, "regression": reg_output},
        name="ResNet50_DualHead_DefectNet",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "classification": "sparse_categorical_crossentropy",
            "regression": "mse",
        },
        loss_weights={
            "classification": 1.0,
            "regression": 5.0,
        },
        metrics={
            "classification": ["accuracy"],
            "regression": ["mae"],
        },
    )
    return model, backbone


def evaluate_split(model, X, y_cls, y_reg, split_name: str):
    preds = model.predict(X, batch_size=32, verbose=0)
    cls_probs = preds["classification"]
    reg_preds = preds["regression"].squeeze(-1)

    y_pred_cls = np.argmax(cls_probs, axis=1)

    acc = accuracy_score(y_cls, y_pred_cls)
    prec = precision_score(y_cls, y_pred_cls, average="weighted", zero_division=0)
    rec = recall_score(y_cls, y_pred_cls, average="weighted", zero_division=0)
    f1 = f1_score(y_cls, y_pred_cls, average="weighted", zero_division=0)

    mae = mean_absolute_error(y_reg, reg_preds)
    mse = mean_squared_error(y_reg, reg_preds)
    r2 = r2_score(y_reg, reg_preds)

    print(f"\n[{split_name.upper()}] acc {acc:.4f}  precision {prec:.4f}  recall {rec:.4f}  f1 {f1:.4f}")
    print(f"[{split_name.upper()}] MAE {mae:.4f}  MSE {mse:.5f}  R2 {r2:.4f}")
    print(
        classification_report(
            y_cls,
            y_pred_cls,
            target_names=CLASS_NAMES[: len(np.unique(y_cls))],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "split": split_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mae": float(mae),
        "mse": float(mse),
        "r2": float(r2),
    }


def main():
    set_seeds(42)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    (X_train, y_train_cls, y_train_reg), (X_val, y_val_cls, y_val_reg), (X_test, y_test_cls, y_test_reg) = (
        load_dataset_splits(target_size=(224, 224))
    )

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    model, backbone = build_dual_head_model(num_classes=len(CLASS_NAMES), input_shape=(224, 224, 3))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print("\n--- Phase 1: Training Dual Heads (Frozen ResNet50 Backbone) ---")
    model.fit(
        X_train,
        {"classification": y_train_cls, "regression": y_train_reg},
        validation_data=(X_val, {"classification": y_val_cls, "regression": y_val_reg}),
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune top layers of ResNet50 with a low learning rate
    print("\n--- Phase 2: Fine-Tuning Top Layers of ResNet50 ---")
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "classification": "sparse_categorical_crossentropy",
            "regression": "mse",
        },
        loss_weights={
            "classification": 1.0,
            "regression": 5.0,
        },
        metrics={
            "classification": ["accuracy"],
            "regression": ["mae"],
        },
    )

    model.fit(
        X_train,
        {"classification": y_train_cls, "regression": y_train_reg},
        validation_data=(X_val, {"classification": y_val_cls, "regression": y_val_reg}),
        epochs=8,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    val_metrics = evaluate_split(model, X_val, y_val_cls, y_val_reg, "val")
    test_metrics = evaluate_split(model, X_test, y_test_cls, y_test_reg, "test")

    model_path = os.path.join("models", "dual_head_resnet50.keras")
    model.save(model_path)

    summary = {
        "model": "ResNet50_DualHead",
        "model_path": model_path,
        "val": val_metrics,
        "test": test_metrics,
    }
    out_json = os.path.join("results", "pretrained_dual_head_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {out_json}")


if __name__ == "__main__":
    main()