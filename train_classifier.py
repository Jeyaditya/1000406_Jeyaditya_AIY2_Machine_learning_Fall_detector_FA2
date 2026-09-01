"""
train_classifier.py — SafeFall AI FA-2, Step 5 & 6
====================================================
Run with:  python train_classifier.py

Loads the labeled frames your (fixed) fa1_pipeline.py already produced,
re-extracts pose-geometry feature vectors (via pose_utils.py, so training
and the Streamlit app use IDENTICAL features), trains two candidate
models (Random Forest and a small MLP neural net), compares them, and
saves whichever wins + every chart/screenshot Step 6 of the FA-2 brief
asks for.

Why two models instead of one CNN: your FA-2 brief explicitly lists
Random Forest as an acceptable model choice alongside CNN. A CNN on raw
224x224 images would be slow and unreliable to train on a 940MX; a
compact feature vector (keypoint positions + the same geometry your
storyboard already explains) trains in seconds on CPU and is easy to
explain in your README. The MLP gives you a genuine loss curve for the
"Loss Graph" requirement; the Random Forest is a strong, fast baseline.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from pose_utils import image_to_feature, FEATURE_NAMES

# ============================================================
# CONFIG
# ============================================================

FA1_OUTPUT_ROOT = Path("./fa1_outputs")          # where fa1_pipeline.py wrote its outputs
OUTPUT_ROOT = Path("./fa2_outputs")
CLASSES = ["fall", "walking", "sitting", "standing", "normal"]
RANDOM_SEED = 42
DEVICE = "cpu"   # same reasoning as FA-1 — 940MX isn't worth fighting for this workload


def load_combined_manifest():
    train_csv = FA1_OUTPUT_ROOT / "train_manifest.csv"
    test_csv = FA1_OUTPUT_ROOT / "test_manifest.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Couldn't find {train_csv} / {test_csv}. Run fa1_pipeline.py first."
        )
    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    df = df[df["path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


def extract_features_for_dataset(model, manifest_df):
    """Re-runs YOLO pose on every image and builds the feature matrix.
    This is the one step that actually needs YOLO — everything else in
    this script is plain sklearn and runs fast."""
    features, labels, paths = [], [], []
    skipped = 0

    for i, row in manifest_df.iterrows():
        img = cv2.imread(row["path"])
        if img is None:
            skipped += 1
            continue
        feature, _ = image_to_feature(model, img, device=DEVICE)
        if feature is None:
            skipped += 1
            continue
        features.append(feature)
        labels.append(row["label"])
        paths.append(row["path"])

        if (i + 1) % 200 == 0:
            print(f"  Feature-extracted {i + 1}/{len(manifest_df)} images "
                  f"({skipped} skipped so far)")

    print(f"  Done. {len(features)} usable images, {skipped} skipped "
          f"(no confident person detected).")
    return np.array(features), np.array(labels), paths


def three_way_split(X, y, seed=RANDOM_SEED):
    """70/15/15 train/val/test, stratified, as required by the FA-2 brief
    (Step 5 — different ratio than FA-1's 70/30)."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed,
    )
    # 0.15 / 0.85 of the remainder = 15% of the original total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y_encoded, label_encoder, name):
    preds = model.predict(X)
    acc = accuracy_score(y_encoded, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_encoded, preds, average="macro", zero_division=0,
    )
    print(f"  [{name}] accuracy={acc:.3f}  precision={precision:.3f}  "
          f"recall={recall:.3f}  f1={f1:.3f}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "preds": preds}


def save_confusion_matrix(y_true, y_pred, label_encoder, out_path, title):
    labels = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_loss_curve(mlp_model, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp_model.loss_curve_)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_model_comparison(rf_metrics, mlp_metrics, out_path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    rf_vals = [rf_metrics[m] for m in metrics]
    mlp_vals = [mlp_metrics[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rf_vals, width, label="Random Forest")
    ax.bar(x + width / 2, mlp_vals, width, label="MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (validation set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_prediction_screenshots(model_wrapper, paths, y_true_encoded, y_pred_encoded,
                                 label_encoder, out_dir, n=8):
    """Grabs a handful of test images with their predicted vs actual label
    for the 'Prediction Screenshots' checklist item."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.random.RandomState(RANDOM_SEED).choice(
        len(paths), size=min(n, len(paths)), replace=False
    )
    for k, i in enumerate(idxs):
        img = cv2.imread(paths[i])
        if img is None:
            continue
        actual = label_encoder.inverse_transform([y_true_encoded[i]])[0]
        pred = label_encoder.inverse_transform([y_pred_encoded[i]])[0]
        correct = "CORRECT" if actual == pred else "WRONG"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Predicted: {pred} | Actual: {actual} | {correct}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"prediction_{k}_{correct.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {len(idxs)} prediction screenshots to {out_dir}")


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    (OUTPUT_ROOT / "screenshots").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "models").mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP 1 — Loading FA-1 manifest and YOLO-Pose")
    print("=" * 70)
    manifest_df = load_combined_manifest()
    print(f"  {len(manifest_df)} labeled images found across classes:")
    print(manifest_df["label"].value_counts())

    from ultralytics import YOLO
    yolo_model = YOLO("yolo11n-pose.pt")
    yolo_model.to(DEVICE)

    print("\n" + "=" * 70)
    print("STEP 2 — Extracting pose feature vectors (this is the slow step)")
    print("=" * 70)
    X, y, paths = extract_features_for_dataset(yolo_model, manifest_df)

    if len(X) < 50:
        print("[!] Very few usable samples — check that fa1_pipeline.py actually "
              "populated fa1_outputs/balanced correctly before training.")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)  # fix class order regardless of what's present
    y_encoded = label_encoder.transform(y)

    print("\n" + "=" * 70)
    print("STEP 3 — 70/15/15 train/val/test split")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y_encoded)
    # keep matching path lists for the screenshot step later
    _, paths_temp, _, _ = train_test_split(paths, y_encoded, test_size=0.15,
                                            stratify=y_encoded, random_state=RANDOM_SEED)
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("STEP 4 — Training Random Forest")
    print("=" * 70)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    rf_val_metrics = evaluate(rf, X_val_s, y_val, label_encoder, "RandomForest (val)")

    print("\n" + "=" * 70)
    print("STEP 5 — Training MLP (small neural net)")
    print("=" * 70)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=400, early_stopping=True, random_state=RANDOM_SEED,
    )
    mlp.fit(X_train_s, y_train)
    mlp_val_metrics = evaluate(mlp, X_val_s, y_val, label_encoder, "MLP (val)")

    print("\n" + "=" * 70)
    print("STEP 6 — Picking the winner + final test-set evaluation")
    print("=" * 70)
    if mlp_val_metrics["f1"] >= rf_val_metrics["f1"]:
        winner_name, winner_model = "mlp", mlp
    else:
        winner_name, winner_model = "random_forest", rf
    print(f"  Winner (by validation macro-F1): {winner_name}")

    test_metrics = evaluate(winner_model, X_test_s, y_test, label_encoder, f"{winner_name} (TEST)")
    print("\nFull classification report (test set):")
    print(classification_report(
        y_test, test_metrics["preds"],
        target_names=label_encoder.classes_, zero_division=0,
    ))

    print("\n" + "=" * 70)
    print("STEP 7 — Saving charts + artifacts")
    print("=" * 70)
    save_confusion_matrix(
        y_test, test_metrics["preds"], label_encoder,
        OUTPUT_ROOT / "screenshots" / "confusion_matrix.png",
        f"Confusion Matrix — {winner_name} (test set)",
    )
    save_model_comparison(
        rf_val_metrics, mlp_val_metrics,
        OUTPUT_ROOT / "screenshots" / "model_comparison.png",
    )
    save_loss_curve(mlp, OUTPUT_ROOT / "screenshots" / "mlp_loss_curve.png")
    save_prediction_screenshots(
        winner_model, paths_temp, y_test, test_metrics["preds"], label_encoder,
        OUTPUT_ROOT / "screenshots" / "predictions",
    )

    joblib.dump(winner_model, OUTPUT_ROOT / "models" / "classifier.joblib")
    joblib.dump(scaler, OUTPUT_ROOT / "models" / "scaler.joblib")
    joblib.dump(label_encoder, OUTPUT_ROOT / "models" / "label_encoder.joblib")

    with open(OUTPUT_ROOT / "models" / "model_info.json", "w") as f:
        json.dump({
            "winner": winner_name,
            "val_metrics": {
                "random_forest": {k: v for k, v in rf_val_metrics.items() if k != "preds"},
                "mlp": {k: v for k, v in mlp_val_metrics.items() if k != "preds"},
            },
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "preds"},
            "feature_names": FEATURE_NAMES,
            "classes": list(label_encoder.classes_),
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE. Winning model: {winner_name}")
    print(f"Artifacts saved in: {(OUTPUT_ROOT / 'models').resolve()}")
    print(f"Screenshots saved in: {(OUTPUT_ROOT / 'screenshots').resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
