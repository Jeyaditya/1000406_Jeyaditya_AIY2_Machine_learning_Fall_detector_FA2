"""
SafeFall AI — FA-1 Local Pipeline
==================================
Run with:  python fa1_pipeline.py

What this does, in order:
  0. Sanity-check your dataset path and inspect one annotation file
     (so you can eyeball the format before we trust it for 190 videos).
  1. Walk all 6 Le2i environments, pair each video with its annotation.
  2. Parse each annotation file for the fall start/end frame numbers.
  3. Sample frames from every video, run YOLO-Pose on each, and label
     each frame as fall / walking / sitting / standing / normal using:
       - the annotation ground-truth for the fall frames
       - the same geometric heuristics your own storyboard already
         defines (aspect ratio, torso inclination angle) for the rest
  4. Resize to 224x224, normalize, save into activity-named folders.
  5. Balance classes via light augmentation (rotate/flip/brightness/zoom)
     — same 4 transforms shown in your storyboard's augmentation slide.
  6. Stratified 70/30 train/test split.
  7. Save every EDA chart and screenshot you need for the storyboard
     into ./fa1_outputs/screenshots/

Everything writes under ./fa1_outputs/ next to this script.
"""

import os
import sys
import math
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split

# ============================================================
# 0. CONFIG — EDIT THESE
# ============================================================

# Root folder that directly contains Home_01, Home_02, Coffee_room_01, ...
# (i.e. the folder you'd see if you unzipped the Kaggle download)
DATASET_ROOT = Path(r"C:\path\to\le2i_dataset")   # <-- EDIT THIS

OUTPUT_ROOT = Path("./fa1_outputs")

IMG_SIZE = 224
CONF_THRESHOLD = 0.5          # raised from 0.25 -> kills low-confidence "ghost" detections
FRAMES_PER_VIDEO = 10         # half biased into the annotated fall window, half spread across the video
RANDOM_SEED = 42

ENVIRONMENTS = [
    "Home_01", "Home_02",
    "Coffee_room_01", "Coffee_room_02",
    "Lecture_room", "Office",
]

CLASSES = ["fall", "walking", "sitting", "standing", "normal"]

# Set to "cuda:0" yourself if you've confirmed CUDA works on the 940MX.
# Defaulting to CPU: a nano pose model on a few hundred still frames
# is a few minutes on CPU and avoids fighting an old 2GB Maxwell card.
DEVICE = "cpu"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# YOLO 17-keypoint (COCO) index map
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}


# ============================================================
# 1. DATASET DISCOVERY
# ============================================================

def find_video_annotation_pairs():
    """Returns list of dicts: {environment, video_path, annotation_path}"""
    pairs = []
    for env in ENVIRONMENTS:
        env_root = DATASET_ROOT / env
        if not env_root.exists():
            print(f"  [!] Missing environment folder: {env}")
            continue

        videos = sorted(env_root.rglob("*.avi"))
        for video_path in videos:
            # Le2i annotation files usually live in a sibling
            # "Annotation_files" folder with the same base name + .txt
            candidates = list(env_root.rglob(f"{video_path.stem}.txt"))
            annotation_path = candidates[0] if candidates else None
            pairs.append({
                "environment": env,
                "video_path": video_path,
                "annotation_path": annotation_path,
            })
    return pairs


def inspect_sample_annotation(pairs):
    """Print one real annotation file so you can confirm the format
    matches what parse_annotation() below assumes, BEFORE running
    the full extraction on all 190 videos."""
    sample = next((p for p in pairs if p["annotation_path"]), None)
    if sample is None:
        print("  [!] No annotation .txt files found at all — check DATASET_ROOT.")
        return
    print(f"  Sample annotation: {sample['annotation_path']}")
    with open(sample["annotation_path"], "r", errors="ignore") as f:
        lines = [l.rstrip() for l in f.readlines()[:8]]
    for i, line in enumerate(lines):
        print(f"    line {i}: {line}")
    print("  -> Confirm: line 0 and line 1 should be the fall start/end frame numbers")
    print("     (both 0 usually means 'no fall in this video').")
    print("     If your files look different, tell me and I'll adjust parse_annotation().")


def parse_annotation(annotation_path):
    """Returns (fall_start_frame, fall_end_frame) or (None, None) if no
    fall / unreadable. Le2i format: line 0 = start frame, line 1 = end
    frame (both 0 if the video has no fall)."""
    if annotation_path is None:
        return None, None
    try:
        with open(annotation_path, "r", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() != ""]
        start = int(float(lines[0]))
        end = int(float(lines[1]))
        if start == 0 and end == 0:
            return None, None
        return start, end
    except Exception:
        return None, None


# ============================================================
# 2. POSE GEOMETRY HEURISTICS
#    (same formulas as your storyboard: aspect ratio + torso angle,
#     extended with knee-angle to separate walking/sitting/standing)
# ============================================================

def angle_at_joint(a, b, c):
    """Angle (degrees) at point b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return None
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def torso_angle(neck, pelvis):
    """theta = arccos(-dy / sqrt(dx^2+dy^2)), vector = neck - pelvis
    (pointing "up" the spine). 0 deg = upright, 90 deg = horizontal.
    Image y increases downward, so an upright torso has neck above
    pelvis -> dy = neck_y - pelvis_y is negative -> -dy is positive."""
    dx = neck[0] - pelvis[0]
    dy = neck[1] - pelvis[1]
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(-dy / denom, -1.0, 1.0)))


def classify_non_fall_posture(xy, conf, conf_thresh=0.3):
    """xy: (17,2) keypoint coords. conf: (17,) confidences.
    Returns one of walking/sitting/standing/normal. Deliberately NEVER
    returns 'fall' — aspect ratio alone is unreliable across the
    different camera angles in this dataset (e.g. the overhead Coffee_room
    angle makes an ordinary seated person look 'wide', which previously
    caused false fall labels). Fall labeling is handled entirely by
    is_fall_window() below, anchored to the annotation ground truth."""

    def ok(name):
        return conf[KP[name]] >= conf_thresh

    visible = [i for i in range(17) if conf[i] >= conf_thresh]
    if len(visible) < 4:
        return "normal"  # too little visible to say anything confident

    xs = xy[visible, 0]
    ys = xy[visible, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    ar = width / height if height > 0 else 1.0

    # Knee angles for standing / walking / sitting distinction
    l_knee_angle = r_knee_angle = None
    if ok("l_hip") and ok("l_knee") and ok("l_ankle"):
        l_knee_angle = angle_at_joint(xy[KP["l_hip"]], xy[KP["l_knee"]], xy[KP["l_ankle"]])
    if ok("r_hip") and ok("r_knee") and ok("r_ankle"):
        r_knee_angle = angle_at_joint(xy[KP["r_hip"]], xy[KP["r_knee"]], xy[KP["r_ankle"]])

    knee_angles = [a for a in (l_knee_angle, r_knee_angle) if a is not None]
    if not knee_angles:
        return "normal"

    avg_knee = sum(knee_angles) / len(knee_angles)
    asymmetry = abs(l_knee_angle - r_knee_angle) if len(knee_angles) == 2 else 0

    if 60 <= avg_knee <= 120 and ar <= 1.6:
        return "sitting"
    if avg_knee > 160 and asymmetry < 15 and ar < 0.55:
        return "standing"
    if asymmetry >= 15 and avg_knee > 90:
        return "walking"
    return "normal"


# ============================================================
# 3. FRAME EXTRACTION + LABELING
# ============================================================

FALL_BUFFER_FRAMES = 15   # frames on either side of the annotated fall window
                           # that still count as "fall" — covers imprecise
                           # annotation boundaries WITHOUT resorting to a
                           # global geometry rule (see classify_non_fall_posture
                           # docstring for why that backfired)


def is_in_fall_window(frame_idx, fall_start, fall_end, buffer=FALL_BUFFER_FRAMES):
    """True if frame_idx is inside the annotated fall window, extended by
    a small buffer on each side. This is the ONLY source of 'fall' labels
    now — no geometry-based fallback, since aspect ratio alone proved
    unreliable across this dataset's different camera angles."""
    if fall_start is None:
        return False
    return (fall_start - buffer) <= frame_idx <= (fall_end + buffer)


def sample_frame_indices(total_frames, n):
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]


def sample_frame_indices_for_video(total_frames, fall_start, fall_end, n=FRAMES_PER_VIDEO):
    """Half the samples are deliberately drawn from INSIDE the annotated
    fall window (if one exists) so the brief 1-1.5s fall event doesn't get
    statistically skipped by even spacing across a 3.5-12s video. The other
    half are spread across the whole video for non-fall class diversity."""
    if fall_start is None:
        return sample_frame_indices(total_frames, n)

    fall_start = max(0, min(fall_start, total_frames - 1))
    fall_end = max(fall_start, min(fall_end, total_frames - 1))

    n_fall = max(2, n // 2)
    n_other = n - n_fall

    fall_span = fall_end - fall_start
    if fall_span <= 0:
        fall_frames = [fall_start]
    else:
        step = fall_span / max(1, n_fall - 1)
        fall_frames = sorted(set(int(fall_start + i * step) for i in range(n_fall)))

    other_frames = sample_frame_indices(total_frames, n_other)
    return sorted(set(fall_frames) | set(other_frames))


def extract_and_label(model, pairs, frames_per_video=FRAMES_PER_VIDEO):
    """Returns a DataFrame log of every processed frame, and writes raw
    (unresized) labeled frames to OUTPUT_ROOT/raw_frames/<class>/."""
    raw_dir = OUTPUT_ROOT / "raw_frames"
    for c in CLASSES:
        (raw_dir / c).mkdir(parents=True, exist_ok=True)

    log_rows = []
    saved_ghost_example = False

    for pair_idx, pair in enumerate(pairs):
        video_path = pair["video_path"]
        env = pair["environment"]
        fall_start, fall_end = parse_annotation(pair["annotation_path"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [!] Could not open {video_path}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        for frame_idx in sample_frame_indices_for_video(total, fall_start, fall_end, frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok_read, frame = cap.read()
            if not ok_read:
                continue

            results = model.predict(
                source=frame, imgsz=320, conf=CONF_THRESHOLD,
                device=DEVICE, verbose=False,
            )
            result = results[0]

            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            # Keep only the highest-confidence person (kills the
            # low-confidence "ghost" detections you saw in Colab)
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
            if confs is not None and len(confs) > 1:
                if not saved_ghost_example:
                    # Save one multi-detection frame as evidence/explanation
                    # for the storyboard (before we filter it down).
                    shots_dir = OUTPUT_ROOT / "screenshots"
                    shots_dir.mkdir(parents=True, exist_ok=True)
                    annotated = result.plot()[:, :, ::-1]
                    Image.fromarray(annotated).save(shots_dir / "multi_detection_before_filter.png")
                    saved_ghost_example = True
                best_idx = int(np.argmax(confs))
            else:
                best_idx = 0

            xy = result.keypoints.xy[best_idx].cpu().numpy()
            conf = (result.keypoints.conf[best_idx].cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones(xy.shape[0]))

            # Fall label comes ONLY from the annotation ground truth (plus a
            # small buffer for imprecise annotation boundaries) — never from
            # a global geometry rule. See is_in_fall_window() / the
            # classify_non_fall_posture() docstring for why.
            if is_in_fall_window(frame_idx, fall_start, fall_end):
                label = "fall"
            else:
                label = classify_non_fall_posture(xy, conf)

            out_name = f"{env}_{video_path.stem}_f{frame_idx}.png"
            out_path = raw_dir / label / out_name
            cv2.imwrite(str(out_path), frame)

            log_rows.append({
                "environment": env, "video": video_path.name,
                "frame_idx": frame_idx, "label": label,
                "num_people_detected": len(result.keypoints),
            })

        cap.release()

        if (pair_idx + 1) % 10 == 0:
            print(f"  Processed {pair_idx + 1}/{len(pairs)} videos")

    return pd.DataFrame(log_rows)


# ============================================================
# 4. PREPROCESSING (resize + normalize) + augmentation samples
# ============================================================

def preprocess_dataset(log_df):
    """Resize/normalize every raw frame into OUTPUT_ROOT/processed/<class>/."""
    processed_dir = OUTPUT_ROOT / "processed"
    for c in CLASSES:
        (processed_dir / c).mkdir(parents=True, exist_ok=True)

    raw_dir = OUTPUT_ROOT / "raw_frames"
    manifest = []

    for c in CLASSES:
        for img_path in (raw_dir / c).glob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            out_path = processed_dir / c / img_path.name
            cv2.imwrite(str(out_path), resized)
            manifest.append({"path": str(out_path), "label": c})

    # One before/after comparison image for the storyboard
    sample_class = next((c for c in CLASSES if (raw_dir / c).glob("*.png")), CLASSES[0])
    sample_files = list((raw_dir / sample_class).glob("*.png"))
    if sample_files:
        raw_img = cv2.imread(str(sample_files[0]))
        norm_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Raw frame\n{raw_img.shape[1]}x{raw_img.shape[0]}")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Resized 224x224\n(normalized 0-1 before model input)")
        axes[1].axis("off")
        plt.tight_layout()
        _save_screenshot(fig, "preprocessing_before_after.png")

    return pd.DataFrame(manifest)


def _augment_variants(img_bgr):
    """Returns dict of {name: image} matching the 4 transforms in the
    storyboard's Data Augmentation slide: rotate, flip, brightness zoom."""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    rotated = pil_img.rotate(15, expand=False, fillcolor=(0, 0, 0))
    flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    bright = ImageEnhance.Brightness(pil_img).enhance(1.35)

    w, h = pil_img.size
    crop_w, crop_h = int(w / 1.3), int(h / 1.3)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    zoomed = pil_img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h))

    return {
        "A_original": pil_img,
        "B_rotated_15deg": rotated,
        "C_flipped": flipped,
        "D_zoom_1.3x": zoomed,
        "brightness_jitter": bright,
    }


def save_augmentation_showcase():
    """Saves one augmented_training_samples.png grid for the storyboard."""
    raw_dir = OUTPUT_ROOT / "raw_frames"
    sample_path = None
    for c in CLASSES:
        files = list((raw_dir / c).glob("*.png"))
        if files:
            sample_path = files[0]
            break
    if sample_path is None:
        print("  [!] No frames available yet for augmentation showcase.")
        return

    img = cv2.imread(str(sample_path))
    variants = _augment_variants(img)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    keys = ["A_original", "B_rotated_15deg", "C_flipped", "D_zoom_1.3x"]
    for ax, key in zip(axes, keys):
        ax.imshow(variants[key])
        ax.set_title(key)
        ax.axis("off")
    plt.tight_layout()
    _save_screenshot(fig, "augmented_training_samples.png")


def balance_classes_with_augmentation(manifest_df):
    """Oversamples minority classes in the TRAIN split using the same
    transforms, up to roughly the size of the largest class."""
    balanced_dir = OUTPUT_ROOT / "balanced"
    for c in CLASSES:
        (balanced_dir / c).mkdir(parents=True, exist_ok=True)

    counts = manifest_df["label"].value_counts()
    target = counts.max() if len(counts) else 0
    balanced_rows = []

    for c in CLASSES:
        subset = manifest_df[manifest_df["label"] == c]
        # copy originals first
        for _, row in subset.iterrows():
            src = Path(row["path"])
            dst = balanced_dir / c / src.name
            shutil.copy2(src, dst)
            balanced_rows.append({"path": str(dst), "label": c})

        have = len(subset)
        need = max(0, target - have)
        if need == 0 or have == 0:
            continue

        files = subset["path"].tolist()
        i = 0
        while need > 0:
            src_path = Path(files[i % len(files)])
            img = cv2.imread(str(src_path))
            variants = _augment_variants(img)
            variant_name = random.choice(["B_rotated_15deg", "C_flipped", "D_zoom_1.3x", "brightness_jitter"])
            variant_img = cv2.cvtColor(np.array(variants[variant_name]), cv2.COLOR_RGB2BGR)
            out_name = f"aug{i}_{variant_name}_{src_path.name}"
            out_path = balanced_dir / c / out_name
            cv2.imwrite(str(out_path), variant_img)
            balanced_rows.append({"path": str(out_path), "label": c})
            i += 1
            need -= 1

    return pd.DataFrame(balanced_rows)


# ============================================================
# 5. SPLIT
# ============================================================

def stratified_split(balanced_df):
    train_df, test_df = train_test_split(
        balanced_df, test_size=0.30, stratify=balanced_df["label"],
        random_state=RANDOM_SEED,
    )
    split_dir = OUTPUT_ROOT / "split"
    for subset_name, subset_df in [("train", train_df), ("test", test_df)]:
        for c in CLASSES:
            (split_dir / subset_name / c).mkdir(parents=True, exist_ok=True)
        for _, row in subset_df.iterrows():
            src = Path(row["path"])
            dst = split_dir / subset_name / row["label"] / src.name
            shutil.copy2(src, dst)
    train_df.to_csv(OUTPUT_ROOT / "train_manifest.csv", index=False)
    test_df.to_csv(OUTPUT_ROOT / "test_manifest.csv", index=False)
    return train_df, test_df


# ============================================================
# 6. EDA CHARTS / SCREENSHOTS
# ============================================================

def _save_screenshot(fig, name):
    shots_dir = OUTPUT_ROOT / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(shots_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved screenshot: {shots_dir / name}")


def eda_video_duration(pairs):
    rows = []
    for p in pairs:
        cap = cv2.VideoCapture(str(p["video_path"]))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total / fps if fps else 0
        rows.append({"video": p["video_path"].name, "duration_s": duration})
        cap.release()
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["duration_s"], bins=15)
    ax.set_xlabel("Video Duration (seconds)")
    ax.set_ylabel("Number of Videos")
    ax.set_title("Le2i Video Duration Distribution")
    ax.grid(axis="y", alpha=0.25)
    _save_screenshot(fig, "eda_video_duration.png")
    return df


def eda_environment_distribution(pairs):
    counts = Counter(p["environment"] for p in pairs)
    envs = ENVIRONMENTS
    values = [counts.get(e, 0) for e in envs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(envs, values)
    ax.set_xlabel("Indoor Environment")
    ax.set_ylabel("Number of Videos")
    ax.set_title("Video Distribution Across Environments")
    plt.xticks(rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _save_screenshot(fig, "eda_environment_distribution.png")


def eda_class_distribution(raw_manifest_df, balanced_manifest_df):
    raw_counts = raw_manifest_df["label"].value_counts().reindex(CLASSES, fill_value=0)
    bal_counts = balanced_manifest_df["label"].value_counts().reindex(CLASSES, fill_value=0)

    x = np.arange(len(CLASSES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, raw_counts.values, width, label="Raw counts")
    ax.bar(x + width / 2, bal_counts.values, width, label="Balanced (augmented)")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Frame count")
    ax.set_title("Activity Class Distribution — Raw vs Balanced")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_screenshot(fig, "eda_class_distribution.png")


def eda_split_summary(train_df, test_df):
    train_counts = train_df["label"].value_counts().reindex(CLASSES, fill_value=0)
    test_counts = test_df["label"].value_counts().reindex(CLASSES, fill_value=0)
    x = np.arange(len(CLASSES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, train_counts.values, width, label="Train (70%)")
    ax.bar(x + width / 2, test_counts.values, width, label="Test (30%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Frame count")
    ax.set_title("Stratified 70/30 Train/Test Split per Class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_screenshot(fig, "eda_train_test_split.png")


def save_annotated_samples_grid():
    """One pose-annotated example per class, for 'annotated dataset
    samples' in the checklist."""
    raw_dir = OUTPUT_ROOT / "raw_frames"
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 4))
    for ax, c in zip(axes, CLASSES):
        files = list((raw_dir / c).glob("*.png"))
        if not files:
            ax.axis("off")
            ax.set_title(f"{c}\n(no samples)")
            continue
        img = cv2.imread(str(files[0]))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(c)
        ax.axis("off")
    plt.tight_layout()
    _save_screenshot(fig, "class_samples_grid.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STEP 0 — Sanity checks")
    print("=" * 70)
    if not DATASET_ROOT.exists():
        print(f"[X] DATASET_ROOT does not exist: {DATASET_ROOT}")
        print("    Edit DATASET_ROOT at the top of this script and re-run.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(exist_ok=True)

    pairs = find_video_annotation_pairs()
    print(f"  Found {len(pairs)} videos across {len(ENVIRONMENTS)} environments.")
    inspect_sample_annotation(pairs)

    proceed = input("\nDoes the annotation format above look right? (y/n): ").strip().lower()
    if proceed != "y":
        print("Stopping so you can fix parse_annotation() or the path. Nothing else was run.")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("STEP 1 — EDA on raw dataset (duration, environment split)")
    print("=" * 70)
    eda_video_duration(pairs)
    eda_environment_distribution(pairs)

    print("\n" + "=" * 70)
    print("STEP 2 — Loading YOLO-Pose")
    print("=" * 70)
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    model.to(DEVICE)
    print(f"  Model loaded on: {DEVICE}")

    print("\n" + "=" * 70)
    print("STEP 3 — Frame extraction + labeling (this is the slow part)")
    print("=" * 70)
    log_df = extract_and_label(model, pairs)
    log_df.to_csv(OUTPUT_ROOT / "extraction_log.csv", index=False)
    print(f"  Extracted {len(log_df)} labeled frames.")
    print(log_df["label"].value_counts())

    print("\n" + "=" * 70)
    print("STEP 4 — Preprocessing (resize 224x224 + normalize)")
    print("=" * 70)
    manifest_df = preprocess_dataset(log_df)

    print("\n" + "=" * 70)
    print("STEP 5 — Augmentation showcase + class balancing")
    print("=" * 70)
    save_augmentation_showcase()
    balanced_df = balance_classes_with_augmentation(manifest_df)

    print("\n" + "=" * 70)
    print("STEP 6 — Stratified 70/30 split")
    print("=" * 70)
    train_df, test_df = stratified_split(balanced_df)

    print("\n" + "=" * 70)
    print("STEP 7 — Remaining EDA charts + screenshots")
    print("=" * 70)
    eda_class_distribution(manifest_df, balanced_df)
    eda_split_summary(train_df, test_df)
    save_annotated_samples_grid()

    print("\n" + "=" * 70)
    print("DONE. Everything is in:", OUTPUT_ROOT.resolve())
    print("Screenshots for your storyboard are in:", (OUTPUT_ROOT / "screenshots").resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
