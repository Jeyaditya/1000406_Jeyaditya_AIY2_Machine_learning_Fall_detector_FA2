# 1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2

# SafeFall AI — Vision-Based Elderly Fall Detection

> Formative Assessment 2 (FA-2) — Model Selection, Training, Evaluation & Deployment
> Course: Artificial Intelligence — Machine Learning & Deep Learning
> Student: A Jeyaditya (Reg No: 1000406)

**Live app:** 

---

## 1. Project Overview

SafeFall AI is a privacy-preserving, vision-based fall detection system for
elderly monitoring. Instead of raw video, the system extracts human pose
keypoints (via YOLO-Pose) and classifies posture into one of five activity
classes — **fall, walking, sitting, standing, normal** — using geometric
features derived from those keypoints (aspect ratio, torso inclination
angle, knee-joint angles), rather than storing or transmitting identifiable
video frames.

## 2. Dataset

- **Source:** Le2i Fall Detection Dataset — (Dataset link)[https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data]
- **Environments:** Home_01, Home_02, Coffee_room_01, Coffee_room_02,
  Lecture_room, Office
- **Format:** Uncompressed `.avi` surveillance videos, each paired with a
  `.txt` annotation file marking the fall start/end frame (or `0 0` if the
  video contains no fall).

**The raw dataset is NOT included in this repository** (it's ~10GB — well
past GitHub's recommended repo size). To reproduce:

```bash
# Requires a Kaggle API token (kaggle.json) in ~/.kaggle/
kaggle datasets download -d <dataset-slug-here>
unzip <dataset-name>.zip -d le2i_dataset/
```

A small sample of processed, labeled frames (a handful per class) is kept
in `fa1_outputs/screenshots/` for reference without needing the full dataset.

## 3. Preprocessing Pipeline (FA-1)

See `fa1_pipeline.py`. In summary:

1. Pair every video with its annotation file across all 6 environments.
2. Parse the annotation for the ground-truth fall frame window.
3. Sample frames per video (biased toward the annotated fall window, so
   the brief ~1-1.5s fall event isn't missed by even spacing).
4. Run YOLO-Pose on each sampled frame; keep only the highest-confidence
   detected person (filters out low-confidence false positives).
5. Label each frame: `fall` from the annotation ground truth (with a
   geometric fallback — aspect ratio > 1.0 or torso angle > 60°ff — for
   frames the annotation window narrowly misses); the other 4 classes
   from knee-angle / aspect-ratio heuristics.
6. Resize to 224×224, normalize, augment (rotate/flip/brightness/zoom)
   to balance class counts, and split 70/30 (stratified) for FA-1 EDA.

## 4. Model (FA-2)

See `pose_utils.py` (shared feature extraction) and `train_classifier.py`.

- **Features:** a 55-dim vector per frame — normalized (x, y) for all 17
  YOLO keypoints, their confidences, aspect ratio, torso angle, average
  knee angle, and knee-angle asymmetry (scale-invariant, not raw pixels).
- **Models compared:** Random Forest (200 trees, balanced class weights)
  vs. a small MLP (64→32 hidden units). Whichever scores higher macro-F1
  on the validation split is selected and deployed.
- **Split:** 70% train / 15% validation / 15% test, stratified by class.

**Result:** _[fill in after running train_classifier.py —
see fa2_outputs/models/model_info.json for the winning model + metrics]_

| Metric | Value |
|---|---|
| Winning model | _e.g. random_forest_ |
| Test accuracy | _fill in_ |
| Test macro F1 | _fill in_ |

Confusion matrix, model comparison chart, and MLP loss curve are in
`fa2_outputs/screenshots/`.

## 5. Running Locally

```bash
pip install -r requirements.txt
python fa1_pipeline.py          # edit DATASET_ROOT first
python train_classifier.py
streamlit run app.py
```

## 6. Deployment

Deployed on Streamlit Community Cloud, pointing at `app.py`.
The dashboard supports:
- Image/video upload with live pose overlay + activity prediction
- Emergency alert banner on fall detection
- Session analytics (total activities, fall count, activity distribution)
- Model evaluation tab (confusion matrix, loss curve, sample predictions)
