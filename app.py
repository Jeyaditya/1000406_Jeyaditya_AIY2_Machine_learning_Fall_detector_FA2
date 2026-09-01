"""
app.py — SafeFall AI Streamlit Dashboard (FA-2, Step 7)
=========================================================
Run locally with:   streamlit run app.py
Deploy to Streamlit Cloud pointing at this file, with requirements.txt
(the GitHub/headless version) in the repo root.

Needs fa2_outputs/models/{classifier,scaler,label_encoder}.joblib to
exist — run train_classifier.py first.
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from PIL import Image

from pose_utils import image_to_feature

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("./fa2_outputs/models")
CLASSES_COLORS = {
    "fall": "#e63946",
    "walking": "#2a9d8f",
    "sitting": "#e9c46a",
    "standing": "#457b9d",
    "normal": "#8d99ae",
}
DEVICE = "cpu"

st.set_page_config(page_title="SafeFall AI", page_icon="🚨", layout="wide")


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    model.to(DEVICE)
    return model


@st.cache_resource
def load_classifier():
    clf_path = MODEL_DIR / "classifier.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    if not (clf_path.exists() and scaler_path.exists() and encoder_path.exists()):
        return None, None, None
    return (joblib.load(clf_path), joblib.load(scaler_path), joblib.load(encoder_path))


# ============================================================
# SESSION STATE (dashboard counters persist across uploads
# within the same browser session)
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {time, source, label, confidence}


def predict_frame(frame_bgr, yolo_model, clf, scaler, label_encoder):
    """Returns (label, confidence, annotated_rgb_image, prob_breakdown) or
    (None, None, original, None) if no confident person detection.
    prob_breakdown is a dict {class_name: probability} for every class —
    useful for spotting whether a wrong prediction is a confident mistake
    (one class way ahead) or a close, uncertain call (probabilities bunched
    together) while you're still tuning the model."""
    feature, result = image_to_feature(yolo_model, frame_bgr, device=DEVICE)
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    if feature is None:
        return None, None, annotated, None

    feature_s = scaler.transform(feature.reshape(1, -1))
    probs = clf.predict_proba(feature_s)[0]
    pred_idx = int(np.argmax(probs))
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    prob_breakdown = dict(zip(label_encoder.classes_, probs.tolist()))
    return label, confidence, annotated, prob_breakdown


def log_prediction(source, label, confidence):
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "label": label,
        "confidence": confidence,
    })


# ============================================================
# SIDEBAR — DASHBOARD ANALYTICS
# ============================================================

with st.sidebar:
    st.title("🚨 SafeFall AI")
    st.caption("Vision-based elderly fall monitoring — CareVision HealthTech")

    history_df = pd.DataFrame(st.session_state.history)
    total = len(history_df)
    fall_count = int((history_df["label"] == "fall").sum()) if total else 0
    normal_count = int((history_df["label"] == "normal").sum()) if total else 0

    st.metric("Total activities detected", total)
    st.metric("Fall events detected", fall_count)
    st.metric("Normal activity count", normal_count)

    if total:
        st.subheader("Activity distribution")
        counts = history_df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = [CLASSES_COLORS.get(c, "#999999") for c in counts.index]
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    if st.button("Reset session analytics"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# MAIN AREA
# ============================================================

yolo_model = load_yolo()
clf, scaler, label_encoder = load_classifier()

if clf is None:
    st.error(
        "No trained model found at `fa2_outputs/models/`. "
        "Run `python train_classifier.py` first, then restart this app."
    )
    st.stop()

tab_predict, tab_metrics = st.tabs(["🎥 Live Prediction", "📊 Model Evaluation"])

with tab_predict:
    st.info(
        "**Known limitation:** video prediction is stable; single-image "
        "prediction is still being tuned — under active development.",
        icon="🛠️",
    )
    st.header("Upload an image or video")
    upload_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if upload_type == "Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            label, confidence, annotated, prob_breakdown = predict_frame(
                frame_bgr, yolo_model, clf, scaler, label_encoder
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Uploaded image", width='stretch')
            with col2:
                st.image(annotated, caption="Pose detection output", width='stretch')

            if label is None:
                st.warning("No confident person detection in this image — try a clearer frame.")
            else:
                log_prediction(uploaded.name, label, confidence)
                if label == "fall" and confidence >= 0.55:
                    st.error(f"🚨 **FALL DETECTED** — confidence {confidence:.1%}. "
                             f"Emergency alert would be dispatched to caregiver.")
                elif label == "fall":
                    st.warning(f"Possible fall (low confidence: {confidence:.1%}) — "
                               f"flagged but below the alert threshold.")
                else:
                    st.success(f"Activity: **{label}** — confidence {confidence:.1%}")

                with st.expander("🔍 Full prediction breakdown (debugging)"):
                    breakdown_df = pd.DataFrame(
                        sorted(prob_breakdown.items(), key=lambda x: -x[1]),
                        columns=["class", "probability"],
                    )
                    st.dataframe(breakdown_df, width='stretch')
                    st.caption(
                        "If 'fall' and the correct class are close together here, "
                        "it's a genuinely hard/borderline example — worth adding more "
                        "similar training frames. If 'fall' is far ahead when it "
                        "shouldn't be, that's a systematic bug worth re-checking "
                        "in fa1_pipeline.py's labeling."
                    )

    else:  # Video
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        sample_every_n = st.slider(
            "Process every Nth frame (higher = faster, coarser)", 5, 60, 15
        )
        fall_confidence_gate = st.slider(
            "Minimum confidence to count as a fall event", 0.3, 0.9, 0.55,
            help="Raise this if you're seeing false alarms; lower it if real "
                 "falls are being missed.",
        )
        event_cooldown_s = st.slider(
            "Merge fall predictions within N seconds into one event", 1, 10, 3,
            help="One real fall usually produces several consecutive 'fall' "
                 "frames as it's sampled — this groups them into a single "
                 "event instead of counting each sampled frame separately.",
        )
        if uploaded is not None:
            tmp_path = Path(f"./_tmp_{uploaded.name}")
            tmp_path.write_bytes(uploaded.read())

            cap = cv2.VideoCapture(str(tmp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            progress = st.progress(0.0)
            status = st.empty()
            fall_events = []          # de-duplicated events shown to the user
            last_fall_frame = None    # for cooldown grouping
            frame_idx = 0
            processed = 0

            preview_slot = st.empty()

            while True:
                ok_read, frame = cap.read()
                if not ok_read:
                    break
                if frame_idx % sample_every_n == 0:
                    label, confidence, annotated, _ = predict_frame(
                        frame, yolo_model, clf, scaler, label_encoder
                    )
                    if label is not None:
                        log_prediction(uploaded.name, label, confidence)
                        preview_slot.image(
                            annotated,
                            caption=f"Frame {frame_idx} ({frame_idx / fps:.1f}s) — {label} ({confidence:.0%})",
                            width='stretch',
                        )
                        # Only count as a fall if confidence clears the gate,
                        # AND merge it into the previous event if it's within
                        # the cooldown window (same physical fall, sampled
                        # more than once) rather than logging a new event.
                        if label == "fall" and confidence >= fall_confidence_gate:
                            within_cooldown = (
                                last_fall_frame is not None
                                and (frame_idx - last_fall_frame) / fps <= event_cooldown_s
                            )
                            if within_cooldown:
                                # extend the ongoing event's end time / keep peak confidence
                                fall_events[-1]["end_timestamp_s"] = round(frame_idx / fps, 2)
                                fall_events[-1]["confidence"] = max(
                                    fall_events[-1]["confidence"], confidence
                                )
                            else:
                                fall_events.append({
                                    "frame": frame_idx,
                                    "start_timestamp_s": round(frame_idx / fps, 2),
                                    "end_timestamp_s": round(frame_idx / fps, 2),
                                    "confidence": confidence,
                                })
                            last_fall_frame = frame_idx
                    processed += 1
                frame_idx += 1
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0))
                status.text(f"Processed {processed} sampled frames / {frame_idx} total frames")

            cap.release()
            tmp_path.unlink(missing_ok=True)

            st.subheader("Result")
            if fall_events:
                st.error(f"🚨 {len(fall_events)} fall event(s) detected in this video.")
                st.dataframe(pd.DataFrame(fall_events))
            else:
                st.success("No fall events detected in this video.")

with tab_metrics:
    st.header("Training-time evaluation (from train_classifier.py)")
    screenshots_dir = Path("./fa2_outputs/screenshots")
    info_path = MODEL_DIR / "model_info.json"

    if info_path.exists():
        import json
        info = json.loads(info_path.read_text())
        st.write(f"**Winning model:** `{info['winner']}`")
        st.write("**Test-set metrics:**")
        st.json(info["test_metrics"])

    col1, col2 = st.columns(2)
    cm_path = screenshots_dir / "confusion_matrix.png"
    cmp_path = screenshots_dir / "model_comparison.png"
    loss_path = screenshots_dir / "mlp_loss_curve.png"

    if cm_path.exists():
        col1.image(str(cm_path), caption="Confusion Matrix (test set)", width='stretch')
    if cmp_path.exists():
        col2.image(str(cmp_path), caption="Random Forest vs MLP", width='stretch')
    if loss_path.exists():
        st.image(str(loss_path), caption="MLP Training Loss Curve", width='stretch')

    pred_dir = screenshots_dir / "predictions"
    if pred_dir.exists():
        st.subheader("Sample predictions from the held-out test set")
        pred_files = sorted(pred_dir.glob("*.png"))[:8]
        cols = st.columns(4)
        for i, p in enumerate(pred_files):
            cols[i % 4].image(str(p), width='stretch')

    if not info_path.exists():
        st.info("Run `python train_classifier.py` to generate evaluation charts.")
