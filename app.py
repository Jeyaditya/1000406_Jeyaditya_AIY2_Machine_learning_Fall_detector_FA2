"""
app.py — SafeFall AI — Professional Healthcare Monitoring Dashboard
====================================================================
AI-Powered Elderly Fall Detection & Activity Monitoring System.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud pointing at this file, with the
repo-root requirements.txt (the CPU / headless version).

This dashboard preserves the REAL machine-learning pipeline:
    YOLO11n-Pose  ->  17 keypoints  ->  55-dim pose/geometric features
    ->  Random Forest classifier  ->  activity label + confidence
    ->  fall detection + emergency alert.

It only redesigns the USER INTERFACE around that pipeline. The shared
feature-extraction module (pose_utils.py) is used UNCHANGED so the
feature vector a model is TRAINED on matches the one it is FED at
inference time.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pose_utils import image_to_feature

# Optional deps — kept optional so the app still boots if missing.
try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("./fa2_outputs/models")
SCREENSHOTS_DIR = Path("./fa2_outputs/screenshots")
INFO_PATH = MODEL_DIR / "model_info.json"

MAX_FRAME_DIMENSION = 480      # every frame downscaled to this before inference
MAX_FRAMES_PER_VIDEO = 40      # hard ceiling on inference calls per video
GC_EVERY_N_FRAMES = 5          # periodic garbage collection in the video loop
HISTORY_CAP = 200              # retain the most recent N prediction records
FALL_DEFAULT_GATE = 0.55       # default fall-confidence threshold
FALL_DEFAULT_COOLDOWN = 3      # default event cooldown (seconds)

# Single restrained accent palette (dark clinical technology aesthetic)
ACCENT = "#22d3ee"          # cyan/teal accent
ACCENT_SOFT = "#0e7490"
DANGER = "#ef4444"          # red — fall / alert
WARNING = "#f59e0b"         # amber — caution
SUCCESS = "#22c55e"         # green — safe / normal
SURFACE = "#111827"         # card surface (slightly lighter than bg)
BORDER = "#1f2937"

# Activity colors (used consistently across charts & cards)
CLASSES_COLORS = {
    "fall": "#ef4444",
    "walking": "#22d3ee",
    "sitting": "#f59e0b",
    "standing": "#60a5fa",
    "normal": "#22c55e",
}

# Display order + title-case labels for the five classes
CLASS_DISPLAY_ORDER = ["fall", "walking", "sitting", "standing", "normal"]
CLASS_DISPLAY = {
    "fall": "Fall",
    "walking": "Walking",
    "sitting": "Sitting",
    "standing": "Standing",
    "normal": "Normal",
}

DEVICE = "cpu"

st.set_page_config(
    page_title="SafeFall AI — Elderly Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM — custom CSS (dark clinical technology theme)
# ============================================================

def _design_css() -> str:
    return """
    <style>
    /* ---- Global tokens ---- */
    :root {
        --bg: #0b1020;
        --surface: #111827;
        --surface-2: #0f1626;
        --border: #1f2937;
        --text: #e5e7eb;
        --text-dim: #9ca3af;
        --accent: #22d3ee;
        --accent-soft: #0e7490;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #22c55e;
        --radius: 14px;
    }

    /* ---- App background ---- */
    .stApp, .stApp > header {
        background-color: var(--bg);
        color: var(--text);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(900px 500px at 85% -10%, rgba(34,211,238,0.08), transparent 60%),
            radial-gradient(700px 500px at 0% 110%, rgba(14,116,144,0.07), transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    .stApp > * { position: relative; z-index: 1; }

    /* ---- Brand header bar ---- */
    .sf-header {
        display: flex; align-items: center; justify-content: space-between;
        gap: 16px; padding: 18px 22px; margin-bottom: 14px;
        background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(15,22,38,0.96));
        border: 1px solid var(--border); border-radius: var(--radius);
        box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    }
    .sf-brand { display: flex; align-items: center; gap: 14px; }
    .sf-logo {
        width: 46px; height: 46px; border-radius: 12px; flex: 0 0 auto;
        background: linear-gradient(135deg, var(--accent), var(--accent-soft));
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; color: #04141a; font-weight: 800;
        box-shadow: 0 0 0 1px rgba(34,211,238,0.35), 0 0 22px rgba(34,211,238,0.30);
    }
    .sf-title { font-size: 22px; font-weight: 800; letter-spacing: .3px; color: var(--text); }
    .sf-subtitle { font-size: 12.5px; color: var(--text-dim); margin-top: 2px; }

    .sf-status {
        display: flex; align-items: center; gap: 9px;
        padding: 8px 14px; border-radius: 999px;
        background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.35);
        font-size: 12.5px; color: #86efac; font-weight: 600; white-space: nowrap;
    }
    .sf-status .dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: var(--success); box-shadow: 0 0 0 0 rgba(34,197,94,0.6);
        animation: sf-pulse 2s infinite;
    }
    .sf-status.offline { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.35); color: #fca5a5; }
    .sf-status.offline .dot { background: var(--danger); animation: none; }
    @keyframes sf-pulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }
        70% { box-shadow: 0 0 0 9px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }

    /* ---- Cards ---- */
    .sf-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25); transition: border-color .2s ease;
    }
    .sf-card:hover { border-color: rgba(34,211,238,0.30); }
    .sf-card-title {
        font-size: 11px; text-transform: uppercase; letter-spacing: 1.4px;
        color: var(--text-dim); font-weight: 700; margin: 0 0 10px 0;
    }

    /* ---- KPI metric cards ---- */
    .sf-kpi {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 16px 18px; position: relative; overflow: hidden;
    }
    .sf-kpi::before {
        content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
        background: var(--accent);
    }
    .sf-kpi.danger::before { background: var(--danger); }
    .sf-kpi.success::before { background: var(--success); }
    .sf-kpi.warn::before { background: var(--warning); }
    .sf-kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-dim); font-weight: 700; }
    .sf-kpi-value { font-size: 30px; font-weight: 800; margin-top: 6px; color: var(--text); line-height: 1.1; }
    .sf-kpi-sub { font-size: 11.5px; color: var(--text-dim); margin-top: 4px; }

    /* ---- Status pill (system component readiness) ---- */
    .sf-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
        background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.30); color: #86efac;
    }
    .sf-pill.warn { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.30); color: #fcd34d; }
    .sf-pill.danger { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.30); color: #fca5a5; }
    .sf-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
    .sf-pill.warn .dot { background: var(--warning); }
    .sf-pill.danger .dot { background: var(--danger); }

    /* ---- Prediction hero card ---- */
    .sf-pred {
        border-radius: var(--radius); padding: 20px; text-align: center;
        border: 1px solid var(--border);
    }
    .sf-pred-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1.4px; color: var(--text-dim); }
    .sf-pred-activity { font-size: 34px; font-weight: 800; margin: 6px 0 2px 0; }
    .sf-pred-conf { font-size: 15px; color: var(--text-dim); }
    .sf-pred.normal { background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(17,24,39,0.6)); border-color: rgba(34,197,94,0.30); }
    .sf-pred.fall   { background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(17,24,39,0.6)); border-color: rgba(239,68,68,0.45); }
    .sf-pred.warn   { background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(17,24,39,0.6)); border-color: rgba(245,158,11,0.35); }

    /* ---- Fall alert banner ---- */
    .sf-alert-fall {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.06));
        border: 1.5px solid rgba(239,68,68,0.55); border-radius: var(--radius);
        padding: 18px 20px; color: #fecaca;
        box-shadow: 0 0 28px rgba(239,68,68,0.20);
        animation: sf-flash 1.6s ease-in-out infinite alternate;
    }
    @keyframes sf-flash { from { box-shadow: 0 0 14px rgba(239,68,68,0.15);} to { box-shadow: 0 0 34px rgba(239,68,68,0.35);} }
    .sf-alert-fall .head { font-size: 20px; font-weight: 800; display:flex; align-items:center; gap:10px; color:#fff; }
    .sf-alert-fall .meta { font-size: 13px; margin-top: 6px; color:#fca5a5; }

    /* ---- Confidence meter ---- */
    .sf-meter { height: 10px; border-radius: 999px; background: #1f2937; overflow: hidden; }
    .sf-meter > span { display:block; height:100%; border-radius:999px; transition: width .4s ease; }

    /* ---- Pipeline flow diagram ---- */
    .sf-pipe { display:flex; flex-wrap:wrap; gap:10px; align-items:stretch; }
    .sf-step {
        flex:1 1 0; min-width: 120px; padding: 12px 12px; border-radius: 12px;
        background: var(--surface-2); border: 1px solid var(--border); text-align:center;
    }
    .sf-step .n { font-size: 11px; color: var(--accent); font-weight:700; letter-spacing:1px; }
    .sf-step .t { font-size: 13px; font-weight: 700; margin-top: 4px; color: var(--text); }
    .sf-step .d { font-size: 11px; color: var(--text-dim); margin-top: 3px; }
    .sf-arrow { display:flex; align-items:center; color: var(--accent); font-size: 18px; }

    /* ---- Empty state ---- */
    .sf-empty {
        text-align:center; padding: 36px 18px; color: var(--text-dim);
        background: var(--surface-2); border: 1px dashed var(--border); border-radius: var(--radius);
    }
    .sf-empty .big { font-size: 30px; opacity:.5; margin-bottom: 8px; }

    /* ---- Timeline ---- */
    .sf-tl { position: relative; padding-left: 18px; }
    .sf-tl::before { content:""; position:absolute; left:6px; top:4px; bottom:4px; width:2px; background: var(--border); }
    .sf-tl-row { position: relative; padding: 6px 0 6px 14px; }
    .sf-tl-row::before {
        content:""; position:absolute; left:-12px; top:11px; width:10px; height:10px; border-radius:50%;
        background: var(--accent); box-shadow: 0 0 0 3px rgba(34,211,238,0.18);
    }
    .sf-tl-row.fall::before { background: var(--danger); box-shadow: 0 0 0 3px rgba(239,68,68,0.22); }
    .sf-tl-row .t { font-size: 12px; color: var(--text-dim); }
    .sf-tl-row .a { font-size: 14px; font-weight: 700; }

    /* ---- Streamlit overrides ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 10px 10px 0 0;
        padding: 9px 16px; color: var(--text-dim); font-weight:600;
        border-bottom: 3px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text) !important; border-bottom-color: var(--accent) !important;
        background: rgba(34,211,238,0.06);
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    .stTabs [data-baseweb="tab-border"] { display:none; }

    section[data-testid="stSidebar"] {
        background: #0a1020; border-right: 1px solid var(--border);
    }
    .stMetric { background: var(--surface); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
    .stMetric label { color: var(--text-dim) !important; font-size: 11px !important; letter-spacing:.6px; }
    .stMetric [data-testid="stMetricValue"] { color: var(--text) !important; }

    .stFileUploader > div { background: var(--surface); border:1px solid var(--border); border-radius: 12px; }
    .stAlert { border-radius: 12px; }

    /* hide the default red top-right menu if it overlaps */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .st-emotion-cache-1dp5ir5 { display: none; }

    /* small helper text */
    .sf-note { font-size: 12px; color: var(--text-dim); }
    .sf-divider { height:1px; background: var(--border); margin: 14px 0; border:none; }
    .sf-tag { font-size:11px; padding:3px 9px; border-radius:999px; background:rgba(34,211,238,0.10); border:1px solid rgba(34,211,238,0.30); color:#a5f3fc; }
    </style>
    """


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def _html(html: str) -> None:
    """Safely render an HTML fragment in Streamlit."""
    st.markdown(html, unsafe_allow_html=True)


def brand_header(online: bool = True) -> None:
    """Top brand bar with live system-status indicator."""
    status_cls = "" if online else "offline"
    status_text = "SYSTEM ONLINE" if online else "SYSTEM DEGRADED"
    _html(f"""
    <div class="sf-header">
        <div class="sf-brand">
            <div class="sf-logo">🛡️</div>
            <div>
                <div class="sf-title">SAFEFALL AI</div>
                <div class="sf-subtitle">AI-Powered Elderly Fall Detection &amp; Activity Monitoring</div>
            </div>
        </div>
        <div class="sf-status {status_cls}">
            <span class="dot"></span>● {status_text}
        </div>
    </div>
    """)


def kpi_card(label: str, value, sub: str = "", variant: str = "") -> None:
    """A polished KPI metric card. variant: '' | 'danger' | 'success' | 'warn'."""
    cls = variant if variant in ("danger", "success", "warn") else ""
    sub_html = f'<div class="sf-kpi-sub">{sub}</div>' if sub else ""
    _html(f"""
    <div class="sf-kpi {cls}">
        <div class="sf-kpi-label">{label}</div>
        <div class="sf-kpi-value">{value}</div>
        {sub_html}
    </div>
    """)


def status_pill(label: str, ok: bool = True, kind: str = "") -> str:
    """Return HTML for a small readiness pill."""
    if kind:
        cls = kind
    else:
        cls = "" if ok else "danger"
    dot = "<span class='dot'></span>"
    return f"<span class='sf-pill {cls}'>{dot} {label}</span>"


def prediction_hero(label: str | None, confidence: float | None) -> str:
    """Return HTML for the big prediction card used in image/video analysis."""
    if label is None:
        return """
        <div class="sf-pred warn">
            <div class="sf-pred-label">Prediction</div>
            <div class="sf-pred-activity" style="font-size:22px;color:#fcd34d;">No person detected</div>
            <div class="sf-pred-conf">Pose engine could not confidently detect a subject in this frame.</div>
        </div>
        """
    disp = CLASS_DISPLAY.get(label, label.title())
    if label == "fall" and (confidence or 0) >= FALL_DEFAULT_GATE:
        variant = "fall"
    elif label == "fall":
        variant = "warn"
    else:
        variant = "normal"
    color = CLASSES_COLORS.get(label, ACCENT)
    conf_txt = f"{(confidence or 0)*100:.1f}% confidence" if confidence is not None else "—"
    return f"""
    <div class="sf-pred {variant}">
        <div class="sf-pred-label">Predicted Activity</div>
        <div class="sf-pred-activity" style="color:{color};">{disp}</div>
        <div class="sf-pred-conf">{conf_txt}</div>
    </div>
    """


def confidence_meter(confidence: float, color: str = ACCENT) -> str:
    pct = max(0.0, min(1.0, float(confidence))) * 100
    return f"""
    <div class="sf-meter"><span style="width:{pct:.1f}%;background:{color};"></span></div>
    """


def fall_alert_banner(timestamp_s: float | None, confidence: float, frame: int | None = None) -> str:
    ts = f"{timestamp_s:.1f}s" if timestamp_s is not None else "—"
    fr = f"Frame {frame}" if frame is not None else ""
    return f"""
    <div class="sf-alert-fall">
        <div class="head">⚠ FALL DETECTED</div>
        <div class="meta">Potential fall event detected at {ts} &nbsp;·&nbsp; Confidence: {confidence*100:.1f}% &nbsp;·&nbsp; {fr}</div>
        <div class="meta" style="margin-top:8px;">Emergency alert generated — caregiver notification flagged for review.</div>
    </div>
    """


def pipeline_flow(steps: list[tuple[str, str, str]]) -> str:
    """Render the conceptual ML pipeline as a horizontal flow diagram."""
    parts = []
    for i, (n, t, d) in enumerate(steps):
        parts.append(f"""
        <div class="sf-step">
            <div class="n">{n}</div>
            <div class="t">{t}</div>
            <div class="d">{d}</div>
        </div>""")
        if i < len(steps) - 1:
            parts.append('<div class="sf-arrow">→</div>')
    return f'<div class="sf-pipe">{"".join(parts)}</div>'


def empty_state(big: str, msg: str) -> str:
    return f"""
    <div class="sf-empty">
        <div class="big">{big}</div>
        <div>{msg}</div>
    </div>
    """


def section_title(title: str, tag: str = "") -> None:
    tag_html = f'<span class="sf-tag" style="margin-left:10px;">{tag}</span>' if tag else ""
    _html(f"<div style='display:flex;align-items:center;margin:18px 0 12px 0;'><span style='font-size:16px;font-weight:800;letter-spacing:.4px;color:var(--text);'>{title}</span>{tag_html}</div><hr class='sf-divider'/>")


def timeline_html(events: list[dict]) -> str:
    """events: list of {time, label, confidence}."""
    if not events:
        return empty_state("—", "No activity recorded yet.")
    rows = []
    for e in events[-14:]:
        cls = "fall" if e["label"] == "fall" else ""
        disp = CLASS_DISPLAY.get(e["label"], str(e["label"]).title())
        conf = e.get("confidence")
        conf_txt = f"{conf*100:.0f}%" if conf is not None else ""
        rows.append(f"""
        <div class="sf-tl-row {cls}">
            <div class="t">{e['time']}</div>
            <div class="a">{disp} <span class="sf-note">· {conf_txt}</span></div>
        </div>""")
    return f'<div class="sf-tl">{"".join(rows)}</div>'


# ============================================================
# CACHED LOADERS  (heavyweight, never reload on Streamlit reruns)
# ============================================================

@st.cache_resource
def load_yolo():
    """Load YOLO11n-Pose once and keep it cached for the session."""
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    model.to(DEVICE)
    return model


@st.cache_resource
def load_classifier():
    """Load the trained RF classifier + scaler + label encoder.
    Returns (clf, scaler, label_encoder) or (None, None, None) if missing."""
    if joblib is None:
        return None, None, None
    clf_path = MODEL_DIR / "classifier.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    if not (clf_path.exists() and scaler_path.exists() and encoder_path.exists()):
        return None, None, None
    return (joblib.load(clf_path), joblib.load(scaler_path), joblib.load(encoder_path))


@st.cache_data(show_spinner=False)
def load_model_info() -> dict | None:
    """Load the REAL evaluation metadata from model_info.json (cached)."""
    if not INFO_PATH.exists():
        return None
    try:
        return json.loads(INFO_PATH.read_text())
    except Exception:
        return None


# ============================================================
# SESSION STATE
# ============================================================

def init_state() -> None:
    ss = st.session_state
    if "history" not in ss:
        ss.history = []  # {time, source, label, confidence, timestamp_s?}
    if "fall_events" not in ss:
        ss.fall_events = []  # {frame, start_timestamp_s, end_timestamp_s, confidence}


def log_prediction(source: str, label: str, confidence: float, timestamp_s: float | None = None) -> None:
    """Append a prediction to history and enforce the session-memory cap."""
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "label": label,
        "confidence": float(confidence),
        "timestamp_s": timestamp_s,
    })
    if len(st.session_state.history) > HISTORY_CAP:
        st.session_state.history = st.session_state.history[-HISTORY_CAP:]


# ============================================================
# MEMORY / PERFORMANCE HELPERS
# ============================================================

def current_memory_mb() -> float | None:
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def downscale_frame(frame_bgr, max_dim: int = MAX_FRAME_DIMENSION):
    """Shrink the frame before YOLO/feature extraction to bound memory & CPU."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame_bgr
    scale = max_dim / longest
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))


# ============================================================
# CORE PREDICTION (preserved ML pipeline)
# ============================================================

def predict_frame(frame_bgr, yolo_model, clf, scaler, label_encoder):
    """Run the REAL pipeline on one frame.

    Returns (label, confidence, annotated_rgb, prob_breakdown) or
    (None, None, annotated, None) when no confident person is detected.

    prob_breakdown is a dict {class_name: probability} for every class,
    straight from the REAL classifier — never fabricated.
    """
    # torch.inference_mode() is applied inside ultralytics' predict() for the
    # YOLO model; the classifier is sklearn (no torch graph).
    feature, result = image_to_feature(yolo_model, frame_bgr, device=DEVICE)
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB for display

    if feature is None:
        return None, None, annotated, None

    feature_s = scaler.transform(feature.reshape(1, -1))
    probs = clf.predict_proba(feature_s)[0]
    pred_idx = int(np.argmax(probs))
    label = label_encoder.inverse_transform([pred_idx])[0]
    # label may come back as numpy str — normalise to plain str
    label = str(label)
    confidence = float(probs[pred_idx])
    prob_breakdown = {str(k): float(v) for k, v in zip(label_encoder.classes_, probs.tolist())}
    return label, confidence, annotated, prob_breakdown


# ============================================================
# SESSION ANALYTICS HELPERS
# ============================================================

def history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=["time", "source", "label", "confidence", "timestamp_s"])
    return pd.DataFrame(st.session_state.history)


def activity_counts() -> dict:
    """Count of each class in session history — only REAL predictions."""
    counts = {c: 0 for c in CLASS_DISPLAY_ORDER}
    for h in st.session_state.history:
        counts[h["label"]] = counts.get(h["label"], 0) + 1
    return counts


def avg_confidence() -> float | None:
    if not st.session_state.history:
        return None
    return float(np.mean([h["confidence"] for h in st.session_state.history]))


def activity_distribution_chart():
    """Horizontal bar chart of REAL activity counts. Returns matplotlib fig or None."""
    if not _HAS_MPL:
        return None
    counts = activity_counts()
    labels = [CLASS_DISPLAY[c] for c in CLASS_DISPLAY_ORDER]
    values = [counts[c] for c in CLASS_DISPLAY_ORDER]
    colors = [CLASSES_COLORS[c] for c in CLASS_DISPLAY_ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.62)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.tick_params(colors="#9ca3af", labelsize=11)
    ax.set_xlabel("Predictions", color="#9ca3af", fontsize=11)
    for b, v in zip(bars, values):
        if v > 0:
            ax.text(b.get_width() + max(values) * 0.01 + 0.3, b.get_y() + b.get_height() / 2,
                    str(int(v)), va="center", color="#e5e7eb", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#1f2937", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def confidence_breakdown_bars(prob_breakdown: dict):
    """Render the REAL per-class probability breakdown as labelled meters."""
    if not prob_breakdown:
        return
    items = sorted(prob_breakdown.items(), key=lambda kv: -kv[1])
    rows = []
    for cls, p in items:
        disp = CLASS_DISPLAY.get(cls, cls.title())
        color = CLASSES_COLORS.get(cls, ACCENT)
        pct = p * 100
        rows.append(f"""
        <div style='margin-bottom:9px;'>
            <div style='display:flex;justify-content:space-between;font-size:12px;color:var(--text-dim);margin-bottom:3px;'>
                <span style='color:{color};font-weight:600;'>{disp}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class='sf-meter'><span style='width:{pct:.1f}%;background:{color};'></span></div>
        </div>""")
    _html("".join(rows))


# ============================================================
# SAFE VIDEO HANDLING (AVI / MP4 / MOV robustness)
# ============================================================

def stream_upload_to_tmp(uploaded, base_name: str) -> Path:
    """Stream uploaded bytes to a safely-named temp file inside /tmp.

    Uses a sanitised, fixed base name (NOT the raw uploaded filename) so an
    uploaded filename can never escape the temp directory or overwrite
    arbitrary files. The original extension is preserved only after
    validation against an allow-list.
    """
    safe_ext = ""
    name = (uploaded.name or "").lower()
    for ext in (".avi", ".mp4", ".mov", ".m4v", ".mkv"):
        if name.endswith(ext):
            safe_ext = ext
            break
    if not safe_ext:
        safe_ext = ".avi"  # fallback; OpenCV will tell us if it can't read it
    tmp_path = Path("/tmp") / f"{base_name}{safe_ext}"
    with open(tmp_path, "wb") as f:
        # stream in 1 MB chunks — avoids materialising the whole file in RAM
        for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
            f.write(chunk)
    return tmp_path


def open_video_robustly(tmp_path: Path):
    """Open a video with OpenCV. Returns (cap, total_frames, fps) or
    (None, 0, 0.0) when the file can't be read at all."""
    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        return None, 0, 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cap, total_frames, fps


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(model_ok: bool, clf_ok: bool, yolo_ok: bool) -> None:
    with st.sidebar:
        _html("""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
            <div style='width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#22d3ee,#0e7490);display:flex;align-items:center;justify-content:center;font-size:18px;'>🛡️</div>
            <div>
                <div style='font-size:15px;font-weight:800;letter-spacing:.4px;'>SAFEFALL AI</div>
                <div style='font-size:10.5px;color:#9ca3af;'>Elderly Monitoring Console</div>
            </div>
        </div>
        <hr class='sf-divider'/>
        """)
        st.caption("Navigation is via the tabs in the main panel — Overview, Image Analysis, Video Monitoring, Analytics, and Model Information.")

        st.markdown("**SYSTEM STATUS**")
        _html(f"""
        <div style='display:flex;flex-direction:column;gap:8px;margin-top:4px;'>
            {status_pill("Pose engine ready", yolo_ok)}
            {status_pill("Classifier ready", clf_ok, kind="" if clf_ok else "danger")}
            {status_pill("Model loaded", model_ok, kind="" if model_ok else "danger")}
            <span class='sf-pill'><span class='dot' style='background:#60a5fa;'></span> Inference: CPU</span>
        </div>
        <hr class='sf-divider'/>
        """)

        # live session KPIs in the sidebar
        counts = activity_counts()
        total = sum(counts.values())
        c1, c2 = st.columns(2)
        c1.metric("Total predictions", total)
        c2.metric("Fall events", counts["fall"])
        st.caption(f"Inference device: **CPU** · No GPU required.")

        mem = current_memory_mb()
        if mem is not None:
            st.caption(f"Memory in use: **{mem:.0f} MB**")

        if st.button("↺ Reset session analytics", use_container_width=True):
            st.session_state.history = []
            st.session_state.fall_events = []
            st.rerun()


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview():
    section_title("Overview", "DASHBOARD")
    counts = activity_counts()
    total = sum(counts.values())
    falls = counts["fall"]
    normal_total = counts["normal"] + counts["walking"] + counts["sitting"] + counts["standing"]
    conf = avg_confidence()

    # ---- KPI row (REAL session data) ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Activities", total if total else 0,
                 "predictions this session" if total else "awaiting input")
    with c2:
        kpi_card("Fall Events", falls if falls else 0,
                 "potential fall incidents" if falls else "none detected",
                 variant="danger" if falls else "")
    with c3:
        kpi_card("Normal Activities", normal_total if normal_total else 0,
                 "non-fall predictions" if normal_total else "awaiting input",
                 variant="success" if normal_total else "")
    with c4:
        conf_txt = f"{conf*100:.1f}%" if conf is not None else "—"
        kpi_card("Avg Confidence", conf_txt,
                 "across all predictions" if conf is not None else "no predictions yet")

    st.markdown("")

    # ---- Monitoring status ----
    if total:
        latest = st.session_state.history[-1]
        status_msg = f"Monitoring active — last activity: {CLASS_DISPLAY.get(latest['label'], latest['label'])} at {latest['time']}"
        _html(f"""
        <div class='sf-card'>
            <div class='sf-card-title'>Monitoring Status</div>
            <div style='font-size:15px;color:var(--text);'>● {status_msg}</div>
            <div class='sf-note' style='margin-top:6px;'>Session-driven analytics update live as images and videos are analyzed.</div>
        </div>
        """)
    else:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Monitoring Status</div>
            <div style='font-size:15px;color:var(--text);'>● Ready for analysis</div>
            <div class='sf-note' style='margin-top:6px;'>Upload an image or video to begin monitoring. Analytics and fall alerts will populate from real model predictions.</div>
        </div>
        """)

    st.markdown("")

    # ---- System explanation + pipeline ----
    left, right = st.columns([1.05, 1])
    with left:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>About SafeFall AI</div>
            <div style='font-size:14px;line-height:1.6;color:#cbd5e1;'>
                SafeFall AI uses human pose estimation and activity classification
                to identify potentially dangerous falls and monitor common movement
                states. The system extracts 17 body keypoints per frame, derives a
                55-dimensional geometric feature vector, and classifies the activity
                into one of five states — fall, walking, sitting, standing, or
                normal — generating an emergency alert when a fall is detected.
            </div>
            <div class='sf-note' style='margin-top:10px;'>
                Academic / prototype AI monitoring system. Not a certified medical device.
            </div>
        </div>
        """)
    with right:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Detection Pipeline</div>
        """)
        _html(pipeline_flow([
            ("01", "Frame", "image / video"),
            ("02", "Pose Estimation", "YOLO11n-Pose"),
            ("03", "17 Keypoints", "skeleton"),
            ("04", "55 Features", "geometry"),
            ("05", "Classification", "Random Forest"),
            ("06", "Fall Detection", "alert"),
        ]))
        _html("</div>")

    st.markdown("")

    # ---- Activity distribution + recent activity ----
    a1, a2 = st.columns([1, 1.15])
    with a1:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution</div>")
        if total:
            fig = activity_distribution_chart()
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            _html(empty_state("📊", "No activity data yet.<br/>Upload an image or video to begin monitoring."))
        _html("</div>")

    with a2:
        _html("<div class='sf-card'><div class='sf-card-title'>Recent Activity</div>")
        if st.session_state.history:
            _html(timeline_html(st.session_state.history))
        else:
            _html(empty_state("—", "No predictions logged yet."))
        _html("</div>")


# ============================================================
# PAGE: IMAGE ANALYSIS
# ============================================================

def page_image_analysis(yolo_model, clf, scaler, label_encoder):
    section_title("Image Analysis", "SINGLE FRAME")
    _html("""
    <div class='sf-note' style='margin-bottom:14px;'>
        Upload an image to run pose estimation and activity classification.
        The pipeline: input image → YOLO skeleton → 55-dim features → activity label & confidence.
    </div>
    """)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    if uploaded is None:
        _html(empty_state("🖼️", "Ready for monitoring.<br/>Upload a JPG, JPEG, or PNG image to analyze a single frame."))
        return

    # ---- Decode image ----
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        _html("<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Unable to read image</div><div class='meta'>The file may be corrupt or use an unsupported format. Please try another image.</div></div>")
        return

    # ---- Run the REAL pipeline ----
    try:
        label, confidence, annotated, prob_breakdown = predict_frame(
            frame_bgr, yolo_model, clf, scaler, label_encoder
        )
    except Exception as e:
        _html(f"<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Analysis error</div><div class='meta'>Pose estimation or classification failed on this image. Try a clearer frame.</div></div>")
        return

    # ---- Workflow visual hierarchy: input → pose → classification ----
    _html("<div class='sf-card' style='margin-bottom:14px;'><div class='sf-card-title'>Input Image</div></div>")
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_img, use_container_width=True)
        _html("<div class='sf-note' style='text-align:center;'>Raw input frame</div>")
    with c2:
        st.image(annotated, use_container_width=True)
        _html("<div class='sf-note' style='text-align:center;'>Pose estimation — YOLO11n-Pose skeleton overlay</div>")

    _html("<div style='text-align:center;color:var(--accent);font-size:20px;margin:6px 0 10px 0;'>↓</div>")

    # ---- Prediction card ----
    left, right = st.columns([1, 1.2])
    with left:
        _html(prediction_hero(label, confidence))
        if label is not None:
            _html("<div style='margin-top:12px;'>" + confidence_meter(confidence, CLASSES_COLORS.get(label, ACCENT)) + "</div>")
            _html(f"<div class='sf-note' style='text-align:center;margin-top:4px;'>Confidence meter — real classifier output</div>")
    with right:
        if label is not None:
            _html("<div class='sf-card'><div class='sf-card-title'>Class Probability Breakdown</div>")
            confidence_breakdown_bars(prob_breakdown)
            _html("<div class='sf-note' style='margin-top:8px;'>Real per-class probabilities from the Random Forest classifier.</div></div>")
        else:
            _html("<div class='sf-card'><div class='sf-card-title'>Class Probability Breakdown</div>" + empty_state("—", "No confident person detection — no classification performed.") + "</div>")

    # ---- Fall / normal alert treatment ----
    st.markdown("")
    if label is None:
        _html("<div class='sf-pred warn' style='text-align:left;'>⚠ No confident person detection in this image — try a clearer frame with a visible subject.</div>")
    else:
        log_prediction(uploaded.name, label, confidence)
        if label == "fall" and confidence >= FALL_DEFAULT_GATE:
            _html(fall_alert_banner(None, confidence))
        elif label == "fall":
            _html(f"""
            <div class='sf-pred warn' style='text-align:left;'>
                <div class='sf-pred-label'>Caution</div>
                <div class='sf-pred-activity' style='font-size:18px;color:#fcd34d;'>Possible fall — below alert threshold</div>
                <div class='sf-pred-conf'>Flagged at {confidence*100:.1f}% confidence (threshold {FALL_DEFAULT_GATE*100:.0f}%). No emergency alert generated.</div>
            </div>
            """)
        else:
            _html(f"""
            <div class='sf-pred normal' style='text-align:left;'>
                <div class='sf-pred-label'>✓ Normal Activity</div>
                <div class='sf-pred-activity' style='font-size:18px;color:#86efac;'>{CLASS_DISPLAY.get(label, label.title())}</div>
                <div class='sf-pred-conf'>No fall detected — activity classified as normal movement.</div>
            </div>
            """)


# ============================================================
# PAGE: VIDEO MONITORING
# ============================================================

def page_video_monitoring(yolo_model, clf, scaler, label_encoder):
    section_title("Video Monitoring", "LIVE CONSOLE")

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sample_every_n = st.slider(
            "Sampling interval (every Nth frame)",
            5, 60, 15,
            help="Higher = faster, coarser analysis. Lower = denser sampling.",
        )
    with ctrl2:
        fall_confidence_gate = st.slider(
            "Fall confidence threshold",
            0.30, 0.90, FALL_DEFAULT_GATE, 0.01,
            format="%.0f%%",
            help="Higher threshold = fewer but more confident alerts.",
        )
    with ctrl3:
        event_cooldown_s = st.slider(
            "Event cooldown (seconds)",
            1, 10, FALL_DEFAULT_COOLDOWN,
            help="Merges repeated fall frames into one event.",
        )

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "m4v", "mkv"],
        key="vid_uploader",
        label_visibility="collapsed",
    )
    _html("<div class='sf-note' style='margin-top:2px;'>Supported formats: AVI &nbsp;•&nbsp; MP4 &nbsp;•&nbsp; MOV &nbsp;•&nbsp; M4V &nbsp;•&nbsp; MKV</div>")

    if uploaded is None:
        _html(empty_state("🎬", "Ready for monitoring.<br/>Upload an AVI, MP4, or MOV video to run the live monitoring console."))
        return

    # ---- Analyze button so expensive inference is deliberate ----
    file_size_mb = uploaded.size / (1024 * 1024)
    file_info = f"File: {uploaded.name} &nbsp;·&nbsp; {file_size_mb:.1f} MB"
    st.markdown(f"**{file_info}**")
    if file_size_mb > 60:
        _html("<div class='sf-note'>ℹ Larger videos may take longer to analyze. Frames are sampled and downscaled to 480px before inference for efficiency.</div>")

    if not st.button("▶ Analyze Video", type="primary", use_container_width=False):
        _html("<div class='sf-note'>Configure the sampling interval, threshold, and cooldown above, then click <b>Analyze Video</b> to begin.</div>")
        return

    # ---- Stream upload to a safe temp file ----
    tmp_path = stream_upload_to_tmp(uploaded, "safefall_upload")
    cap, total_frames, fps = open_video_robustly(tmp_path)

    if cap is None or total_frames <= 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        _html("<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Unable to read this video</div><div class='meta'>The file may use an unsupported codec, be corrupt, or contain no readable frames. Try another file or format.</div></div>")
        return

    # ---- Console layout: pose visualizer | live prediction ----
    _html("<div class='sf-card' style='margin-bottom:12px;'><div class='sf-card-title'>Live Monitoring Console</div></div>")
    console_left, console_right = st.columns([1.35, 1])
    with console_left:
        preview_slot = st.empty()  # live annotated skeleton updates here
        _html("<div class='sf-note' style='text-align:center;'>Pose visualizer — skeleton follows the subject frame-by-frame</div>")
    with console_right:
        pred_slot = st.empty()      # live prediction card
        progress_slot = st.empty()  # processing panel
        status_slot = st.empty()    # status text

    fall_events: list[dict] = []
    last_fall_frame = None
    frame_idx = 0
    processed = 0
    hit_frame_cap = False
    duration_s = total_frames / fps if fps else 0.0

    try:
        while True:
            ok_read, frame = cap.read()
            if not ok_read:
                break
            if frame_idx % sample_every_n == 0:
                if processed >= MAX_FRAMES_PER_VIDEO:
                    hit_frame_cap = True
                    break

                small_frame = downscale_frame(frame)
                del frame  # drop the full-resolution copy immediately

                label, confidence, annotated, _ = predict_frame(
                    small_frame, yolo_model, clf, scaler, label_encoder
                )

                if label is not None:
                    log_prediction(uploaded.name, label, confidence, round(frame_idx / fps, 2))
                    preview_slot.image(
                        annotated,
                        caption=f"Frame {frame_idx} ({frame_idx / fps:.1f}s) — {CLASS_DISPLAY.get(label, label.title())} ({confidence*100:.0f}%)",
                        use_container_width=True,
                    )
                    pred_slot.markdown(prediction_hero(label, confidence), unsafe_allow_html=True)
                    progress_slot.markdown(f"""
                    <div class='sf-card'>
                        <div class='sf-card-title'>Analysis In Progress</div>
                        <div style='font-size:14px;color:var(--text);'>Processing frame <b>{frame_idx}</b> / {total_frames}</div>
                        <div class='sf-note' style='margin-top:6px;'>Activity: <b style='color:{CLASSES_COLORS.get(label, ACCENT)};'>{CLASS_DISPLAY.get(label, label.title())}</b></div>
                        <div class='sf-note'>Confidence: {confidence*100:.1f}%</div>
                        <div class='sf-note'>Frames analyzed: {processed + 1}</div>
                        <div class='sf-note'>Current status: Monitoring…</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Fall detection with confidence gate + cooldown grouping
                    if label == "fall" and confidence >= fall_confidence_gate:
                        within_cooldown = (
                            last_fall_frame is not None
                            and (frame_idx - last_fall_frame) / fps <= event_cooldown_s
                        )
                        if within_cooldown:
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
                else:
                    # No person detected — keep console informative
                    progress_slot.markdown(f"""
                    <div class='sf-card'>
                        <div class='sf-card-title'>Analysis In Progress</div>
                        <div style='font-size:14px;color:var(--text);'>Processing frame <b>{frame_idx}</b> / {total_frames}</div>
                        <div class='sf-note' style='margin-top:6px;'>Activity: <b style='color:#9ca3af;'>No person detected</b></div>
                        <div class='sf-note'>Frames analyzed: {processed + 1}</div>
                        <div class='sf-note'>Current status: Scanning…</div>
                    </div>
                    """, unsafe_allow_html=True)

                del annotated
                processed += 1

                if processed % GC_EVERY_N_FRAMES == 0:
                    gc.collect()
            else:
                del frame
            frame_idx += 1
            if total_frames:
                status_slot.progress(min(frame_idx / total_frames, 1.0))
                status_slot.caption(f"Processed {processed} sampled frames / {frame_idx} total frames · Memory: {_mem_txt()}")
    except MemoryError:
        status_slot.empty()
        _html("<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Memory pressure</div><div class='meta'>Ran low on memory partway through. Try a shorter clip, a larger sampling interval, or run locally.</div></div>")
    finally:
        cap.release()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()

    # ---- Persist fall events to session for Analytics page ----
    if fall_events:
        st.session_state.fall_events = fall_events

    # ================= RESULTS SUMMARY =================
    st.markdown("")
    section_title("Analysis Results", "SUMMARY")

    avg_conf = float(np.mean([h["confidence"] for h in st.session_state.history[-processed:]])) if processed else 0.0
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        kpi_card("Frames Analyzed", processed)
    with rc2:
        kpi_card("Fall Events", len(fall_events), variant="danger" if fall_events else "")
    with rc3:
        kpi_card("Avg Confidence", f"{avg_conf*100:.1f}%" if processed else "—")
    with rc4:
        kpi_card("Video Duration", f"{duration_s:.1f}s")

    # ---- Primary result ----
    st.markdown("")
    if fall_events:
        latest_fall = max(fall_events, key=lambda e: e["confidence"])
        _html(fall_alert_banner(latest_fall["start_timestamp_s"], latest_fall["confidence"], latest_fall["frame"]))
    else:
        _html("""
        <div class='sf-pred normal' style='text-align:left;'>
            <div class='sf-pred-label'>✓ Safe</div>
            <div class='sf-pred-activity' style='font-size:20px;color:#86efac;'>No potential fall events detected</div>
            <div class='sf-pred-conf'>All sampled frames classified as normal activity or below the fall-confidence threshold.</div>
        </div>
        """)

    # ---- Frame cap notice (contextual, not scary) ----
    if hit_frame_cap:
        _html(f"<div class='sf-note' style='margin-top:10px;'>ℹ Sampling stopped at {MAX_FRAMES_PER_VIDEO} frames (cloud stability cap). Increase the sampling interval to cover more of a long video within the same limit.</div>")

    # ---- Fall event table + timeline ----
    if fall_events:
        st.markdown("")
        col_t, col_tbl = st.columns([1, 1.2])
        with col_t:
            _html("<div class='sf-card'><div class='sf-card-title'>Fall Event Timeline</div>")
            tl_events = [
                {"time": f"00:{e['start_timestamp_s']:05.1f}", "label": "fall", "confidence": e["confidence"]}
                for e in fall_events
            ]
            _html(timeline_html(tl_events))
            _html("</div>")
        with col_tbl:
            _html("<div class='sf-card'><div class='sf-card-title'>Emergency Alert Status — Potential Fall Events</div>")
            df_events = pd.DataFrame(fall_events)
            df_events["Event"] = ["Fall"] * len(df_events)
            df_events["Timestamp (s)"] = df_events["start_timestamp_s"]
            df_events["Confidence"] = df_events["confidence"].apply(lambda c: f"{c*100:.1f}%")
            df_events["Duration (s)"] = (df_events["end_timestamp_s"] - df_events["start_timestamp_s"]).round(2)
            st.dataframe(
                df_events[["Event", "Timestamp (s)", "Confidence", "Duration (s)"]],
                use_container_width=True,
                hide_index=True,
            )
            _html("<div class='sf-note'>Emergency alert generated — potential fall event detected. No emergency service was automatically contacted (prototype monitoring system).</div></div>")
    else:
        st.markdown("")
        _html("<div class='sf-card'><div class='sf-card-title'>Fall Event Log</div>" + empty_state("✓", "No potential fall events detected in this video.") + "</div>")

    # ---- Post-analysis activity distribution ----
    st.markdown("")
    col_dist, col_hist = st.columns([1, 1.15])
    with col_dist:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution (this session)</div>")
        if st.session_state.history:
            fig = activity_distribution_chart()
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            _html(empty_state("📊", "No activity data yet."))
        _html("</div>")
    with col_hist:
        _html("<div class='sf-card'><div class='sf-card-title'>Recent Predictions</div>")
        if st.session_state.history:
            _html(timeline_html(st.session_state.history))
        else:
            _html(empty_state("—", "No predictions logged."))
        _html("</div>")


def _mem_txt() -> str:
    mem = current_memory_mb()
    return f"{mem:.0f} MB" if mem is not None else "n/a"


# ============================================================
# PAGE: ANALYTICS
# ============================================================

def page_analytics():
    section_title("Analytics", "SESSION INSIGHTS")
    counts = activity_counts()
    total = sum(counts.values())

    # ---- KPI strip ----
    cols = st.columns(6)
    metrics = [
        ("Total", total, ""),
        ("Fall", counts["fall"], "danger" if counts["fall"] else ""),
        ("Walking", counts["walking"], ""),
        ("Sitting", counts["sitting"], ""),
        ("Standing", counts["standing"], ""),
        ("Normal", counts["normal"], "success" if counts["normal"] else ""),
    ]
    for col, (lbl, val, var) in zip(cols, metrics):
        with col:
            kpi_card(lbl, val, variant=var)

    st.markdown("")

    # ---- Activity distribution + confidence summary ----
    a1, a2 = st.columns([1.1, 1])
    with a1:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution</div>")
        if total:
            fig = activity_distribution_chart()
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            _html(empty_state("📊", "No activity data yet.<br/>Upload an image or video to begin monitoring."))
        _html("</div>")
    with a2:
        _html("<div class='sf-card'><div class='sf-card-title'>Confidence Summary</div>")
        if total:
            confs = [h["confidence"] for h in st.session_state.history]
            avg = float(np.mean(confs))
            mn = float(np.min(confs))
            mx = float(np.max(confs))
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Average", f"{avg*100:.1f}%")
            cm2.metric("Minimum", f"{mn*100:.1f}%")
            cm3.metric("Maximum", f"{mx*100:.1f}%")
            _html(confidence_meter(avg, ACCENT))
            _html("<div class='sf-note' style='margin-top:6px;'>Average confidence across all session predictions.</div>")
        else:
            _html(empty_state("—", "No confidence data yet."))
        _html("</div>")

    st.markdown("")

    # ---- Fall event table ----
    section_title("Fall Events", "ALERT LOG")
    if st.session_state.fall_events:
        df_events = pd.DataFrame(st.session_state.fall_events)
        df_events["Event"] = ["Fall"] * len(df_events)
        df_events["Timestamp (s)"] = df_events["start_timestamp_s"]
        df_events["Confidence"] = df_events["confidence"].apply(lambda c: f"{c*100:.1f}%")
        df_events["Duration (s)"] = (df_events["end_timestamp_s"] - df_events["start_timestamp_s"]).round(2)
        st.dataframe(
            df_events[["Event", "Timestamp (s)", "Confidence", "Duration (s)"]],
            use_container_width=True,
            hide_index=True,
        )
        _html("<div class='sf-note'>Emergency alert generated — potential fall event detected. Prototype monitoring system; no emergency service contacted automatically.</div>")
    else:
        _html("<div class='sf-card'>" + empty_state("✓", "No potential fall events detected yet.") + "</div>")

    # ---- History table (bounded) ----
    st.markdown("")
    section_title("Activity History", f"LAST {HISTORY_CAP} RECORDS")
    if st.session_state.history:
        df = history_df()
        df_disp = df.copy()
        df_disp["Activity"] = df_disp["label"].map(lambda l: CLASS_DISPLAY.get(l, l.title()))
        df_disp["Confidence"] = df_disp["confidence"].apply(lambda c: f"{c*100:.1f}%")
        df_disp["Source"] = df_disp["source"]
        df_disp["Timestamp"] = df_disp["time"]
        st.dataframe(
            df_disp[["Timestamp", "Source", "Activity", "Confidence"]],
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        _html(f"<div class='sf-note'>History capped at {HISTORY_CAP} most recent records to bound session memory.</div>")
    else:
        _html("<div class='sf-card'>" + empty_state("—", "No activity history yet. Upload an image or video to start logging predictions.") + "</div>")


# ============================================================
# PAGE: MODEL INFORMATION
# ============================================================

def page_model_information():
    section_title("Model Information", "ARCHITECTURE")

    info = load_model_info()

    # ---- Architecture cards ----
    a1, a2, a3 = st.columns(3)
    with a1:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Pose Estimation</div>
            <div style='font-size:18px;font-weight:800;color:var(--text);'>YOLO11n-Pose</div>
            <div class='sf-note' style='margin-top:8px;line-height:1.5;'>
                Ultralytics YOLO11n-Pose detects 17 COCO body keypoints per
                person — shoulders, elbows, wrists, hips, knees, ankles, and
                facial landmarks — forming the skeleton used for activity
                inference.
            </div>
            <div style='margin-top:10px;'><span class='sf-tag'>17 keypoints</span></div>
        </div>
        """)
    with a2:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Classification</div>
            <div style='font-size:18px;font-weight:800;color:var(--text);'>Random Forest</div>
            <div class='sf-note' style='margin-top:8px;line-height:1.5;'>
                A Random Forest classifier (200 trees, balanced class weights)
                predicts the activity from the geometric feature vector. Five
                activity classes:
            </div>
            <div style='margin-top:8px;'>
                <span class='sf-tag' style='color:#fca5a5;border-color:rgba(239,68,68,0.4);background:rgba(239,68,68,0.10);'>Fall</span>
                <span class='sf-tag'>Walking</span>
                <span class='sf-tag'>Sitting</span>
                <span class='sf-tag'>Standing</span>
                <span class='sf-tag'>Normal</span>
            </div>
        </div>
        """)
    with a3:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Feature Extraction</div>
            <div style='font-size:18px;font-weight:800;color:var(--text);'>55-Dim Vector</div>
            <div class='sf-note' style='margin-top:8px;line-height:1.5;'>
                Normalized (x, y) for all 17 keypoints, their confidences,
                aspect ratio, torso inclination angle, average knee angle,
                and knee-angle asymmetry — a scale-invariant geometric
                representation, not raw pixels.
            </div>
            <div style='margin-top:10px;'><span class='sf-tag'>scale-invariant</span></div>
        </div>
        """)

    st.markdown("")

    # ---- Pipeline diagram ----
    _html("<div class='sf-card'><div class='sf-card-title'>Conceptual Pipeline</div>")
    _html(pipeline_flow([
        ("01", "Frame", "image / video"),
        ("02", "Pose Estimation", "YOLO11n-Pose"),
        ("03", "Keypoints", "17 landmarks"),
        ("04", "Feature Extraction", "55 features"),
        ("05", "Classification", "Random Forest"),
        ("06", "Fall Detection", "alert"),
    ]))
    _html("</div>")

    st.markdown("")

    # ---- Inference info ----
    i1, i2 = st.columns(2)
    with i1:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Inference</div>
            <div style='font-size:15px;color:var(--text);line-height:1.7;'>
                ● CPU-compatible inference<br/>
                ● No GPU / CUDA required<br/>
                ● Cached model loading (loaded once per session)<br/>
                ● Frames downscaled to 480px before inference<br/>
                ● Periodic garbage collection during video loops
            </div>
        </div>
        """)
    with i2:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Dataset</div>
            <div style='font-size:15px;color:var(--text);line-height:1.7;'>
                ● Le2i Fall Detection Dataset<br/>
                ● 6 environments: Home, Office, Coffee room, Lecture room<br/>
                ● Surveillance video with fall annotations<br/>
                ● 70% train / 15% validation / 15% test (stratified)
            </div>
        </div>
        """)

    # ================= EVALUATION / ANALYTICS =================
    st.markdown("")
    section_title("Evaluation Metrics", "REAL RESULTS")

    if info is None:
        _html("<div class='sf-card'>" + empty_state("📉", "Evaluation artifact unavailable.<br/>Run train_classifier.py to generate model_info.json and evaluation charts.") + "</div>")
    else:
        tm = info.get("test_metrics", {})
        winner = info.get("winner", "—")

        # ---- Real metric cards ----
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Accuracy", f"{tm.get('accuracy', 0)*100:.1f}%", "test set", variant="success")
        with m2:
            kpi_card("Precision", f"{tm.get('precision', 0)*100:.1f}%", "test set")
        with m3:
            kpi_card("Recall", f"{tm.get('recall', 0)*100:.1f}%", "test set")
        with m4:
            kpi_card("F1 Score", f"{tm.get('f1', 0)*100:.1f}%", "test set")

        st.markdown("")
        _html(f"""
        <div class='sf-card' style='margin-bottom:14px;'>
            <div class='sf-card-title'>Winning Model</div>
            <div style='font-size:16px;color:var(--text);'>Selected classifier: <b style='color:var(--accent);'>{winner}</b> &nbsp;(highest macro-F1 on validation split)</div>
        </div>
        """)

        # ---- Real evaluation charts (only if files exist) ----
        cm_path = SCREENSHOTS_DIR / "confusion_matrix.png"
        cmp_path = SCREENSHOTS_DIR / "model_comparison.png"
        loss_path = SCREENSHOTS_DIR / "mlp_loss_curve.png"

        ec1, ec2 = st.columns(2)
        with ec1:
            _html("<div class='sf-card'><div class='sf-card-title'>Confusion Matrix (Test Set)</div>")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                _html(empty_state("—", "Confusion matrix artifact unavailable."))
            _html("</div>")
        with ec2:
            _html("<div class='sf-card'><div class='sf-card-title'>Model Comparison — Random Forest vs MLP</div>")
            if cmp_path.exists():
                st.image(str(cmp_path), use_container_width=True)
            else:
                _html(empty_state("—", "Model comparison artifact unavailable."))
            _html("</div>")

        st.markdown("")
        _html("<div class='sf-card' style='margin-bottom:14px;'><div class='sf-card-title'>MLP Training Loss Curve</div>")
        if loss_path.exists():
            st.image(str(loss_path), use_container_width=True)
        else:
            _html(empty_state("—", "Loss curve artifact unavailable."))
        _html("</div>")

        # ---- Validation metrics detail (REAL, from model_info.json) ----
        with st.expander("Detailed validation metrics (per model)"):
            vm = info.get("val_metrics", {})
            if vm:
                rows = []
                for model_name, m in vm.items():
                    rows.append({
                        "Model": model_name,
                        "Accuracy": f"{m.get('accuracy',0)*100:.1f}%",
                        "Precision": f"{m.get('precision',0)*100:.1f}%",
                        "Recall": f"{m.get('recall',0)*100:.1f}%",
                        "F1": f"{m.get('f1',0)*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                _html(empty_state("—", "No validation metrics available."))

        # ---- Sample predictions (REAL screenshots) ----
        pred_dir = SCREENSHOTS_DIR / "predictions"
        if pred_dir.exists():
            st.markdown("")
            _html("<div class='sf-card'><div class='sf-card-title'>Sample Predictions — Held-Out Test Set</div>")
            pred_files = sorted(pred_dir.glob("*.png"))[:8]
            if pred_files:
                pcols = st.columns(4)
                for i, p in enumerate(pred_files):
                    with pcols[i % 4]:
                        st.image(str(p), use_container_width=True, caption=p.stem)
            else:
                _html(empty_state("—", "No sample prediction screenshots available."))
            _html("</div>")

    # ---- Deployment challenges (rubric: discuss real-world challenges) ----
    st.markdown("")
    with st.expander("Real-world deployment challenges & future improvements"):
        _html("""
        <div style='font-size:14px;line-height:1.7;color:#cbd5e1;'>
            <b>Challenges observed:</b> lighting variation across the six Le2i
            environments, camera angle differences, partial occlusions, similar
            body postures (e.g. sitting vs. a low fall), and occasional false
            fall detections when the torso angle is borderline.<br/><br/>
            <b>Future improvements:</b> adding more activity videos per class,
            improving pose-estimation confidence under low light, reducing false
            alerts with temporal smoothing, supporting real-time CCTV feeds, and
            periodic retraining with new annotated healthcare data.
        </div>
        """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    init_state()

    # ---- Load models (cached) ----
    yolo_load_error = None
    try:
        yolo_model = load_yolo()
        yolo_ok = True
    except Exception as e:
        yolo_model = None
        yolo_ok = False
        yolo_load_error = str(e)

    clf, scaler, label_encoder = load_classifier()
    clf_ok = clf is not None
    model_ok = clf_ok and yolo_ok

    # ---- Inject design system ----
    _html(_design_css())

    # ---- Brand header ----
    brand_header(online=model_ok)

    # ---- Sidebar ----
    render_sidebar(model_ok, clf_ok, yolo_ok)

    # ---- Hard stop if classifier artifacts are missing ----
    if clf is None:
        _html("""
        <div class='sf-alert-fall'>
            <div class='head'>⚠ Model artifacts not found</div>
            <div class='meta'>No trained model found at <code>fa2_outputs/models/</code>. Run <code>python train_classifier.py</code> first, then restart the app.</div>
        </div>
        """)
        st.stop()

    if yolo_model is None:
        _html(f"""
        <div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'>
            <div class='head' style='color:#fcd34d;'>⚠ Pose engine unavailable</div>
            <div class='meta'>YOLO11n-Pose could not be loaded. {yolo_load_error or ''}</div>
        </div>
        """)
        st.stop()

    # ---- Main tabs ----
    tab_overview, tab_image, tab_video, tab_analytics, tab_model = st.tabs([
        "▣ Overview",
        "🖼️ Image Analysis",
        "🎬 Video Monitoring",
        "📊 Analytics",
        "ℹ️ Model Information",
    ])

    with tab_overview:
        page_overview()
    with tab_image:
        page_image_analysis(yolo_model, clf, scaler, label_encoder)
    with tab_video:
        page_video_monitoring(yolo_model, clf, scaler, label_encoder)
    with tab_analytics:
        page_analytics()
    with tab_model:
        page_model_information()


if __name__ == "__main__":
    main()
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
import gc

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from PIL import Image

from pose_utils import image_to_feature

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("./fa2_outputs/models")
MAX_FRAME_DIMENSION = 480      # every frame is downscaled to this before
                                 # pose inference, regardless of source
                                 # video resolution — caps memory/CPU per
                                 # frame on Streamlit Cloud's ~1GB RAM cap
MAX_FRAMES_PER_VIDEO = 40       # hard ceiling on total inference calls per
                                 # video upload, independent of the slider
GC_EVERY_N_FRAMES = 5           # force garbage collection periodically —
                                 # torch/ultralytics don't always release
                                 # memory promptly between calls in a loop
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


def current_memory_mb():
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def downscale_frame(frame_bgr, max_dim=MAX_FRAME_DIMENSION):
    """Shrinks the frame before it ever reaches YOLO/feature extraction.
    Keeps memory and CPU cost bounded regardless of how large the source
    video's resolution is."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame_bgr
    scale = max_dim / longest
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))


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
            file_size_mb = uploaded.size / (1024 * 1024)
            st.caption(f"Uploaded file: {file_size_mb:.1f} MB")
            if file_size_mb > 60:
                st.warning(
                    "Large video detected. SafeFall AI will sample frames and resize them before AI inference to keep processing efficient."
                )

            tmp_path = Path(f"./_tmp_{uploaded.name}")
            # Stream the upload to disk in chunks instead of materializing the
            # whole file as one big bytes object in memory first.
            with open(tmp_path, "wb") as f:
                for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
                    f.write(chunk)

            cap = cv2.VideoCapture(str(tmp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            progress = st.progress(0.0)
            status = st.empty()
            mem_status = st.empty() if _HAS_PSUTIL else None
            fall_events = []          # de-duplicated events shown to the user
            last_fall_frame = None    # for cooldown grouping
            frame_idx = 0
            processed = 0
            hit_frame_cap = False

            preview_slot = st.empty()

            try:
                while True:
                    ok_read, frame = cap.read()
                    if not ok_read:
                        break
                    if frame_idx % sample_every_n == 0:
                        if processed >= MAX_FRAMES_PER_VIDEO:
                            hit_frame_cap = True
                            break

                        small_frame = downscale_frame(frame)
                        del frame  # drop the full-resolution copy immediately

                        label, confidence, annotated, _ = predict_frame(
                            small_frame, yolo_model, clf, scaler, label_encoder
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
                        del annotated
                        processed += 1

                        if processed % GC_EVERY_N_FRAMES == 0:
                            gc.collect()
                            if mem_status is not None:
                                mem_status.caption(f"Memory in use: {current_memory_mb():.0f} MB")
                    else:
                        del frame
                    frame_idx += 1
                    if total_frames:
                        progress.progress(min(frame_idx / total_frames, 1.0))
                    status.text(f"Processed {processed} sampled frames / {frame_idx} total frames")

            except MemoryError:
                st.error(
                    "Ran out of memory partway through this video. Try a shorter "
                    "clip, a larger 'process every Nth frame' value, or test "
                    "locally where memory isn't capped at ~1GB."
                )
            finally:
                cap.release()
                tmp_path.unlink(missing_ok=True)
                gc.collect()

            if hit_frame_cap:
                st.info(
                    f"Stopped after {MAX_FRAMES_PER_VIDEO} sampled frames (safety "
                    f"limit for the cloud deployment) — the video may be longer "
                    f"than what was scanned. Increase 'process every Nth frame' "
                    f"to cover more of a long video within the same cap."
                )

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
