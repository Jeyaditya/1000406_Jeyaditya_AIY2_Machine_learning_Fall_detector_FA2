from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pose_utils import image_to_feature

try:
    import joblib
except ImportError:
    joblib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("./fa2_outputs/models")
SCREENSHOTS_DIR = Path("./fa2_outputs/screenshots")
INFO_PATH = MODEL_DIR / "model_info.json"

MAX_FRAME_DIMENSION = 480
MAX_FRAMES_PER_VIDEO = 300
GC_EVERY_N_FRAMES = 5
HISTORY_CAP = 200
FALL_DEFAULT_GATE = 0.55
FALL_DEFAULT_COOLDOWN = 3

ACCENT = "#22d3ee"
ACCENT_SOFT = "#0e7490"
DANGER = "#ef4444"
WARNING = "#f59e0b"
SUCCESS = "#22c55e"
SURFACE = "#111827"
BORDER = "#1f2937"

CLASSES_COLORS = {
    "fall": "#ef4444",
    "walking": "#22d3ee",
    "sitting": "#f59e0b",
    "standing": "#60a5fa",
    "normal": "#22c55e",
}

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
# DESIGN SYSTEM
# ============================================================

def _design_css() -> str:
    return textwrap.dedent("""
    <style>
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

    .sf-meter { height: 10px; border-radius: 999px; background: #1f2937; overflow: hidden; }
    .sf-meter > span { display:block; height:100%; border-radius:999px; transition: width .4s ease; }

    .sf-pipe { display:flex; flex-wrap:wrap; gap:10px; align-items:stretch; }
    .sf-step {
        flex:1 1 0; min-width: 120px; padding: 12px 12px; border-radius: 12px;
        background: var(--surface-2); border: 1px solid var(--border); text-align:center;
    }
    .sf-step .n { font-size: 11px; color: var(--accent); font-weight:700; letter-spacing:1px; }
    .sf-step .t { font-size: 13px; font-weight: 700; margin-top: 4px; color: var(--text); }
    .sf-step .d { font-size: 11px; color: var(--text-dim); margin-top: 3px; }
    .sf-arrow { display:flex; align-items:center; color: var(--accent); font-size: 18px; }

    .sf-empty {
        text-align:center; padding: 36px 18px; color: var(--text-dim);
        background: var(--surface-2); border: 1px dashed var(--border); border-radius: var(--radius);
    }
    .sf-empty .big { font-size: 30px; opacity:.5; margin-bottom: 8px; }

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

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .st-emotion-cache-1dp5ir5 { display: none; }

    .sf-note { font-size: 12px; color: var(--text-dim); }
    .sf-divider { height:1px; background: var(--border); margin: 14px 0; border:none; }
    .sf-tag { font-size:11px; padding:3px 9px; border-radius:999px; background:rgba(34,211,238,0.10); border:1px solid rgba(34,211,238,0.30); color:#a5f3fc; }
    </style>
    """)


# ============================================================
# UI HELPERS
# ============================================================

def _html(html_str: str) -> None:
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


def brand_header(online: bool = True) -> None:
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
    cls = kind if kind else ("" if ok else "danger")
    dot = "<span class='dot'></span>"
    return f"<span class='sf-pill {cls}'>{dot} {label}</span>"


def prediction_hero(label: str | None, confidence: float | None) -> str:
    if label is None:
        return (
            "<div class='sf-pred warn'>"
            "<div class='sf-pred-label'>Prediction</div>"
            "<div class='sf-pred-activity' style='font-size:22px;color:#fcd34d;'>No person detected</div>"
            "<div class='sf-pred-conf'>Pose engine could not confidently detect a subject in this frame.</div>"
            "</div>"
        )
    disp = CLASS_DISPLAY.get(label, label.title())
    if label == "fall" and (confidence or 0) >= FALL_DEFAULT_GATE:
        variant = "fall"
    elif label == "fall":
        variant = "warn"
    else:
        variant = "normal"
    color = CLASSES_COLORS.get(label, ACCENT)
    conf_txt = f"{(confidence or 0)*100:.1f}% confidence" if confidence is not None else "—"
    return (
        f"<div class='sf-pred {variant}'>"
        f"<div class='sf-pred-label'>Predicted Activity</div>"
        f"<div class='sf-pred-activity' style='color:{color};'>{disp}</div>"
        f"<div class='sf-pred-conf'>{conf_txt}</div>"
        f"</div>"
    )


def confidence_meter(confidence: float, color: str = ACCENT) -> str:
    pct = max(0.0, min(1.0, float(confidence))) * 100
    return f"<div class='sf-meter'><span style='width:{pct:.1f}%;background:{color};'></span></div>"


def fall_alert_banner(timestamp_s: float | None, confidence: float, frame: int | None = None) -> str:
    ts = f"{timestamp_s:.1f}s" if timestamp_s is not None else "—"
    fr = f"Frame {frame}" if frame is not None else ""
    return (
        f"<div class='sf-alert-fall'>"
        f"<div class='head'>⚠ FALL DETECTED</div>"
        f"<div class='meta'>Potential fall event detected at {ts} &nbsp;·&nbsp; Confidence: {confidence*100:.1f}% &nbsp;·&nbsp; {fr}</div>"
        f"<div class='meta' style='margin-top:8px;'>Emergency alert generated — caregiver notification flagged for review.</div>"
        f"</div>"
    )


def pipeline_flow(steps: list[tuple[str, str, str]]) -> str:
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
        ss.history = []
    if "fall_events" not in ss:
        ss.fall_events = []


def log_prediction(source: str, label: str, confidence: float, timestamp_s: float | None = None) -> None:
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "label": label,
        "confidence": float(confidence),
        "timestamp_s": timestamp_s,
    })
    if len(st.session_state.history) > HISTORY_CAP:
        st.session_state.history = st.session_state.history[-HISTORY_CAP:]


def merge_fall_events(new_events: list[dict]) -> list[dict]:
    seen = {
        (e.get("source"), e.get("frame"), e.get("start_timestamp_s"))
        for e in st.session_state.fall_events
    }
    for e in new_events:
        key = (e.get("source"), e.get("frame"), e.get("start_timestamp_s"))
        if key in seen:
            continue
        st.session_state.fall_events.append(dict(e))
        seen.add(key)
    return st.session_state.fall_events


def format_ts(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if seconds < 0:
        seconds = 0.0
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


# ============================================================
# MEMORY & PREDICTION HELPERS
# ============================================================

def current_memory_mb() -> float | None:
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def downscale_frame(frame_bgr, max_dim: int = MAX_FRAME_DIMENSION):
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame_bgr
    scale = max_dim / longest
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))


def predict_frame(frame_bgr, yolo_model, clf, scaler, label_encoder):
    feature, result = image_to_feature(yolo_model, frame_bgr, device=DEVICE)
    annotated = result.plot()[:, :, ::-1]

    if feature is None:
        return None, None, annotated, None

    feature_s = scaler.transform(feature.reshape(1, -1))
    probs = clf.predict_proba(feature_s)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    if confidence < 0.30:
        label = "normal"
    else:
        label = str(label_encoder.inverse_transform([pred_idx])[0])

    prob_breakdown = {str(k): float(v) for k, v in zip(label_encoder.classes_, probs.tolist())}
    return label, confidence, annotated, prob_breakdown


def history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=["time", "source", "label", "confidence", "timestamp_s"])
    return pd.DataFrame(st.session_state.history)


def activity_counts() -> dict:
    counts = {c: 0 for c in CLASS_DISPLAY_ORDER}
    for h in st.session_state.history:
        counts[h["label"]] = counts.get(h["label"], 0) + 1
    return counts


def avg_confidence() -> float | None:
    if not st.session_state.history:
        return None
    return float(np.mean([h["confidence"] for h in st.session_state.history]))


def activity_distribution_chart(data_counts=None, title_label="Predictions"):
    if not _HAS_MPL:
        return None
    counts = data_counts if data_counts is not None else activity_counts()
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
    ax.set_xlabel(title_label, color="#9ca3af", fontsize=11)
    for b, v in zip(bars, values):
        if v > 0:
            ax.text(b.get_width() + max(values) * 0.01 + 0.3, b.get_y() + b.get_height() / 2,
                    str(int(v)), va="center", color="#e5e7eb", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#1f2937", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def confidence_breakdown_bars(prob_breakdown: dict):
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


def stream_upload_to_tmp(uploaded, base_name: str) -> Path:
    try:
        uploaded.seek(0)
    except Exception:
        pass

    safe_ext = ""
    name = (uploaded.name or "").lower()
    for ext in (".avi", ".mp4", ".mov", ".m4v", ".mkv"):
        if name.endswith(ext):
            safe_ext = ext
            break
    if not safe_ext:
        safe_ext = ".avi"

    fd, tmp_name = tempfile.mkstemp(prefix=f"sf_{base_name}_", suffix=safe_ext)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
                f.write(chunk)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return tmp_path


def open_video_robustly(tmp_path: Path):
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
        st.caption("Navigation is via tabs in the main panel.")

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

        counts = activity_counts()
        total = sum(counts.values())
        c1, c2 = st.columns(2)
        c1.metric("Total predictions", total)
        c2.metric("Fall incidents", len(st.session_state.fall_events))
        st.caption("Inference device: **CPU** · No GPU required.")

        mem = current_memory_mb()
        if mem is not None:
            st.caption(f"Memory in use: **{mem:.0f} MB**")

        if st.button("↺ Reset session analytics", width="stretch"):
            st.session_state.history = []
            st.session_state.fall_events = []
            st.session_state.pop("last_image_analysis", None)
            st.rerun()


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview():
    section_title("Overview", "DASHBOARD")
    counts = activity_counts()
    total = sum(counts.values())
    fall_incidents = len(st.session_state.fall_events)
    fall_predictions = counts["fall"]
    nonfall_total = counts["normal"] + counts["walking"] + counts["sitting"] + counts["standing"]
    conf = avg_confidence()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Activities", total if total else 0,
                 "predictions this session" if total else "awaiting input")
    with c2:
        kpi_card("Fall Incidents", fall_incidents if fall_incidents else 0,
                 f"grouped events · {fall_predictions} fall predictions" if fall_predictions
                 else ("no incidents logged" if total else "none detected"),
                 variant="danger" if fall_incidents else "")
    with c3:
        kpi_card("Non-Fall Activities", nonfall_total if nonfall_total else 0,
                 "walking · sitting · standing · normal" if nonfall_total else "awaiting input",
                 variant="success" if nonfall_total else "")
    with c4:
        conf_txt = f"{conf*100:.1f}%" if conf is not None else "—"
        kpi_card("Avg Confidence", conf_txt,
                 "across all predictions" if conf is not None else "no predictions yet")

    st.markdown("")

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

    a1, a2 = st.columns([1, 1.15])
    with a1:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution</div>")
        if total:
            fig = activity_distribution_chart()
            if fig is not None:
                st.pyplot(fig, width="stretch")
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

    content_hash = None
    try:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
            hasher.update(chunk)
        uploaded.seek(0)
        content_hash = hasher.hexdigest()
    except Exception:
        content_hash = None

    cached = st.session_state.get("last_image_analysis") if content_hash else None
    if cached is not None and cached.get("hash") == content_hash:
        pil_img = cached["pil_img"]
        label = cached["label"]
        confidence = cached["confidence"]
        annotated = cached["annotated"]
        prob_breakdown = cached["prob_breakdown"]
        _html(
            "<div class='sf-note' style='margin-bottom:12px;'>"
            "ℹ This image was already analyzed in this session — showing stored result."
            "</div>"
        )
    else:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            _html("<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Unable to read image</div><div class='meta'>The file may be corrupt or use an unsupported format. Please try another image.</div></div>")
            return

        try:
            label, confidence, annotated, prob_breakdown = predict_frame(
                frame_bgr, yolo_model, clf, scaler, label_encoder
            )
        except Exception:
            _html("<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'><div class='head' style='color:#fcd34d;'>⚠ Analysis error</div><div class='meta'>Pose estimation or classification failed on this image. Try a clearer frame.</div></div>")
            return
        del frame_bgr

        if label is not None:
            log_prediction(uploaded.name, label, confidence)
        st.session_state.last_image_analysis = {
            "hash": content_hash,
            "name": uploaded.name,
            "pil_img": pil_img,
            "label": label,
            "confidence": confidence,
            "annotated": annotated,
            "prob_breakdown": prob_breakdown,
        }

    _html("<div class='sf-card' style='margin-bottom:14px;'><div class='sf-card-title'>Input Image</div></div>")
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_img, width="stretch")
        _html("<div class='sf-note' style='text-align:center;'>Raw input frame</div>")
    with c2:
        st.image(annotated, width="stretch")
        _html("<div class='sf-note' style='text-align:center;'>Pose estimation — YOLO11n-Pose skeleton overlay</div>")

    _html("<div style='text-align:center;color:var(--accent);font-size:20px;margin:6px 0 10px 0;'>↓</div>")

    left, right = st.columns([1, 1.2])
    with left:
        _html(prediction_hero(label, confidence))
        if label is not None:
            _html(f"<div style='margin-top:12px;'>{confidence_meter(confidence, CLASSES_COLORS.get(label, ACCENT))}</div>")
            _html("<div class='sf-note' style='text-align:center;margin-top:4px;'>Confidence meter — real classifier output</div>")
    with right:
        if label is not None:
            _html("<div class='sf-card'><div class='sf-card-title'>Class Probability Breakdown</div>")
            confidence_breakdown_bars(prob_breakdown)
            _html("<div class='sf-note' style='margin-top:8px;'>Real per-class probabilities from the Random Forest classifier.</div></div>")
        else:
            _html("<div class='sf-card'><div class='sf-card-title'>Class Probability Breakdown</div>" + empty_state("—", "No confident person detection — no classification performed.") + "</div>")

    st.markdown("")
    if label is None:
        _html("<div class='sf-pred warn' style='text-align:left;'>⚠ No confident person detection in this image — try a clearer frame with a visible subject.</div>")
    else:
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
# PAGE: VIDEO MONITORING (SYNCHRONIZED METRICS)
# ============================================================

def page_video_monitoring(yolo_model, clf, scaler, label_encoder):
    section_title("Video Monitoring", "LIVE CONSOLE")

    ctrl1, ctrl2, ctrl3 = st.columns(3)

    with ctrl1:
        sample_every_n = st.slider(
            "Sampling interval (every Nth frame)",
            2, 30, 6,
            help="Controls analysis density. Lower = smoother continuous monitoring of falls.",
        )

    with ctrl2:
        fall_gate_pct = st.slider(
            "Fall confidence threshold",
            30, 90, int(round(FALL_DEFAULT_GATE * 100)), 1,
            format="%d%%",
            help="Higher threshold = fewer but more confident alerts.",
        )
        fall_confidence_gate = fall_gate_pct / 100.0

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

    _html(
        "<div class='sf-note' style='margin-top:2px;'>"
        "Supported formats: AVI &nbsp;•&nbsp; MP4 &nbsp;•&nbsp; MOV &nbsp;•&nbsp; M4V &nbsp;•&nbsp; MKV"
        "</div>"
    )

    if uploaded is None:
        _html(empty_state("🎬", "Ready for monitoring.<br/>Upload an AVI, MP4, or MOV video to run the live monitoring console."))
        return

    file_size_mb = uploaded.size / (1024 * 1024)
    st.markdown(f"**File: {uploaded.name} &nbsp;·&nbsp; {file_size_mb:.1f} MB**")

    if not st.button("▶ Analyze Video", type="primary", width="content"):
        return

    tmp_path = stream_upload_to_tmp(uploaded, "safefall_upload")
    cap, total_frames, fps = open_video_robustly(tmp_path)

    if cap is None:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        _html(
            "<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'>"
            "<div class='head' style='color:#fcd34d;'>⚠ Unable to read this video</div>"
            "<div class='meta'>The file may use an unsupported codec, be corrupt, or contain no readable frames.</div>"
            "</div>"
        )
        return

    total_known = total_frames > 0
    duration_s = total_frames / fps if total_known and fps and fps > 0 else 0.0

    _html("<div class='sf-card' style='margin-bottom:12px;'><div class='sf-card-title'>Live Monitoring Console</div></div>")

    console_left, console_right = st.columns([1.35, 1])
    with console_left:
        preview_slot = st.empty()
        _html("<div class='sf-note' style='text-align:center;'>Pose visualizer — skeleton follows the subject frame-by-frame</div>")

    with console_right:
        pred_slot = st.empty()
        progress_slot = st.empty()
        status_slot = st.empty()

    # Synchronized per-video metrics
    this_video_history: list[dict] = []
    fall_events: list[dict] = []
    last_fall_frame = None
    frame_idx = 0
    processed = 0
    video_confidences: list[float] = []

    try:
        while True:
            ok_read, frame = cap.read()
            if not ok_read:
                break

            # Stride-based sampling (processes full video without artificial caps)
            if frame_idx % sample_every_n == 0:
                small_frame = downscale_frame(frame)
                del frame

                label, confidence, annotated, _ = predict_frame(
                    small_frame,
                    yolo_model,
                    clf,
                    scaler,
                    label_encoder,
                )

                frame_total_txt = f" / {total_frames}" if total_known else ""
                timestamp_s = frame_idx / fps if fps and fps > 0 else 0.0

                if label is not None:
                    display_frame = annotated
                    cap_text = f"Frame {frame_idx} ({timestamp_s:.1f}s) — {CLASS_DISPLAY.get(label, label.title())} ({confidence * 100:.0f}%)"

                    pred_entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "source": uploaded.name,
                        "label": label,
                        "confidence": float(confidence),
                        "timestamp_s": round(timestamp_s, 2),
                    }
                    this_video_history.append(pred_entry)
                    log_prediction(uploaded.name, label, confidence, round(timestamp_s, 2))
                    video_confidences.append(float(confidence))

                    if label == "fall" and confidence >= fall_confidence_gate:
                        within_cooldown = (
                            last_fall_frame is not None
                            and fps
                            and fps > 0
                            and ((frame_idx - last_fall_frame) / fps) <= event_cooldown_s
                        )

                        if within_cooldown:
                            fall_events[-1]["end_timestamp_s"] = round(timestamp_s, 2)
                            fall_events[-1]["confidence"] = max(
                                fall_events[-1]["confidence"],
                                confidence,
                            )
                        else:
                            fall_events.append({
                                "source": uploaded.name,
                                "frame": frame_idx,
                                "start_timestamp_s": round(timestamp_s, 2),
                                "end_timestamp_s": round(timestamp_s, 2),
                                "confidence": confidence,
                            })

                        last_fall_frame = frame_idx
                else:
                    display_frame = small_frame[:, :, ::-1]
                    cap_text = f"Frame {frame_idx} ({timestamp_s:.1f}s) — Scanning / No person detected"

                preview_slot.image(display_frame, caption=cap_text, width="stretch")

                # Live Hero display: keep displaying the confirmed fall incident once detected
                if fall_events:
                    top_ev = max(fall_events, key=lambda e: e["confidence"])
                    pred_slot.markdown(
                        prediction_hero("fall", top_ev["confidence"]),
                        unsafe_allow_html=True,
                    )
                elif label is not None:
                    pred_slot.markdown(
                        prediction_hero(label, confidence),
                        unsafe_allow_html=True,
                    )

                del annotated
                processed += 1

                if processed % GC_EVERY_N_FRAMES == 0:
                    gc.collect()

            else:
                del frame

            frame_idx += 1

            if total_known:
                status_slot.progress(min(frame_idx / total_frames, 1.0))
                status_slot.caption(
                    f"Processed {processed} sampled frames / {frame_idx} total frames · Memory: {_mem_txt()}"
                )
            else:
                status_slot.caption(
                    f"Processed {processed} sampled frames · {frame_idx} total frames read · Memory: {_mem_txt()}"
                )

    except MemoryError:
        status_slot.empty()
        _html(
            "<div class='sf-alert-fall' style='border-color:rgba(245,158,11,0.5);background:linear-gradient(135deg,rgba(245,158,11,0.16),rgba(17,24,39,0.6));'>"
            "<div class='head' style='color:#fcd34d;'>⚠ Memory pressure</div>"
            "<div class='meta'>Ran low on memory partway through. Try a shorter clip or run locally.</div>"
            "</div>"
        )
    finally:
        cap.release()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()
        progress_slot.empty()

    if fall_events:
        merge_fall_events(fall_events)

    status_slot.caption(f"✓ Video complete · {processed} frames analyzed across {frame_idx} total frames.")

    # ================= RESULTS SUMMARY (EXACT SYNCHRONIZATION) =================
    st.markdown("")
    section_title("Analysis Results", "SUMMARY")

    # Lock Hero Card to the highest confidence incident of this video
    if fall_events:
        peak_fall = max(fall_events, key=lambda e: e["confidence"])
        pred_slot.markdown(
            prediction_hero("fall", peak_fall["confidence"]),
            unsafe_allow_html=True,
        )

    # Activity counts isolated strictly to this video
    vid_counts = {c: 0 for c in CLASS_DISPLAY_ORDER}
    for h in this_video_history:
        vid_counts[h["label"]] = vid_counts.get(h["label"], 0) + 1

    total_valid_frames = len(this_video_history)
    avg_conf = float(np.mean(video_confidences)) if video_confidences else 0.0

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        kpi_card("Frames Analyzed", total_valid_frames, f"{processed} sampled frames processed")
    with rc2:
        kpi_card("Fall Events", len(fall_events), "grouped incidents (cooldown-merged)" if fall_events else "none detected",
                 variant="danger" if fall_events else "")
    with rc3:
        kpi_card("Avg Confidence", f"{avg_conf*100:.1f}%" if video_confidences else "—",
                 "this video only" if video_confidences else "no predictions")
    with rc4:
        actual_duration = frame_idx / fps if fps and fps > 0 else duration_s
        kpi_card("Video Duration", f"{actual_duration:.1f}s" if actual_duration else "unknown")

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

    if fall_events:
        st.markdown("")
        col_t, col_tbl = st.columns([1, 1.2])
        with col_t:
            _html("<div class='sf-card'><div class='sf-card-title'>Fall Event Timeline</div>")
            tl_events = [
                {"time": format_ts(e["start_timestamp_s"]), "label": "fall", "confidence": e["confidence"]}
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
                width="stretch",
                hide_index=True,
            )
            _html("<div class='sf-note'>Emergency alert generated — potential fall event detected. Prototype monitoring system; no emergency service contacted automatically.</div></div>")
    else:
        st.markdown("")
        _html("<div class='sf-card'><div class='sf-card-title'>Fall Event Log</div>" + empty_state("✓", "No potential fall events detected in this video.") + "</div>")

    st.markdown("")
    col_dist, col_hist = st.columns([1, 1.15])
    with col_dist:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution (This Video)</div>")
        if total_valid_frames > 0:
            fig = activity_distribution_chart(data_counts=vid_counts, title_label="Predictions (This Video)")
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        else:
            _html(empty_state("📊", "No activity data for this video."))
        _html("</div>")

    with col_hist:
        _html("<div class='sf-card'><div class='sf-card-title'>Video Frame Predictions</div>")
        if this_video_history:
            _html(timeline_html(this_video_history))
        else:
            _html(empty_state("—", "No predictions logged for this video."))
        _html("</div>")


def _mem_txt() -> str:
    mem = current_memory_mb()
    return f"{mem:.0f} MB" if mem is not None else "n/a"


# ============================================================
# PAGE: ANALYTICS (SESSION SUMMARY)
# ============================================================

def page_analytics():
    section_title("Analytics", "SESSION INSIGHTS")
    counts = activity_counts()
    total = sum(counts.values())

    cols = st.columns(6)
    metrics = [
        ("Total", total, ""),
        ("Fall Predictions", counts["fall"], "danger" if counts["fall"] else ""),
        ("Walking", counts["walking"], ""),
        ("Sitting", counts["sitting"], ""),
        ("Standing", counts["standing"], ""),
        ("Normal", counts["normal"], "success" if counts["normal"] else ""),
    ]
    for col, (lbl, val, var) in zip(cols, metrics):
        with col:
            kpi_card(lbl, val, variant=var)

    st.markdown("")

    a1, a2 = st.columns([1.1, 1])
    with a1:
        _html("<div class='sf-card'><div class='sf-card-title'>Activity Distribution (All Videos)</div>")
        if total:
            fig = activity_distribution_chart(data_counts=counts, title_label="Total Predictions (Session)")
            if fig is not None:
                st.pyplot(fig, width="stretch")
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

    section_title("Fall Events", "ALERT LOG")
    if st.session_state.fall_events:
        df_events = pd.DataFrame(st.session_state.fall_events)
        df_events["Event"] = ["Fall"] * len(df_events)
        df_events["Timestamp (s)"] = df_events["start_timestamp_s"]
        df_events["Confidence"] = df_events["confidence"].apply(lambda c: f"{c*100:.1f}%")
        df_events["Duration (s)"] = (df_events["end_timestamp_s"] - df_events["start_timestamp_s"]).round(2)
        if "source" in df_events.columns:
            df_events["Source Video"] = df_events["source"]
            show_cols = ["Event", "Source Video", "Timestamp (s)", "Confidence", "Duration (s)"]
        else:
            show_cols = ["Event", "Timestamp (s)", "Confidence", "Duration (s)"]
        st.dataframe(
            df_events[show_cols],
            width="stretch",
            hide_index=True,
        )
        _html("<div class='sf-note'>Fall events are grouped incidents, accumulated across the session's videos.</div>")
    else:
        _html("<div class='sf-card'>" + empty_state("✓", "No potential fall events detected yet.") + "</div>")

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
            width="stretch",
            hide_index=True,
            height=360,
        )
        _html(f"<div class='sf-note'>History capped at {HISTORY_CAP} most recent records.</div>")
    else:
        _html("<div class='sf-card'>" + empty_state("—", "No activity history yet. Upload an image or video to start logging predictions.") + "</div>")


# ============================================================
# PAGE: MODEL INFORMATION
# ============================================================

def page_model_information():
    section_title("Model Information", "ARCHITECTURE")

    info = load_model_info()

    a1, a2, a3 = st.columns(3)
    with a1:
        _html("""
        <div class='sf-card'>
            <div class='sf-card-title'>Pose Estimation</div>
            <div style='font-size:18px;font-weight:800;color:var(--text);'>YOLO11n-Pose</div>
            <div class='sf-note' style='margin-top:8px;line-height:1.5;'>
                Ultralytics YOLO11n-Pose detects 17 COCO body keypoints per person.
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
                A Random Forest classifier (200 trees, balanced class weights) predicts activity states.
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
                Normalized keypoint coordinates, confidence scores, bounding-box aspect ratio, and torso angle.
            </div>
            <div style='margin-top:10px;'><span class='sf-tag'>scale-invariant</span></div>
        </div>
        """)

    st.markdown("")

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
                ● 6 environments: Home ×2, Coffee room ×2, Lecture room, Office<br/>
                ● Surveillance video with fall annotations<br/>
                ● 70% train / 15% validation / 15% test (stratified)
            </div>
        </div>
        """)

    st.markdown("")
    section_title("Evaluation Metrics", "REAL RESULTS")

    if info is None:
        _html("<div class='sf-card'>" + empty_state("📉", "Evaluation artifact unavailable.<br/>Run train_classifier.py to generate model_info.json and evaluation charts.") + "</div>")
    else:
        tm = info.get("test_metrics", {})
        winner = info.get("winner", "—")

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

        cm_path = SCREENSHOTS_DIR / "confusion_matrix.png"
        cmp_path = SCREENSHOTS_DIR / "model_comparison.png"
        loss_path = SCREENSHOTS_DIR / "mlp_loss_curve.png"

        ec1, ec2 = st.columns(2)
        with ec1:
            _html("<div class='sf-card'><div class='sf-card-title'>Confusion Matrix (Test Set)</div>")
            if cm_path.exists():
                st.image(str(cm_path), width="stretch")
            else:
                _html(empty_state("—", "Confusion matrix artifact unavailable."))
            _html("</div>")
        with ec2:
            _html("<div class='sf-card'><div class='sf-card-title'>Model Comparison — Random Forest vs MLP</div>")
            if cmp_path.exists():
                st.image(str(cmp_path), width="stretch")
            else:
                _html(empty_state("—", "Model comparison artifact unavailable."))
            _html("</div>")

        st.markdown("")
        _html("<div class='sf-card' style='margin-bottom:14px;'><div class='sf-card-title'>MLP Training Loss Curve</div>")
        if loss_path.exists():
            st.image(str(loss_path), width="stretch")
        else:
            _html(empty_state("—", "Loss curve artifact unavailable."))
        _html("</div>")

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
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                _html(empty_state("—", "No validation metrics available."))


# ============================================================
# MAIN
# ============================================================

def main():
    init_state()

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

    _html(_design_css())
    brand_header(online=model_ok)
    render_sidebar(model_ok, clf_ok, yolo_ok)

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
