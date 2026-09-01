"""
pose_utils.py — shared feature extraction for SafeFall AI (FA-2)

Both train_classifier.py and app.py import from here so the feature
vector a model is TRAINED on is guaranteed to match the feature vector
it's FED at inference time in Streamlit. (A very common bug is these
two silently drifting apart — keeping one shared module prevents that.)
"""

import math
import numpy as np

# YOLO 17-keypoint (COCO) index map — same as fa1_pipeline.py
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
NUM_KEYPOINTS = 17
CONF_THRESHOLD = 0.5   # same threshold used in fa1_pipeline.py — keep in sync

FEATURE_NAMES = (
    [f"kp{i}_x" for i in range(NUM_KEYPOINTS)]
    + [f"kp{i}_y" for i in range(NUM_KEYPOINTS)]
    + [f"kp{i}_conf" for i in range(NUM_KEYPOINTS)]
    + ["aspect_ratio", "torso_angle", "avg_knee_angle", "knee_asymmetry"]
)


def angle_at_joint(a, b, c):
    """Angle (degrees) at point b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def torso_angle(neck, pelvis):
    """0 deg = upright, 90 deg = horizontal. Same formula as fa1_pipeline.py."""
    dx = neck[0] - pelvis[0]
    dy = neck[1] - pelvis[1]
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(-dy / denom, -1.0, 1.0)))


def get_best_person(result, conf_threshold=CONF_THRESHOLD):
    """Given one ultralytics pose Result, return (xy, conf) for the
    highest-confidence detected person, or (None, None) if nobody was
    detected above threshold. This is the same 'kill the ghost' filter
    used during FA-1 extraction — keep it identical here so training
    and inference see the same kind of input."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return None, None

    if result.boxes is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        if confs.max() < conf_threshold:
            return None, None
        best_idx = int(np.argmax(confs))
    else:
        best_idx = 0

    xy = result.keypoints.xy[best_idx].cpu().numpy()
    conf = (result.keypoints.conf[best_idx].cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(xy.shape[0]))
    return xy, conf


def extract_feature_vector(xy, conf, conf_threshold=CONF_THRESHOLD):
    """xy: (17,2) keypoint coords in pixel space. conf: (17,) confidences.
    Returns a fixed-length, scale-invariant feature vector, or None if
    too few keypoints are visible to say anything useful."""

    def ok(name):
        return conf[KP[name]] >= conf_threshold

    visible = [i for i in range(NUM_KEYPOINTS) if conf[i] >= conf_threshold]
    if len(visible) < 4:
        return None

    xs = xy[visible, 0]
    ys = xy[visible, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)

    # Scale-invariant keypoint coordinates (0-1 within the person's own
    # bounding box) — same idea as your storyboard's "anonymized numerical
    # coordinate" slide: position matters, absolute pixels don't.
    norm_x = (xy[:, 0] - x_min) / width
    norm_y = (xy[:, 1] - y_min) / height
    # Zero out coords for keypoints below threshold so noise doesn't leak in
    mask = (conf >= conf_threshold).astype(np.float32)
    norm_x = norm_x * mask
    norm_y = norm_y * mask

    ar = width / height

    if ok("l_shoulder") and ok("r_shoulder") and ok("l_hip") and ok("r_hip"):
        neck = (xy[KP["l_shoulder"]] + xy[KP["r_shoulder"]]) / 2
        pelvis = (xy[KP["l_hip"]] + xy[KP["r_hip"]]) / 2
        theta = torso_angle(neck, pelvis)
    else:
        theta = 0.0

    l_knee_angle = r_knee_angle = None
    if ok("l_hip") and ok("l_knee") and ok("l_ankle"):
        l_knee_angle = angle_at_joint(xy[KP["l_hip"]], xy[KP["l_knee"]], xy[KP["l_ankle"]])
    if ok("r_hip") and ok("r_knee") and ok("r_ankle"):
        r_knee_angle = angle_at_joint(xy[KP["r_hip"]], xy[KP["r_knee"]], xy[KP["r_ankle"]])

    knee_angles = [a for a in (l_knee_angle, r_knee_angle) if a is not None]
    avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 180.0
    asymmetry = abs(l_knee_angle - r_knee_angle) if len(knee_angles) == 2 else 0.0

    feature = np.concatenate([
        norm_x, norm_y, conf,
        [ar, theta, avg_knee, asymmetry],
    ]).astype(np.float32)

    return feature


def image_to_feature(model, image_bgr, conf_threshold=CONF_THRESHOLD, imgsz=320, device="cpu"):
    """Convenience wrapper: raw BGR image -> feature vector (or None).
    Also returns the raw ultralytics result so callers can .plot() it
    for an annotated screenshot without running inference twice."""
    results = model.predict(source=image_bgr, imgsz=imgsz, conf=conf_threshold,
                             device=device, verbose=False)
    result = results[0]
    xy, conf = get_best_person(result, conf_threshold)
    if xy is None:
        return None, result
    feature = extract_feature_vector(xy, conf, conf_threshold)
    return feature, result
