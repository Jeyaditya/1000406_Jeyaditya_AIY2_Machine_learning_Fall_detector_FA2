import math
import numpy as np

KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
NUM_KEYPOINTS = 17

# Lowered from 0.5 to 0.20 so floor poses and horizontal falls are not killed
CONF_THRESHOLD = 0.20  

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
    """0 deg = upright, 90 deg = horizontal."""
    dx = neck[0] - pelvis[0]
    dy = neck[1] - pelvis[1]
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(-dy / denom, -1.0, 1.0)))


def get_best_person(result, conf_threshold=CONF_THRESHOLD):
    """Return (xy, conf) for the highest-confidence detected person,
    with a lower threshold to prevent dropping people lying on the ground."""
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
    """Returns a fixed-length, scale-invariant feature vector."""
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

    norm_x = (xy[:, 0] - x_min) / width
    norm_y = (xy[:, 1] - y_min) / height
    mask = (conf >= conf_threshold).astype(np.float32)
    norm_x = norm_x * mask
    norm_y = norm_y * mask

    ar = width / height

    # Robust torso calculation: fallback to any visible shoulder/hip pair if occluded
    shoulders = [xy[KP[s]] for s in ("l_shoulder", "r_shoulder") if ok(s)]
    hips = [xy[KP[h]] for h in ("l_hip", "r_hip") if ok(h)]

    if shoulders and hips:
        neck = np.mean(shoulders, axis=0)
        pelvis = np.mean(hips, axis=0)
        theta = torso_angle(neck, pelvis)
    else:
        # If torso landmarks are completely hidden, fall back to aspect ratio clue
        theta = 90.0 if ar > 1.2 else 0.0

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
    """Runs YOLO pose and returns the feature vector and plotted result."""
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf_threshold,
        device=device,
        verbose=False
    )
    result = results[0]
    xy, conf = get_best_person(result, conf_threshold)
    if xy is None:
        return None, result
    feature = extract_feature_vector(xy, conf, conf_threshold)
    return feature, result
