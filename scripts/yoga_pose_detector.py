import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

import os

# ── Config ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "models"))

def _resolve_model_path(filename):
    for candidate in [
        os.path.join(_MODELS_DIR, filename),
        os.path.join(_SCRIPT_DIR, filename),
        filename,
    ]:
        if os.path.exists(candidate):
            return candidate
    return filename

MODEL_PATH      = _resolve_model_path('yoga_pose_classifier.tflite')
LANDMARKER_PATH = _resolve_model_path('pose_landmarker_heavy.task')
CLASS_NAMES_PATH = os.path.join(_SCRIPT_DIR, 'class_names.txt') if os.path.exists(os.path.join(_SCRIPT_DIR, 'class_names.txt')) else 'class_names.txt'
CAMERA_INDEX    = 0
CONF_THRESHOLD  = 0.6   # min confidence to show prediction

# ── Constants ─────────────────────────────────────────────────────────────────
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
N_LANDMARKS        = 33

# MediaPipe pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]


# ── Load class names ──────────────────────────────────────────────────────────
def load_class_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ── Normalization (must match Colab training exactly) ─────────────────────────
def normalize_landmarks(lm_xy: np.ndarray) -> np.ndarray:
    """
    lm_xy: (33, 2) array of x, y coordinates
    Returns: (66,) normalized flat embedding
    """
    left_hip  = lm_xy[IDX_LEFT_HIP]
    right_hip = lm_xy[IDX_RIGHT_HIP]
    hip_center = (left_hip + right_hip) * 0.5

    # Center on hip midpoint
    centered = lm_xy - hip_center

    # Compute torso size
    left_shoulder  = lm_xy[IDX_LEFT_SHOULDER]
    right_shoulder = lm_xy[IDX_RIGHT_SHOULDER]
    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    torso_size = np.linalg.norm(shoulder_center - hip_center)

    # Max dist from center
    dists    = np.linalg.norm(centered, axis=1)
    max_dist = np.max(dists)

    scale = max(torso_size * 2.5, max_dist, 1e-6)
    normalized = centered / scale

    return normalized.flatten().astype(np.float32)  # (66,)


# ── TFLite inference ──────────────────────────────────────────────────────────
class PoseClassifier:
    def __init__(self, model_path):
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input  = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        print(f'Model loaded — input shape: {self._input["shape"]}')

    def predict(self, embedding: np.ndarray):
        """embedding: (66,) float32"""
        inp = np.expand_dims(embedding, axis=0).astype(np.float32)
        self._interpreter.set_tensor(self._input['index'], inp)
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._output['index'])[0]
        return probs


# ── MediaPipe landmarker ──────────────────────────────────────────────────────
class LandmarkDetector:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray):
        """
        frame_rgb: RGB numpy array
        Returns: (lm_xy (33,2), lm_full (33,4)) or (None, None)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result   = self._detector.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None

        lm_list = result.pose_landmarks[0]

        # Visibility check
        vis_scores = [lm.visibility for lm in lm_list]
        if np.mean(vis_scores) < 0.3:
            return None, None

        lm_xy   = np.array([[lm.x, lm.y] for lm in lm_list], dtype=np.float32)
        lm_full = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lm_list], dtype=np.float32)

        return lm_xy, lm_full

    def close(self):
        self._detector.close()


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm_xy, h, w):
    """Draw all 33 joints and connections on frame."""
    # Connections
    for a, b in POSE_CONNECTIONS:
        if a < len(lm_xy) and b < len(lm_xy):
            pt1 = (int(lm_xy[a][0] * w), int(lm_xy[a][1] * h))
            pt2 = (int(lm_xy[b][0] * w), int(lm_xy[b][1] * h))
            cv2.line(frame, pt1, pt2, (0, 200, 100), 2, cv2.LINE_AA)

    # Joints
    for i, (x, y) in enumerate(lm_xy):
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 150, 80),    1,  cv2.LINE_AA)


def draw_prediction(frame, class_names, probs, threshold):
    """Draw prediction label and confidence bar."""
    h, w = frame.shape[:2]
    top_idx  = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_name = class_names[top_idx]

    # Background panel
    cv2.rectangle(frame, (0, 0), (280, 44 + 28 * len(class_names)), (20, 20, 20), -1)

    # Main prediction
    color = (0, 220, 120) if top_prob >= threshold else (80, 80, 200)
    label = f'{top_name.upper()}  {top_prob*100:.1f}%'
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # All class bars
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        y    = 50 + i * 28
        bar  = int(prob * 200)
        bcol = (0, 180, 100) if i == top_idx else (60, 60, 60)
        cv2.rectangle(frame, (10, y), (10 + bar, y + 16), bcol, -1)
        cv2.putText(frame, f'{name:<16} {prob*100:5.1f}%',
                    (10, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS (bottom right)
    return top_name, top_prob


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f'FPS: {fps:.1f}', (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)


def draw_status(frame, msg, color=(80, 80, 220)):
    h, w = frame.shape[:2]
    cv2.putText(frame, msg, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print('Loading models...')
    class_names = ['chair', 'cobra', 'dog', 'no_pose', 'shoudler_stand', 'traingle', 'tree', 'warrior']
    detector    = LandmarkDetector(LANDMARKER_PATH)
    classifier  = PoseClassifier(MODEL_PATH)
    print(f'Classes: {class_names}')
    print('Press Q to quit.')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed.')
            break

        frame = cv2.flip(frame, 1)   # mirror
        h, w  = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detect landmarks ──
        lm_xy, lm_full = detector.detect(frame_rgb)

        if lm_xy is not None:
            # Draw skeleton
            draw_skeleton(frame, lm_xy, h, w)

            # Normalize → classify
            embedding = normalize_landmarks(lm_xy)
            probs     = classifier.predict(embedding)

            # Draw prediction panel
            pred_name, pred_conf = draw_prediction(frame, class_names, probs, CONF_THRESHOLD)

            status_msg   = f'Detected: {pred_name}  ({pred_conf*100:.1f}%)'
            status_color = (0, 220, 120) if pred_conf >= CONF_THRESHOLD else (80, 150, 220)
            draw_status(frame, status_msg, status_color)
        else:
            draw_status(frame, 'No pose detected — step into frame', (80, 80, 220))

        # FPS
        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow('Yoga Pose Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print('Done.')


if __name__ == '__main__':
    main()
