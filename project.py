import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import math

# ===============================
# Paths
# ===============================
yolo_model_path = "best.pt"                   # YOLOv8 weapon model
keras_model_path = "trained_keras_model2.h5"  # Trained posture model
output_dir = "threat_clips"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# Load Models
# ===============================
yolo = YOLO(yolo_model_path)
print("✅ YOLO model loaded.")

# Debug: Print class names for verification
print("YOLO Class Names:", yolo.names)

posture_model = load_model(keras_model_path, safe_mode=False)
print("✅ CNN-LSTM posture model loaded.")

# ===============================
# Initialize MediaPipe
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Settings
# ===============================
SEQ_LEN = 50
CONF_THRESHOLD = 0.35
RECORD_SECONDS = 5
COOLDOWN_TIME = 10
ROLLING_AVG_WINDOW = 10
FRAME_SKIP = 2

frame_buffer = deque(maxlen=SEQ_LEN)
threat_history = deque(maxlen=ROLLING_AVG_WINDOW)
recording = False
cooldown_start = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# ===============================
# Enhanced is_aiming Function
# ===============================
def is_aiming(results, weapon_boxes=None, frame_w=1, frame_h=1, aim_angle_thresh=25, over_box_thresh=0.1):
    """
    Robust aiming detection:
    - Checks if arm is straight
    - Checks if pointing direction is generally horizontal
    - Optionally: Checks if wrist is inside/near any weapon box
    Returns True if aiming gesture detected, else False.
    """
    if not results.pose_landmarks:
        return False
    lm = results.pose_landmarks.landmark

    # Helper
    def get_pt(name):
        p = lm[mp_pose.PoseLandmark[name].value]
        return np.array([p.x * frame_w, p.y * frame_h])

    arms = [
        ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')
    ]
    for shoulder, elbow, wrist in arms:
        s, e, w = get_pt(shoulder), get_pt(elbow), get_pt(wrist)
        upper, lower = e-s, w-e
        norm_upper, norm_lower = np.linalg.norm(upper), np.linalg.norm(lower)
        if norm_upper < 1e-3 or norm_lower < 1e-3:
            continue

        # Check straightness via cosine similarity
        cos_sim = np.dot(upper, lower) / (norm_upper * norm_lower)
        straight = cos_sim < -0.96  # very straight (-1)

        # Angle of wrist-to-shoulder vector (degrees, relative to horizontal)
        dx, dy = w - s
        angle = math.degrees(math.atan2(dy, dx))
        horizontal = abs(angle) < aim_angle_thresh or abs(angle-180) < aim_angle_thresh or abs(angle+180) < aim_angle_thresh

        # Optionally: Is hand on or near any weapon bbox?
        hand_on_weapon = False
        if weapon_boxes:
            for (x1, y1, x2, y2) in weapon_boxes:
                buffer = over_box_thresh * frame_w
                if (x1-buffer) <= w[0] <= (x2+buffer) and (y1-buffer) <= w[1] <= (y2+buffer):
                    hand_on_weapon = True
                    break

        if straight and (horizontal or hand_on_weapon):
            return True
    return False

# ===============================
# Camera Setup
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("🎥 Smart Threat Detection started. Press 'q' to exit.")

frame_count = 0

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb)

    # Pose keypoints
    keypoints = []
    if results_pose.pose_landmarks:
        mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for lm in results_pose.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints = [0] * (33 * 3)
    frame_buffer.append(keypoints)

    # Posture model
    threat_prob = 0
    if len(frame_buffer) == SEQ_LEN:
        input_seq = np.array(frame_buffer).flatten().reshape(1, -1).astype(np.float32)
        preds = posture_model.predict(input_seq, verbose=0)
        threat_prob = float(preds[0][1]) if preds.shape[1] > 1 else float(preds[0][0])
        threat_history.append(threat_prob)
        threat_prob = np.mean(threat_history)

    # YOLO weapon detection (binary 0 or 1)
    results_yolo = yolo.predict(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
    weapon_detected = False
    weapon_boxes = []
    weapon_keywords = ["gun", "pistol", "knife", "rifle", "weapon", "shotgun"]

    for r in results_yolo:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id] if cls_id in yolo.names else "object"
            conf = float(box.conf[0])
            print(f"Detected: {label}, conf: {conf}")

            if any(w in label.lower() for w in weapon_keywords) and conf > CONF_THRESHOLD:
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                weapon_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # BINARY weapon score logic
    weapon_score = 1 if weapon_detected else 0

    # Aiming logic with weapon context
    aiming = is_aiming(results_pose, weapon_boxes, frame_w, frame_h)

    # ========== MODIFIED THREAT LOGIC ==========
    # Threat: YES only if posture>0.6, weapon=1, aiming=YES
    threat_detected = (weapon_score == 1) and (threat_prob > 0.6) and aiming
    # ===========================================

    # Recording logic
    current_time = time.time()
    if threat_detected and not recording and (current_time - cooldown_start > COOLDOWN_TIME):
        recording = True
        start_time = time.time()
        filename = os.path.join(output_dir, f"threat_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_w, frame_h))
        print(f"⚠ Threat detected! Recording: {filename}")

    if recording:
        out.write(frame)
        if time.time() - start_time > RECORD_SECONDS:
            recording = False
            cooldown_start = time.time()
            out.release()
            print("✅ Clip saved successfully.")

    # Overlay info (Weapon: 0 or 1 ONLY)
    cv2.putText(frame, f"Weapon: {weapon_score:d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Posture: {threat_prob:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Aiming: {'YES' if aiming else 'NO'}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Threat: {'YES' if threat_detected else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if threat_detected else (0, 255, 0), 2)

    cv2.imshow("🧠 Context-Aware Threat Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if 'out' in locals() and recording:
    out.release()
cv2.destroyAllWindows()
print("👋 Session ended.")