import cv2
import time
import numpy as np
from ultralytics import YOLO
from flirpy.camera.lepton import Lepton

# Load segmentation model
model = YOLO("yolov8n-seg.pt")  # Segmentation model

# Open RGB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open RGB camera")

# Open FLIR Lepton thermal camera
lepton = Lepton()

# Constants
DEAD_TEMP_THRESHOLD = 30000  # 30.0°C in Lepton units (x100)
MOTION_TEMP_DIFF = 500       # 0.5°C = 500
DEAD_TIME_SECONDS = 5

prev_thermal = None
person_tracker = {}

# Resize RGB mask to thermal mask
def resize_mask(mask, from_shape, to_shape):
    return cv2.resize(mask.astype(np.uint8), (to_shape[1], to_shape[0]), interpolation=cv2.INTER_NEAREST)

# Detect motion using masked region
def detect_motion(curr, prev):
    if curr.shape != prev.shape:
        return False
    diff = np.abs(curr.astype(np.int32) - prev.astype(np.int32))
    return np.mean(diff) > MOTION_TEMP_DIFF

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_h, rgb_w = frame.shape[:2]

    # Run segmentation
    results = model(frame, verbose=False)[0]

    # Capture thermal frame
    thermal_raw, _ = lepton.capture()
    lepton_h, lepton_w = thermal_raw.shape

    # Create thermal visualization
    thermal_vis = cv2.normalize(thermal_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thermal_vis = cv2.applyColorMap(cv2.resize(thermal_vis, (320, 240)), cv2.COLORMAP_INFERNO)

    timestamp = time.time()
    new_tracker = {}
    detected = 0
    possibly_dead = 0

    for i, mask in enumerate(results.masks.data):
        class_id = int(results.boxes.cls[i])
        if class_id != 0:  # Not a person
            continue

        detected += 1
        person_id = f"seg-{i}"

        # Convert segmentation mask to full image size
        full_mask = mask.cpu().numpy()
        full_mask_resized = resize_mask(full_mask, (rgb_h, rgb_w), (lepton_h, lepton_w))

        # Mask thermal region
        roi_curr = thermal_raw * full_mask_resized
        roi_prev = prev_thermal * full_mask_resized if prev_thermal is not None else roi_curr
        valid_pixels = full_mask_resized > 0
        if np.sum(valid_pixels) == 0:
            continue

        avg_temp = np.mean(roi_curr[valid_pixels])
        moved = detect_motion(roi_curr[valid_pixels], roi_prev[valid_pixels])

        last_move = person_tracker.get(person_id, 0)
        if moved:
            last_move = timestamp
        new_tracker[person_id] = last_move

        time_inactive = timestamp - last_move
        is_dead = (avg_temp < DEAD_TEMP_THRESHOLD) and (time_inactive > DEAD_TIME_SECONDS)

        # Label and Color
        if is_dead:
            label = f"Dead ({avg_temp / 100:.1f}°C)"
            color = (0, 0, 255)
            possibly_dead += 1
        elif moved:
            label = f"Alive ({avg_temp / 100:.1f}°C)"
            color = (0, 255, 0)
        else:
            label = f"No Motion ({avg_temp / 100:.1f}°C)"
            color = (0, 255, 255)

        # Draw mask on RGB
        contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Also draw on thermal image
        full_mask_thermal = resize_mask(full_mask, (rgb_h, rgb_w), (240, 320))
        contours_thermal, _ = cv2.findContours(full_mask_thermal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thermal_vis, contours_thermal, -1, color, 2)
        cv2.putText(thermal_vis, label, (10, 20 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    prev_thermal = thermal_raw.copy()
    person_tracker = new_tracker

    # Show stacked output
    rgb_resized = cv2.resize(frame, (320, 240))
    stacked = np.hstack([rgb_resized, thermal_vis])
    cv2.putText(stacked, f"Detected: {detected}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(stacked, f"Dead: {possibly_dead}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Segmented Human + Thermal Detection", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
lepton.close()
cv2.destroyAllWindows()
