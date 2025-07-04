import time
import cv2
import numpy as np
from ultralytics import YOLO
import board
import busio
import adafruit_mlx90640

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# RGB camera init
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open RGB camera")

# MLX90640 thermal camera init
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
thermal_frame = np.zeros((24 * 32,), dtype=float)
prev_thermal = np.zeros((24, 32), dtype=float)

# Parameters
MOTION_THRESHOLD = 0.5     # °C
DEAD_TEMP_THRESHOLD = 30.0 # °C
DEAD_TIME_SECONDS = 5
tracker = {}

# Resize and map RGB mask to thermal mask
def resize_mask_to_thermal(mask, rgb_shape, thermal_shape=(24, 32)):
    resized = cv2.resize(mask.astype(np.uint8), (thermal_shape[1], thermal_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)

# Detect motion
def detect_motion(curr_roi, prev_roi):
    if curr_roi.shape != prev_roi.shape:
        return False
    diff = np.abs(curr_roi - prev_roi)
    return np.any(diff > MOTION_THRESHOLD)

# Convert thermal data to displayable color image
def thermal_to_colormap(thermal):
    vmin = np.percentile(thermal, 5)
    vmax = np.percentile(thermal, 95)
    scaled = np.clip((thermal - vmin) / (vmax - vmin), 0, 1)
    thermal_8bit = (scaled * 255).astype(np.uint8)
    thermal_resized = cv2.resize(thermal_8bit, (320, 240), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(thermal_resized, cv2.COLORMAP_INFERNO)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_h, rgb_w = frame.shape[:2]

    # Read thermal frame
    try:
        mlx.getFrame(thermal_frame)
    except Exception as e:
        print("Thermal read error:", e)
        continue
    curr_thermal = np.array(thermal_frame).reshape((24, 32))
    thermal_vis = thermal_to_colormap(curr_thermal)

    # YOLOv8 segmentation
    results = model(frame, verbose=False)[0]
    timestamp = time.time()
    new_tracker = {}
    people_detected = 0
    possibly_dead = 0

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()  # shape: [num_masks, H, W]
        for i, mask in enumerate(masks):
            class_id = int(results.boxes.cls[i])
            if class_id != 0:  # Only detect humans
                continue

            people_detected += 1
            person_id = f"seg-{i}"
            full_mask = mask > 0.5

            # Resize mask to thermal size
            thermal_mask = resize_mask_to_thermal(full_mask, (rgb_h, rgb_w))

            # Extract ROI in thermal
            roi_curr = curr_thermal[thermal_mask]
            roi_prev = prev_thermal[thermal_mask] if prev_thermal is not None else roi_curr
            avg_temp = np.mean(roi_curr)

            # Motion detection
            moved = detect_motion(roi_curr, roi_prev)
            last_move = tracker.get(person_id, 0)
            if moved:
                last_move = timestamp
            new_tracker[person_id] = last_move

            time_inactive = timestamp - last_move
            is_dead = (avg_temp < DEAD_TEMP_THRESHOLD) and (time_inactive > DEAD_TIME_SECONDS)

            # Label and color
            if is_dead:
                label = f"Dead ({avg_temp:.1f}°C)"
                color = (0, 0, 255)
                possibly_dead += 1
            elif moved:
                label = f"Alive ({avg_temp:.1f}°C)"
                color = (0, 255, 0)
            else:
                label = f"No Motion ({avg_temp:.1f}°C)"
                color = (0, 255, 255)

            # Draw segmentation on RGB
            contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)
            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw same contour on thermal view
            thermal_mask_vis = cv2.resize(full_mask.astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
            contours_t, _ = cv2.findContours(thermal_mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(thermal_vis, contours_t, -1, color, 2)
            cv2.putText(thermal_vis, label, (10, 20 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update previous
    prev_thermal = curr_thermal.copy()
    tracker = new_tracker

    # Stack views
    rgb_small = cv2.resize(frame, (320, 240))
    stacked = np.hstack((rgb_small, thermal_vis))
    cv2.putText(stacked, f"Humans: {people_detected}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(stacked, f"Dead: {possibly_dead}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Segmented RGB + Thermal View", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
