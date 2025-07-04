import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 COCO model (you can switch to yolov8s.pt/yolov8m.pt for better accuracy)
model = YOLO("yolov8n.pt")

# Class IDs to detect
TARGET_CLASS_IDS = {
    0: "person",          # for motion tracking
    56: "chair",          # just detection
    60: "dining table"    # just detection
}

# Motion detection parameters
MOTION_THRESHOLD = 7
DEAD_TIME_SECONDS = 5
human_tracker = {}

# Open webcam
cap = cv2.VideoCapture(0)
prev_gray = None

# Function to detect motion in a region
def is_moving(current_roi, previous_roi):
    if current_roi.shape != previous_roi.shape:
        return False
    diff = cv2.absdiff(current_roi, previous_roi)
    _, motion_mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    return np.sum(motion_mask > 0) > 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()

    # Convert to grayscale for motion tracking
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Run YOLOv8 detection
    results = model(frame, verbose=False)[0]

    humans_detected = 0
    possibly_dead = 0
    new_tracker = {}

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        if class_id not in TARGET_CLASS_IDS:
            continue

        label_name = TARGET_CLASS_IDS[class_id]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = (255, 255, 255)  # default color

        if class_id == 0:  # Person: motion tracking
            humans_detected += 1
            roi_now = gray[y1:y2, x1:x2]
            person_id = f"{x1}-{y1}-{x2}-{y2}"

            moved = False
            if prev_gray is not None:
                roi_prev = prev_gray[y1:y2, x1:x2]
                moved = is_moving(roi_now, roi_prev)

            last_move_time = human_tracker.get(person_id, 0)
            if moved:
                last_move_time = timestamp

            new_tracker[person_id] = last_move_time
            time_inactive = timestamp - last_move_time
            is_dead = time_inactive > DEAD_TIME_SECONDS

            label = "Dead" if is_dead else ("Alive" if moved else "No Motion")
            color = (0, 0, 255) if is_dead else ((0, 255, 255) if not moved else (0, 255, 0))

            if is_dead:
                possibly_dead += 1
        else:
            label = label_name
            color = (0, 128, 255)

        # Draw detection and label on original RGB image
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display stats on frame
    cv2.putText(frame, f"Humans: {humans_detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Possibly Dead: {possibly_dead}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show final frame
    cv2.imshow("YOLOv8: Person Motion + Chair/Table Detection", frame)

    prev_gray = gray.copy()
    human_tracker = new_tracker

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
