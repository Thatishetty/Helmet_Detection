import os
import cv2
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ====== CONFIG ======
RTSP_URL = "rtsp://admin:Sunnet1q2w@192.168.0.64:554/stream"  # Adjust as needed
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO_NAME = "helmet_detection_output.mp4"
IOU_THRESHOLD = 0.2
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "no_helmet_log.csv")
SHOW_HELMET_BOXES = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== LOAD MODELS ======
print("📦 Loading models...")
person_model = YOLO("models/yolov8n.pt")
helmet_model = YOLO("models/hemletYoloV8_100epochs.pt")  # Use the better performing model

# ====== IOU FUNCTION ======
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ====== OPEN VIDEO CAPTURE ======
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise Exception("❌ Unable to open RTSP stream.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25  # fallback

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME), fourcc, fps, (width, height))

# ====== INIT CSV LOG FILE ======
with open(CSV_LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Person_Box", "Violation"])

print("🎥 Starting video processing... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = frame.copy()

    # ==== Detect persons ====
    person_results = person_model(image)[0]
    person_boxes = []
    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        label = person_results.names[cls_id].lower()
        if label == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            person_boxes.append((x1, y1, x2, y2))

    # ==== Detect helmets ====
    helmet_results = helmet_model(image)[0]
    helmet_boxes = []
    for box in helmet_results.boxes:
        cls_id = int(box.cls[0])
        label = helmet_results.names[cls_id].lower()
        if "helmet" in label:
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0].cpu().numpy())
            helmet_boxes.append((hx1, hy1, hx2, hy2))

            if SHOW_HELMET_BOXES:
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)

    # ==== Match persons to helmets (head region only) ====
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for (x1, y1, x2, y2) in person_boxes:
        head_region = (x1, y1, x2, y1 + int((y2 - y1) / 3))
        has_helmet = any(iou(head_region, hbox) > IOU_THRESHOLD for hbox in helmet_boxes)
        
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        label = "Helmet" if has_helmet else "No Helmet"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Log violation
        if not has_helmet:
            with open(CSV_LOG_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, f"{x1},{y1},{x2},{y2}", "No Helmet"])

    # ==== Display and Save Frame ====
    out.write(image)
    cv2.imshow("Helmet Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== CLEANUP ======
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video processing complete. Output saved in 'outputs/' folder.")
