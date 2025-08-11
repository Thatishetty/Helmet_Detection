import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ====== CONFIG ======
VIDEO_SOURCE = 0  # Use 0 for webcam or provide path to video file
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO_NAME = "helmet_detection_output.mp4"
IOU_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== LOAD MODELS ======
person_model = YOLO("models/yolov8n.pt")
helmet_models = [
    YOLO("models/hemletYoloV8_100epochs.pt"),
    YOLO("models/hemletYoloV8_25epochs.pt")
]





def reconnect_stream(source, max_retries=5, wait_sec=5):
    for attempt in range(max_retries):
        print(f"🔄 Attempting to reconnect to stream (try {attempt+1}/{max_retries})...")
        cap = cv2.VideoCapture(source)
        time.sleep(1)
        if cap.isOpened():
            print("✅ Reconnected to the stream.")
            return cap
        time.sleep(wait_sec)
    print("❌ Failed to reconnect to stream after multiple attempts.")
    return None

# ====== IoU Function ======
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ====== Open Video Capture ======
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.65:554/stream")#Server Room

cap = cv2.VideoCapture("rtsp://admin:Sunnet1q2w@192.168.0.64:554/stream", cv2.CAP_FFMPEG)#main Room


# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter object
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME), fourcc, fps, (width, height))

print("🎥 Starting video processing... Press 'q' to quit.")

while True:
    if not cap.isOpened():
        cap = reconnect_stream("rtsp://admin:Sunnet1q2w@192.168.0.64:554/stream")
        if cap is None:
            break

    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Frame read failed, stream might have dropped.")
        cap.release()
        cap = reconnect_stream("rtsp://admin:Sunnet1q2w@192.168.0.64:554/stream")
        if cap is None:
            break
        continue

    # continue processing frame...


    image = frame.copy()

    # ==== Detect persons ====
    person_results = person_model(image)[0]
    person_boxes = []
    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        label = person_results.names[cls_id]
        if label.lower() == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            person_boxes.append((x1, y1, x2, y2))

    # ==== Detect helmets ====
    helmet_boxes = []
    for model in helmet_models:
        result = model(image)[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            if label.lower() == 'helmet':
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                helmet_boxes.append((x1, y1, x2, y2))

    # ==== Match persons to helmets ====
    for (x1, y1, x2, y2) in person_boxes:
        has_helmet = any(iou((x1, y1, x2, y2), hbox) > IOU_THRESHOLD for hbox in helmet_boxes)
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        label = "Helmet" if has_helmet else "No Helmet"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ==== Display and Save Frame ====
    out.write(image)
    cv2.imshow("Helmet Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Cleanup ======
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video processing complete. Saved to 'outputs/' folder.")
