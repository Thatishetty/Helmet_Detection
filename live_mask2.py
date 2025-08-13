import sys
import os
import cv2
import csv
import torch
from datetime import datetime
from ultralytics import YOLO

# Add YOLOv5 repo path
sys.path.append('yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# ===== CONFIG =====
CAMERA_SOURCE = 0  # Webcam index or RTSP/HTTP link
OUTPUT_DIR = "output_mask_live"
MASK_MODEL_PATH = "models/mask_yolov5.pt"
PERSON_MODEL_PATH = "yolov8n.pt"
CONF_PERSON = 0.5
CONF_MASK = 0.4

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "no_mask_log.csv")

# Write CSV header if not exists
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "FaceBox", "MaskLabel"])

# Load models
print("Loading models...")
person_model = YOLO(PERSON_MODEL_PATH)
device = select_device('')
mask_model = DetectMultiBackend(MASK_MODEL_PATH, device=device)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit()

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    img_vis = frame.copy()

    # Detect persons
    results = person_model(frame, conf=CONF_PERSON, classes=[0], verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop person
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        # Prepare input for YOLOv5 mask detection
        im = cv2.resize(person_crop, (640, 640))
        im = torch.from_numpy(im).to(device).float() / 255.0
        im = im.permute(2, 0, 1).unsqueeze(0)

        pred = mask_model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, CONF_MASK, 0.45)

        # Loop over face detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], person_crop.shape).round()
                for *xyxy, conf, cls in det:
                    mask_label = mask_model.names[int(cls)]
                    px1, py1, px2, py2 = map(int, xyxy)

                    # Map face coordinates to original frame
                    fx1, fy1 = x1 + px1, y1 + py1
                    fx2, fy2 = x1 + px2, y1 + py2

                    # Color & logging
                    if "mask" in mask_label.lower() and "without" not in mask_label.lower():
                        color = (0, 255, 0)  # Green
                    else:
                        color = (0, 0, 255)  # Red
                        with open(CSV_LOG_PATH, 'a', newline='') as f:
                            csv.writer(f).writerow([datetime.now().isoformat(),
                                                    f"{fx1},{fy1},{fx2},{fy2}",
                                                    mask_label])

                    # Draw face-level box
                    cv2.rectangle(img_vis, (fx1, fy1), (fx2, fy2), color, 2)
                    cv2.putText(img_vis, f"{mask_label} {conf:.2f}", (fx1, fy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show output
    cv2.imshow("Mask Detection Live - Face Level", img_vis)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
