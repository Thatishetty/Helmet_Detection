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
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# ===== CONFIG =====
OUTPUT_DIR = "output_mask_webcam"
MASK_MODEL_PATH = "models/mask_detector_model.h5"
PERSON_MODEL_PATH = "yolov8n.pt"
CONF_PERSON = 0.5
CONF_MASK = 0.4
CAM_INDEX = 0  # webcam index

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "no_mask_log.csv")

# Write CSV header if not exists
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "PersonBox", "MaskLabel"])

# Load models
print("Loading models...")
person_model = YOLO(PERSON_MODEL_PATH)
device = select_device('')
mask_model = DetectMultiBackend(MASK_MODEL_PATH, device=device)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit()

print("Press 'q' to quit.")
while True:
    ret, img0 = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = person_model.predict(img0, conf=CONF_PERSON, classes=[0], verbose=False)
    img_vis = img0.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        person_crop = img0[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        # Prepare temp crop for YOLOv5
        temp_path = os.path.join(OUTPUT_DIR, "temp.jpg")
        cv2.imwrite(temp_path, person_crop)

        dataset = LoadImages(temp_path, img_size=640, stride=32)
        for _, im, im0_, _, _ in dataset:
            im = torch.from_numpy(im).to(device).float() / 255.0
            if im.ndim == 3:
                im = im.unsqueeze(0)

            pred = mask_model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, CONF_MASK, 0.45)

            label = "No Mask"
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_.shape).round()
                    for *xyxy, conf, cls in det:
                        mask_label = mask_model.names[int(cls)]
                        label = f"{mask_label} {conf:.2f}"
                        color = (0, 255, 0) if "mask" in mask_label.lower() and "without" not in mask_label.lower() else (0, 0, 255)
                        cv2.rectangle(img_vis,
                                      (x1 + int(xyxy[0]), y1 + int(xyxy[1])),
                                      (x1 + int(xyxy[2]), y1 + int(xyxy[3])),
                                      color, 2)
                        cv2.putText(img_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log if no mask detected
            if "without mask" in label.lower():
                with open(CSV_LOG_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow([datetime.now().isoformat(), f"{x1},{y1},{x2},{y2}", label])

    # Show webcam output
    cv2.imshow("Mask Detection", img_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Mask detection stopped. CSV saved at:", CSV_LOG_PATH)
