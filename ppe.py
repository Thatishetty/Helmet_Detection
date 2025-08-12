import sys
import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

# ===== CONFIG =====
IMAGE_DIR = "ppe_images"
OUTPUT_DIR = "output_ppe"
PPE_MODEL_PATH = "models/best.pt"  # your trained PPE model
CONF_PPE = 0.4

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "ppe_violation_log.csv")

# Write CSV header
with open(CSV_LOG_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(["Timestamp", "Image", "Violation"])

# Load model
print("Loading PPE model...")
ppe_model = YOLO(PPE_MODEL_PATH)

# Process images in folder
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"Could not read {img_name}")
        continue

    results = ppe_model(img_path, conf=CONF_PPE, verbose=False)
    img_vis = img0.copy()

    violation_detected = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = ppe_model.names[cls]
        conf = box.conf[0]
        
        # Color coding
        if "no" in label.lower():
            color = (0, 0, 255)  # red for violation
            violation_detected = True
        else:
            color = (0, 255, 0)  # green for compliance

        # Draw
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Log if violation
    if violation_detected:
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), img_name, "PPE Violation"])

    # Save and show
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img_vis)
    cv2.imshow("PPE Detection", img_vis)
    key = cv2.waitKey(2000) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
print("PPE detection complete. Check outputs in", OUTPUT_DIR)
