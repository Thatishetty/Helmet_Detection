import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ====== CONFIG ======
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
IOU_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== LOAD MODELS ======
person_model = YOLO("models/yolov8n.pt")
helmet_models = [
    YOLO("models/hemletYoloV8_100epochs.pt"),
    YOLO("models/hemletYoloV8_25epochs.pt")
]

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

# ====== Run on Images ======
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="Processing"):
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)
    if image is None:
        continue
    original = image.copy()

    # ==== Detect persons ====
    person_results = person_model(image)[0]
    person_boxes = []
    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        label = person_results.names[cls_id]
        if label.lower() == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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

    # ==== Save output ====
    save_path = os.path.join(OUTPUT_DIR, f"check_{img_file}")
    cv2.imwrite(save_path, image)

print("✅ All images processed and saved to 'outputs/' folder.")
