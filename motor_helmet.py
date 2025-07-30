import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# === CONFIG ===
IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
IOU_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === MODELS ===
person_model = YOLO("models/yolov8n.pt")
helmet_models = [
    YOLO("models/hemletYoloV8_100epochs.pt"),
    YOLO("models/hemletYoloV8_25epochs.pt")
]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# === PROCESS IMAGES ===
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="Processing"):
    path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(path)
    if image is None:
        continue

    annotated = image.copy()

    # ==== Person + Motorcycle Detection ====
    detections = person_model(image)[0]
    person_boxes = []
    motorcycle_detected = False

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        label = detections.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if label.lower() == 'person':
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif label.lower() == 'motorcycle':
            motorcycle_detected = True
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(annotated, "Motorcycle", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    # ==== Helmet Detection only if motorcycle exists ====
    if motorcycle_detected:
        helmet_boxes = []
        for model in helmet_models:
            result = model(image)[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                if label.lower() == 'helmet':
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    helmet_boxes.append((x1, y1, x2, y2))

        for (x1, y1, x2, y2) in person_boxes:
            # Define head region (top 1/3rd)
            head_box = (x1, y1, x2, y1 + int((y2 - y1) / 3))
            has_helmet = any(iou(head_box, hbox) > IOU_THRESHOLD for hbox in helmet_boxes)

            color = (0, 255, 0) if has_helmet else (0, 0, 255)
            label = "Helmet" if has_helmet else "No Helmet"

            cv2.rectangle(annotated, (head_box[0], head_box[1]), (head_box[2], head_box[3]), color, 2)
            cv2.putText(annotated, label, (head_box[0], head_box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ==== Show 3-sec pop-up ====
    cv2.imshow("Helmet Detection", annotated)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # ==== Save image ====
    save_path = os.path.join(OUTPUT_DIR, f"helmetcheck_{img_file}")
    cv2.imwrite(save_path, annotated)

print("✅ Done! Images saved to 'outputs/' with motorcycle-based helmet check.")
