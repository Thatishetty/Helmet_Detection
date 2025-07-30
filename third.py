import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

# ====== CONFIG ======
IMAGE_DIR = "images"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
LOG_FILE = "logs/best_model_log.csv"
IOU_THRESHOLD = 0.2  # adjust if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ====== LOAD MODELS ======
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
models = []

for name in model_files:
    path = os.path.join(MODEL_DIR, name)
    try:
        model = YOLO(path)
        models.append((name, model))
        print(f"✅ Loaded model: {name}")
    except Exception as e:
        print(f"⚠️ Skipping model {name}: {e}")

if not models:
    print("❌ No valid models found. Exiting.")
    exit()

log_data = []

# ====== HELPER: IoU ======
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ====== PROCESS IMAGES ======
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Can't read image: {img_file}")
        continue

    results_data = []
    side_by_side_imgs = []

    for model_name, model in models:
        result = model(image)[0]
        boxes = result.boxes
        names = result.names
        annotated = image.copy()

        helmet_boxes = []
        person_boxes = []

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if label.lower() == 'helmet':
                helmet_boxes.append((x1, y1, x2, y2))
            elif label.lower() == 'person':
                person_boxes.append((x1, y1, x2, y2))

        # Draw boxes
        for (x1, y1, x2, y2) in person_boxes:
            has_helmet = any(iou((x1, y1, x2, y2), hbox) > IOU_THRESHOLD for hbox in helmet_boxes)
            color = (0, 255, 0) if has_helmet else (0, 0, 255)  # Green if helmet overlaps, else Red
            label = "Person (Helmet)" if has_helmet else "Person (No Helmet)"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (x1, y1, x2, y2) in helmet_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Score = total detections + sum confidence
        score = len(boxes) + sum(float(b.conf[0]) for b in boxes)
        cv2.putText(annotated, f"Model: {model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        results_data.append((model_name, score, annotated))
        side_by_side_imgs.append(cv2.resize(annotated, (640, 480)))

    results_data.sort(key=lambda x: x[1], reverse=True)
    best_model_name, best_score, best_output = results_data[0]

    log_data.append({
        "image": img_file,
        "best_model": best_model_name,
        "score": best_score,
        "timestamp": datetime.now().isoformat()
    })

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"best_{img_file}"), best_output)

    # Show side-by-side view
    combined = np.hstack(side_by_side_imgs)
    cv2.imshow("Helmet Detection - Comparison", combined)
    key = cv2.waitKey(500)
    if key == ord('q'):
        break

# ====== SAVE & SUMMARY ======
df = pd.DataFrame(log_data)
df.to_csv(LOG_FILE, index=False)

summary = df['best_model'].value_counts().reset_index()
summary.columns = ['Model Name', 'Times Selected as Best']
print("\n📊 Best Model Summary:\n")
print(summary.to_string(index=False))

cv2.destroyAllWindows()
print("✅ Finished processing.")
