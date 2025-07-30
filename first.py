import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# ====== CONFIGURATION ======
IMAGE_DIR = "images"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
LOG_FILE = "logs/best_model_log.csv"

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
    except Exception as e:
        print(f"⚠️ Skipping {name}: {e}")

# ====== INIT LOGGING ======
log_data = []

# ====== PROCESS IMAGES ======
for img_file in os.listdir(IMAGE_DIR):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)

    results_data = []
    side_by_side_imgs = []

    for model_name, model in models:
        result = model(image)[0]
        boxes = result.boxes
        annotated = result.plot()

        # Add label of model
        cv2.putText(annotated, f"Model: {model_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Score: number of detections + sum of confidences
        score = len(boxes) + sum(float(b.conf[0]) for b in boxes)
        results_data.append((model_name, score, annotated))

        # Resize for visualization
        side_by_side_imgs.append(cv2.resize(annotated, (640, 480)))

    # Sort by best score
    results_data.sort(key=lambda x: x[1], reverse=True)
    best_model_name, best_score, best_output = results_data[0]

    # Log
    log_data.append({
        "image": img_file,
        "best_model": best_model_name,
        "score": best_score,
        "timestamp": datetime.now().isoformat()
    })

    # Save best output image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"best_{img_file}"), best_output)

    # Show all results side-by-side
    combined_image = np.hstack(side_by_side_imgs)
    cv2.imshow("Helmet Detection - Comparison", combined_image)

    key = cv2.waitKey(1500)
    if key == ord('q'):
        break

# ====== SAVE LOG ======
df = pd.DataFrame(log_data)
df.to_csv(LOG_FILE, index=False)

cv2.destroyAllWindows()
print("✅ Done. Logs saved and outputs generated.")
