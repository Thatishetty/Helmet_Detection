import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
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
        print(f"✅ Loaded model: {name}")
    except Exception as e:
        print(f"⚠️ Skipping model {name}: {e}")

if not models:
    print("❌ No valid YOLOv8 models found. Exiting.")
    exit()

# ====== INIT LOGGING ======
log_data = []

# ====== PROCESS IMAGES ======
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Skipping unreadable image: {img_file}")
        continue

    results_data = []
    side_by_side_imgs = []

    for model_name, model in models:
        result = model(image)[0]
        boxes = result.boxes
        annotated = result.plot()

        # Add model label
        cv2.putText(annotated, f"Model: {model_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Scoring = count + confidence sum
        score = len(boxes) + sum(float(b.conf[0]) for b in boxes)
        results_data.append((model_name, score, annotated))

        # Resize for visual comparison
        side_by_side_imgs.append(cv2.resize(annotated, (640, 480)))

    # Get best model
    results_data.sort(key=lambda x: x[1], reverse=True)
    best_model_name, best_score, best_output = results_data[0]

    # Log best model
    log_data.append({
        "image": img_file,
        "best_model": best_model_name,
        "score": best_score,
        "timestamp": datetime.now().isoformat()
    })

    # Save best annotated image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"best_{img_file}"), best_output)

    # Show side-by-side result briefly (non-blocking)
    combined_image = np.hstack(side_by_side_imgs)
    cv2.imshow("Helmet Detection - Comparison", combined_image)
    key = cv2.waitKey(500)  # Show for 500 ms
    if key == ord('q'):
        break

# ====== SAVE LOG ======
df = pd.DataFrame(log_data)
df.to_csv(LOG_FILE, index=False)

# ====== PRINT SUMMARY ======
summary = df['best_model'].value_counts().reset_index()
summary.columns = ['Model Name', 'Times Selected as Best']

print("\n📊 Best Model Summary:\n")
print(summary.to_string(index=False))

cv2.destroyAllWindows()
print("\n✅ Finished processing all images.")
