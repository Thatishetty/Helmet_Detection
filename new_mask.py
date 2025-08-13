import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime
from ultralytics import YOLO
from yolov5.utils.torch_utils import select_device

# CONFIG
OUTPUT_DIR = "output_mask_webcam"
MASK_MODEL_PATH = "models/mask_detector_model.h5"
PERSON_MODEL_PATH = "yolov8n.pt"
CONF_PERSON = 0.5
INPUT_SIZE = (224, 224)  # Adjust if your model expects another size
CLASS_NAMES = ["with_mask", "without_mask"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "no_mask_log.csv")
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "PersonBox", "MaskLabel"])

print("Loading models...")
person_model = YOLO(PERSON_MODEL_PATH)
mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)
device = select_device('')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    results = person_model.predict(frame, conf=CONF_PERSON, classes=[0], verbose=False)
    vis = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        face = cv2.resize(crop, INPUT_SIZE)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = mask_model.predict(face, verbose=0)[0]
        conf = np.max(preds)
        label = CLASS_NAMES[np.argmax(preds)]
        color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

        cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label == "without_mask":
            with open(CSV_LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now().isoformat(), f"{x1},{y1},{x2},{y2}", label])

    cv2.imshow("Mask Detection", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Mask detection stopped. CSV saved at:", CSV_LOG_PATH)
