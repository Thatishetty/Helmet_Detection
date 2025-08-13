import cv2
import glob
from ultralytics import YOLO

# ===== CONFIG =====
MODEL1_PATH = "ppe_detection1.pt"   # First model
MODEL2_PATH = "ppe_detection2.pt"   # Second model
IMAGE_FOLDER = "ppeimages/images"   # Folder containing test images
CONF_THRESHOLD = 0.5

# Expected PPE items
PPE_ITEMS = ["Helmet", "Vest", "Boots"]

# Load both models
model1 = YOLO("models/best.pt")
model2 = YOLO("models/yolov8n.pt")

# Get all image paths
image_paths = glob.glob(f"{IMAGE_FOLDER}/*.jpg") + glob.glob(f"{IMAGE_FOLDER}/*.png")

for img_path in image_paths:
    img = cv2.imread(img_path)

    # Run both models
    results1 = model1(img, conf=CONF_THRESHOLD)[0]
    results2 = model2(img, conf=CONF_THRESHOLD)[0]

    # Merge detections
    detections = []
    for result in [results1, results2]:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((label, conf, (x1, y1, x2, y2)))

    # Group PPE by person (simplified for image testing)
    person_ppe = {}
    person_id = 0
    for label, conf, (x1, y1, x2, y2) in detections:
        if label == "Person":
            person_ppe[person_id] = {"bbox": (x1, y1, x2, y2), "ppe": set()}
            person_id += 1
        else:
            # Assign PPE to closest person bbox
            for pid in person_ppe:
                px1, py1, px2, py2 = person_ppe[pid]["bbox"]
                if px1 < (x1 + x2) / 2 < px2 and py1 < (y1 + y2) / 2 < py2:
                    person_ppe[pid]["ppe"].add(label)

    # Draw results
    for pid, pdata in person_ppe.items():
        x1, y1, x2, y2 = pdata["bbox"]
        missing = [item for item in PPE_ITEMS if item not in pdata["ppe"]]
        color = (0, 255, 0) if not missing else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Missing: {', '.join(missing) if missing else 'None'}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show result
    cv2.imshow("PPE Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
