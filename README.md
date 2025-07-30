
# 🪖 Helmet Detection on Road Surveillance using YOLOv8

Detects **persons** and checks for **helmets** using custom-trained YOLOv8 models. Automatically highlights:

- 🟦 People with blue bounding boxes  
- 🟩 Helmets (Green Box on head)
- 🟥 No Helmet (Red Box on head)
- 🟧 Motorcycle (Orange Box with label at the bottom)

---

## 📸 Example Output

![example](outputs/headcheck_sample.jpg) <!-- Add your actual image here -->

---

## 🔧 Features

- Uses **YOLOv8n.pt** to detect persons and motorcycles
- Uses two custom helmet detection models:
  - `hemletYoloV8_100epochs.pt`
  - `hemletYoloV8_25epochs.pt`
- Helmet check only if a **motorcycle is present**
- Saves results to `/outputs` and displays each image for 3 seconds
- Annotates head region for helmet detection

---

## 📁 Folder Structure

```
Helmet_Detection/
├── images/               # Input test images
├── models/               # YOLOv8 .pt files (ignored in git)
├── outputs/              # Annotated results
├── person_helmet_check.py  # Main detection script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/helmet-detection-yolov8.git
cd helmet-detection-yolov8
```

### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place files

- Add test images to `images/`
- Add `.pt` model files to `models/`:
  - `yolov8n.pt` (official)
  - `hemletYoloV8_100epochs.pt`, `hemletYoloV8_25epochs.pt` (custom)

### 5. Run the detection

```bash
python person_helmet_check.py
```

---

## 🧠 Model Logic

1. Detect all persons and motorcycles using YOLOv8n
2. If a motorcycle is present:
   - Detect helmets using 2 models
   - Crop top 1/3 of each person box as the head region
   - Check for overlapping helmets
   - Annotate as:
     - ✅ Green: Helmet present
     - ❌ Red: No Helmet

---

##Sample Sceernshot 
<img width="3420" height="2214" alt="image" src="https://github.com/user-attachments/assets/04fd7e9b-24e4-4924-9e9d-3305b7c0c398" />


## 📦 Requirements

- Python 3.10+
- OpenCV
- Ultralytics (YOLOv8)
- tqdm

Install via:
```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License. Use it freely for research and development.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Helmet dataset contributors (GitHub and Kaggle)
