````markdown
# 📚 BookVision AI – Day 24

A custom book detection application built using **YOLO11n** and **Streamlit**.

##  Features

- Book detection using a custom-trained YOLO11n model
- Upload your own images
- Test using sample book images
- Adjustable detection confidence
- Displays detected books and confidence scores
- Shows original and detected images side by side

##  Model

- Model: YOLO11n
- Task: Object Detection
- Classes: 1
- Class: `Book`
- Image Size: 640 × 640
- Training Epochs: 50

## 📊 Model Results

Validation results:

| Metric | Result |
|---|---:|
| mAP@50 | 59.6% |
| mAP@50-95 | 37.8% |
| Precision | 67.5% |
| Recall | 59.7% |

## 🛠️ Technologies

- Python
- Ultralytics YOLO
- YOLO11n
- Streamlit
- PIL

## ▶️ Run the Application

Install dependencies:

```bash
pip install ultralytics streamlit pillow
````

Run:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
Day-24/
├── app.py
├── best.pt
├── original_images/
├── dataset/
└── scripts/
```

