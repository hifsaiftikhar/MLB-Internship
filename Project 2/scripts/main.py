from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.pt"

model = YOLO(MODEL_PATH)

image_path = BASE_DIR / "sample_test_images" / "pothole_sample_1.jpg"

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("Detection completed!")