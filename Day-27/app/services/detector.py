import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("models/best.pt")


class YOLODetector:
    def __init__(self):
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(str(MODEL_PATH))
        self.model_name = MODEL_PATH.name
        print("Model loaded successfully.")

    def detect_frame(self, frame: np.ndarray, confidence: float = 0.25):
        """
        Run YOLO on a single frame.
        Returns annotated frame and list of detections.
        """
        results = self.model(frame, conf=confidence, verbose=False)[0]
        annotated = results.plot()

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            conf = round(float(box.conf[0].item()), 4)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return annotated, detections


# Single instance loaded at startup
detector = YOLODetector()