import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("models/yolov8n.pt")

class YOLODetector:
    def __init__(self):
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(str(MODEL_PATH))
        self.model_name = MODEL_PATH.name
        print("Model loaded successfully.")

    def predict(self, image_bytes: bytes, confidence: float = 0.25):
        """
        Run YOLO inference on image bytes.
        Returns list of detections with class, confidence, and bbox.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image.")

        # Run inference
        results = self.model(img, conf=confidence, verbose=False)[0]

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

        return detections

    def predict_with_image(self, image_bytes: bytes, confidence: float = 0.25):
        """
        Run YOLO inference and return annotated image as bytes.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image.")

        results = self.model(img, conf=confidence, verbose=False)[0]
        annotated = results.plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        return buffer.tobytes()


# Single instance loaded at startup
detector = YOLODetector()