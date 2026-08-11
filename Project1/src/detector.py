import cv2
import numpy as np
from ultralytics import YOLO
import src.config as config
from src.geometry import calculate_overlap_ios
from src.traditional_cv import run_traditional_cv_pipeline, calculate_slot_density

class ParkingOccupancyDetector:
    def __init__(self, model_name=config.YOLO_MODEL_NAME):
        """Initialize the YOLO-based object detector."""
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
    def detect_vehicles(self, frame, conf_threshold=0.25):
        """
        Run YOLO inference and filter predictions to keep only vehicle class IDs.
        Returns a list of bboxes: [x_min, y_min, x_max, y_max, confidence, class_id]
        """
        results = self.model(frame, verbose=False)[0]
        vehicles = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if class_id in config.VEHICLE_CLASS_IDS and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                vehicles.append([x1, y1, x2, y2, conf, class_id])
                
        return vehicles

    def analyze_occupancy(
        self, 
        frame, 
        slots, 
        yolo_overlap_thresh=config.YOLO_OVERLAP_THRESHOLD,
        cv_thresh=config.CV_OCCUPANCY_THRESHOLD,
        cv_fallback_thresh=config.CV_FALLBACK_THRESHOLD,
        low_threshold=config.CANNY_LOW_THRESHOLD,
        high_threshold=config.CANNY_HIGH_THRESHOLD
    ):
        """
        Analyze occupancy of predefined slots by combining YOLO detections and Traditional CV density.
        """
        cv_results = run_traditional_cv_pipeline(frame, low_threshold, high_threshold)
        dilated_edges = cv_results["dilated"]
        
        vehicles = self.detect_vehicles(frame)
        
        analyzed_slots = []
        
        for slot in slots:
            slot_id = slot["id"]
            points = slot["points"]
            
            cv_density = calculate_slot_density(dilated_edges, points)
            
            max_overlap = 0.0
            best_vehicle = None
            
            for vehicle in vehicles:
                bbox = vehicle[:4]
                overlap = calculate_overlap_ios(points, bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_vehicle = vehicle
            
            is_occupied = False
            decision_reason = "Vacant"
            
            if max_overlap >= yolo_overlap_thresh:
                is_occupied = True
                decision_reason = f"YOLO (Overlap: {max_overlap:.2f})"
            elif max_overlap >= 0.10 and cv_density >= cv_thresh:
                is_occupied = True
                decision_reason = f"Hybrid (YOLO: {max_overlap:.2f}, CV: {cv_density:.2f})"
            elif cv_density >= cv_fallback_thresh:
                is_occupied = True
                decision_reason = f"CV Fallback (CV: {cv_density:.2f})"
            else:
                decision_reason = f"Vacant (YOLO: {max_overlap:.2f}, CV: {cv_density:.2f})"
                
            analyzed_slots.append({
                "id": slot_id,
                "points": points,
                "is_occupied": is_occupied,
                "cv_density": cv_density,
                "yolo_overlap": max_overlap,
                "reason": decision_reason
            })
            
        return analyzed_slots, vehicles, cv_results
