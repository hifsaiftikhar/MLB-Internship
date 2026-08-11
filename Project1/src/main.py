import cv2
import json
import os
import sys

# Ensure project directories are in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.detector import ParkingOccupancyDetector

def main():
    print("==================================================")
    st_time = cv2.getTickCount()
    print("Starting Smart Parking Lot Occupancy Analyzer...")
    print("==================================================")
    
    # 1. Verify and Load Image
    if not os.path.exists(config.IMAGE_PATH):
        print(f"Error: Sample image not found at {config.IMAGE_PATH}")
        print("Please run python scripts/download_assets.py first.")
        sys.exit(1)
        
    img = cv2.imread(config.IMAGE_PATH)
    if img is None:
        print(f"Error: Could not read image at {config.IMAGE_PATH}")
        sys.exit(1)
        
    print(f"Loaded test image: {config.IMAGE_PATH} ({img.shape[1]}x{img.shape[0]})")
    
    # 2. Verify and Load Coordinates
    if not os.path.exists(config.SLOTS_CONFIG_PATH):
        print(f"Error: Parking slots configuration not found at {config.SLOTS_CONFIG_PATH}")
        print("Please run python scripts/download_assets.py to prepare slots.")
        sys.exit(1)
        
    with open(config.SLOTS_CONFIG_PATH, "r") as f:
        slots = json.load(f)
    print(f"Loaded {len(slots)} predefined parking slots.")
    
    # 3. Load YOLO and Hybrid Analyzer
    detector = ParkingOccupancyDetector(config.YOLO_MODEL_NAME)
    
    # 4. Run Analysis
    print("\nRunning hybrid detection pipeline...")
    analyzed_slots, vehicles, cv_results = detector.analyze_occupancy(
        img, 
        slots, 
        yolo_overlap_thresh=config.YOLO_OVERLAP_THRESHOLD,
        cv_thresh=config.CV_OCCUPANCY_THRESHOLD,
        cv_fallback_thresh=config.CV_FALLBACK_THRESHOLD
    )
    
    # 5. Calculate Occupancy Statistics
    total_slots = len(analyzed_slots)
    occupied_count = sum([1 for s in analyzed_slots if s["is_occupied"]])
    vacant_count = total_slots - occupied_count
    occupancy_rate = (occupied_count / total_slots) * 100 if total_slots > 0 else 0
    
    print("\n--- Occupancy Analysis Summary ---")
    print(f"Total Parking Slots : {total_slots}")
    print(f"Occupied Spaces     : {occupied_count}")
    print(f"Vacant Spaces       : {vacant_count}")
    print(f"Occupancy Rate      : {occupancy_rate:.2f}%")
    print("----------------------------------\n")
    
    # 6. Save Pipeline Steps (Aesthetic Outputs for Project Report)
    pipeline_steps_dir = os.path.join(config.RESULTS_DIR, "pipeline_steps")
    os.makedirs(pipeline_steps_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(pipeline_steps_dir, "01_grayscale.png"), cv_results["gray"])
    cv2.imwrite(os.path.join(pipeline_steps_dir, "02_clahe.png"), cv_results["enhanced"])
    cv2.imwrite(os.path.join(pipeline_steps_dir, "03_gaussian_blur.png"), cv_results["blurred"])
    cv2.imwrite(os.path.join(pipeline_steps_dir, "04_canny_edges.png"), cv_results["edges"])
    cv2.imwrite(os.path.join(pipeline_steps_dir, "05_morphology.png"), cv_results["dilated"])
    print(f"Saved traditional CV pipeline steps to: {pipeline_steps_dir}")
    
    # 7. Draw and Save Final Annotated Image
    overlay = img.copy()
    annotated = img.copy()
    
    for slot in analyzed_slots:
        points = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        
        # Red = Occupied, Green = Vacant
        color = (0, 0, 255) if slot["is_occupied"] else (0, 255, 0)
        
        # Transparent polygon fill
        cv2.fillPoly(overlay, [points], color)
        # Outline
        cv2.polylines(annotated, [points], True, color, 2)
        
        # Centroid coordinate for ID text
        cx = int(sum([p[0] for p in slot["points"]]) / 4)
        cy = int(sum([p[1] for p in slot["points"]]) / 4)
        cv2.putText(annotated, f"#{slot['id']}", (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlays
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    
    # Save the output
    output_path = os.path.join(config.RESULTS_DIR, "annotated_image.png")
    cv2.imwrite(output_path, annotated)
    print(f"Saved final annotated result to: {output_path}")
    
    # Compute elapsed time
    elapsed = (cv2.getTickCount() - st_time) / cv2.getTickFrequency()
    print(f"\nPipeline successfully completed in {elapsed:.2f} seconds.")
    print("==================================================")

import numpy as np

if __name__ == "__main__":
    main()
