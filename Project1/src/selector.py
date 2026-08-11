import cv2
import json
import os
import numpy as np
import src.config as config
from src.geometry import is_point_inside_polygon

current_points = []
slots = []
img_display = None

def load_slots():
    global slots
    if os.path.exists(config.SLOTS_CONFIG_PATH):
        try:
            with open(config.SLOTS_CONFIG_PATH, "r") as f:
                slots = json.load(f)
            print(f"Loaded {len(slots)} slots from {config.SLOTS_CONFIG_PATH}")
        except Exception as e:
            print(f"Error loading config file: {e}. Starting fresh.")
            slots = []
    else:
        print("No configuration file found. Starting fresh.")
        slots = []

def save_slots():
    with open(config.SLOTS_CONFIG_PATH, "w") as f:
        json.dump(slots, f, indent=4)
    print(f"Successfully saved {len(slots)} slots to {config.SLOTS_CONFIG_PATH}")

def mouse_callback(event, x, y, flags, param):
    global current_points, slots, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        print(f"Point added: ({x}, {y}). Current slot points: {len(current_points)}/4")
        
        if len(current_points) == 4:
            new_id = max([s["id"] for s in slots]) + 1 if slots else 1
            slots.append({
                "id": new_id,
                "points": list(current_points)
            })
            print(f"Slot #{new_id} defined!")
            current_points.clear()
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_pt = [x, y]
        removed = False
        for i, slot in enumerate(slots):
            if is_point_inside_polygon(clicked_pt, slot["points"]):
                print(f"Removed Slot #{slot['id']}")
                slots.pop(i)
                removed = True
                break
        if not removed:
            print("Right-click did not hit any slot.")

def draw_overlay(img):
    overlay = img.copy()
    
    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.polylines(img, [pts], True, (0, 200, 0), 2)
        
        centroid_x = int(sum([p[0] for p in slot["points"]]) / 4)
        centroid_y = int(sum([p[1] for p in slot["points"]]) / 4)
        cv2.putText(img, f"#{slot['id']}", (centroid_x - 15, centroid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"#{slot['id']}", (centroid_x - 15, centroid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    for i, pt in enumerate(current_points):
        cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(img, tuple(current_points[i-1]), tuple(pt), (0, 0, 255), 2)
            
    instructions = [
        "Instructions:",
        "- Left Click: Add 4 corner points (clockwise) to create a slot",
        "- Right Click inside a slot: Delete it",
        "- Backspace or 'c': Cancel current points",
        "- 's': Save slots to JSON",
        "- 'q': Exit / Done"
    ]
    
    y_offset = 30
    for inst in instructions:
        cv2.putText(img, inst, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, inst, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20

def main():
    global current_points, slots
    
    load_slots()
    
    if not os.path.exists(config.IMAGE_PATH):
        print(f"Error: Reference image {config.IMAGE_PATH} not found.")
        print("Please run scripts/download_assets.py first.")
        return
        
    img = cv2.imread(config.IMAGE_PATH)
    if img is None:
        print(f"Error: Could not read image at {config.IMAGE_PATH}")
        return
        
    window_name = "Parking Slot Selector (Perspective/4-Point Quadrilateral)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- Parking Slot Selector Ready ---")
    print("Click 4 corners of a parking slot (clockwise) to add a slot.")
    print("Press 's' to Save, 'q' to Quit, Backspace/'c' to clear current points.\n")
    
    while True:
        img_display = img.copy()
        draw_overlay(img_display)
        
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_slots()
        elif key == 8 or key == ord('c'):
            if current_points:
                print("Cleared current drawing points.")
                current_points.clear()
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
