from ultralytics import YOLO

print("===== OBJECT DETECTION - VEHICLE DETECTION USING YOLO =====\n")

# 1. Load a pre-trained YOLO model.
model = YOLO("yolov8n.pt")

# 2. Run inference on the sample vehicle images
image_files = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]

results = model.predict(source=image_files, save=True, name="vehicle_detection", conf=0.25)

# Vehicle-related classes 
vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}

print("\n===== Detection Results =====\n")
total_vehicles_detected = 0
class_counts = {}

for i, r in enumerate(results):
    print(f"Image: {image_files[i]}")
    if len(r.boxes) == 0:
        print("  No objects detected.")
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"  {class_name}: confidence {confidence:.2f}")

        if class_name in vehicle_classes:
            total_vehicles_detected += 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    print()

print("===== Summary =====")
print(f"Total vehicle detections across all images: {total_vehicles_detected}")
print("Breakdown by vehicle type:")
for vehicle_type, count in class_counts.items():
    print(f"  {vehicle_type}: {count}")

print("\nOutput images with bounding boxes saved under runs/detect/vehicle_detection/")