from ultralytics import YOLO

# 1. Load a pre-trained YOLO model.
model = YOLO("yolov8n.pt")

print("Model loaded: yolov8n.pt")
print(f"Number of classes: {len(model.names)}")
print("Class names:", model.names)

# 2. Perform object detection on a single image
# Place a real photo named "sample1.jpg" in this same folder before running.
print("\n===== Detection on a single image =====")
results = model.predict(source="sample1.jpg", save=True, name="single_image")

for r in results:
    print(f"\nImage: {r.path}")
    print(f"Detected {len(r.boxes)} object(s):")
    for box in r.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {model.names[class_id]}: confidence {confidence:.2f}, "
              f"bounding box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# 3. Perform object detection on multiple images
print("\n===== Detection on multiple images =====")
image_list = ["sample1.jpg", "sample2.jpg", "sample3.jpg"]
results_multi = model.predict(source=image_list, save=True, name="multiple_images")

for r in results_multi:
    print(f"\nImage: {r.path}")
    print(f"Detected {len(r.boxes)} object(s):")
    for box in r.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"  {model.names[class_id]}: confidence {confidence:.2f}")

print("\nAll prediction results (images with bounding boxes drawn) saved under runs/detect/")