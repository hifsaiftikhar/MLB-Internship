# Day-13: Object Detection with YOLO

## What I worked on

Today's focus was Object Detection - identifying what objects are in an image and where they are located, using a pre-trained YOLO model. I practiced running inference with YOLOv8, built a vehicle detection mini project, and created a Gradio app so the model can be used interactively through a browser.

## What is Object Detection?

Object Detection identifies multiple objects within a single image and draws a bounding box (a rectangle) around each one, along with a class label (what the object is) and a confidence score (how sure the model is, from 0 to 1). Unlike simpler tasks, it can find and locate several different objects in the same image at once, not just describe the image as a whole.

## How is Object Detection different from Image Classification?

Image Classification (Days 10-12) answers "what is the single main subject of this image?" and produces one label for the entire image. Object Detection answers "what objects are present, and exactly where are they?" - it can detect multiple objects in one image, each with its own bounding box, label, and confidence score, rather than reducing the whole image to a single answer.

## What is YOLO?

YOLO ("You Only Look Once") is a family of object detection models that process the entire image in a single pass to predict all bounding boxes, class labels, and confidence scores at once. This is why YOLO is fast enough for real-time detection (video, live camera feeds) - earlier object detection approaches scanned an image in multiple passes or regions, which was much slower. Today I used YOLOv8n ("nano"), the smallest and fastest YOLOv8 variant, pre-trained on the COCO dataset (80 everyday object classes).

## Coding practice

**yolo_practice.py:**
- Loaded the pre-trained yolov8n model and explored its 80 COCO classes.
- Ran detection on a single image and on multiple images at once, printing each detected object's class, confidence score, and bounding box coordinates.
- Tested on my own photos: a portrait (correctly detected the main subject at 0.93 confidence, plus a smaller, correctly lower-confidence detection of a partially visible person in the blurred background at 0.31), a laptop screen close-up, and a bag/passport photo (correctly detected a person's arm/hand at 0.82).
- Saved all annotated output images (with bounding boxes drawn) under runs/detect/.

## Mini Project: Vehicle Detection

### Which dataset did I use?

I chose Vehicle Detection using pre-trained YOLOv8n directly, without any custom training or specialized dataset. Vehicle-related classes (car, truck, bus, motorcycle, bicycle) are already part of YOLO's 80 COCO classes, so no fine-tuning was needed - this fit the task's note that training was not required today. I used 6 real photos (img1.jpg to img6.jpg) showing street scenes, highways, and vehicles.

### What objects were detected?

Across the 6 images, 37 total vehicle detections: 30 cars, 3 trucks, 3 buses, and 1 motorcycle, plus a few incidental person detections in busy scenes.

### Observations about the detection results

The clearest pattern across all images was that **detection confidence correlated closely with how large and clearly visible each vehicle was in the frame**. In a busy highway traffic scene, cars close to the camera scored high confidence (0.70-0.82), while small, distant, or partially obscured cars in the background scored much lower (0.25-0.40). This shows the model appropriately reflects its own uncertainty rather than guessing with false confidence on harder cases.

In a cleaner image of two trucks facing the camera, both were detected with tight, accurate bounding boxes even though they were close together and partially overlapping in the frame (0.86 and 0.73 confidence) - showing YOLO handles closely-spaced similar objects well when each is reasonably large and clearly visible.

## Gradio App (app.py)

The app lets a user upload any image through a browser and see YOLO's detection results without running any code themselves.

**How it works:**
1. The user uploads an image through the Gradio interface.
2. Clicking "Detect Objects" sends the image to the detect_objects() function, which runs the same YOLOv8n model used in the practice and mini project scripts.
3. The model's results are drawn directly onto the image (bounding boxes, class labels, confidence scores) using YOLO's built-in .plot() method, and a text summary of each detection is generated.
4. Both the annotated image and the text summary are displayed back to the user.

**Exception handling included:**
- If no image is uploaded, the app displays "Please upload an image before proceeding." instead of crashing.
- If an invalid or unreadable file is provided, the app displays "Invalid file. Please upload a valid image."
- Any other unexpected error during detection is caught with a try/except block and shown as a plain message - no raw Python traceback ever reaches the interface.
- On success, the app displays "Detection completed successfully." along with the list of detected classes and confidence scores.

**Sharing the app:** since Gradio apps only run locally by default, I used ngrok to create a public URL that tunnels to my local app, so it can be accessed and tested from any browser while my machine and the app are running.

## Files

- yolo_practice.py: Practice 1 and 2 - loading YOLO, exploring its classes, running detection on single/multiple images, testing on my own photos
- vehicle_detection.py: mini project - vehicle detection on 6 sample images
- app.py: Gradio app for interactive object detection, with full exception handling
- sample1.jpg, sample2.jpg, sample3.jpg: sample images used in practice
- img1.jpg - img6.jpg: sample images used in the vehicle detection mini project
- runs/detect/: output images with bounding boxes drawn, from both the practice script and the mini project
- README.md: this file

## Author

Hifsa Iftikhar
