import os
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)

# Load the pre-trained YOLO model once, when the app starts, so it doesn't
# reload on every single prediction request.
model = YOLO(resolve("yolov8n.pt")) if os.path.exists(resolve("yolov8n.pt")) else YOLO("yolov8n.pt")


def detect_objects(input_image):
    """
    Takes an uploaded image, runs YOLO object detection on it, and returns
    the annotated image plus a text summary. Handles missing input, invalid
    files, and unexpected errors gracefully instead of crashing or showing
    a raw traceback.
    """
    # Case 1: no image uploaded at all
    if input_image is None:
        return None, "Please upload an image before proceeding."

    # Case 2: uploaded file isn't a valid, readable image
    try:
        # Gradio's Image(type="numpy") already converts a valid image into
        # a NumPy array, but I still validate its shape here in case
        # something unexpected slips through (e.g. an unreadable/corrupt file).
        if not isinstance(input_image, np.ndarray) or input_image.ndim not in (2, 3):
            return None, "Invalid file. Please upload a valid image."

        # A quick sanity check that this can actually be treated as an image
        Image.fromarray(input_image)

    except Exception:
        return None, "Invalid file. Please upload a valid image."

    # Case 3: run detection, catching any unexpected error so no Python
    # traceback ever reaches the interface
    try:
        results = model.predict(source=input_image, conf=0.25, verbose=False)
        annotated_image = results[0].plot()

        detections = results[0].boxes
        if len(detections) == 0:
            summary = "No objects detected. Try a different image."
        else:
            lines = []
            for box in detections:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                lines.append(f"{class_name}: {confidence:.2f}")
            summary = "Detection completed successfully.\n\n" + "\n".join(lines)

        return annotated_image, summary

    except Exception as e:
        return None, f"Something went wrong while processing the image: {e}"


# Build the Gradio interface
with gr.Blocks(title="YOLO Object Detection") as demo:
    gr.Markdown("# YOLO Object Detection")
    gr.Markdown(
        """
        **What this app does:** This app uses a pre-trained YOLOv8 model to detect common
        everyday objects in any image you provide - things like people, cars, animals,
        laptops, phones, and more (80 object categories in total).

        **How to use it:**
        1. Upload your own image using the box on the left, or click one of the sample
           images below to try the app instantly without uploading anything.
        2. Click the **Detect Objects** button.
        3. The image on the right will show the same picture with bounding boxes drawn
           around each detected object, along with its name and confidence score.
        4. The detection results will also be listed as text below the images.
        5. Click **Clear** to reset and try another image.
        """
    )

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload an image")
        output_image = gr.Image(type="numpy", label="Detected Objects")

    gr.Examples(
        examples=[
            [resolve("sample1.jpg")],
            [resolve("sample2.jpg")],
            [resolve("sample3.jpg")],
            [resolve("img1.jpg")],
            [resolve("img2.jpg")],
            [resolve("img6.jpg")],
        ],
        inputs=input_image,
        label="Try a sample image",
    )

    output_text = gr.Textbox(label="Detection Results", lines=6)

    submit_btn = gr.Button("Detect Objects", variant="primary")
    clear_btn = gr.Button("Clear")

    submit_btn.click(
        fn=detect_objects,
        inputs=input_image,
        outputs=[output_image, output_text],
    )

    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=None,
        outputs=[input_image, output_image, output_text],
    )

if __name__ == "__main__":
    demo.launch()