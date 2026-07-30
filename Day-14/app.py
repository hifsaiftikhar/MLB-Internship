import os
from datetime import date

import numpy as np
import cv2
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


def process_image(input_image, operation, resize_w, resize_h, rotate_choice,
                   flip_choice, crop_x1, crop_y1, crop_x2, crop_y2,
                   name_text, brightness, contrast):
    """
    Applies the selected operation to the uploaded image. Handles missing
    input and unexpected errors gracefully instead of crashing or showing
    a raw traceback.
    """
    # Case 1: no image uploaded
    if input_image is None:
        return None, "Please upload an image before proceeding."

    # Case 2: not a valid image array
    try:
        if not isinstance(input_image, np.ndarray) or input_image.ndim not in (2, 3):
            return None, "Invalid file. Please upload a valid image."
    except Exception:
        return None, "Invalid file. Please upload a valid image."

    try:
        # Gradio gives images in RGB; OpenCV's functions here work fine on
        # RGB arrays directly (color order doesn't affect geometric ops
        # like resize/rotate/flip/crop, only color-specific ones).
        img = input_image.copy()

        if operation == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        elif operation == "Resize":
            w, h = int(resize_w), int(resize_h)
            if w <= 0 or h <= 0:
                return None, "Width and height must be positive numbers."
            result = cv2.resize(img, (w, h))

        elif operation == "Rotate":
            if rotate_choice == "90 degrees":
                result = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_choice == "180 degrees":
                result = cv2.rotate(img, cv2.ROTATE_180)
            elif rotate_choice == "270 degrees":
                result = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                return None, "Please select a rotation angle."

        elif operation == "Flip":
            if flip_choice == "Horizontal":
                result = cv2.flip(img, 1)
            elif flip_choice == "Vertical":
                result = cv2.flip(img, 0)
            elif flip_choice == "Both":
                result = cv2.flip(img, -1)
            else:
                return None, "Please select a flip direction."

        elif operation == "Crop":
            h, w = img.shape[:2]
            x1, y1, x2, y2 = int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)
            if not (0 <= x1 < x2 <= w) or not (0 <= y1 < y2 <= h):
                return None, "Invalid crop region. Please check the coordinates fit within the image size."
            result = img[y1:y2, x1:x2]

        elif operation == "Draw Shapes":
            result = img.copy()
            height, width = result.shape[:2]
            cv2.rectangle(result, (10, 10), (width // 3, height // 3), (0, 255, 0), 2)
            cv2.circle(result, (width // 2, height // 2), min(width, height) // 6, (255, 0, 0), 2)
            cv2.line(result, (0, height - 1), (width - 1, 0), (0, 0, 255), 2)
            points = np.array([
                [width // 2, height - 10],
                [width - 10, height // 2],
                [width // 2, 10],
            ], dtype=np.int32)
            cv2.polylines(result, [points], isClosed=True, color=(0, 255, 255), thickness=2)

        elif operation == "Add Text":
            result = img.copy()
            label = name_text.strip() if name_text and name_text.strip() else "Hifsa Iftikhar"
            today = date.today().strftime("%Y-%m-%d")
            text = f"{label} - {today}"
            height = result.shape[0]
            cv2.putText(result, text, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        elif operation == "Brightness/Contrast":
            # brightness added directly; contrast scales pixel values around
            # their midpoint. alpha=contrast (1.0 = no change), beta=brightness
            alpha = float(contrast)
            beta = float(brightness)
            result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        else:
            return None, "Please select an operation."

        return result, "Processing completed successfully."

    except Exception as e:
        return None, f"Something went wrong while processing the image: {e}"


def update_visible_controls(operation):
    """Shows only the input controls relevant to the selected operation."""
    return (
        gr.update(visible=(operation == "Resize")),
        gr.update(visible=(operation == "Resize")),
        gr.update(visible=(operation == "Rotate")),
        gr.update(visible=(operation == "Flip")),
        gr.update(visible=(operation == "Crop")),
        gr.update(visible=(operation == "Crop")),
        gr.update(visible=(operation == "Crop")),
        gr.update(visible=(operation == "Crop")),
        gr.update(visible=(operation == "Add Text")),
        gr.update(visible=(operation == "Brightness/Contrast")),
        gr.update(visible=(operation == "Brightness/Contrast")),
    )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
#title-block {
    text-align: center;
    padding: 1.5rem 1rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
#title-block h1 {
    color: white !important;
    font-size: 2rem !important;
    margin-bottom: 0.3rem !important;
}
#title-block p {
    color: #e0e7ff !important;
    font-size: 0.95rem !important;
}
.panel {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    padding: 1rem !important;
}
#process-btn {
    font-size: 1.05rem !important;
    height: 48px !important;
}
"""

with gr.Blocks(title="Image Processing Toolkit", theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="title-block"):
        gr.Markdown("# 🖼️ Image Processing Toolkit")
        gr.Markdown(
            "Upload an image, pick an operation, and preview the result instantly - built with OpenCV."
        )

    with gr.Accordion("How to use this app", open=False):
        gr.Markdown(
            """
            1. Upload an image below.
            2. Choose an operation from the dropdown - extra controls will appear if needed.
            3. Click **Process Image**.
            4. Preview the result on the right, and use the download icon on the output image to save it.
            """
        )

    with gr.Row():
        with gr.Column(elem_classes=["panel"]):
            input_image = gr.Image(type="numpy", label="Upload an image")
        with gr.Column(elem_classes=["panel"]):
            output_image = gr.Image(type="numpy", label="Processed Image")

    gr.Examples(
        examples=[
            [resolve("input/landscape.jpg")],
            [resolve("input/person.jpg")],
            [resolve("input/car.jpg")],
            [resolve("input/document.jpg")],
            [resolve("input/object.jpg")],
        ],
        inputs=input_image,
        label="Try a sample image",
    )

    operation = gr.Dropdown(
        choices=["Grayscale", "Resize", "Rotate", "Flip", "Crop",
                 "Draw Shapes", "Add Text", "Brightness/Contrast"],
        label="Select operation",
        value="Grayscale",
    )

    with gr.Row():
        resize_w = gr.Number(label="Resize width", value=300, visible=False)
        resize_h = gr.Number(label="Resize height", value=300, visible=False)
        rotate_choice = gr.Radio(["90 degrees", "180 degrees", "270 degrees"],
                                  label="Rotation angle", visible=False)
        flip_choice = gr.Radio(["Horizontal", "Vertical", "Both"],
                                label="Flip direction", visible=False)

    with gr.Row():
        crop_x1 = gr.Number(label="Crop x1 (left)", value=0, visible=False)
        crop_y1 = gr.Number(label="Crop y1 (top)", value=0, visible=False)
        crop_x2 = gr.Number(label="Crop x2 (right)", value=100, visible=False)
        crop_y2 = gr.Number(label="Crop y2 (bottom)", value=100, visible=False)

    name_text = gr.Textbox(label="Name to display on image", value="Hifsa Iftikhar", visible=False)

    with gr.Row():
        brightness = gr.Slider(-100, 100, value=0, label="Brightness", visible=False)
        contrast = gr.Slider(0.5, 3.0, value=1.0, label="Contrast", visible=False)

    operation.change(
        fn=update_visible_controls,
        inputs=operation,
        outputs=[resize_w, resize_h, rotate_choice, flip_choice,
                 crop_x1, crop_y1, crop_x2, crop_y2, name_text,
                 brightness, contrast],
    )

    output_text = gr.Textbox(label="Status", lines=2)

    with gr.Row():
        process_btn = gr.Button("✨ Process Image", variant="primary", elem_id="process-btn")
        clear_btn = gr.Button("Clear")

    process_btn.click(
        fn=process_image,
        inputs=[input_image, operation, resize_w, resize_h, rotate_choice,
                flip_choice, crop_x1, crop_y1, crop_x2, crop_y2,
                name_text, brightness, contrast],
        outputs=[output_image, output_text],
    )

    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=None,
        outputs=[input_image, output_image, output_text],
    )

if __name__ == "__main__":
    demo.launch()