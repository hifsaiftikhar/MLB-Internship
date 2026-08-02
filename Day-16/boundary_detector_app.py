import os
import numpy as np
import cv2
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


def detect_boundary(input_image):
    if input_image is None:
        return None, "Please upload an image before proceeding."

    try:
        if not isinstance(input_image, np.ndarray) or input_image.ndim not in (2, 3):
            return None, "Invalid file. Please upload a valid image."
    except Exception:
        return None, "Invalid file. Please upload a valid image."

    try:
        img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # Grayscale, blur, Canny
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological closing - fills small gaps in the edge lines
        # so the document outline forms one continuous contour
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find the largest contour - assumed to be the document
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = img_bgr.shape[0] * img_bgr.shape[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        result = img_bgr.copy()
        boundary_found = False

        for contour in contours:
            if cv2.contourArea(contour) < 0.15 * image_area:
                continue
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
            boundary_found = True
            break

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if boundary_found:
            status = "Processing completed successfully. Document boundary detected."
        else:
            status = "Processing completed. No confident document boundary found - showing original image."

        return result_rgb, status

    except Exception as e:
        return None, f"Something went wrong while processing the image: {e}"


with gr.Blocks(title="Document Boundary Detection Tool", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# Document Boundary Detection Tool")
    gr.Markdown(
        "Upload a document photo - this app detects its boundary using Canny edge "
        "detection and morphological operations, then draws it on the original image."
    )

    input_image = gr.Image(type="numpy", label="Upload a document photo")

    gr.Examples(
        examples=[
            [resolve("input/document.jpg")],
        ],
        inputs=input_image,
        label="Try a sample image",
    )

    output_image = gr.Image(type="numpy", label="Detected Boundary")
    status_box = gr.Textbox(label="Status", lines=2)

    process_btn = gr.Button("Detect Boundary", variant="primary")
    process_btn.click(fn=detect_boundary, inputs=input_image, outputs=[output_image, status_box])

if __name__ == "__main__":
    demo.launch()
