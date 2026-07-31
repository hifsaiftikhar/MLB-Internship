import os
import numpy as np
import cv2
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


def order_points(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]   # bottom-left: largest y-x
    return rect


def find_document_contour(img):
    """Finds the largest 4-sided contour - the classic document scanner approach."""
    image_area = img.shape[0] * img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        # Skip anything too small to plausibly be the actual document -
        # avoids mistaking a small flat background patch for the page.
        if cv2.contourArea(contour) < 0.2 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def correct_perspective(img):
    """Detects a document's corners and warps it to a straightened view."""
    corners = find_document_contour(img)
    if corners is None:
        return img, False

    rect = order_points(corners.astype("float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))
    return warped, True


def enhance_document(input_image):
    if input_image is None:
        return None, None, "Please upload an image before proceeding."

    try:
        if not isinstance(input_image, np.ndarray) or input_image.ndim not in (2, 3):
            return None, None, "Invalid file. Please upload a valid image."
    except Exception:
        return None, None, "Invalid file. Please upload a valid image."

    try:
        # Gradio uses RGB, OpenCV uses BGR - convert for processing
        img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # Correct perspective if tilted
        corrected, was_corrected = correct_perspective(img_bgr)

        # Convert to grayscale
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        # Reduce noise, keep text edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhance brightness and contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=15)

        # Sharpen
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

        final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

        status = "Processing completed successfully."
        status += " Perspective was corrected." if was_corrected else \
            " No clear document edges detected - perspective correction skipped."

        return corrected_rgb, final_rgb, status

    except Exception as e:
        return None, None, f"Something went wrong while processing the image: {e}"


with gr.Blocks(title="Document Image Enhancement Tool", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 📄 Document Image Enhancement Tool")
    gr.Markdown(
        "Upload a photo of a document (even if tilted) - this app automatically "
        "detects its edges, straightens it, then reduces noise and enhances "
        "brightness, contrast, and sharpness for a clean, scanned look."
    )

    input_image = gr.Image(type="numpy", label="Upload a document photo")

    gr.Examples(
        examples=[
            [resolve("input/sample1.jpg")],
            [resolve("input/sample2.jpg")],
            [resolve("input/sample3.jpg")],
        ],
        inputs=input_image,
        label="Try a sample image",
    )

    with gr.Row():
        corrected_image = gr.Image(type="numpy", label="Perspective-Corrected")
        final_image = gr.Image(type="numpy", label="Final Enhanced")

    status_box = gr.Textbox(label="Status", lines=2)

    process_btn = gr.Button("Enhance Document", variant="primary")
    process_btn.click(
        fn=enhance_document,
        inputs=input_image,
        outputs=[corrected_image, final_image, status_box],
    )

if __name__ == "__main__":
    demo.launch()