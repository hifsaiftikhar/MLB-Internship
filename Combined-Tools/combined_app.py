import os
import numpy as np
import cv2
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


# ===== Shared helper =====

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_contour(img, min_area_ratio=0.2):
    image_area = img.shape[0] * img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in contours:
        if cv2.contourArea(contour) < min_area_ratio * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


# ===== Tab 1: Day-15 =====

def correct_perspective(img):
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
    dst = np.array([[0, 0], [max_width - 1, 0],
                    [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
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
        img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        corrected, was_corrected = correct_perspective(img_bgr)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=15)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        status = "Processing completed successfully."
        status += " Perspective was corrected." if was_corrected else \
            " No clear document edges detected - perspective correction skipped."
        return corrected_rgb, final_rgb, status
    except Exception as e:
        return None, None, f"Something went wrong: {e}"


# ===== Tab 2: Day-16 =====

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
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
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
        status = "Document boundary detected." if boundary_found else \
            "No confident document boundary found - showing original image."
        return result_rgb, status
    except Exception as e:
        return None, f"Something went wrong: {e}"


# ===== Build example lists =====

tab1_examples = []
tab2_examples = []

for i in (1, 2, 3):
    p1 = resolve(f"input/tab1_sample{i}.jpg")
    p2 = resolve(f"input/tab2_sample{i}.jpg")
    if os.path.exists(p1):
        tab1_examples.append(p1)
    if os.path.exists(p2):
        tab2_examples.append(p2)

print("Tab1 examples:", tab1_examples)
print("Tab2 examples:", tab2_examples)


# ===== Combined app =====

with gr.Blocks(title="Document Tools") as demo:
    gr.Markdown("# Document Processing Tools")
    gr.Markdown("Two tools in one app - switch tabs to try either.")

    with gr.Tabs():
        with gr.TabItem("Day-15: Document Image Enhancement"):
            gr.Markdown(
                "Upload a document photo - automatically detects tilt, straightens it, "
                "then reduces noise and enhances brightness, contrast, and sharpness."
            )
            input_1 = gr.Image(type="numpy", label="Upload a document photo")
            if tab1_examples:
                gr.Examples(
                    examples=[[p] for p in tab1_examples],
                    inputs=input_1,
                    label="Try a sample image",
                    examples_per_page=3,
                    cache_examples=False
                )
            with gr.Row():
                corrected_out = gr.Image(type="numpy", label="Perspective-Corrected")
                final_out = gr.Image(type="numpy", label="Final Enhanced")
            status_1 = gr.Textbox(label="Status", lines=2)
            btn_1 = gr.Button("Enhance Document", variant="primary")
            btn_1.click(fn=enhance_document, inputs=input_1, outputs=[corrected_out, final_out, status_1])

        with gr.TabItem("Day-16: Document Boundary Detection"):
            gr.Markdown(
                "Upload a document photo - detects its boundary using Canny edge detection "
                "and morphological operations, then draws it on the original image."
            )
            input_2 = gr.Image(type="numpy", label="Upload a document photo")
            if tab2_examples:
                with gr.Accordion("Try a sample image", open=True):
                    sample_gallery = gr.Gallery(
                        value=tab2_examples,
                        columns=3,
                        height=150,
                        label=None,
                        show_label=False,
                        interactive=False
                    )
                    def load_tab2_sample(evt: gr.SelectData):
                        img = cv2.imread(tab2_examples[evt.index])
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    sample_gallery.select(
                        fn=load_tab2_sample,
                        inputs=None,
                        outputs=input_2
                    )
            boundary_out = gr.Image(type="numpy", label="Detected Boundary")
            status_2 = gr.Textbox(label="Status", lines=2)
            btn_2 = gr.Button("Detect Boundary", variant="primary")
            btn_2.click(fn=detect_boundary, inputs=input_2, outputs=[boundary_out, status_2])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"))