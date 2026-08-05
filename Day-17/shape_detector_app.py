import os
import numpy as np
import cv2
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    corners = len(approx)

    if corners == 3:
        return "Triangle"
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif corners > 4:
        return "Circle"
    return "Polygon"


def detect_shapes(input_image):
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
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = img_bgr.copy()
        lines = []
        count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perimeter = cv2.arcLength(c, True)
            shape = classify_shape(c)
            x, y, w, h = cv2.boundingRect(c)

            cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
            cv2.putText(result, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            lines.append(f"{shape}: area={area:.0f}, perimeter={perimeter:.0f}")
            count += 1

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        status = f"Detected {count} shape(s).\n\n" + "\n".join(lines) if count else \
            "No shapes detected. Try an image with clearer shapes against a plain background."
        return result_rgb, status

    except Exception as e:
        return None, f"Something went wrong: {e}"


with gr.Blocks(title="Shape Detection System", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# Shape Detection System")
    gr.Markdown(
        "Upload an image with simple shapes - detects each shape, labels it "
        "(Circle, Rectangle, Square, Triangle, Polygon), and shows its area and perimeter."
    )

    input_image = gr.Image(type="numpy", label="Upload an image")

    examples = []
    for i in (1, 2, 3):
        for ext in (".jpg", ".png", ".jpeg"):
            path = resolve(f"input/sample{i}{ext}")
            if os.path.exists(path):
                examples.append(path)
                break
    gr.Examples(examples=[[p] for p in examples], inputs=input_image, label="Try a sample image")

    output_image = gr.Image(type="numpy", label="Detected Shapes")
    status_box = gr.Textbox(label="Results", lines=6)

    process_btn = gr.Button("Detect Shapes", variant="primary")
    process_btn.click(fn=detect_shapes, inputs=input_image, outputs=[output_image, status_box])

if __name__ == "__main__":
    demo.launch()
