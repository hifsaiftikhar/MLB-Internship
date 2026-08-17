import easyocr
import gradio as gr
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

reader = easyocr.Reader(['en'])
os.makedirs("output", exist_ok=True)

# Load sample images
def get_samples():
    samples = []
    for i in (1, 2, 3):
        p = f"input/sample{i}.jpg"
        if os.path.exists(p):
            samples.append(p)
    return samples

def extract_text(image, apply_preprocessing):
    if image is None:
        return "Upload an image to get started.", None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if apply_preprocessing:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        processed = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        input_img = processed
    else:
        input_img = img_bgr

    results = reader.readtext(input_img)
    lines = [text for (_, text, conf) in results if conf > 0.3]
    extracted = "\n".join(lines) if lines else "No text detected with confidence above 30%."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/extracted_{timestamp}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted)

    return extracted, output_path

samples = get_samples()

with gr.Blocks(title="OCR Document Reader", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # OCR Document Reader
    **Extract text from any image instantly using EasyOCR.**

    Upload a photo of a document, receipt, signboard, book page, or any image containing text.
    The app reads all visible text and lets you download the result as a `.txt` file.

    **How it works:**
    - Upload your image or pick a sample below
    - Optionally enable preprocessing to improve results on dark or blurry images
    - Click Extract Text — results appear on the right
    - Download the extracted text as a file
    """)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(type="numpy", label="Image Input")

            if samples:
                gr.Markdown("**Try a sample:**")
                gr.Examples(
                    examples=[[s] for s in samples],
                    inputs=input_image,
                    label="Sample Images"
                )

            preprocess = gr.Checkbox(
                label="Apply Preprocessing",
                value=False,
                info="Converts to grayscale and enhances contrast. Helps with dark or low-quality images."
            )
            btn = gr.Button("Extract Text", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Extracted Text")
            output_text = gr.Textbox(
                label="Result",
                lines=18,
                placeholder="Extracted text will appear here after clicking Extract Text..."
            )
            output_file = gr.File(label="Download as .txt")

    gr.Markdown("---")
    gr.Markdown("""
    **Tips for better results:**
    - Use clear, well-lit images
    - Avoid heavy shadows or glare
    - Enable preprocessing for faded or dark text
    - Higher resolution images give more accurate results
    """)

    btn.click(
        fn=extract_text,
        inputs=[input_image, preprocess],
        outputs=[output_text, output_file]
    )

if __name__ == "__main__":
    demo.launch()