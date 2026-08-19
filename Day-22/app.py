import cv2
import numpy as np
import gradio as gr
import os


# Create output directory
os.makedirs("output", exist_ok=True)


# --------------------------------------------------
# Image Segmentation Function
# --------------------------------------------------

def segment_image(image):

    if image is None:
        return None, None, None, None

    # Gradio gives RGB image
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binary Thresholding
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # 2. Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 3. Otsu Thresholding
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Foreground / Background Segmentation
    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel
    )

    foreground = cv2.bitwise_and(
        img,
        img,
        mask=cleaned
    )

    # Convert outputs back to RGB
    binary_rgb = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2RGB
    )

    adaptive_rgb = cv2.cvtColor(
        adaptive,
        cv2.COLOR_GRAY2RGB
    )

    otsu_rgb = cv2.cvtColor(
        otsu,
        cv2.COLOR_GRAY2RGB
    )

    foreground_rgb = cv2.cvtColor(
        foreground,
        cv2.COLOR_BGR2RGB
    )

    # Save best segmentation result
    output_path = "output/best_segmentation.png"

    cv2.imwrite(
        output_path,
        foreground
    )

    return (
        binary_rgb,
        adaptive_rgb,
        otsu_rgb,
        foreground_rgb
    )


# --------------------------------------------------
# Sample Images
# --------------------------------------------------

sample_images = []

sample_files = [
    "sample01.jpg",
    "sample02.jpg",
    "sample04.png"
]

for filename in sample_files:

    path = os.path.join("input", filename)

    if os.path.exists(path):
        sample_images.append(path)


# --------------------------------------------------
# Gradio Application
# --------------------------------------------------

with gr.Blocks(
    title="Document & Object Segmentation Tool"
) as app:

    gr.Markdown(
        """
        # 🖼️ Document & Object Segmentation Tool

        ### Image Segmentation using OpenCV

        Upload an image or select one of the sample images
        to compare different thresholding and segmentation
        techniques.

        **Methods:** Binary • Adaptive • Otsu • Foreground Segmentation
        """
    )

    gr.Markdown("---")

    # Input Section
    gr.Markdown("## 📥 Input Image")

    input_image = gr.Image(
        type="numpy",
        label="Upload Image",
        height=400
    )

    if sample_images:

        gr.Markdown(
            "### 🖼️ Try a Sample Image"
        )

        gr.Examples(
            examples=sample_images,
            inputs=input_image,
            label="Sample Images"
        )

    segment_button = gr.Button(
        "🔍 Segment Image",
        variant="primary",
        size="lg"
    )

    gr.Markdown("---")

    # Results
    gr.Markdown(
        """
        ## 📊 Segmentation Results

        Compare how each thresholding method handles
        the selected image.
        """
    )

    with gr.Row():

        binary_output = gr.Image(
            label="1. Binary Thresholding",
            height=350
        )

        adaptive_output = gr.Image(
            label="2. Adaptive Thresholding",
            height=350
        )

    with gr.Row():

        otsu_output = gr.Image(
            label="3. Otsu Thresholding",
            height=350
        )

        foreground_output = gr.Image(
            label="4. Foreground Segmentation",
            height=350
        )

    gr.Markdown("---")

    gr.Markdown(
        """
        ### 📌 Method Information

        **Binary Thresholding:** Uses a fixed threshold value of 127.

        **Adaptive Thresholding:** Calculates a local threshold
        for different regions of the image.

        **Otsu Thresholding:** Automatically calculates an optimal
        global threshold.

        **Foreground Segmentation:** Uses Otsu thresholding followed
        by morphological operations to reduce noise.
        """
    )

    gr.Markdown(
        """
        ### 💾 Best Result

        The foreground segmentation result is automatically saved
        to the `output/` folder as `best_segmentation.png`.
        """
    )

    # Button action
    segment_button.click(
        fn=segment_image,
        inputs=input_image,
        outputs=[
            binary_output,
            adaptive_output,
            otsu_output,
            foreground_output
        ]
    )


# --------------------------------------------------
# Launch Application
# --------------------------------------------------

if __name__ == "__main__":
    app.launch()