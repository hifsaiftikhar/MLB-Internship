import cv2
import numpy as np
import os
from pathlib import Path

# Create output directory
os.makedirs("output", exist_ok=True)

def process_image(image_path):
    """Apply all three thresholding methods and save comparison."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path.name}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binary Thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 2. Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 3. Otsu Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Simple Foreground/Background Segmentation using Otsu + morphology
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Create foreground mask on original image
    mask = cleaned
    foreground = cv2.bitwise_and(img, img, mask=mask)

    # Save all outputs
    stem = image_path.stem
    cv2.imwrite(f"output/{stem}_1_binary.png", binary)
    cv2.imwrite(f"output/{stem}_2_adaptive.png", adaptive)
    cv2.imwrite(f"output/{stem}_3_otsu.png", otsu)
    cv2.imwrite(f"output/{stem}_4_foreground.png", foreground)

    print(f"Processed: {image_path.name}")
    print(f"  Binary threshold: 127 (fixed)")
    print(f"  Adaptive: Gaussian, block=11")
    print(f"  Otsu: auto-calculated threshold")
    print(f"  Saved 4 output images")
    print()

# Process all images in input folder
input_folder = Path("input")
images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))

if not images:
    print("No images found in input/ folder. Add images and run again.")
else:
    print(f"Found {len(images)} images. Processing...\n")
    for image_path in sorted(images):
        process_image(image_path)
    print("Done. All outputs saved to output/ folder.")
