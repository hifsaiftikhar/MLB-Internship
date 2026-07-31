import cv2
import numpy as np
import os

image_path = "input/document.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    # Brightness adjustment
    brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
    darker = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
    cv2.imwrite("output/6_brighter.jpg", brighter)
    cv2.imwrite("output/6_darker.jpg", darker)
    print("Brightness increased and decreased")

    # Contrast adjustment
    high_contrast = cv2.convertScaleAbs(img, alpha=1.8, beta=0)
    low_contrast = cv2.convertScaleAbs(img, alpha=0.6, beta=0)
    cv2.imwrite("output/7_high_contrast.jpg", high_contrast)
    cv2.imwrite("output/7_low_contrast.jpg", low_contrast)
    print("Contrast increased and decreased")

    # Gaussian Blur - smooths everything including edges
    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite("output/8_gaussian_blur.jpg", gaussian)

    # Median Blur - better at keeping edges while removing noise
    median = cv2.medianBlur(img, 7)
    cv2.imwrite("output/8_median_blur.jpg", median)

    # Bilateral Filter - reduces noise while preserving edges
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite("output/8_bilateral_filter.jpg", bilateral)
    print("Gaussian Blur, Median Blur, and Bilateral Filter applied")

    # Sharpening kernel - emphasizes edges and text
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    sharpened = cv2.filter2D(img, -1, sharpen_kernel)
    cv2.imwrite("output/9_sharpened.jpg", sharpened)
    print("Sharpening applied")

    print("\nAll enhancement results saved to output/")