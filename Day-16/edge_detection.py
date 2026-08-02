import cv2
import numpy as np
import os

image_path = "input/document.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    # 1. Convert to grayscale - edge detection works on single-channel intensity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/1_grayscale.jpg", gray)

    # 2. Gaussian Blur - reduces noise before edge detection, since raw
    # pixel noise creates lots of tiny false edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("output/2_blurred.jpg", blurred)
    print("Grayscale and blur applied")

    # 3. Sobel - gradient in X and Y direction, combined
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    cv2.imwrite("output/3_sobel.jpg", sobel_combined)

    # Laplacian - second-derivative edge detector, all directions at once
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    cv2.imwrite("output/3_laplacian.jpg", laplacian)

    # Canny - multi-step detector, generally cleanest result
    canny = cv2.Canny(blurred, 50, 150)
    cv2.imwrite("output/3_canny.jpg", canny)

    print("Sobel, Laplacian, and Canny edge detection applied")

    # Side-by-side comparison of all three
    comparison = np.hstack([
        cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR),
    ])
    cv2.imwrite("output/3_comparison.jpg", comparison)
    print("Comparison image saved (Sobel | Laplacian | Canny)")
