import cv2
import numpy as np
import os

image_path = "input/document.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite("output/4_before_morph.jpg", edges)

    kernel = np.ones((5, 5), np.uint8)

    # Erosion - shrinks white regions, removes small noise
    erosion = cv2.erode(edges, kernel, iterations=1)
    cv2.imwrite("output/5_erosion.jpg", erosion)

    # Dilation - grows white regions, fills small gaps
    dilation = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite("output/5_dilation.jpg", dilation)

    # Opening - erosion then dilation, removes noise while keeping shape
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("output/5_opening.jpg", opening)

    # Closing - dilation then erosion, fills small holes/gaps
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("output/5_closing.jpg", closing)

    # Morphological Gradient - dilation minus erosion, highlights outline
    gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite("output/5_gradient.jpg", gradient)

    # Top Hat - original minus opening, highlights small bright details
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite("output/5_tophat.jpg", tophat)

    # Black Hat - closing minus original, highlights small dark details
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite("output/5_blackhat.jpg", blackhat)

    print("All 7 morphological operations applied")

    # Before/after comparison (Canny edges vs closing, which is most
    # relevant for the document boundary tool built next)
    comparison = np.hstack([edges, closing])
    cv2.imwrite("output/5_before_after.jpg", comparison)
    print("Before/after comparison saved (edges | closing)")
