import cv2
import os

image_path = "input/shapes.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    # Grayscale, then threshold to get a binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("output/1_threshold.jpg", thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    # Draw all contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("output/2_all_contours.jpg", contour_img)

    # Area, perimeter, and bounding rectangle for each
    result = img.copy()
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(f"Contour {i}: area={area:.0f}, perimeter={perimeter:.0f}, "
              f"bounding box=({x},{y},{w},{h})")

    cv2.imwrite("output/3_bounding_rects.jpg", result)
    print("\nSaved threshold, all-contours, and bounding-rectangle images to output/")
