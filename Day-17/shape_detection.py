import cv2
import os

image_path = "input/shapes.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def detect_shape(contour):
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
    else:
        return "Polygon"


if img is not None:
    result = img.copy()
    for c in contours:
        shape = detect_shape(c)
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
        cv2.putText(result, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
        print(f"Detected: {shape} at ({x},{y})")

    cv2.imwrite("output/4_shapes_labeled.jpg", result)
    print("\nLabeled shape image saved to output/4_shapes_labeled.jpg")
