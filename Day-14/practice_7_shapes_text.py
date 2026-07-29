import cv2
import os
from datetime import date

image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)
    height, width = img.shape[:2]

    # Work on a copy so the original image stays untouched -
    # cv2 drawing functions modify the array in place.
    canvas = img.copy()

    # 1. Rectangle
    # cv2.rectangle(image, top_left_point, bottom_right_point, color_BGR, thickness)
    cv2.rectangle(canvas, (10, 10), (width // 3, height // 3), (0, 255, 0), 2)

    # 2. Circle
    # cv2.circle(image, center_point, radius, color_BGR, thickness)
    center = (width // 2, height // 2)
    cv2.circle(canvas, center, min(width, height) // 6, (255, 0, 0), 2)

    # 3. Line
    cv2.line(canvas, (0, height - 1), (width - 1, 0), (0, 0, 255), 2)

    # 4. Polygon (a simple triangle here, using a set of points)
    import numpy as np
    points = np.array([
        [width // 2, height - 10],
        [width - 10, height // 2],
        [width // 2, 10],
    ], dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 255), thickness=2)

    # 5. Custom text: name and today's date
    today = date.today().strftime("%Y-%m-%d")
    text = f"Hifsa Iftikhar - {today}"
    cv2.putText(
        canvas, text, (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    output_path = "output/shapes_and_text.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"Rectangle, circle, line, polygon, and text drawn.")
    print(f"Saved to {output_path}")
