import cv2
import numpy as np
import os
from datetime import date

# The 5 required images for the Challenge Task
images = {
    "landscape": "input/landscape.jpg",
    "person": "input/person.jpg",
    "vehicle": "input/car.jpg",
    "document": "input/document.jpg",
    "object": "input/object.jpg",
}


def process_all_operations(img, output_dir):
    """Applies every toolkit operation to one image and saves each result
    into its own subfolder under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    height, width = img.shape[:2]

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/grayscale.jpg", gray)

    # 2. Resize (to 3 different resolutions)
    cv2.imwrite(f"{output_dir}/resized_small.jpg", cv2.resize(img, (150, 100)))
    cv2.imwrite(f"{output_dir}/resized_medium.jpg", cv2.resize(img, (400, 300)))
    cv2.imwrite(f"{output_dir}/resized_large.jpg", cv2.resize(img, (800, 600)))

    # 3. Crop (center region)
    cy1, cy2 = height // 4, 3 * height // 4
    cx1, cx2 = width // 4, 3 * width // 4
    cropped = img[cy1:cy2, cx1:cx2]
    cv2.imwrite(f"{output_dir}/cropped_center.jpg", cropped)

    # 4. Rotate (90, 180, 270)
    cv2.imwrite(f"{output_dir}/rotated_90.jpg", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(f"{output_dir}/rotated_180.jpg", cv2.rotate(img, cv2.ROTATE_180))
    cv2.imwrite(f"{output_dir}/rotated_270.jpg", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 5. Flip (horizontal, vertical)
    cv2.imwrite(f"{output_dir}/flipped_horizontal.jpg", cv2.flip(img, 1))
    cv2.imwrite(f"{output_dir}/flipped_vertical.jpg", cv2.flip(img, 0))

    # 6. Draw shapes (rectangle, circle, line, polygon)
    shapes_img = img.copy()
    cv2.rectangle(shapes_img, (10, 10), (width // 3, height // 3), (0, 255, 0), 2)
    cv2.circle(shapes_img, (width // 2, height // 2), min(width, height) // 6, (255, 0, 0), 2)
    cv2.line(shapes_img, (0, height - 1), (width - 1, 0), (0, 0, 255), 2)
    points = np.array([
        [width // 2, height - 10],
        [width - 10, height // 2],
        [width // 2, 10],
    ], dtype=np.int32)
    cv2.polylines(shapes_img, [points], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.imwrite(f"{output_dir}/shapes.jpg", shapes_img)

    # 7. Add custom text (name + today's date)
    text_img = img.copy()
    today = date.today().strftime("%Y-%m-%d")
    text = f"Hifsa Iftikhar - {today}"
    cv2.putText(text_img, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{output_dir}/text.jpg", text_img)

    # Bonus: brightness/contrast adjustment
    bright_contrast = cv2.convertScaleAbs(img, alpha=1.3, beta=25)
    cv2.imwrite(f"{output_dir}/brightness_contrast.jpg", bright_contrast)

    return 11  # number of output files generated per image


print("===== CHALLENGE TASK: PROCESSING 5 IMAGES =====\n")

for label, path in images.items():
    img = cv2.imread(path)
    if img is None:
        print(f"[{label}] Error: could not read image at {path} - skipping.")
        continue

    output_dir = f"output/challenge_{label}"
    count = process_all_operations(img, output_dir)
    print(f"[{label}] Loaded from {path} ({img.shape[1]}x{img.shape[0]}) "
          f"-> {count} processed images saved to {output_dir}/")

print("\nChallenge Task complete. All 5 images processed with every operation.")
