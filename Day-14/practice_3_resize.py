import cv2
import os

image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)
    original_height, original_width = img.shape[:2]
    print(f"Original size: {original_width} x {original_height}")

    # Resize to a few fixed resolutions.
    # cv2.resize takes (width, height) - opposite order from img.shape.
    sizes = {
        "small": (150, 100),
        "medium": (400, 300),
        "large": (800, 600),
    }

    for name, (w, h) in sizes.items():
        resized = cv2.resize(img, (w, h))
        output_path = f"output/resized_{name}.jpg"
        cv2.imwrite(output_path, resized)
        print(f"Resized to {w}x{h} -> saved as {output_path}")

    # Resize by scale factor instead of fixed size
    scale = 0.5
    scaled = cv2.resize(img, None, fx=scale, fy=scale)
    scaled_h, scaled_w = scaled.shape[:2]
    print(f"\nResized by {scale}x scale -> {scaled_w}x{scaled_h}")
    cv2.imwrite("output/resized_half_scale.jpg", scaled)