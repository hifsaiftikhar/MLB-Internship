import cv2
import os

image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)

    # cv2.flip(img, flip_code):
    # flip_code = 1  -> horizontal flip (mirror left-right)
    # flip_code = 0  -> vertical flip (mirror top-bottom)
    # flip_code = -1 -> both at once (180-degree-like flip on both axes)

    flipped_horizontal = cv2.flip(img, 1)
    cv2.imwrite("output/flipped_horizontal.jpg", flipped_horizontal)
    print("Horizontal flip saved (mirrored left-right).")

    flipped_vertical = cv2.flip(img, 0)
    cv2.imwrite("output/flipped_vertical.jpg", flipped_vertical)
    print("Vertical flip saved (mirrored top-bottom).")

    flipped_both = cv2.flip(img, -1)
    cv2.imwrite("output/flipped_both.jpg", flipped_both)
    print("Both-axis flip saved (mirrored left-right AND top-bottom).")

    print("\nNote: dimensions stay exactly the same for all flips - "
          "only the pixel arrangement changes, unlike rotation by 90/270 "
          "which swaps width and height.")
