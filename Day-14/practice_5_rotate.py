import cv2
import os

image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)
    original_h, original_w = img.shape[:2]
    print(f"Original size: {original_w} x {original_h}")

    # OpenCV has built-in constants for the 3 most common fixed rotations,
    # which is simpler and faster than computing a rotation matrix manually.

    # 90 degrees clockwise
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("output/rotated_90.jpg", rotated_90)
    print(f"Rotated 90 deg: {rotated_90.shape[1]}x{rotated_90.shape[0]} "
          f"(width and height swap for a 90-degree turn)")

    # 180 degrees
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite("output/rotated_180.jpg", rotated_180)
    print(f"Rotated 180 deg: {rotated_180.shape[1]}x{rotated_180.shape[0]} "
          f"(dimensions stay the same, image is upside down)")

    # 270 degrees clockwise (equivalent to 90 degrees counter-clockwise)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite("output/rotated_270.jpg", rotated_270)
    print(f"Rotated 270 deg: {rotated_270.shape[1]}x{rotated_270.shape[0]} "
          f"(width and height swap again)")
