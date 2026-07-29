import cv2
import os

image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)
    height, width = img.shape[:2]
    print(f"Original size: {width} x {height}")

    # Cropping in OpenCV is just slicing the image array:
    # img[y_start:y_end, x_start:x_end]
    # Note the order: ROWS (height/y) come first, then COLUMNS (width/x) -
    # opposite of how we normally think of (x, y) coordinates.

    # 1. Top-left quarter
    top_left = img[0:height // 2, 0:width // 2]
    cv2.imwrite("output/crop_top_left.jpg", top_left)

    # 2. Center region
    center_y1, center_y2 = height // 4, 3 * height // 4
    center_x1, center_x2 = width // 4, 3 * width // 4
    center = img[center_y1:center_y2, center_x1:center_x2]
    cv2.imwrite("output/crop_center.jpg", center)

    # 3. Bottom-right quarter
    bottom_right = img[height // 2:height, width // 2:width]
    cv2.imwrite("output/crop_bottom_right.jpg", bottom_right)

    # 4. A custom, specific region (top strip across the full width)
    top_strip = img[0:height // 3, 0:width]
    cv2.imwrite("output/crop_top_strip.jpg", top_strip)

    print("Saved 4 cropped regions:")
    print(f"  Top-left quarter: {top_left.shape[1]}x{top_left.shape[0]}")
    print(f"  Center region: {center.shape[1]}x{center.shape[0]}")
    print(f"  Bottom-right quarter: {bottom_right.shape[1]}x{bottom_right.shape[0]}")
    print(f"  Top strip: {top_strip.shape[1]}x{top_strip.shape[0]}")
