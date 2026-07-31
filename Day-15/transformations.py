import cv2
import numpy as np
import os

image_path = "input/document.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    os.makedirs("output", exist_ok=True)
    height, width = img.shape[:2]
    print(f"Original size: {width} x {height}")

    # Translation - shift image left/right and up/down
    tx, ty = 50, 30
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    cv2.imwrite("output/1_translated.jpg", translated)
    print(f"Translated by ({tx}, {ty}) pixels")

    # Rotation by an arbitrary angle
    angle = 30
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite("output/2_rotated_30deg.jpg", rotated)
    print(f"Rotated by {angle} degrees")

    # Scaling up and down
    scaled_up = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite("output/3_scaled_up.jpg", scaled_up)
    cv2.imwrite("output/3_scaled_down.jpg", scaled_down)
    print(f"Scaled up to {scaled_up.shape[1]}x{scaled_up.shape[0]}, "
          f"scaled down to {scaled_down.shape[1]}x{scaled_down.shape[0]}")

    # Affine transformation - maps 3 points, preserves parallel lines
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    dst_points = np.float32([[0, 0], [int(0.9 * (width - 1)), int(0.1 * height)],
                              [int(0.1 * width), height - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    affine_result = cv2.warpAffine(img, affine_matrix, (width, height))
    cv2.imwrite("output/4_affine.jpg", affine_result)
    print("Affine transformation applied")

    # Perspective transformation - maps 4 points, straightens tilted documents
    src_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst_pts = np.float32([[50, 20], [width - 30, 0], [width - 1, height - 40], [20, height - 1]])
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    perspective_result = cv2.warpPerspective(img, perspective_matrix, (width, height))
    cv2.imwrite("output/5_perspective.jpg", perspective_result)
    print("Perspective transformation applied")

    print("\nAll 5 transformations saved to output/")