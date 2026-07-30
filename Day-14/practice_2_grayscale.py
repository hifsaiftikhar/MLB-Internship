import cv2
import os

# 1. Read the original color image
image_path = "input/test_sample.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    print("Original image shape (Height, Width, Channels):", img.shape)

    # Convert BGR image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Grayscale image shape (Height, Width):", gray_img.shape)
    # No channel count - grayscale has one value per pixel, not three.

    # 3. Save the grayscale result
    os.makedirs("output", exist_ok=True)
    output_path = "output/grayscale_result.jpg"
    cv2.imwrite(output_path, gray_img)
    print(f"Grayscale image saved to {output_path}")