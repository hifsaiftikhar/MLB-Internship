import cv2
import os

# 1. Read the original color image
image_path = "input/landscape.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    print("Original image shape (Height, Width, Channels):", img.shape)

    # 2. Convert color (BGR) image to grayscale
    # cv2.cvtColor converts between color spaces. BGR2GRAY collapses the
    # 3 color channels down to a single brightness channel.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Grayscale image shape (Height, Width):", gray_img.shape)
    # Notice: grayscale has only 2 dimensions (no channel count), since
    # there's only one value per pixel instead of three.

    # 3. Save the grayscale result
    os.makedirs("output", exist_ok=True)
    output_path = "output/grayscale_result.jpg"
    cv2.imwrite(output_path, gray_img)
    print(f"Grayscale image saved to {output_path}")
