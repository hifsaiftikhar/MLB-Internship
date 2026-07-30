import cv2
import os

# 1. Read an image
image_path = "input/test_sample.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: could not read image at {image_path}")
else:
    # Dimensions and channel count
    height, width, channels = img.shape
    print(f"Image: {image_path}")
    print(f"Height: {height} pixels")
    print(f"Width: {width} pixels")
    print(f"Channels: {channels} (Blue, Green, Red)")

    # File size on disk
    file_size_bytes = os.path.getsize(image_path)
    file_size_kb = file_size_bytes / 1024
    print(f"File size: {file_size_bytes} bytes ({file_size_kb:.2f} KB)")