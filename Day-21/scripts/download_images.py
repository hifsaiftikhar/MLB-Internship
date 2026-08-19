import os
import urllib.request
import cv2
import numpy as np
import sys

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def rotate_and_scale(img, angle=25, scale=0.85):
    """Apply rotation and scaling to create a matching pair image."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Perform affine transformation
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def perspective_warp(img):
    """Apply homography warp to simulate a different viewing perspective."""
    h, w = img.shape[:2]
    # Source points: 4 corners of image
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    # Destination points: Shift corners inward to skew perspective
    dest_pts = np.float32([
        [int(w * 0.1), int(h * 0.1)],
        [int(w * 0.9), int(h * 0.05)],
        [int(w * 0.85), int(h * 0.95)],
        [int(w * 0.15), int(h * 0.9)]
    ])
    # Get homography matrix
    H = cv2.getPerspectiveTransform(src_pts, dest_pts)
    # Warp image
    warped = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "input"))
    os.makedirs(input_dir, exist_ok=True)
    
    BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
    
    # 1. Real Image Pairs
    real_pairs = [
        # Pair 1: Box (book) and box in scene
        ("box.png", "box_in_scene.png", "pair01_1.png", "pair01_2.png"),
        # Pair 2: Graffiti texture
        ("graf1.png", "graf3.png", "pair02_1.png", "pair02_2.png"),
        # Pair 3: Stereo views
        ("left.jpg", "right.jpg", "pair03_1.png", "pair03_2.png"),
        # Pair 4: Aloe plant
        ("aloeL.jpg", "aloeR.jpg", "pair04_1.png", "pair04_2.png"),
        # Pair 5: Objects in scene
        ("pic1.png", "pic2.png", "pair05_1.png", "pair05_2.png")
    ]
    
    print("--- Downloading Real Image Pairs ---")
    for src1, src2, dst1, dst2 in real_pairs:
        p1 = os.path.join(input_dir, dst1)
        p2 = os.path.join(input_dir, dst2)
        if not os.path.exists(p1):
            download_file(BASE_URL + src1, p1)
        if not os.path.exists(p2):
            download_file(BASE_URL + src2, p2)
            
    # 2. Programmatic Image Pairs (Transformations)
    single_images = [
        # Src image, dst base name, transformation type
        ("starry_night.jpg", "pair06", "rotate"),
        ("building.jpg", "pair07", "warp"),
        ("fruits.jpg", "pair08", "rotate"),
        ("opencv-logo.png", "pair09", "warp"),
        ("baboon.jpg", "pair10", "rotate")
    ]
    
    print("\n--- Downloading and Generating Programmatic Pairs ---")
    for src, dst_prefix, transform_type in single_images:
        p1 = os.path.join(input_dir, f"{dst_prefix}_1.png")
        p2 = os.path.join(input_dir, f"{dst_prefix}_2.png")
        
        # Download base image
        temp_dest = os.path.join(input_dir, "temp_" + src)
        if not os.path.exists(p1) or not os.path.exists(p2):
            if download_file(BASE_URL + src, temp_dest):
                img = cv2.imread(temp_dest)
                if img is not None:
                    # Save first image (original)
                    cv2.imwrite(p1, img)
                    # Apply transformation for the second image
                    if transform_type == "rotate":
                        transformed_img = rotate_and_scale(img, angle=20, scale=0.8)
                    else:
                        transformed_img = perspective_warp(img)
                    cv2.imwrite(p2, transformed_img)
                    print(f"Generated transformed pair for {dst_prefix}")
                
                # Cleanup temp file
                if os.path.exists(temp_dest):
                    os.remove(temp_dest)
            else:
                print(f"Failed to prepare programmatic pair for {src}", file=sys.stderr)
                
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
