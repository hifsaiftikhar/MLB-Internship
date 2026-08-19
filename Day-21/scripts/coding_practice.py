import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(current_dir, "..", "input"))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    img1_path = os.path.join(input_dir, "pair01_1.png")
    img2_path = os.path.join(input_dir, "pair01_2.png")
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Error: Input images not found. Make sure download_images.py ran successfully.")
        return
        
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    print("==================================================")
    print("      DAY 21: FEATURE DETECTION & MATCHING        ")
    print("==================================================")
    
    # ------------------ Task 1: Harris Corner Detection ------------------
    print("\n[Task 1] Running Harris Corner Detection on Image 1...")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)
    
    start_time = time.time()
    # Harris Corner Detection: blockSize=2, ksize=3, free parameter k=0.04
    harris_dst = cv2.cornerHarris(gray1, blockSize=2, ksize=3, k=0.04)
    harris_time = (time.time() - start_time) * 1000  # in ms
    
    # Dilate corner results to make them visible
    harris_dst = cv2.dilate(harris_dst, None)
    
    # Threshold for an optimal value (corners marked in Red)
    img_harris = img1.copy()
    img_harris[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]
    
    harris_count = np.sum(harris_dst > 0.01 * harris_dst.max())
    cv2.imwrite(os.path.join(output_dir, "harris_corners.png"), img_harris)
    print(f"-> Harris Corner Detection complete. Time taken: {harris_time:.2f} ms")
    print(f"-> Detected corners above threshold: {harris_count}")
    print(f"-> Saved Harris visualization to output/harris_corners.png")
    
    # ------------------ Task 2 & 3: ORB Keypoint Detection & Visualisation ------------------
    print("\n[Task 2 & 3] Running ORB Feature Detection on Image 1...")
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=1000)
    
    start_time = time.time()
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    orb_time = (time.time() - start_time) * 1000  # in ms
    
    # Draw keypoints with rich circles showing size and orientation
    img_orb_kps = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite(os.path.join(output_dir, "orb_keypoints.png"), img_orb_kps)
    print(f"-> ORB Detection complete. Time taken: {orb_time:.2f} ms")
    print(f"-> Detected keypoints: {len(kp1)}")
    print(f"-> Saved ORB visualization to output/orb_keypoints.png")
    
    # ------------------ Task 4 & 5: Feature Matching with ORB & Brute Force Matcher ------------------
    print("\n[Task 4 & 5] Matching features between Image 1 and Image 2...")
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher object with Hamming distance (since ORB uses binary BRIEF descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Perform KNN Match (k=2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    
    # Filter matches using Lowe's ratio test (ratio=0.75)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches[:50], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0), 
        singlePointColor=(0, 0, 255)
    )
    
    cv2.imwrite(os.path.join(output_dir, "matched_features.png"), img_matches)
    print(f"-> Matching complete. Total raw matches: {len(raw_matches)}")
    print(f"-> Total good matches after Lowe's Ratio Test: {len(good_matches)}")
    print(f"-> Saved Matched Features visualization to output/matched_features.png")
    
    # ------------------ Task 6: Performance Comparison ------------------
    print("\n[Task 6] Performance Benchmarking & Comparison:")
    print("-" * 55)
    print(f"{'Metric':<30} | {'Harris Corners':<10} | {'ORB Features':<10}")
    print("-" * 55)
    print(f"{'Detection Speed (ms)':<30} | {harris_time:<14.2f} | {orb_time:<12.2f}")
    print(f"{'Detected Features Count':<30} | {harris_count:<14} | {len(kp1):<12}")
    print(f"{'Scale Invariance':<30} | {'NO':<14} | {'YES (Pyramids)':<12}")
    print(f"{'Rotation Invariance':<30} | {'YES (Partially)':<14} | {'YES (Intensity)':<12}")
    print(f"{'Descriptor Provided':<30} | {'NO':<14} | {'YES (BRIEF)':<12}")
    print("-" * 55)
    
    print("\nSummary Analysis:")
    print("1. Speed: ORB is exceptionally fast and optimized for real-time mobile/embedded devices, often beating Harris on complex scene grids.")
    print("2. Scale Invariance: Harris Corner Detection fails when scaling because corner contours look like straight lines at close zoom. ORB uses an image scale-pyramid to solve this.")
    print("3. Matching Capability: Harris Corners only find local position spikes and do not output descriptors. ORB produces binary BRIEF descriptors, enabling direct Brute-Force or FLANN distance matching.")
    print("==================================================\n")

if __name__ == "__main__":
    main()
