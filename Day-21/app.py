import gradio as gr
import cv2
import numpy as np
import os
import time

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, "input")

# Human-readable pair mapping
PRESET_PAIRS = {
    "Pair 01: Book Cover (Perspective Match)": ("pair01_1.png", "pair01_2.png"),
    "Pair 02: Graffiti Wall (Rotation & Scale)": ("pair02_1.png", "pair02_2.png"),
    "Pair 03: Stereo Views (Baseline Offset)": ("pair03_1.png", "pair03_2.png"),
    "Pair 04: Aloe Plant (Stereo Offset)": ("pair04_1.png", "pair04_2.png"),
    "Pair 05: Objects in Scene (Scale & Pose)": ("pair05_1.png", "pair05_2.png"),
    "Pair 06: Starry Night (Rotation & Scaling)": ("pair06_1.png", "pair06_2.png"),
    "Pair 07: Landmark Building (Perspective Warp)": ("pair07_1.png", "pair07_2.png"),
    "Pair 08: Fruits Bowl (Affine Rotation)": ("pair08_1.png", "pair08_2.png"),
    "Pair 09: OpenCV Logo (Homography Distortion)": ("pair09_1.png", "pair09_2.png"),
    "Pair 10: Baboon Face (Skew & Rotate)": ("pair10_1.png", "pair10_2.png")
}

def load_preset(pair_name):
    """Load preset image files when chosen in the dropdown."""
    if not pair_name or pair_name not in PRESET_PAIRS:
        return None, None
    img1_file, img2_file = PRESET_PAIRS[pair_name]
    img1_path = os.path.join(INPUT_DIR, img1_file)
    img2_path = os.path.join(INPUT_DIR, img2_file)
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        return cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    return None, None

def match_features(img1, img2, max_features=500, ratio_test=0.75):
    """
    Perform ORB feature detection, BRIEF extraction, and Brute Force KNN matching.
    """
    if img1 is None or img2 is None:
        return None, 0, 0, 0, "Please upload or select both images."
        
    start_time = time.time()
    
    # Convert RGB (Gradio format) to BGR for OpenCV
    bgr_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    bgr_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    # Initialize ORB Detector
    orb = cv2.ORB_create(nfeatures=int(max_features))
    
    # Detect Keypoints and Compute Descriptors
    kp1, des1 = orb.detectAndCompute(bgr_img1, None)
    kp2, des2 = orb.detectAndCompute(bgr_img2, None)
    
    num_kp1 = len(kp1)
    num_kp2 = len(kp2)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        # Create output image showing original inputs side-by-side
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1+w2] = img2
        return canvas, num_kp1, num_kp2, 0, f"No descriptors detected. Check image texture."
        
    # Create BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors using KNN (k=2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filter matches using Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
                
    # Sort matches by distance (best matches first)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    # Draw matches (draw top 100 for readability)
    drawn_matches = cv2.drawMatches(
        bgr_img1, kp1, 
        bgr_img2, kp2, 
        good_matches[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255)
    )
    
    # Convert back to RGB for Gradio display
    rgb_output = cv2.cvtColor(drawn_matches, cv2.COLOR_BGR2RGB)
    elapsed_time = (time.time() - start_time) * 1000
    
    status = f"Extraction and matching completed in {elapsed_time:.1f} ms."
    return rgb_output, num_kp1, num_kp2, len(good_matches), status

def run_batch_analysis(max_features=500, ratio_test=0.75):
    """
    Run matching across all 10 preset pairs and compile a tabular summary.
    """
    results = []
    for pair_title, (f1, f2) in PRESET_PAIRS.items():
        p1 = os.path.join(INPUT_DIR, f1)
        p2 = os.path.join(INPUT_DIR, f2)
        
        if not os.path.exists(p1) or not os.path.exists(p2):
            results.append([pair_title.split(":")[0], "N/A", "N/A", "N/A", "Assets missing"])
            continue
            
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        
        start = time.time()
        orb = cv2.ORB_create(nfeatures=int(max_features))
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < ratio_test * n.distance]
            good_count = len(good)
        else:
            good_count = 0
            
        elapsed = (time.time() - start) * 1000
        results.append([
            pair_title.split(":")[0] + ": " + pair_title.split(":")[1].split(" (")[0], 
            len(kp1), 
            len(kp2), 
            good_count, 
            f"{elapsed:.1f} ms"
        ])
    return results

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Image Feature Matching System
        Compare two images using ORB features and Brute Force KNN matching.
        """
    )

    with gr.Tab("Feature Matching"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                preset_dd = gr.Dropdown(
                    choices=list(PRESET_PAIRS.keys()),
                    label="Preset Image Pair",
                )

                max_feat = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=500,
                    step=50,
                    label="Max ORB Features",
                )

                ratio_val = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    value=0.75,
                    step=0.05,
                    label="Lowe Ratio Threshold",
                )

                with gr.Row():
                    match_btn = gr.Button("Match Features", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=3):
                with gr.Row():
                    img1_input = gr.Image(label="Image 1", type="numpy")
                    img2_input = gr.Image(label="Image 2", type="numpy")

        gr.Markdown("### Results")

        with gr.Row():
            kps1_stat = gr.Number(
                label="Keypoints - Image 1", value=0, precision=0
            )
            kps2_stat = gr.Number(
                label="Keypoints - Image 2", value=0, precision=0
            )
            good_matches_stat = gr.Number(
                label="Good Matches", value=0, precision=0
            )

        status_text = gr.Textbox(
            label="Status",
            value="Ready to match.",
            interactive=False,
        )

        output_matches = gr.Image(
            label="Matched Features",
            type="numpy",
        )

        preset_dd.change(
            fn=load_preset,
            inputs=[preset_dd],
            outputs=[img1_input, img2_input],
        )

        match_btn.click(
            fn=match_features,
            inputs=[img1_input, img2_input, max_feat, ratio_val],
            outputs=[
                output_matches,
                kps1_stat,
                kps2_stat,
                good_matches_stat,
                status_text,
            ],
        )

        def clear_fields():
            return None, None, None, 0, 0, 0, "Cleared."

        clear_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[
                img1_input,
                img2_input,
                output_matches,
                kps1_stat,
                kps2_stat,
                good_matches_stat,
                status_text,
            ],
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown(
            """
            Run feature matching on all 10 preset image pairs and compare the
            number of keypoints, good matches, and processing time.
            """
        )

        batch_btn = gr.Button("Run Batch Analysis", variant="primary")

        batch_table = gr.Dataframe(
            headers=[
                "Pair Name",
                "Keypoints (Img 1)",
                "Keypoints (Img 2)",
                "Good Matches",
                "Inference Time",
            ],
            datatype=["str", "number", "number", "number", "str"],
            row_count=10,
            col_count=5,
            label="Batch Results",
        )

        batch_btn.click(
            fn=run_batch_analysis,
            inputs=[max_feat, ratio_val],
            outputs=[batch_table],
        )

    with gr.Tab("Documentation"):
        gr.Markdown(
            """
            ## Feature Detection Theory

            ### Harris Corner Detection
            Harris detects strong local changes in image intensity and is
            useful for finding corner-like features.

            ### ORB
            ORB combines FAST keypoint detection with a binary descriptor.
            It is designed to be fast and works well for practical feature
            matching tasks.

            ### Lowe's Ratio Test
            For each keypoint, the two closest descriptor matches are compared.
            A match is kept when the best match is sufficiently better than
            the second-best match.

            In this project, the default ratio threshold is **0.75**.
            """
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())