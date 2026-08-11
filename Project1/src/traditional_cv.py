import cv2
import numpy as np
import src.config as config

def enhance_image(frame):
    """
    Step 1 & 2: Grayscale conversion and Contrast Enhancement (CLAHE)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, 
        tileGridSize=config.CLAHE_GRID_SIZE
    )
    enhanced = clahe.apply(gray)
    return gray, enhanced

def detect_edges(gray_frame, low_threshold=config.CANNY_LOW_THRESHOLD, high_threshold=config.CANNY_HIGH_THRESHOLD):
    """
    Step 3 & 4: Gaussian Blur noise reduction and Canny Edge Detection
    """
    blurred = cv2.GaussianBlur(gray_frame, config.GAUSSIAN_BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return blurred, edges

def apply_morphology(edge_frame, kernel_size=config.MORPHOLOGY_KERNEL_SIZE, iterations=config.MORPHOLOGY_ITERATIONS):
    """
    Step 5: Morphological Operations (Dilation to close gaps between edges)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(edge_frame, kernel, iterations=iterations)
    return dilated

def calculate_slot_density(dilated_edges, slot_points):
    """
    Step 6: Compute pixel density of edges inside the slot polygon.
    """
    mask = np.zeros(dilated_edges.shape, dtype=np.uint8)
    poly_pts = np.array(slot_points, dtype=np.int32)
    cv2.fillPoly(mask, [poly_pts], 255)
    
    slot_area = np.sum(mask > 0)
    if slot_area == 0:
        return 0.0
        
    active_edges = cv2.bitwise_and(dilated_edges, mask)
    active_pixel_count = np.sum(active_edges > 0)
    
    density = float(active_pixel_count) / float(slot_area)
    return density

def run_traditional_cv_pipeline(frame, low_threshold=config.CANNY_LOW_THRESHOLD, high_threshold=config.CANNY_HIGH_THRESHOLD):
    """
    Orchestrate the complete traditional CV pipeline and return intermediary results
    """
    gray, enhanced = enhance_image(frame)
    blurred, edges = detect_edges(enhanced, low_threshold, high_threshold)
    dilated = apply_morphology(edges)
    
    return {
        "gray": gray,
        "enhanced": enhanced,
        "blurred": blurred,
        "edges": edges,
        "dilated": dilated
    }
