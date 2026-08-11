import cv2
import numpy as np

def points_to_array(points):
    """Convert JSON points list of lists/dicts to a numpy array of shape (N, 1, 2) for cv2."""
    if isinstance(points[0], dict):
        pts = [[p['x'], p['y']] for p in points]
    else:
        pts = points
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

def get_slot_bounding_box(poly_pts):
    """Get the bounding box of a polygon."""
    x_coords = [p[0] for p in poly_pts]
    y_coords = [p[1] for p in poly_pts]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def calculate_overlap_ios(slot_points, bbox):
    """
    Calculate the Intersection-over-Slot (IoS) ratio.
    slot_points: list of [x, y] coordinates (4 points)
    bbox: list/tuple of [x_min, y_min, x_max, y_max] from YOLO
    """
    poly_pts = np.array(slot_points, dtype=np.int32)
    bx1, by1, bx2, by2 = map(int, bbox)
    
    sx1, sy1, sx2, sy2 = get_slot_bounding_box(poly_pts)
    
    x_min = min(sx1, bx1)
    y_min = min(sy1, by1)
    x_max = max(sx2, bx2)
    y_max = max(sy2, by2)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        return 0.0
        
    slot_mask = np.zeros((height, width), dtype=np.uint8)
    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    
    shifted_poly = poly_pts - [x_min, y_min]
    shifted_bbox_p1 = (bx1 - x_min, by1 - y_min)
    shifted_bbox_p2 = (bx2 - x_min, by2 - y_min)
    
    cv2.fillPoly(slot_mask, [shifted_poly], 255)
    cv2.rectangle(bbox_mask, shifted_bbox_p1, shifted_bbox_p2, 255, -1)
    
    intersection = cv2.bitwise_and(slot_mask, bbox_mask)
    intersection_area = np.sum(intersection > 0)
    slot_area = np.sum(slot_mask > 0)
    
    if slot_area == 0:
        return 0.0
        
    return float(intersection_area) / float(slot_area)

def is_point_inside_polygon(point, poly_points):
    """Check if a point (x, y) is inside a polygon using OpenCV's pointPolygonTest."""
    poly_pts = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(poly_pts, (float(point[0]), float(point[1])), False)
    return dist >= 0
