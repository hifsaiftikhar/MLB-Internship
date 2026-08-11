import os

# Centralized configuration paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default asset paths
IMAGE_PATH = os.path.join(DATA_DIR, "carParkImg.png")
VIDEO_PATH = os.path.join(DATA_DIR, "carPark.mp4")
SLOTS_CONFIG_PATH = os.path.join(DATA_DIR, "parking_slots.json")

# Traditional CV Pipeline Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (3, 3)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MORPHOLOGY_KERNEL_SIZE = (3, 3)
MORPHOLOGY_ITERATIONS = 1

# Occupancy Decisions
CV_OCCUPANCY_THRESHOLD = 0.10
CV_FALLBACK_THRESHOLD = 0.45

# YOLO Model Settings
YOLO_MODEL_NAME = "yolov8n.pt"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]   # COCO Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_OVERLAP_THRESHOLD = 0.35
