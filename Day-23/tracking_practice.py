import cv2
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Process all videos
# -----------------------------
video_files = sorted(INPUT_DIR.glob("*.mp4"))

if not video_files:
    print("No videos found in input folder.")
    exit()

for video_path in video_files:

    print("\n" + "=" * 50)
    print(f"Processing: {video_path.name}")
    print("=" * 50)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = f"tracked_{video_path.stem}.mp4"
    output_path = OUTPUT_DIR / output_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    unique_ids = set()
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Track objects using ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Get tracking IDs
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().tolist()
            unique_ids.update(ids)

        # Draw bounding boxes, classes, confidence and IDs
        annotated = results[0].plot()

        out.write(annotated)

        frame_count += 1

    cap.release()
    out.release()

    print(f"Frames processed: {frame_count}")
    print(f"Unique objects tracked: {len(unique_ids)}")
    print(f"IDs: {sorted(unique_ids)}")
    print(f"Saved: {output_path}")

print("\nAll videos processed successfully!")