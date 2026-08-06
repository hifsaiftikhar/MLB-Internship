import cv2
import os

os.makedirs("output", exist_ok=True)

# List of videos to process for the challenge task
video_sources = [
    ("input/sample_video.mp4", "output/processed_video.mp4"),
    ("input/sample_video2.mp4", "output/processed_video2.mp4"),
    ("input/sample_video3.mp4", "output/processed_video3.mp4")
]

for video_path, output_path in video_sources:
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found. Skipping.")
        continue

    print(f"\nProcessing {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: could not open video at {video_path}")
        continue

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps} | Resolution: {width}x{height} | Total Frames: {total_frames}")

    # Writer for the processed output (Canny edges on grayscale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        out.write(edges)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Successfully processed {frame_count} frames. Saved to {output_path}")
