import cv2
import os

video_path = "input/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: could not open video at {video_path}")
else:
    os.makedirs("output", exist_ok=True)

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"Width x Height: {width}x{height}")
    print(f"Total frames: {total_frames}")

    # Writer for the processed output (Canny edges on grayscale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output/processed_video.mp4", fourcc, fps, (width, height), isColor=False)

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
    print(f"\nProcessed {frame_count} frames")
    print("Saved to output/processed_video.mp4")
