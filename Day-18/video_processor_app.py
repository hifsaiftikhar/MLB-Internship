import os
import cv2
import tempfile
import gradio as gr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(filename):
    return os.path.join(SCRIPT_DIR, filename)


def process_video(video_path, enable_gray, blur_kernel, canny_thresh1, canny_thresh2):
    if not video_path:
        return None, "No video provided. Please upload a video or record using your webcam."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Error opening video file: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output to a temporary file
    temp_dir = tempfile.gettempdir()
    # OpenCV VideoWriter output needs standard mp4 containers
    output_path = os.path.join(temp_dir, "processed_output.mp4")

    # Use standard mp4v codec for writing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # If edge detection is applied, the output frame is single channel
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=(not enable_gray))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = frame

        # Convert to Grayscale
        if enable_gray:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur (kernel must be odd and positive)
            if blur_kernel > 1:
                ksize = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

            # Apply Canny Edge Detection
            if canny_thresh1 > 0 or canny_thresh2 > 0:
                processed = cv2.Canny(processed, canny_thresh1, canny_thresh2)

        out.write(processed)
        frame_count += 1

    cap.release()
    out.release()

    # Re-encode to H.264 if we want wide browser compatibility,
    # but standard Gradio video output handles mp4v cleanly or falls back.
    status = f"Processed {frame_count} frames of {total_frames} total. Saved results successfully."
    return output_path, status


# GUI setup
with gr.Blocks(title="Real-Time Video Processing Tool", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# Real-Time Video Processing Tool")
    gr.Markdown(
        "Upload a video or record from your webcam, apply filters (Grayscale, Gaussian Blur, Canny Edge Detection), and download the processed file."
    )

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video (Upload or Webcam)")

            enable_gray = gr.Checkbox(label="Convert to Grayscale", value=True)
            blur_kernel = gr.Slider(minimum=1, maximum=21, step=2, value=5, label="Gaussian Blur Kernel Size")
            canny_thresh1 = gr.Slider(minimum=0, maximum=255, step=1, value=50, label="Canny Threshold 1")
            canny_thresh2 = gr.Slider(minimum=0, maximum=255, step=1, value=150, label="Canny Threshold 2")

            process_btn = gr.Button("Process Video & Save", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Processed Video Output")
            status_box = gr.Textbox(label="Status Logs", lines=3)

    examples = [resolve(f"input/sample_video{i}.mp4") for i in ("", "2", "3")
                if os.path.exists(resolve(f"input/sample_video{i}.mp4"))]
    if examples:
        gr.Examples(examples=[[p] for p in examples], inputs=input_video, label="Try a sample video")

    process_btn.click(
        fn=process_video,
        inputs=[input_video, enable_gray, blur_kernel, canny_thresh1, canny_thresh2],
        outputs=[output_video, status_box]
    )

if __name__ == "__main__":
    demo.launch()
