import cv2
import gradio as gr
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)


# ============================================================
# SAMPLE VIDEOS
# ============================================================

SAMPLE_VIDEOS = [
    INPUT_DIR / "video_1.mp4",
    INPUT_DIR / "video_2.mp4",
    INPUT_DIR / "video_3.mp4",
]


# ============================================================
# LOAD SAMPLE VIDEO
# ============================================================

def load_sample_video(sample_number):

    if sample_number is None:
        return None

    try:
        index = int(sample_number) - 1
    except ValueError:
        return None

    if index < 0 or index >= len(SAMPLE_VIDEOS):
        return None

    video_path = SAMPLE_VIDEOS[index]

    if video_path.exists():
        return str(video_path)

    return None


# ============================================================
# TRACK VIDEO
# ============================================================

def track_video(video_path, progress=gr.Progress()):

    if video_path is None:
        return (
            None,
            "Please upload a video or select a sample video.",
            "—",
            "—",
            "Ready"
        )

    video_path = Path(video_path)

    if not video_path.exists():
        return (
            None,
            "The selected video could not be found.",
            "—",
            "—",
            "Error"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return (
            None,
            "Unable to open the selected video.",
            "—",
            "—",
            "Error"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / f"tracked_{video_path.stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # Tracking data
    # --------------------------------------------------------

    unique_ids = set()
    frame_count = 0

    # Reset tracker for a new video
    model.predictor = None

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        # Collect tracking IDs
        if result.boxes.id is not None:

            ids = result.boxes.id.int().tolist()

            unique_ids.update(ids)

        # Draw:
        # - bounding boxes
        # - class names
        # - confidence
        # - tracking IDs
        annotated_frame = result.plot()

        out.write(annotated_frame)

        frame_count += 1

        # Update progress
        if total_frames > 0 and frame_count % 10 == 0:

            progress(
                frame_count / total_frames,
                desc="Processing video..."
            )

    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()
    out.release()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    object_count = len(unique_ids)

    summary = f"""
## Tracking Complete

| Metric | Result |
|---|---:|
| Video | `{video_path.name}` |
| Frames processed | **{frame_count:,}** |
| Unique objects | **{object_count}** |
| Detection model | **YOLOv8n** |
| Tracking algorithm | **ByteTrack** |

The processed video contains bounding boxes, class labels,
confidence scores, and persistent tracking IDs.
"""

    return (
        str(output_path),
        summary,
        str(object_count),
        str(frame_count),
        "Completed"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """

/* =========================================================
   PAGE
   ========================================================= */

.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;

    background: #f4f5f7 !important;

    color: #1f2937 !important;

    font-family:
        Inter,
        Arial,
        sans-serif !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.header {
    background: #ffffff;

    border: 1px solid #e1e4e8;

    border-radius: 12px;

    padding: 28px 32px;

    margin-bottom: 20px;
}

.header-title {
    font-size: 30px;

    font-weight: 700;

    color: #111827;

    margin: 0;
}

.header-subtitle {
    margin-top: 7px;

    color: #6b7280;

    font-size: 15px;

    line-height: 1.5;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: #ffffff !important;

    border: 1px solid #e1e4e8 !important;

    border-radius: 12px !important;

    padding: 20px !important;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section-title {
    font-size: 18px;

    font-weight: 650;

    color: #111827;

    margin-bottom: 5px;
}

.section-description {
    color: #6b7280;

    font-size: 14px;

    margin-bottom: 15px;

    line-height: 1.5;
}


/* =========================================================
   VIDEO UPLOAD
   ========================================================= */

.upload-area {
    border: 1px dashed #cbd5e1 !important;

    border-radius: 10px !important;

    background: #f8fafc !important;
}


/* =========================================================
   SAMPLE VIDEO BUTTONS
   ========================================================= */

.sample-button {
    min-height: 42px !important;

    border-radius: 8px !important;

    background: #ffffff !important;

    border: 1px solid #d1d5db !important;

    color: #374151 !important;

    font-weight: 500 !important;
}

.sample-button:hover {
    background: #f3f4f6 !important;

    border-color: #9ca3af !important;
}


/* =========================================================
   TRACK BUTTON
   ========================================================= */

.track-button {
    background: #2563eb !important;

    border: none !important;

    color: white !important;

    border-radius: 8px !important;

    font-weight: 600 !important;

    min-height: 46px !important;

    margin-top: 14px !important;
}

.track-button:hover {
    background: #1d4ed8 !important;
}


/* =========================================================
   STATISTICS
   ========================================================= */

.stat-card {
    background: #ffffff !important;

    border: 1px solid #e1e4e8 !important;

    border-radius: 10px !important;

    padding: 14px !important;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: #ffffff;

    border: 1px solid #e1e4e8;

    border-radius: 10px;

    padding: 18px;

    min-height: 120px;
}

.info-title {
    font-size: 16px;

    font-weight: 650;

    color: #111827;

    margin-bottom: 6px;
}

.info-text {
    color: #6b7280;

    font-size: 13px;

    line-height: 1.5;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #9ca3af;

    font-size: 13px;

    padding: 25px 0 10px;
}


/* =========================================================
   REMOVE STRONG SHADOWS
   ========================================================= */

* {
    box-shadow: none !important;
}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Smart Object Tracking",
    css=CUSTOM_CSS,
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="header">

            <div class="header-title">
                Smart Object Tracking
            </div>

            <div class="header-subtitle">
                Detect and track objects across video frames
                using YOLOv8 and ByteTrack.
            </div>

        </div>
        """
    )


    # ========================================================
    # MAIN VIDEO AREA
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Input Video
                </div>

                <div class="section-description">
                    Upload your own video or select one of the
                    sample videos below.
                </div>
                """
            )

            video_input = gr.Video(
                label="Video",
                sources=["upload"],
                elem_classes="upload-area"
            )


            gr.Markdown("**Sample Videos**")


            # ------------------------------------------------
            # SAMPLE VIDEO BUTTONS
            # ------------------------------------------------

            with gr.Row():

                sample_1 = gr.Button(
                    "Sample 1",
                    elem_classes="sample-button"
                )

                sample_2 = gr.Button(
                    "Sample 2",
                    elem_classes="sample-button"
                )

                sample_3 = gr.Button(
                    "Sample 3",
                    elem_classes="sample-button"
                )


            gr.Markdown(
                """
                Select a sample to load it into the input area,
                then start tracking.
                """
            )


            track_button = gr.Button(
                "Start Tracking",
                variant="primary",
                elem_classes="track-button"
            )


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Tracking Output
                </div>

                <div class="section-description">
                    Processed video with detection and tracking
                    annotations.
                </div>
                """
            )

            video_output = gr.Video(
                label="Processed Video"
            )


    # ========================================================
    # STATISTICS
    # ========================================================

    gr.HTML(
        """
        <div style="
            margin-top: 24px;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: 650;
            color: #111827;
        ">
            Tracking Statistics
        </div>
        """
    )


    with gr.Row():

        with gr.Column(
            elem_classes="stat-card"
        ):

            total_objects = gr.Textbox(
                label="Unique Objects",
                value="—",
                interactive=False
            )


        with gr.Column(
            elem_classes="stat-card"
        ):

            total_frames = gr.Textbox(
                label="Frames Processed",
                value="—",
                interactive=False
            )


        with gr.Column(
            elem_classes="stat-card"
        ):

            status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    with gr.Row():

        with gr.Column(
            elem_classes="card"
        ):

            summary_output = gr.Markdown(
                """
### Ready

Upload a video or select one of the sample videos.

The system will detect objects, assign tracking IDs,
and generate a processed video.
"""
            )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    gr.HTML(
        """
        <div style="
            margin-top: 24px;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: 650;
            color: #111827;
        ">
            System Information
        </div>
        """
    )


    with gr.Row():

        with gr.Column(
            elem_classes="info-card"
        ):

            gr.HTML(
                """
                <div class="info-title">
                    Detection Model
                </div>

                <div class="info-text">
                    YOLOv8n<br>
                    Real-time object detection.
                </div>
                """
            )


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.HTML(
                """
                <div class="info-title">
                    Tracking Algorithm
                </div>

                <div class="info-text">
                    ByteTrack<br>
                    Maintains object identities across frames.
                </div>
                """
            )


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.HTML(
                """
                <div class="info-title">
                    Tracking Information
                </div>

                <div class="info-text">
                    Object IDs and confidence scores are displayed
                    on detected objects.
                </div>
                """
            )


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.HTML(
                """
                <div class="info-title">
                    Output
                </div>

                <div class="info-text">
                    MP4 processed video with tracking annotations.
                </div>
                """
            )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">
            Day-23 Object Tracking Project
            &nbsp;•&nbsp;
            YOLOv8 + ByteTrack
        </div>
        """
    )


    # ========================================================
    # SAMPLE BUTTON EVENTS
    # ========================================================

    sample_1.click(
        fn=lambda: load_sample_video("1"),
        inputs=None,
        outputs=video_input
    )

    sample_2.click(
        fn=lambda: load_sample_video("2"),
        inputs=None,
        outputs=video_input
    )

    sample_3.click(
        fn=lambda: load_sample_video("3"),
        inputs=None,
        outputs=video_input
    )


    # ========================================================
    # TRACKING EVENT
    # ========================================================

    track_button.click(
        fn=track_video,
        inputs=video_input,
        outputs=[
            video_output,
            summary_output,
            total_objects,
            total_frames,
            status
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch()
