import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BookVision AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS - LIGHT THEME
# --------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f8fc;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #e8f4ff, #eefcf9);
        padding: 30px 35px;
        border-radius: 18px;
        border: 1px solid #d9e7f3;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #526777;
        line-height: 1.6;
    }

    /* Section titles */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #173f5f;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dce6ef;
        box-shadow: 0 3px 12px rgba(30, 70, 100, 0.06);
        margin-bottom: 15px;
    }

    /* Metrics */
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #dce6ef;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #087f8c;
    }

    .metric-label {
        font-size: 14px;
        color: #667784;
        margin-top: 4px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 14px;
        padding: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #edf5fa;
    }

    /* Success / warning */
    .result-box {
        padding: 15px;
        border-radius: 12px;
        background: #eefaf6;
        border: 1px solid #c9eee2;
        color: #176b59;
        margin-top: 12px;
    }

    .no-result-box {
        padding: 15px;
        border-radius: 12px;
        background: #fff7e8;
        border: 1px solid #f3dfb0;
        color: #7a5b17;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">📚 BookVision AI</div>
    <div class="hero-subtitle">
        Intelligent book detection powered by a custom-trained
        <b>YOLO11n</b> computer vision model.
        Upload an image or choose from the sample images to detect books instantly.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("## ⚙️ Detection Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher values make the detector more strict."
    )

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("**Model:** YOLO11n")
    st.write("**Classes:** 1")
    st.write("**Class:** Book")
    st.write("**Image Size:** 640 × 640")

    st.divider()

    st.caption(
        "Day-24 Custom Object Detection Project"
    )

# --------------------------------------------------
# IMAGE SOURCE
# --------------------------------------------------
st.markdown(
    '<div class="section-title">🖼️ Choose an Image</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📤 Upload Image", "🧪 Sample Images"])

image = None
image_name = None

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
with tab1:

    uploaded_file = st.file_uploader(
        "Upload a book image",
        type=["jpg", "jpeg", "png"],
        help="Upload one image at a time."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

# --------------------------------------------------
# SAMPLE IMAGES
# --------------------------------------------------
# --------------------------------------------------
# SAMPLE IMAGES
# --------------------------------------------------
with tab2:

    sample_folder = "original_images"

    sample_images = [
        "book31.jpeg",
        "book32.jpeg",
        "book33.jpeg",
        "book34.jpeg",
        "book35.jpeg",
        "book36.jpeg",
        "book37.jpeg",
        "book38.jpeg",
        "book39.jpeg",
        "book40.jpeg"
    ]

    if os.path.exists(sample_folder):

        # Keep only the selected 10 images
        sample_images = [
            file for file in sample_images
            if os.path.exists(
                os.path.join(sample_folder, file)
            )
        ]

    if sample_images:

        selected_sample = st.selectbox(
            "Choose a sample image",
            sample_images
        )

        sample_path = os.path.join(
            sample_folder,
            selected_sample
        )

        image = Image.open(sample_path).convert("RGB")
        image_name = selected_sample

        st.image(
            image,
            caption=f"Selected sample: {selected_sample}",
            use_container_width=True
        )

    else:

        st.error(
            "The selected sample images could not be found "
            "in the original_images folder."
        )

# --------------------------------------------------
# DETECTION
# --------------------------------------------------
if image is not None:

    st.markdown("---")

    if st.button(
        "🔍 Detect Books",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

        result = results[0]

        # Detection image
        result_image = result.plot()

        boxes = result.boxes

        # --------------------------------------------------
        # RESULTS HEADER
        # --------------------------------------------------
        st.markdown(
            '<div class="section-title">📊 Detection Results</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        if boxes is not None and len(boxes) > 0:

            detection_count = len(boxes)

            confidences = boxes.conf.cpu().tolist()

            average_confidence = (
                sum(confidences) / len(confidences)
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {detection_count}
                        </div>
                        <div class="metric-label">
                            Books Detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {average_confidence:.1%}
                        </div>
                        <div class="metric-label">
                            Average Confidence
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            640
                        </div>
                        <div class="metric-label">
                            Input Size
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # --------------------------------------------------
            # BEFORE / AFTER
            # --------------------------------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(
                    image,
                    use_container_width=True
                )

            with col2:
                st.markdown("### Detection Result")
                st.image(
                    result_image,
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="result-box">
                    ✅ <b>{detection_count} book(s)</b> detected
                    successfully.
                </div>
                """,
                unsafe_allow_html=True
            )

            # --------------------------------------------------
            # CONFIDENCE DETAILS
            # --------------------------------------------------
            with st.expander("📋 Detection Details"):

                for i, confidence_value in enumerate(
                    confidences,
                    start=1
                ):
                    st.write(
                        f"Book {i}: "
                        f"**{confidence_value:.1%} confidence**"
                    )

        else:

            st.markdown(
                """
                <div class="no-result-box">
                    ⚠️ No books were detected in this image.
                    Try lowering the confidence threshold or
                    uploading a clearer image.
                </div>
                """,
                unsafe_allow_html=True
            )

else:

    st.info(
        "👆 Upload an image or select a sample image above "
        "to start detection."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#71808c; padding:10px;">
        📚 <b>BookVision AI</b> &nbsp;|&nbsp;
        YOLO11n Custom Object Detection &nbsp;|&nbsp;
        Day-24 Computer Vision Project
    </div>
    """,
    unsafe_allow_html=True
)