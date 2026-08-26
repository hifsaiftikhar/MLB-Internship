import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import tempfile

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #172033, #263a5c);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 17px;
        color: #d9e2f2;
    }

    .metric-card {
        padding: 20px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e9f0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-label {
        color: #687386;
        font-size: 14px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .status {
        padding: 12px 16px;
        border-radius: 10px;
        background: #eef7ee;
        color: #287a32;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
TEST_IMAGE_DIR = BASE_DIR / "sample_test_images"


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------
CLASS_NAMES = {
    0: "D00 - Longitudinal Crack",
    1: "D10 - Transverse Crack",
    2: "D20 - Alligator Crack",
    3: "D40 - Pothole",
    4: "D43 - Repair",
    5: "D44 - Rutting",
    6: "D50 - Other Damage"
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


model = load_model()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("## 🛣️ RoadGuard AI")

    st.markdown("---")

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.markdown("---")

    st.markdown("### Model")

    if model is not None:
        st.success("Custom YOLO11s model loaded")
    else:
        st.error("best.pt not found")

    st.markdown("---")

    st.caption("AI-based Road Damage Detection")
    st.caption("YOLO11s • 7 damage classes")


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛣️ RoadGuard AI</h1>
    <p>
        Intelligent road damage detection using a custom-trained YOLO11s model.
        Upload a road image or test the system using the provided sample images.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------
if model is None:

    st.error(
        "⚠️ Custom model not found. "
        "Please place your trained `best.pt` file inside the Project 2 folder."
    )

    st.stop()


# --------------------------------------------------
# TEST IMAGES
# --------------------------------------------------
st.markdown(
    '<div class="section-title">🧪 Quick Test Images</div>',
    unsafe_allow_html=True
)

test_images = []

if TEST_IMAGE_DIR.exists():

    for image_path in sorted(TEST_IMAGE_DIR.glob("*")):

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            test_images.append(image_path)


if test_images:

    cols = st.columns(len(test_images))

    for i, image_path in enumerate(test_images):

        with cols[i]:

            image = Image.open(image_path)

            st.image(
                image,
                caption=image_path.name,
                use_container_width=True
            )

            if st.button(
                "Analyze",
                key=f"test_{image_path.name}"
            ):

                st.session_state["selected_image"] = str(image_path)


else:

    st.info("No sample images found.")


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------
st.markdown(
    '<div class="section-title">📤 Upload Your Own Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing road damage."
)

if uploaded_file is not None:

    st.session_state["uploaded_image"] = uploaded_file


# --------------------------------------------------
# DETERMINE IMAGE
# --------------------------------------------------
image_source = None

if "uploaded_image" in st.session_state:

    image_source = st.session_state["uploaded_image"]

elif "selected_image" in st.session_state:

    image_source = st.session_state["selected_image"]


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if image_source is not None:

    st.markdown(
        '<div class="section-title">🔍 Detection Result</div>',
        unsafe_allow_html=True
    )

    if isinstance(image_source, str):

        image = Image.open(image_source)

    else:

        image = Image.open(image_source)


    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Image")

        st.image(
            image,
            use_container_width=True
        )


    # ----------------------------------------------
    # RUN YOLO
    # ----------------------------------------------
    with st.spinner("Analyzing road damage..."):

        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

    result = results[0]

    annotated_image = result.plot()

    # ----------------------------------------------
    # DETECTIONS
    # ----------------------------------------------
    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])

            detections.append({
                "class_id": cls_id,
                "class": CLASS_NAMES.get(
                    cls_id,
                    f"Class {cls_id}"
                ),
                "confidence": conf_score
            })


    with col2:

        st.markdown("### AI Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )


    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------
    st.markdown(
        '<div class="section-title">📊 Detection Summary</div>',
        unsafe_allow_html=True
    )

    total = len(detections)

    if total > 0:

        average_confidence = sum(
            d["confidence"] for d in detections
        ) / total

        highest_confidence = max(
            d["confidence"] for d in detections
        )

    else:

        average_confidence = 0
        highest_confidence = 0


    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {average_confidence:.1%}
                </div>
                <div class="metric-label">Average Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {highest_confidence:.1%}
                </div>
                <div class="metric-label">Highest Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ----------------------------------------------
    # DETECTION DETAILS
    # ----------------------------------------------
    if detections:

        st.markdown(
            '<div class="section-title">📋 Damage Detected</div>',
            unsafe_allow_html=True
        )

        for i, detection in enumerate(detections, 1):

            st.write(
                f"**{i}. {detection['class']}** — "
                f"{detection['confidence']:.1%} confidence"
            )

    else:

        st.info(
            "No road damage was detected above the selected "
            "confidence threshold."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.caption(
    "RoadGuard AI • Custom YOLO11s Road Damage Detection • "
    "7-class detection system"
)