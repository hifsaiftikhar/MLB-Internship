import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from PIL import Image
import sys
from ultralytics import YOLO
import os

if not os.path.exists("yolov8n.pt"):
    import urllib.request
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolovn.pt",
        "yolov8n.pt"
    )
# Ensure project directories are in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.detector import ParkingOccupancyDetector
from src.geometry import points_to_array

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Parking Lot Occupancy Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 5px solid #3B82F6;
        text-align: center;
    }
    .metric-occupied {
        border-left: 5px solid #EF4444;
    }
    .metric-vacant {
        border-left: 5px solid #10B981;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        margin-top: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the YOLO model once (cached)
@st.cache_resource
def get_detector(model_name):
    try:
        return ParkingOccupancyDetector(model_name)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Load defined slot coordinates
def load_slots():
    if os.path.exists(config.SLOTS_CONFIG_PATH):
        try:
            with open(config.SLOTS_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error reading JSON config: {e}")
            return []
    return []

# Draw annotations on image
def draw_occupancy_overlay(img, analyzed_slots, vehicles, draw_yolo=False):
    overlay = img.copy()
    output_img = img.copy()
    
    # Draw parking slots
    for slot in analyzed_slots:
        points = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        
        # Color: Red for occupied, Green for vacant
        color = (0, 0, 255) if slot["is_occupied"] else (0, 255, 0)
        
        # Draw transparent filled polygon
        cv2.fillPoly(overlay, [points], color)
        # Draw solid outline
        cv2.polylines(output_img, [points], True, color, 2)
        
        # Add slot ID text in center
        cx = int(sum([p[0] for p in slot["points"]]) / 4)
        cy = int(sum([p[1] for p in slot["points"]]) / 4)
        cv2.putText(output_img, f"{slot['id']}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend transparent polygons
    cv2.addWeighted(overlay, 0.25, output_img, 0.75, 0, output_img)
    
    # Draw YOLO vehicle bounding boxes if selected
    if draw_yolo:
        for vehicle in vehicles:
            x1, y1, x2, y2, conf, class_id = vehicle
            cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
            label = f"Vehicle: {conf:.2f}"
            cv2.putText(output_img, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1, cv2.LINE_AA)
            
    return output_img

def main():
    st.markdown('<div class="main-header">Smart Parking Lot Occupancy Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Hybrid Computer Vision System using YOLO and Traditional OpenCV Preprocessing</div>', unsafe_allow_html=True)
    
    # 1. Sidebar - Configuration & Parameters
    st.sidebar.header("📁 Configuration & Model Setup")
    
    # Model loader
    detector = get_detector(config.YOLO_MODEL_NAME)
    
    st.sidebar.subheader("🎯 YOLO Bounding Box Settings")
    yolo_conf = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    yolo_overlap = st.sidebar.slider("YOLO Overlap (IoS) Threshold", 0.10, 0.90, config.YOLO_OVERLAP_THRESHOLD, 0.05)
    
    st.sidebar.subheader("⚙️ Traditional CV Settings")
    canny_low = st.sidebar.slider("Canny Low Threshold", 10, 150, config.CANNY_LOW_THRESHOLD, 5)
    canny_high = st.sidebar.slider("Canny High Threshold", 50, 300, config.CANNY_HIGH_THRESHOLD, 5)
    cv_thresh = st.sidebar.slider("CV Edge Density Occupied", 0.02, 0.30, config.CV_OCCUPANCY_THRESHOLD, 0.01)
    cv_fallback = st.sidebar.slider("CV Edge Fallback (YOLO Miss)", 0.05, 0.40, config.CV_FALLBACK_THRESHOLD, 0.01)
    
    # Load Slots Configuration
    slots = load_slots()
    if not slots:
        st.warning("No parking slots configuration file found! Please run the asset downloader script.")
        return
        
    st.sidebar.success(f"Loaded {len(slots)} parking slots.")
    
    # Choose Input Mode
    input_mode = st.radio("Select Analysis Target:", ["Static Image", "Video Stream"], horizontal=True)
    
    # Load test image
    if not os.path.exists(config.IMAGE_PATH):
        st.error(f"Sample image not found at {config.IMAGE_PATH}. Please run python scripts/download_assets.py")
        return
        
    raw_img = cv2.imread(config.IMAGE_PATH)
    
    # Tabs
    tab_dashboard, tab_pipeline, tab_inspector, tab_guide = st.tabs([
        "📊 Occupancy Dashboard", 
        "🌀 CV Pipeline Steps", 
        "🔍 Detail Slot Inspector",
        "📖 Technical Documentation"
    ])
    
    # ------------------ TAB 1: DASHBOARD ------------------
    with tab_dashboard:
        if input_mode == "Static Image":
            # Run analyzer
            analyzed_slots, vehicles, cv_results = detector.analyze_occupancy(
                raw_img, 
                slots, 
                yolo_overlap_thresh=yolo_overlap,
                cv_thresh=cv_thresh,
                cv_fallback_thresh=cv_fallback,
                low_threshold=canny_low,
                high_threshold=canny_high
            )
            
            # Compute stats
            total_slots = len(analyzed_slots)
            occupied_slots = sum([1 for s in analyzed_slots if s["is_occupied"]])
            vacant_slots = total_slots - occupied_slots
            occupancy_rate = (occupied_slots / total_slots) * 100 if total_slots > 0 else 0
            
            # Display Stylized Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Total Spaces</div><div class="metric-value">{total_slots}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card metric-occupied"><div class="metric-title">Occupied</div><div class="metric-value">{occupied_slots}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card metric-vacant"><div class="metric-title">Vacant</div><div class="metric-value">{vacant_slots}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Occupancy Rate</div><div class="metric-value">{occupancy_rate:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Options and Output
            draw_yolo_boxes = st.checkbox("Show YOLO Vehicle Dounding Boxes", value=False)
            
            annotated_frame = draw_occupancy_overlay(raw_img, analyzed_slots, vehicles, draw_yolo=draw_yolo_boxes)
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Analyzed Parking Lot Occupancy Map")
            
        elif input_mode == "Video Stream":
            if not os.path.exists(config.VIDEO_PATH):
                st.error(f"Sample video not found at {config.VIDEO_PATH}. Please run python scripts/download_assets.py")
                return
                
            st.info("Demonstrating video processing loop. You can start/stop using the control buttons below.")
            
            col_play, col_stop, _ = st.columns([1, 1, 8])
            play_btn = col_play.button("▶️ Start Video Feed")
            stop_btn = col_stop.button("⏹️ Stop")
            
            video_placeholder = st.empty()
            stat_placeholder = st.empty()
            
            if play_btn:
                cap = cv2.VideoCapture(config.VIDEO_PATH)
                if not cap.isOpened():
                    st.error("Error opening video file.")
                    return
                
                # Setup video execution loop
                stop_pressed = False
                frame_skip = 5  # Analyze every 5th frame for speed
                frame_count = 0
                
                while cap.isOpened() and not stop_pressed:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                        
                    # Resize frame for quick handling
                    # Frame size of carPark.mp4 is typically 1280x720
                    
                    # Run occupancy detection
                    analyzed_slots, vehicles, cv_results = detector.analyze_occupancy(
                        frame, 
                        slots, 
                        yolo_overlap_thresh=yolo_overlap,
                        cv_thresh=cv_thresh,
                        cv_fallback_thresh=cv_fallback,
                        low_threshold=canny_low,
                        high_threshold=canny_high
                    )
                    
                    # Compute stats
                    total_slots = len(analyzed_slots)
                    occupied_slots = sum([1 for s in analyzed_slots if s["is_occupied"]])
                    vacant_slots = total_slots - occupied_slots
                    occupancy_rate = (occupied_slots / total_slots) * 100 if total_slots > 0 else 0
                    
                    # Display Stats inside stream
                    with stat_placeholder.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Spaces", total_slots)
                        c2.metric("Occupied Spaces", occupied_slots, delta=None, delta_color="inverse")
                        c3.metric("Vacant Spaces", vacant_slots)
                        c4.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")
                    
                    annotated_frame = draw_occupancy_overlay(frame, analyzed_slots, vehicles, draw_yolo=True)
                    video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Small sleep to simulate realistic video frame rate
                    time.sleep(0.01)
                    
                cap.release()
                
    # ------------------ TAB 2: PIPELINE ------------------
    with tab_pipeline:
        st.subheader("Traditional Computer Vision Preprocessing Pipeline")
        st.markdown("""
            Here is the frame-by-frame breakdown of the traditional computer vision pipeline (Days 13–17). 
            Applying these operations allows us to extract edge/contour textures, which serve as a backup to check 
            if a spot is empty or occupied, especially when lighting, shadows, or vehicle orientation degrade YOLO detections.
        """)
        
        # Run CV pipeline
        cv_res = detector.analyze_occupancy(
            raw_img, 
            slots, 
            yolo_overlap_thresh=yolo_overlap,
            cv_thresh=cv_thresh,
            cv_fallback_thresh=cv_fallback,
            low_threshold=canny_low,
            high_threshold=canny_high
        )[2]
        
        # Grid layout for steps
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="1. Original Parking Lot Image", use_container_width=True)
            st.image(cv_res["enhanced"], caption="2. Contrast Enhanced (CLAHE)", use_container_width=True)
        with col2:
            st.image(cv_res["edges"], caption="3. Edge Detection (Canny)", use_container_width=True)
            st.image(cv_res["dilated"], caption="4. Morphological Dilation (Joined Edges)", use_container_width=True)

    # ------------------ TAB 3: SLOT INSPECTOR ------------------
    with tab_inspector:
        st.subheader("Individual Slot Verification & Diagnostics")
        st.markdown("Select a specific slot ID from the dropdown to check its raw cropped view, dilated edge density mask, and telemetry.")
        
        slot_id_to_inspect = st.selectbox("Choose a Parking Slot ID to Inspect:", sorted([s["id"] for s in slots]))
        
        # Re-run analyzer
        analyzed_slots, vehicles, cv_res = detector.analyze_occupancy(
            raw_img, 
            slots, 
            yolo_overlap_thresh=yolo_overlap,
            cv_thresh=cv_thresh,
            cv_fallback_thresh=cv_fallback,
            low_threshold=canny_low,
            high_threshold=canny_high
        )
        
        # Find selected slot data
        selected_slot_data = next((s for s in analyzed_slots if s["id"] == slot_id_to_inspect), None)
        
        if selected_slot_data:
            # Crop slot region from original image and dilated edges
            pts = np.array(selected_slot_data["points"], dtype=np.int32)
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            
            # Crop regions
            crop_raw = raw_img[y_min:y_max, x_min:x_max]
            crop_edges = cv_res["dilated"][y_min:y_max, x_min:x_max]
            
            # Display close-up and details
            col_crop_raw, col_crop_edges, col_metrics = st.columns([1, 1, 2])
            
            with col_crop_raw:
                st.image(cv2.cvtColor(crop_raw, cv2.COLOR_BGR2RGB), caption="Raw Slot Crop", use_container_width=True)
            with col_crop_edges:
                st.image(crop_edges, caption="Edge Mask (Dilated Canny)", use_container_width=True)
            with col_metrics:
                st.markdown("### Telemetry Analysis")
                
                status_color = "🔴 OCCUPIED" if selected_slot_data["is_occupied"] else "🟢 VACANT"
                st.markdown(f"**Status**: `{status_color}`")
                st.markdown(f"**YOLO Vehicle Overlap (IoS)**: `{selected_slot_data['yolo_overlap']:.4f}` (Threshold: {yolo_overlap})")
                st.markdown(f"**CV Edge/Texture Density**: `{selected_slot_data['cv_density']:.4f}` (Thresholds: Hybrid {cv_thresh}, Fallback {cv_fallback})")
                st.markdown(f"**Decision Reason**: `{selected_slot_data['reason']}`")
                
    # ------------------ TAB 4: GUIDE ------------------
    with tab_guide:
        st.subheader("Pipeline & Decision-Making Architecture")
        st.markdown(r"""
        ### System Workflow
        The system implements a double-check occupancy mechanism. Traditional CV methods analyze edge densities, while YOLO is used to detect classes.
        
        1. **YOLO Vehicle Detection**: YOLO runs inference on the full frame to detect `car`, `truck`, `bus`, and `motorcycle` classes.
        2. **Polygon Overlaps**: For each parking slot quadrilateral polygon $S$ and detected vehicle bounding box $B$, the **Intersection over Slot (IoS)** is computed:
           $$\text{IoS} = \frac{\text{Area}(S \cap B)}{\text{Area}(S)}$$
        3. **OpenCV Edge Texture Analysis**: A traditional CV pipeline runs concurrently to highlight edges. The **CV Edge Density** ($\rho$) is computed inside the slot mask:
           $$\rho = \frac{\text{Dilated Edge Pixels in Slot}}{\text{Total Pixels in Slot}}$$
        4. **Hybrid Merging Rules**:
           - **YOLO Strong Occupancy**: If $\text{IoS} \ge \text{YOLO\_OVERLAP\_THRESHOLD}$, slot is marked **Occupied**.
           - **Hybrid Occupancy**: If $0.10 \le \text{IoS} < \text{YOLO\_OVERLAP\_THRESHOLD}$ AND $\rho \ge \text{CV\_OCCUPANCY\_THRESHOLD}$, slot is marked **Occupied** (confirms weak vehicle detections).
           - **CV Fallback Occupancy**: If $\text{IoS} < 0.10$ but $\rho \ge \text{CV\_FALLBACK\_THRESHOLD}$, slot is marked **Occupied** (catches vehicles missed by YOLO due to severe camera angle or shadows).
           - **Vacant**: Otherwise.
        """)
        
if __name__ == "__main__":
    main()
