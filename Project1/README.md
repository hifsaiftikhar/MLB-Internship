# Smart Parking Lot Occupancy Analyzer

A computer vision system that analyzes a parking lot image or video and determines which parking spaces are occupied and which are vacant. It combines YOLOv8 object detection with a traditional OpenCV preprocessing pipeline (grayscale, CLAHE contrast enhancement, Gaussian blur, Canny edge detection, morphological dilation), so the occupancy decision doesn't rely on a single method.

## Project overview

Given a static camera view of a parking lot, the system:
1. Detects vehicles in the frame using a pretrained YOLOv8 model.
2. Compares each predefined parking slot against detected vehicles to measure overlap.
3. Cross-checks uncertain or missed detections using a traditional CV edge-density pipeline.
4. Classifies every slot as occupied or vacant based on a combination of both signals.
5. Displays the result as a color-coded overlay (green = vacant, red = occupied) along with occupancy statistics.

## Dataset used

Reference image and video from Murtaza's Computer Vision Zone Car Parking Space Detection project

Suggested public datasets for extending this project: PKLot Dataset on Roboflow Universe (12,416 surveillance frames across varying weather conditions), and the CNR-Park Dataset (about 150,000 labeled occupancy images).

## Project workflow

**1. Parking slot definition (selector.py)**
Since the camera is static, parking slots are defined once and reused, rather than re-detected on every frame. I built an interactive tool where you click 4 corner points per slot (in order), and it saves the coordinates to a JSON config file. This supports angled, non-rectangular slots, not just straight rectangles, which matters for slots near the edge of a wide-angle camera view where perspective distorts their shape.

**2. Vehicle detection (detector.py)**
Uses a pretrained YOLOv8 model to detect vehicles (car, truck, bus, motorcycle) in the frame - no custom training needed, since these are already standard COCO classes.

**3. Traditional CV pipeline (traditional_cv.py)**
Grayscale conversion, then CLAHE (adaptive contrast enhancement, better than simple brightness/contrast scaling for handling uneven outdoor lighting and shadows), Gaussian blur, Canny edge detection, and morphological dilation to close small gaps in detected edges.

**4. Occupancy decision (geometry.py + detector.py)**
For each slot, I calculate Intersection over Slot (IoS) - what fraction of the slot's area is covered by a detected vehicle's bounding box, using pixel masks so it works correctly with non-rectangular slot polygons, not just axis-aligned rectangles.

The decision combines both signals rather than relying on YOLO alone:
- If IoS is high (>= 0.35), the slot is marked occupied - YOLO confirmed.
- If IoS is moderate (0.10-0.35) and the CV edge density inside the slot is also elevated, it's marked occupied - this catches cases where YOLO's detection is weak or partial.
- If YOLO misses the vehicle entirely (IoS < 0.10) but edge density is high, it's still marked occupied - this is the CV fallback, meant to catch cases where a shadow, camera angle, or partial occlusion causes YOLO to miss a car that traditional edge analysis can still pick up on.
- Otherwise, the slot is marked vacant.

**5. Visualization and statistics (main.py / app.py)**
Draws each slot's outline and a semi-transparent fill (green/red), labels it with its ID, and reports total slots, occupied count, vacant count, and occupancy percentage.

## Technologies used

- Python, OpenCV
- Ultralytics YOLOv8 (pretrained, no custom training)
- NumPy
- Streamlit (interactive dashboard)

## The Streamlit dashboard (app.py)

Four tabs:
- **Occupancy Dashboard** - runs the analysis on a static image or a video feed, shows the annotated result and live KPI cards (total/occupied/vacant/occupancy rate), with adjustable sliders for every threshold in the decision logic.
- **CV Pipeline Steps** - shows each stage of the traditional CV pipeline side by side (grayscale, CLAHE, edges, dilation), so the intermediate processing is visible, not just the final result.
- **Detail Slot Inspector** - pick any individual slot by ID and see a cropped close-up of just that slot, its edge mask, and the exact numbers (IoS value, edge density, which rule triggered the decision) that led to its classification.
- **Technical Documentation** - explains the IoS and edge density formulas and the decision rules in the app itself.

## Results

Running the pipeline on the reference parking lot image (69 predefined slots):
- Total parking slots: 69
- Occupied: 54
- Vacant: 15
- Occupancy rate: 78.3%

Pipeline step images and the final annotated result are saved under results/.

## Challenges faced

- **Shadows and uneven lighting:** bright daylight creates deep shadows that can confuse both thresholding and YOLO's detections. Addressed by adding CLAHE as a preprocessing step before edge detection, which improves local contrast in shadowed regions specifically, rather than adjusting brightness uniformly across the whole image.
- **Camera perspective distortion:** slots farther from the camera or near the frame edges are visibly skewed by the wide-angle lens, so plain axis-aligned rectangles don't match their real shape well. Solved by building a 4-point polygon selector instead of a simple rectangle tool, and using pixel-mask-based overlap calculations (via cv2.fillPoly) so the IoS calculation is accurate for any quadrilateral shape, not just upright rectangles.
- **YOLO missing partially occluded vehicles:** cars partly hidden behind trees, or parked very close together, are sometimes missed by YOLO entirely. The CV fallback rule (high edge density even with no YOLO detection) was added specifically to catch these cases, rather than relying on YOLO as the sole source of truth.

## Future improvements

- Temporal filtering across video frames (e.g. a moving average or simple voting window) to prevent a slot's status from flickering between occupied/vacant on borderline frames.
- Automated slot boundary detection using a segmentation model, to remove the need for manually clicking each slot's corners once per camera setup.
- A small dedicated occupied-vs-vacant classifier trained specifically on cropped slot images, as a third signal alongside YOLO and the CV pipeline.

## Project structure

```
Project1/
    data/
        carParkImg.png          reference parking lot image
        carPark.mp4             reference video
        parking_slots.json      slot coordinates (69 slots)
    results/
        annotated_image.png     final output with occupancy overlay
        pipeline_steps/         intermediate CV pipeline images
    scripts/
        download_assets.py      downloads/prepares the reference image, video, and slot data
    src/
        app.py                  Streamlit dashboard
        config.py                paths, thresholds, class IDs
        detector.py              YOLO detection + hybrid occupancy decision
        geometry.py               polygon/overlap math (IoS calculation)
        main.py                   command-line pipeline runner
        selector.py                interactive slot corner selector tool
    requirements.txt
    README.md
```

## Running the project

Install dependencies:
```
python -m pip install -r requirements.txt
```

Download/prepare reference assets:
```
python scripts/download_assets.py
```

Run the dashboard:
```
python -m streamlit run src/app.py
```

Run the command-line analyzer (prints stats, saves output images):
```
python src/main.py
```

Define or edit parking slots interactively:
```
python src/selector.py
```
Left click adds a corner (4 clicks per slot, clockwise). Right click inside a slot deletes it. Press 's' to save, 'q' to quit.

## Author

Hifsa Iftikhar
