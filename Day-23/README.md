# Smart Object Tracking System

## Overview

This project implements a video-based multi-object tracking system using YOLOv8 and ByteTrack.

The system detects objects in video frames, assigns a unique tracking ID to each detected object, and maintains the identity of objects across consecutive frames. It also counts the total number of unique objects observed in each video.

A Gradio application is included to provide a user-friendly interface for uploading videos, selecting sample videos, running tracking, and viewing the processed results.

## Objectives

* Understand the fundamentals of object tracking.
* Perform multi-object tracking on video.
* Assign unique IDs to detected objects.
* Maintain object identities across consecutive frames.
* Count the total number of unique objects in a video.
* Save processed tracking videos.
* Build a Gradio application for interactive video tracking.

## Object Detection vs Object Tracking

Object detection identifies objects in individual images or video frames. Each frame is processed independently, so the same object may not have the same identity between frames.

Object tracking extends detection by associating detected objects across consecutive frames. Each object is assigned a tracking ID, allowing the system to follow its movement throughout the video.

| Object Detection                              | Object Tracking                                              |
| --------------------------------------------- | ------------------------------------------------------------ |
| Detects objects in each frame                 | Detects and follows objects across frames                    |
| Does not maintain object identity             | Maintains object identity                                    |
| Produces bounding boxes and confidence scores | Produces bounding boxes, confidence scores, and tracking IDs |
| Frames are processed independently            | Information is maintained between frames                     |

## Technologies Used

* Python
* Ultralytics YOLO
* YOLOv8n
* ByteTrack
* OpenCV
* Gradio

## Tracking Algorithm

This project uses ByteTrack for multi-object tracking.

ByteTrack associates detected objects between consecutive frames and assigns tracking IDs. The `persist=True` setting is used during tracking so that the tracker maintains information between frames.

The tracking process is implemented using the Ultralytics tracking interface:

```python
results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)
```

## Project Structure

```text
Day-23/
│
├── app.py
├── tracking_practice.py
├── requirements.txt
│
├── input/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   ├── video_3.mp4
│   ├── video_4.mp4
│   └── video_5.mp4
│
└── output/
    ├── tracked_video_1.mp4
    ├── tracked_video_2.mp4
    ├── tracked_video_3.mp4
    ├── tracked_video_4.mp4
    └── tracked_video_5.mp4
```

## Tracking Practice

The `tracking_practice.py` script processes the videos stored in the `input` directory.

For each video, the script:

1. Opens the video using OpenCV.
2. Reads the video frame by frame.
3. Runs YOLO object detection and ByteTrack tracking.
4. Extracts the tracking IDs.
5. Stores unique IDs in a set.
6. Draws bounding boxes, class labels, confidence scores, and tracking IDs.
7. Saves the processed video in the `output` directory.
8. Reports the number of frames processed and unique objects tracked.

## Gradio Application

The `app.py` file provides an interactive interface for the tracking system.

The application allows users to:

* Upload their own video.
* Select from three sample videos.
* Start object tracking.
* View the processed video.
* View tracking IDs and confidence scores on detected objects.
* See the total number of unique objects.
* See the number of frames processed.
* View information about the YOLO model and ByteTrack algorithm.

## How to Run

### 1. Install Dependencies

Open a terminal inside the `Day-23` directory and run:

```bash
pip install -r requirements.txt
```

### 2. Run Tracking Practice

```bash
python tracking_practice.py
```

The processed videos will be saved in the `output` directory.

### 3. Run the Gradio Application

```bash
python app.py
```

The terminal will provide a local Gradio URL. Open the URL in a browser to use the application.

## Sample Videos

The project includes five sample videos for tracking practice.

The Gradio application provides three of these videos directly through the interface:

* Sample 1: `video_1.mp4`
* Sample 2: `video_2.mp4`
* Sample 3: `video_3.mp4`

The remaining videos are available in the `input` directory for additional testing.

## Output

The processed videos contain:

* Object bounding boxes
* Object class labels
* Confidence scores
* Tracking IDs

The system also calculates the total number of unique tracking IDs observed throughout each video.

## Challenges

One of the main challenges in object tracking is maintaining consistent object identities when objects move, overlap, or temporarily become difficult to detect.

Other challenges include:

* Different object movement speeds.
* Multiple objects appearing simultaneously.
* Objects crossing or overlapping each other.
* Changes in object position between frames.
* Different video resolutions and frame rates.
* Processing time for longer videos.

ByteTrack was used to handle multi-object tracking and maintain object identities across frames.

## Expected Outcome

The completed system demonstrates how YOLO-based object detection can be combined with ByteTrack to perform multi-object tracking on videos.

The project provides both a command-line tracking script and an interactive Gradio application.

