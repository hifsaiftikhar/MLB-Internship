# Day-18: Video Processing with OpenCV

## What I worked on

Today's focus was processing videos with OpenCV - reading video files frame by frame, extracting properties like FPS and resolution, applying image processing techniques (grayscale, Gaussian blur, Canny edge detection) to each frame, and saving the processed result as a new video. I also built webcam capture code, and worked around a hardware limitation (no physical webcam on this machine) with a documented simulation fallback.

## How OpenCV reads videos

OpenCV treats a video as a sequential stream of individual image frames, accessed through the `cv2.VideoCapture` class. Opening a video file or camera returns a capture object, and calling `cap.read()` repeatedly returns one frame at a time as a `(ret, frame)` tuple - `ret` is `True` if a frame was successfully retrieved, and `False` once the video has ended (or if a camera disconnects). This is why almost all video processing code is structured as a `while True: ret, frame = cap.read()` loop that breaks when `ret` is `False`. Each frame is just a regular image (a NumPy array), which is why every image processing technique from the last several days - grayscale conversion, blurring, edge detection, drawing - can be applied directly to a video frame with no special video-specific version of the function needed.

To save a processed video, `cv2.VideoWriter` is used the same way in reverse: instead of reading frames, you call `.write(frame)` once per processed frame, and it assembles them back into a video file using a specified codec (`mp4v` for `.mp4` output, in this case).

## What FPS means

FPS (Frames Per Second) is how many individual frames are displayed per second of video playback. It's read directly from the source video using `cap.get(cv2.CAP_PROP_FPS)`. This number matters specifically when writing the output video: `cv2.VideoWriter` needs to be told the FPS to use, and if it doesn't match the original video's actual frame rate, the processed video will play back faster or slower than intended, even though the same number of frames are present - the timing, not the frame count, is what FPS controls.

## Which processing techniques I applied

- **Grayscale conversion** (`cv2.cvtColor`) - reduces each frame from 3 color channels to 1 brightness channel, since edge detection doesn't need color information and grayscale is significantly faster to process across every frame of a video.
- **Gaussian Blur** (`cv2.GaussianBlur`) - smooths out small pixel-level noise before edge detection, so the result shows real object edges rather than noise-driven false edges.
- **Canny Edge Detection** (`cv2.Canny`) - applied to every frame, producing a clean, thin outline of moving and static objects in the scene.

These three are applied in sequence, frame by frame, inside the main video-reading loop, and each processed frame is written to the output video as it's produced.

## Webcam processing and a hardware limitation

`webcam_processing.py` uses `cv2.VideoCapture(0)` to open the default camera and process its live feed the same way as a video file - grayscale, then Canny, displayed in a live window until the user presses `q`.

When testing this, I found that this machine has no physical webcam attached at all - confirmed both by OpenCV's `Camera index out of range` error across camera indexes 0-3, and by Windows' own Camera app, which returned the explicit error code `0xA00F4244 <NoCamerasAreAttached>`. This isn't a driver or settings problem; there's genuinely no camera hardware present to access.

To still demonstrate the intended real-time processing loop, I added an automatic fallback: the script first attempts to open the webcam, and if that fails, it loads `input/sample_video.mp4` instead and loops it continuously to emulate a live feed, with a visible `"SIMULATED WEBCAM (Fallback)"` label overlaid on the video so it's clearly labeled as a substitute, not real live footage. The actual frame-by-frame capture, processing, and display loop is identical either way - only the source of the frames changes.

## Challenges faced

- **No webcam hardware available** - addressed with the documented, clearly-labeled simulation fallback described above, rather than skipping the webcam requirement entirely or pretending to have tested real live capture.
- **Matching output FPS to input FPS** - initially easy to overlook, but getting this wrong makes the processed video play at the wrong speed even though every frame was processed correctly; fixed by always reading FPS from the source video with `cap.get(cv2.CAP_PROP_FPS)` and passing that same value into `cv2.VideoWriter`.
- **Processing time on high-resolution video** - one of my test videos was 2160x3840 (4K, portrait orientation) with 312 frames; applying grayscale and Canny to every frame at that resolution took noticeably longer than the smaller test videos, which is expected given the much larger number of pixels per frame.

## Challenge Task: 3 video comparison

Processed 3 different videos, saving both the original and the processed (grayscale + Canny) version of each:

| Video | Resolution | Frame count | Observations |
|---|---|---|---|
| sample_video.mp4 | 2160x3840 (portrait) | 312 | Clear outlines overall, but the amount of fine background detail (building textures, wiring) created noticeably more edge clutter than the other two videos. |
| sample_video2.mp4 | 768x432 | 377 | Car traffic scene. Vehicles have strong contrast against the road surface, producing very clean, well-defined outlines with comparatively little noise. |
| sample_video3.mp4 | 768x432 | 596 | Pedestrian walking scene. Moving outlines came through clearly; Gaussian blur noticeably reduced noise from the textured pavement background before edge detection. |

## Files

- video_processing.py: reads a video, prints FPS/resolution/frame count, applies grayscale + Canny to every frame, saves the processed video
- webcam_processing.py: live webcam processing with grayscale + Canny, with an automatic, clearly-labeled fallback to a looped sample video when no webcam is present
- video_processor_app.py: Gradio-based Real-Time Video Processing Tool (mini project)
- input/: sample_video.mp4, sample_video2.mp4, sample_video3.mp4 (original videos)
- output/: processed (grayscale + Canny) versions of each video
- README.md: this file

## Author

Hifsa Iftikhar