# Day-21: Feature Detection & Feature Matching

## What are Image Features?
Features are distinctive points in an image that can be reliably found again even if the image is rotated, scaled, or slightly changed. Think of them as landmarks — corners, edges, and blobs that stand out from their surroundings.

## Techniques Used

### Harris Corner Detection
Finds corners by looking for areas where brightness changes significantly in multiple directions. Fast and simple but has two key limitations — it does not produce descriptors (so you cannot match features between images directly) and it fails when the image is scaled.

### ORB (Oriented FAST and Rotated BRIEF)
Detects keypoints using FAST and describes them using BRIEF binary descriptors. Uses an image pyramid to handle scale changes and orientation assignment for rotation invariance. Free to use, fast, and good enough for most real-world matching tasks.

## Coding Practice Results

| Metric | Harris | ORB |
|--------|--------|-----|
| Detection Speed | 21.38 ms | 229.75 ms |
| Features Detected | 8959 | 865 |
| Scale Invariant | No | Yes |
| Rotation Invariant | Partially | Yes |
| Produces Descriptor | No | Yes (BRIEF) |

**Key observation:** Harris detects far more points (8959 vs 865) but they are raw corner positions with no descriptor — useless for matching. ORB detects fewer but more meaningful keypoints with descriptors attached, enabling actual feature matching.

## Feature Matching
Used Brute Force Matcher with Hamming distance (suited for binary ORB descriptors) followed by Lowe's Ratio Test to filter weak matches.

- Total raw matches: 865
- Good matches after ratio test: 42

Lowe's Ratio Test keeps a match only if the best match is significantly better than the second best — filters out ambiguous matches that could be noise.

## Mini Project: Image Feature Matching System
Gradio app that accepts two images, detects ORB keypoints on both, matches them using Brute Force + ratio test, and displays the matched feature visualization along with keypoint counts and good match count.

## Project Structure

```
Day-21/
├── input/                     — image pairs used for detection and matching
├── output/                    — visualizations (Harris, ORB keypoints, matched features)
├── scripts/
│   ├── coding_practice.py     — Harris and ORB practice with benchmarking
│   └── download_images.py     — image downloader script
├── app.py                     — Gradio feature matching app
├── requirements.txt
└── README.md
```

## How to Run

**Practice script:**
```
python scripts/coding_practice.py
```

**Gradio app:**
```
python app.py
```

## Challenges Faced
- Harris corners cannot be used for matching directly — no descriptor output
- Lowe's ratio test threshold needs tuning — too strict misses valid matches, too loose includes noise
- Image pairs with large viewpoint changes reduce good match count significantly

## Author
Hifsa Iftikhar
