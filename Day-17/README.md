# Day-17: Contours & Shape Detection

## What I worked on

Today's focus was finding objects and shapes in an image using contours - the outlines of shapes, found via thresholding and OpenCV's contour functions. I built practice scripts for contour detection and shape classification, then a Shape Detection System (Gradio app) that labels each detected shape with its area and perimeter.

## What are contours?

A contour is a curve joining all continuous points along a boundary that share similar color or intensity - essentially the outline of a shape or object. `cv2.findContours()` finds these outlines, but it requires a binary (black and white) image as input, not a regular color or grayscale image, since it needs a clear distinction between "object" and "background" to trace an edge.

## How contour detection works

1. Convert the image to grayscale.
2. Apply thresholding (`cv2.threshold`) to convert grayscale into pure black and white, based on a cutoff pixel value - this is what gives `cv2.findContours()` the clean binary input it needs.
3. `cv2.findContours()` traces the outlines of all the white (or black, depending on threshold direction) regions.
4. Each contour can then be measured (`cv2.contourArea`, `cv2.arcLength`) or approximated into a simpler polygon (`cv2.approxPolyDP`), which is the basis for shape detection - counting how many corners the simplified polygon has tells you what shape it likely is.

## Which shapes my program can detect

- **Triangle** - exactly 3 corners after polygon approximation.
- **Square** - 4 corners, with width and height close to equal (aspect ratio between 0.95 and 1.05).
- **Rectangle** - 4 corners, but width and height are clearly different.
- **Circle** - more than 4 corners after approximation (a circle approximates to many small segments).
- **Polygon** - anything else (fallback label).

## Mini Project: Shape Detection System (shape_detector_app.py)

A Gradio app that:
1. Loads an uploaded image.
2. Converts to grayscale and applies thresholding.
3. Finds all contours, filtering out very small ones (area under 100) that are usually just noise.
4. Classifies each remaining contour's shape.
5. Draws the contour outline and shape label directly on the image.
6. Displays each shape's area and perimeter as text.

## Challenges faced

- **Distinguishing squares from rectangles** required an extra step beyond just counting corners, since both have 4 corners after polygon approximation. I compared the bounding rectangle's width to its height - if they're close to equal (within a 5% tolerance), it's classified as a square, otherwise a rectangle.
- **Filtering out noise contours:** thresholding can sometimes pick up tiny specks or JPEG compression artifacts as extremely small contours. I added a minimum area threshold (100 pixels) so these don't get labeled as false "shapes" alongside the real ones.
- **Choosing the right approximation tolerance** for `cv2.approxPolyDP`: too tight, and a slightly imperfect circle gets approximated with too many corners and read as a polygon instead of a circle; too loose, and a rectangle's corners can get simplified away entirely. I used 4% of the contour's perimeter as the tolerance, which worked reliably across my test shapes.

## Challenge Task

Processed 10 different shape images. For each image, output/challenge/ contains:
- `<name>_original.jpg` - the original image
- `<name>_contours.jpg` - all detected contour outlines drawn
- `<name>_final.jpg` - final image with each shape labeled, contours drawn, and its bounding information available

## Files

- contour_detection.py: reads an image, thresholds it, finds contours, calculates area/perimeter, draws bounding rectangles
- shape_detection.py: classifies each contour as Triangle/Square/Rectangle/Circle/Polygon and labels the image
- shape_detector_app.py: Gradio-based Shape Detection System
- challenge_task.py: processes 10 shape images and saves original/contours/final for each
- input/: sample shape images
- output/: results from contour_detection.py and shape_detection.py, plus output/challenge/ for the Challenge Task
- README.md: this file

## Author

Hifsa Iftikhar
