# Day-16: Edge Detection & Morphological Operations

## What I worked on

Today's focus was edge detection (Sobel, Laplacian, Canny) and morphological operations (erosion, dilation, opening, closing, gradient, top hat, black hat), then combining both into a Document Boundary Detection Tool that finds and outlines a document's edge in a photo.

## The difference between Sobel, Laplacian, and Canny

- **Sobel:** calculates the gradient (rate of intensity change) separately in the X and Y directions using small convolution kernels, then combines them. Good at showing edge direction, but sensitive to noise and produces thicker, less precise edges on its own.
- **Laplacian:** a second-derivative operator that detects edges in all directions at once, without needing separate X/Y kernels. More sensitive to noise than Sobel, so it generally needs blurring first, and can pick up more false edges from small pixel variations.
- **Canny:** a multi-step algorithm (blur, gradient calculation, non-maximum suppression, then double thresholding) that produces the cleanest, thinnest, most continuous edges of the three. This is why I used Canny specifically in the Document Boundary Detection Tool - it gave the most reliable single-pixel-wide outline to build a contour from.

In my comparison output (output/3_comparison.jpg), Canny visibly produced the cleanest, least noisy lines, while Sobel and Laplacian both showed thicker, slightly noisier edges around the same document outline.

## The purpose of each morphological operation

- **Erosion:** shrinks white regions in a binary image, removing small isolated noise specks.
- **Dilation:** grows white regions, filling small gaps and connecting nearby broken edge segments.
- **Opening (erosion then dilation):** removes small noise while keeping the overall shape of larger regions intact.
- **Closing (dilation then erosion):** fills small holes and gaps inside a shape while keeping its overall size the same. This is the operation I used in the Document Boundary Detection Tool, since Canny edges are often broken into small disconnected segments - closing stitches them into one continuous outline that can be traced as a single contour.
- **Morphological Gradient:** dilation minus erosion, which highlights just the outline of a shape.
- **Top Hat:** original image minus its opening, which highlights small bright details that are smaller than the structuring element.
- **Black Hat:** closing minus the original image, which highlights small dark details.

## Mini Project: Document Boundary Detection Tool (boundary_detector_app.py)

A Gradio app that:
1. Converts the uploaded document photo to grayscale.
2. Applies Gaussian Blur to reduce noise before edge detection.
3. Detects edges using Canny.
4. Applies morphological closing to fill small gaps in the edge lines, so the document's outline forms one continuous contour instead of several broken segments.
5. Finds the largest contour above a minimum size threshold - assumed to be the document.
6. Draws the detected boundary directly on the original image in green.

This is a different technique from Day-15's Document Image Enhancement Tool: yesterday's app *warped* the document to a straightened, top-down view; today's tool just *outlines* where the document is detected on the original, unmodified photo.

## Which combination of techniques gave the best results

**Grayscale, then Gaussian Blur, then Canny, then morphological Closing.** Canny alone often produces small broken edge segments rather than one continuous outline, especially where the document's edge briefly loses contrast against its background (a shadow, a slightly blurred region, or a busy background). Closing fills exactly these small gaps, which is what allows `cv2.findContours` to trace one single, continuous shape instead of many disconnected fragments. Without closing, contour detection frequently failed to find a clean 4-sided (or large enough) region even on otherwise reasonable photos.

## Challenges faced while detecting document boundaries

Out of 10 test images, 7 had their boundary successfully detected and 3 did not, for two distinct and explainable reasons:

- **Low contrast against the background:** two of the failed images were the same document (one straight, one tilted) photographed on a light gray surface. Since the page itself was also light-colored, there wasn't enough intensity difference at the edge for Canny to produce a strong, continuous boundary line - the detector had too little contrast to work with.
- **A strong shadow competing with the real edge:** the third failed image had a distinct cast shadow crossing the document. The shadow's own edge was high-contrast enough that it competed with the actual page boundary, and the "largest contour" logic isn't aware of which high-contrast line is the real page edge versus an unrelated shadow.

Both cases are genuine limitations of contour-based detection, not implementation bugs: the approach depends on the document having reasonably clear, consistent contrast against its background, and can be confused by other strong edges in the scene, like shadows. The tool correctly reports "no confident boundary found" in these cases rather than drawing an incorrect outline.

## Challenge Task

Processed 10 different document images (straight scans, phone photos, tilted photos, a shadowed photo, and a slightly blurred photo). For each image, output/challenge/ contains:
- `<name>_original.jpg` - the original photo
- `<name>_edges.jpg` - Canny edge detection result
- `<name>_morph.jpg` - after morphological closing
- `<name>_boundary.jpg` - final image with the detected boundary drawn (or the original image if no confident boundary was found)

7 of 10 images had a boundary successfully detected; 3 did not, for the reasons explained above.

## Files

- edge_detection.py: grayscale, Gaussian Blur, and Sobel/Laplacian/Canny comparison
- morphological_ops.py: all 7 morphological operations, with a before/after comparison
- boundary_detector_app.py: Gradio-based Document Boundary Detection Tool
- challenge_task.py: processes 10 document images and saves original/edges/morph/boundary for each
- input/: sample document images
- output/: results from edge_detection.py and morphological_ops.py, plus output/challenge/ for the Challenge Task
- README.md: this file

## Author

Hifsa Iftikhar
