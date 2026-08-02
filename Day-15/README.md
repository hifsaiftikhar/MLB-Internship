# Day-15: Image Transformations & Image Enhancement

## What I worked on

Today's focus was image transformations (translation, rotation, scaling, affine, perspective) and image enhancement (brightness, contrast, noise reduction, sharpening) - the standard preprocessing steps used before feeding images into Computer Vision models. I built practice scripts for both, then a Document Image Enhancement Tool that automatically detects and straightens tilted documents.

## Transformations implemented

- **Translation:** shifts the image left/right and up/down using a translation matrix with `cv2.warpAffine`. Used for centering an object or data augmentation.
- **Rotation (arbitrary angle):** rotates by any angle (not just 90/180/270) using `cv2.getRotationMatrix2D`. Used for correcting slightly tilted photos.
- **Scaling:** resizing up and down, covered again here since it's grouped with the other transformations today.
- **Affine transformation:** maps 3 points to 3 new points using `cv2.getAffineTransform`, preserving parallel lines. Used for skew correction and shearing effects.
- **Perspective transformation:** maps 4 points to 4 new points using `cv2.getPerspectiveTransform`, and is not limited to preserving parallel lines - this is what makes it capable of straightening a document photographed at an angle, which affine transformation cannot do on its own.

## Enhancement techniques implemented

- **Brightness adjustment:** adds/subtracts a constant from every pixel (`beta` in `cv2.convertScaleAbs`).
- **Contrast adjustment:** scales pixel values around their midpoint (`alpha` in `cv2.convertScaleAbs`).
- **Gaussian Blur:** smooths the image, including edges - fast but not edge-aware.
- **Median Blur:** better at removing noise while keeping edges cleaner than Gaussian Blur.
- **Bilateral Filter:** reduces noise while specifically preserving edges - the best choice here since document text edges need to stay sharp while background noise is smoothed out.
- **Sharpening:** a convolution kernel that emphasizes each pixel relative to its neighbors, making text and edges more defined.

## Mini Project: Document Image Enhancement Tool (document_enhancer_app.py)

A Gradio app that:
1. Detects the document's edges automatically using Canny edge detection and contour analysis - the classic "document scanner" approach.
2. If a clear 4-sided document shape is found, warps it to a straightened, top-down view using perspective transformation.
3. Converts to grayscale.
4. Reduces noise with a bilateral filter (keeps text edges sharp).
5. Enhances brightness and contrast.
6. Sharpens the result.
7. Displays the perspective-corrected version and the final enhanced version side by side.

The app gracefully skips perspective correction (with a clear status message) if no confident 4-sided document shape is detected, rather than guessing incorrectly.

## Which transformation had the biggest impact on document quality

**Perspective transformation.** A tilted, skewed photo of a document is genuinely harder to read and would confuse downstream tasks like OCR - grayscale, brightness, and sharpening improve visual quality, but none of them can fix the actual geometric distortion of a document photographed at an angle. In my Challenge Task results, the difference between the original tilted photo and the perspective-corrected version was the single most visually significant change - after that, the enhancement steps (denoise, contrast, sharpen) made further but comparatively smaller improvements to clarity and readability.

## Challenges faced

- **False-positive contour detection:** my first version of the perspective correction sometimes locked onto a small, plain, texture-less patch of background (like a blank part of a table) instead of the actual document, since a flat empty area can produce cleaner edges than a document full of text. I fixed this by requiring any detected contour to cover at least 20% of the total image area before accepting it as the document, which prevents small irrelevant regions from being mistaken for the page.
- **Curved and cluttered scenes don't produce a clean 4-sided contour:** photos of an open book (curved page surface, wrinkled fabric background) or a desk with multiple overlapping items consistently failed to find a confident match. This is a real limitation of the classical contour-based approach, not a bug - the tool correctly reports "no clear document edges detected" and falls back to processing the image without perspective correction, rather than producing an incorrect warp.
- **Getting a clean success on real photos required specific conditions:** a single, flat document with reasonably plain, contrasting background (e.g., a light page on a dark desk) worked reliably; anything with clutter, curvature, or low contrast against the background did not. This shaped how I selected images for the Challenge Task.

## Challenge Task

Processed 5 tilted document images (sample1.jpg - sample5.jpg) through the full pipeline. All 5 had their perspective successfully detected and corrected. For each image, the following are saved under output/challenge/:
- `<name>_original.jpg` - the original tilted photo
- `<name>_corrected.jpg` - after perspective correction
- `<name>_final.jpg` - after grayscale, denoising, brightness/contrast, and sharpening
- `<name>_comparison.jpg` - a labeled side-by-side comparison of all three stages

## Files

- transformations.py: translation, rotation, scaling, affine, and perspective transformation practice
- enhancement.py: brightness, contrast, Gaussian/median/bilateral filtering, and sharpening practice
- document_enhancer_app.py: Gradio-based Document Image Enhancement Tool
- challenge_task.py: processes 5 tilted documents and generates before/after comparisons
- input/: sample document images (sample1.jpg - sample10.jpg)
- output/: results from transformations.py and enhancement.py, plus output/challenge/ for the Challenge Task
- README.md: this file

## Author

Hifsa Iftikhar
