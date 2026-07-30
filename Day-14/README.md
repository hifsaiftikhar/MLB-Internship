# Day-14: OpenCV Fundamentals & Basic Image Processing

## What I worked on

Today's focus was learning how images are represented and manipulated in OpenCV - reading/writing images, understanding image properties, and performing common operations like resizing, cropping, rotating, flipping, and drawing. I built 7 practice programs, a menu-driven Image Processing Toolkit, a Gradio app version of the toolkit, and applied every operation to 5 different real images for the Challenge Task.

## The difference between BGR and RGB

Every color image is stored as 3 channels representing how much Blue, Green, and Red light makes up each pixel. Most tools (Pillow, matplotlib, web browsers) store and expect these channels in **RGB** order - Red first, then Green, then Blue. OpenCV, for historical reasons, stores and reads images in **BGR** order instead - Blue first, then Green, then Red.

This matters in practice: if you read an image with OpenCV and try to display or process it with a tool expecting RGB without converting first, the colors come out wrong - reds and blues get swapped. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converts between the two when needed.

## What grayscale images are and why they are used

A grayscale image has only 1 channel instead of 3 - each pixel is a single brightness value (typically 0-255) instead of separate Blue, Green, and Red values. Grayscale is used because:
- It reduces the amount of data by two-thirds, which speeds up processing.
- Many classical computer vision techniques (edge detection, thresholding, some feature detectors) only care about brightness patterns, not color, so converting to grayscale first removes irrelevant information without losing what actually matters for those tasks.

## Which OpenCV functions I used

- `cv2.imread()` / `cv2.imwrite()` - reading and saving images
- `cv2.cvtColor()` - converting between color spaces (BGR to grayscale, BGR to RGB)
- `cv2.resize()` - resizing to a fixed width/height or by a scale factor
- Array slicing (`img[y1:y2, x1:x2]`) - cropping, since an image is just a NumPy array
- `cv2.rotate()` - fixed 90/180/270 degree rotations
- `cv2.flip()` - horizontal, vertical, and both-axis flipping
- `cv2.rectangle()`, `cv2.circle()`, `cv2.line()`, `cv2.polylines()` - drawing shapes
- `cv2.putText()` - adding text to an image
- `cv2.convertScaleAbs()` - brightness/contrast adjustment (bonus feature)

## Practice Programs (practice_1 to practice_7)

1. Read an image and display dimensions, channels, and file size
2. Convert a color image to grayscale
3. Resize to different resolutions (fixed sizes and by scale factor)
4. Crop different regions (top-left, center, bottom-right, top strip)
5. Rotate by 90, 180, and 270 degrees
6. Flip horizontally and vertically
7. Draw a rectangle, circle, line, and polygon, and add custom text (name + today's date)

## Mini Project: Image Processing Toolkit (image_toolkit.py)

A menu-driven console application: load an image, then repeatedly choose from Grayscale, Resize, Rotate, Flip, Crop, Draw Shapes, Add Text, Save, Load a different image, or Exit. Operations chain on top of each other - for example, converting to grayscale and then rotating applies the rotation to the already-grayscaled image, not the original.

## Gradio App (app.py)

The same toolkit, through a web interface: upload an image (or click one of 5 sample images included with the app), select an operation from a dropdown (extra controls appear only when relevant - e.g., width/height fields only show for Resize), click Process, and preview the result with a built-in download option.

**Exception handling:** no image uploaded, invalid file, and invalid crop coordinates are all caught with clear messages rather than crashing; any unexpected error is caught so no raw Python traceback reaches the interface.

**Bonus features included:** brightness/contrast adjustment (using `cv2.convertScaleAbs` with adjustable alpha/beta sliders), in addition to all required operations.

## Challenge Task (challenge_task.py)

Applied all 7 operations plus the brightness/contrast bonus to 5 different images:
- **Landscape** (landscape.jpg)
- **Person** (person.jpg)
- **Vehicle** (car.jpg)
- **Document** (document.jpg)
- **Object** (object.jpg)

Each image produced 11 output files (grayscale, 3 resize variants, center crop, 3 rotations, 2 flips, shapes, text, brightness/contrast), organized into its own subfolder under output/ (challenge_landscape/, challenge_person/, etc.) - 55 total output images.

## Challenges faced

- One of my source images (car.jpg) had a .jpg extension but wasn't actually a valid JPEG file internally - likely downloaded from a source that serves a different format under a .jpg-looking name. cv2.imread() failed silently (returned None) instead of raising a clear error. I fixed this by re-saving the file through Pillow (Image.open().convert("RGB").save()), which reads based on actual file content rather than the extension, and correctly re-encoded it as a real JPEG.
- Keeping image paths working correctly regardless of which folder the Gradio app is launched from - solved by resolving all file paths relative to the script's own location (os.path.dirname(os.path.abspath(__file__))) instead of assuming a fixed working directory.
- Deciding how to organize 55 output files (5 images x 11 operations) in a way that stays readable - solved by giving each source image its own subfolder under output/, rather than one flat folder with 55 similarly-named files.

## Files

- practice_1_image_properties.py through practice_7_shapes_text.py: coding practice programs
- image_toolkit.py: menu-driven Image Processing Toolkit
- app.py: Gradio app version of the toolkit, with sample images and exception handling
- challenge_task.py: applies all operations to the 5 Challenge Task images
- input/: landscape.jpg, person.jpg, car.jpg, document.jpg, object.jpg
- output/: all processed images, including output/challenge_*/ subfolders for the Challenge Task
- README.md: this file

## Author

Hifsa Iftikhar
