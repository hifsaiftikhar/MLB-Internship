import cv2
import os

def find_image_path(i):
    possible_names = [
        f"input/sample{i}.jpg",
        f"input/sample{i}.png",
        f"input/sample{i}.jpeg",
        f"input/sample {i}.jpg",
        f"input/sample {i}.png",
        f"input/sample {i}.jpeg",
    ]
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None


challenge_images = []
for i in range(1, 11):
    path = find_image_path(i)
    if path:
        challenge_images.append(path)
    else:
        print(f"Warning: Could not find image for sample {i}")


def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    corners = len(approx)

    if corners == 3:
        return "Triangle"
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif corners > 4:
        return "Circle"
    return "Polygon"


print("===== CHALLENGE TASK: 10 SHAPE IMAGES =====\n")
os.makedirs("output/challenge", exist_ok=True)

for path in challenge_images:
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        print(f"[{name}] Error: could not read {path} - skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fallback to Otsu thresholding if standard thresholding finds no shapes
    if sum(1 for c in contours if cv2.contourArea(c) >= 100) == 0:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    final = img.copy()
    shape_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        shape = classify_shape(c)
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(final, [c], -1, (0, 255, 0), 2)
        cv2.putText(final, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
        shape_count += 1

    cv2.imwrite(f"output/challenge/{name}_original.jpg", img)
    cv2.imwrite(f"output/challenge/{name}_contours.jpg", contour_img)
    cv2.imwrite(f"output/challenge/{name}_final.jpg", final)

    print(f"[{name}] {img.shape[1]}x{img.shape[0]} -> {shape_count} shape(s) detected")

print("\nChallenge Task complete. Results saved to output/challenge/")
