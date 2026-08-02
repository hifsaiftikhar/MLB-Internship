import cv2
import numpy as np
import os

challenge_images = [f"input/sample{i}.jpg" for i in range(1, 11)]


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = img.shape[0] * img.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    final = img.copy()
    found = False
    for contour in contours:
        if cv2.contourArea(contour) < 0.15 * image_area:
            continue
        cv2.drawContours(final, [contour], -1, (0, 255, 0), 3)
        found = True
        break

    return edges, closed, final, found


print("===== CHALLENGE TASK: 10 DOCUMENT IMAGES =====\n")
os.makedirs("output/challenge", exist_ok=True)

for path in challenge_images:
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        print(f"[{name}] Error: could not read {path} - skipping.")
        continue

    edges, morph, final, found = process_image(img)

    cv2.imwrite(f"output/challenge/{name}_original.jpg", img)
    cv2.imwrite(f"output/challenge/{name}_edges.jpg", edges)
    cv2.imwrite(f"output/challenge/{name}_morph.jpg", morph)
    cv2.imwrite(f"output/challenge/{name}_boundary.jpg", final)

    status = "boundary detected" if found else "no confident boundary found"
    print(f"[{name}] {img.shape[1]}x{img.shape[0]} -> {status}")

print("\nChallenge Task complete. Results saved to output/challenge/")
