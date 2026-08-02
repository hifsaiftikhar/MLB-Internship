import cv2
import numpy as np
import os
from document_enhancer_app import correct_perspective

# All 5 tilted images required for the Challenge Task
challenge_images = [f"input/sample{i}.jpg" for i in range(1, 6)]


def enhance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.3, beta=15)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


def make_comparison(original, corrected, final, label):
    # Resize all three to the same height so they line up side by side
    target_h = 400
    def resize_to_height(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    o = resize_to_height(original, target_h)
    c = resize_to_height(cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
                          if corrected.ndim == 2 else corrected, target_h)
    f = resize_to_height(cv2.cvtColor(final, cv2.COLOR_GRAY2BGR), target_h)

    labeled = []
    for img, text in [(o, "Original"), (c, "Perspective-Corrected"), (f, "Final Enhanced")]:
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        labeled.append(img)

    comparison = np.hstack(labeled)
    return comparison


print("===== CHALLENGE TASK: 5 TILTED DOCUMENTS =====\n")

os.makedirs("output/challenge", exist_ok=True)

for path in challenge_images:
    name = os.path.splitext(os.path.basename(path))[0]
    original = cv2.imread(path)
    if original is None:
        print(f"[{name}] Error: could not read {path} - skipping.")
        continue

    corrected, was_corrected = correct_perspective(original)
    final = enhance(corrected)

    cv2.imwrite(f"output/challenge/{name}_original.jpg", original)
    cv2.imwrite(f"output/challenge/{name}_corrected.jpg", corrected)
    cv2.imwrite(f"output/challenge/{name}_final.jpg", final)

    comparison = make_comparison(original, corrected, final, name)
    cv2.imwrite(f"output/challenge/{name}_comparison.jpg", comparison)

    status = "perspective corrected" if was_corrected else "correction skipped (no clear edges)"
    print(f"[{name}] {original.shape[1]}x{original.shape[0]} -> {status}")

print("\nChallenge Task complete. Results saved to output/challenge/")
