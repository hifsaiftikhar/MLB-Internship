import easyocr
import cv2
import os
from pathlib import Path

reader = easyocr.Reader(['en'])

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return denoised

os.makedirs("output", exist_ok=True)

input_folder = Path("input")
images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))

for image_path in sorted(images):
    print(f"\n--- Processing: {image_path.name} ---")
    img = cv2.imread(str(image_path))
    if img is None:
        print("Could not read image")
        continue

    # Raw OCR
    results = reader.readtext(img)
    
    # Preprocessed OCR
    preprocessed = preprocess_image(img)
    results_preprocessed = reader.readtext(preprocessed)

    # Save results
    output_file = Path("output") / f"{image_path.stem}_extracted.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== RAW OCR: {image_path.name} ===\n")
        for (bbox, text, conf) in results:
            if conf > 0.3:
                f.write(f"{text} (conf: {conf:.2f})\n")
        f.write(f"\n=== PREPROCESSED OCR ===\n")
        for (bbox, text, conf) in results_preprocessed:
            if conf > 0.3:
                f.write(f"{text} (conf: {conf:.2f})\n")

    # Print summary
    high_conf = [t for (_, t, c) in results if c > 0.5]
    print(f"High confidence words: {high_conf}")
    print(f"Saved to: {output_file}")

print("\nDone. All images processed.")