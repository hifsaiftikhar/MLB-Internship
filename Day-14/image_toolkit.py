import cv2
import numpy as np
import os
from datetime import date


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not read image at {path}")
        return None
    print(f"Image loaded: {path} ({img.shape[1]}x{img.shape[0]})")
    return img


def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert back to 3-channel so it can still be saved/displayed
    # consistently alongside color images in this same toolkit.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def resize_image(img):
    try:
        width = int(input("Enter new width: "))
        height = int(input("Enter new height: "))
        return cv2.resize(img, (width, height))
    except ValueError:
        print("Invalid input. Width and height must be whole numbers.")
        return img


def rotate_image(img):
    print("1. 90 degrees\n2. 180 degrees\n3. 270 degrees")
    choice = input("Choose rotation: ")
    if choice == "1":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif choice == "2":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif choice == "3":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("Invalid choice. No rotation applied.")
        return img


def flip_image(img):
    print("1. Horizontal\n2. Vertical\n3. Both")
    choice = input("Choose flip direction: ")
    if choice == "1":
        return cv2.flip(img, 1)
    elif choice == "2":
        return cv2.flip(img, 0)
    elif choice == "3":
        return cv2.flip(img, -1)
    else:
        print("Invalid choice. No flip applied.")
        return img


def crop_image(img):
    height, width = img.shape[:2]
    print(f"Image size is {width}x{height}. Enter crop region:")
    try:
        x1 = int(input("x1 (left): "))
        y1 = int(input("y1 (top): "))
        x2 = int(input("x2 (right): "))
        y2 = int(input("y2 (bottom): "))

        if not (0 <= x1 < x2 <= width) or not (0 <= y1 < y2 <= height):
            print("Invalid crop region - coordinates out of bounds. No crop applied.")
            return img

        return img[y1:y2, x1:x2]
    except ValueError:
        print("Invalid input. Coordinates must be whole numbers.")
        return img


def draw_shapes(img):
    canvas = img.copy()
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (10, 10), (width // 3, height // 3), (0, 255, 0), 2)
    cv2.circle(canvas, (width // 2, height // 2), min(width, height) // 6, (255, 0, 0), 2)
    cv2.line(canvas, (0, height - 1), (width - 1, 0), (0, 0, 255), 2)
    points = np.array([
        [width // 2, height - 10],
        [width - 10, height // 2],
        [width // 2, 10],
    ], dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 255), thickness=2)
    print("Rectangle, circle, line, and polygon drawn.")
    return canvas


def add_text(img):
    name = input("Enter your name: ").strip()
    if not name:
        name = "Hifsa Iftikhar"
    today = date.today().strftime("%Y-%m-%d")
    text = f"{name} - {today}"
    canvas = img.copy()
    height = canvas.shape[0]
    cv2.putText(canvas, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    print(f"Added text: {text}")
    return canvas


def save_image(img):
    os.makedirs("output", exist_ok=True)
    filename = input("Enter filename to save as (e.g. result.jpg): ").strip()
    if not filename:
        filename = "result.jpg"
    output_path = os.path.join("output", filename)
    cv2.imwrite(output_path, img)
    print(f"Saved to {output_path}")


def main():
    print("===== IMAGE PROCESSING TOOLKIT =====")
    image_path = input("Enter path to an image to load: ").strip()
    img = load_image(image_path)

    if img is None:
        print("Cannot continue without a valid image. Exiting.")
        return

    while True:
        print("\n--- Menu ---")
        print("1. Convert to grayscale")
        print("2. Resize image")
        print("3. Rotate image")
        print("4. Flip image")
        print("5. Crop image")
        print("6. Draw shapes")
        print("7. Add custom text")
        print("8. Save current image")
        print("9. Load a different image")
        print("10. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            img = to_grayscale(img)
        elif choice == "2":
            img = resize_image(img)
        elif choice == "3":
            img = rotate_image(img)
        elif choice == "4":
            img = flip_image(img)
        elif choice == "5":
            img = crop_image(img)
        elif choice == "6":
            img = draw_shapes(img)
        elif choice == "7":
            img = add_text(img)
        elif choice == "8":
            save_image(img)
        elif choice == "9":
            new_path = input("Enter path to new image: ").strip()
            new_img = load_image(new_path)
            if new_img is not None:
                img = new_img
        elif choice == "10":
            print("Exiting. Goodbye.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
