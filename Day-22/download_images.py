import os
import urllib.request
import sys


def download_file(url, destination):
    print(f"Downloading {url}...")

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )

        with urllib.request.urlopen(request) as response:
            data = response.read()

        with open(destination, "wb") as out_file:
            out_file.write(data)

        print(f"Successfully downloaded: {os.path.basename(destination)}")
        return True

    except Exception as e:
        print(
            f"Error downloading {url}: {e}",
            file=sys.stderr
        )
        return False


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input")

    os.makedirs(input_dir, exist_ok=True)

    # OpenCV sample images
    images = {
        "sample01.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",

        "sample02.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",

        "sample03.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/peppers.png",

        "sample04.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/graf1.png",

        "sample05.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/home.jpg",

        "sample06.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",

        "sample07.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png",

        "sample08.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/boat.jpg",

        "sample09.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png",

        "sample10.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/chessboard.png",

        "sample11.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",

        "sample12.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left.jpg",

        "sample13.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right.jpg",

        "sample14.png":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/coins.png",

        "sample15.jpg":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/starry_night.jpg",
    }

    print("=" * 60)
    print("        DAY-22 IMAGE SEGMENTATION DATASET")
    print("=" * 60)

    print(f"\nPreparing {len(images)} sample images...\n")

    successful = 0
    skipped = 0
    failed = 0

    for filename, url in images.items():

        destination = os.path.join(input_dir, filename)

        # Don't download again if file already exists
        if os.path.exists(destination):
            print(f"Already exists: {filename}")
            skipped += 1
            continue

        if download_file(url, destination):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"Newly downloaded : {successful}")
    print(f"Already existed  : {skipped}")
    print(f"Failed           : {failed}")
    print(f"Total images     : {len(images)}")
    print(f"\nImages location: {input_dir}")

    if successful + skipped == len(images):
        print("\nAll images are ready!")
    else:
        print("\nSome images could not be downloaded.")
        print("Please check the URLs and run the script again.")


if __name__ == "__main__":
    main()