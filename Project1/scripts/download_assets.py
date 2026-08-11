import os
import urllib.request
import sys
import pickle
import json

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            chunk_size = 1024 * 1024
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded / (1024*1024):.2f} MB...", end="\r")
            print()
        print(f"Successfully downloaded {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def convert_pickle_to_json(pickle_path, json_path):
    print(f"Converting pickle coordinates {pickle_path} to JSON {json_path}...")
    try:
        with open(pickle_path, 'rb') as f:
            pos_list = pickle.load(f)
        
        width, height = 107, 48
        
        slots = []
        for i, pos in enumerate(pos_list):
            x, y = pos
            points = [
                [int(x), int(y)],
                [int(x + width), int(y)],
                [int(x + width), int(y + height)],
                [int(x), int(y + height)]
            ]
            slots.append({
                "id": i + 1,
                "points": points
            })
            
        with open(json_path, 'w') as f:
            json.dump(slots, f, indent=4)
        print(f"Successfully converted {len(slots)} spots and saved to {json_path}.")
        return True
    except Exception as e:
        print(f"Error converting coordinates: {e}", file=sys.stderr)
        return False

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    image_urls = [
        "https://github.com/harshbafnaa/car-parking-detection/raw/main/carParkImg.png",
        "https://github.com/harshbafnaa/car-parking-detection/raw/master/carParkImg.png",
        "https://github.com/theabess/Parking-Space-Counter/raw/master/carParkImg.png"
    ]
    
    video_urls = [
        "https://github.com/harshbafnaa/car-parking-detection/raw/main/carPark.mp4",
        "https://github.com/harshbafnaa/car-parking-detection/raw/master/carPark.mp4",
        "https://github.com/theabess/Parking-Space-Counter/raw/master/carPark.mp4"
    ]
    
    pickle_urls = [
        "https://github.com/harshbafnaa/car-parking-detection/raw/main/CarParkPos",
        "https://github.com/harshbafnaa/car-parking-detection/raw/master/CarParkPos",
        "https://github.com/theabess/Parking-Space-Counter/raw/master/CarParkPos"
    ]
    
    success = True
    
    image_dest = os.path.join(data_dir, "carParkImg.png")
    image_downloaded = os.path.exists(image_dest)
    if not image_downloaded:
        for url in image_urls:
            if download_file(url, image_dest):
                image_downloaded = True
                break
    if not image_downloaded:
        print("Failed to download carParkImg.png from all sources.", file=sys.stderr)
        success = False
        
    video_dest = os.path.join(data_dir, "carPark.mp4")
    video_downloaded = os.path.exists(video_dest)
    if not video_downloaded:
        for url in video_urls:
            if download_file(url, video_dest):
                video_downloaded = True
                break
    if not video_downloaded:
        print("Failed to download carPark.mp4 from all sources.", file=sys.stderr)
        success = False

    pickle_dest = os.path.join(data_dir, "CarParkPos")
    pickle_downloaded = False
    for url in pickle_urls:
        if download_file(url, pickle_dest):
            pickle_downloaded = True
            break
            
    if pickle_downloaded:
        json_dest = os.path.join(data_dir, "parking_slots.json")
        if convert_pickle_to_json(pickle_dest, json_dest):
            try:
                os.remove(pickle_dest)
            except Exception:
                pass
        else:
            success = False
    else:
        print("Failed to download CarParkPos coordinates from all sources.", file=sys.stderr)
        success = False
            
    if success:
        print("\nAll assets downloaded and prepared successfully!")
    else:
        print("\nSome assets failed to download or prepare.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
