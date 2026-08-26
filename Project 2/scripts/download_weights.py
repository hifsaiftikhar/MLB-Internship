import os
import shutil
import sys
import urllib.request
from huggingface_hub import hf_hub_download

def download_from_url(url, destination):
    print(f"Downloading from {url} to {destination}...")
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
        print(f"Successfully downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Failed to download from {url}: {e}", file=sys.stderr)
        return False

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dest = os.path.join(project_root, "best.pt")
    
    if os.path.exists(dest):
        print(f"Model weights already exist at {dest}")
        return
        
    # Attempt 1: Hugging Face Custom Road Damage Model
    repo_id = "vinothvikas1987/pothole-detection-yolov8"
    filename = "best.pt"
    
    print("Attempting to download pre-trained road damage weights from Hugging Face Hub...")
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=False)
        shutil.copy(cached_path, dest)
        print(f"Successfully saved Hugging Face model weights to {dest}")
        return
    except Exception as e:
        print(f"Hugging Face download failed (e.g., due to rate limiting/HTTP 429): {e}")
        print("Switching to Fallback: Downloading YOLOv8s baseline from GitHub releases...")
        
    # Attempt 2: Fallback to GitHub Releases (no strict 429 rate-limiting)
    fallback_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
    if download_from_url(fallback_url, dest):
        print(f"Successfully saved baseline model weights as fallback to {dest}")
    else:
        print("Error: All weight downloads failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
