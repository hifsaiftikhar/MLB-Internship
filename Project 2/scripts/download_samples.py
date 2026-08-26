import os
import urllib.request
import sys

def download_file(url, destination):
    print(f"Downloading sample image from {url} to {destination}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully saved to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sample_dir = os.path.join(project_root, "sample_test_images")
    os.makedirs(sample_dir, exist_ok=True)
    
    samples = {
        "pothole_sample_1.jpg": "https://images.unsplash.com/photo-1515162305285-0293e4767cc2?q=80&w=640&auto=format&fit=crop",
        "crack_sample_1.jpg": "https://images.unsplash.com/photo-1621243804936-775306a8f2e3?q=80&w=640&auto=format&fit=crop",
        "distress_sample_1.jpg": "https://images.unsplash.com/photo-1599740831114-171888b1d927?q=80&w=640&auto=format&fit=crop",
        "damaged_road_1.jpg": "https://images.unsplash.com/photo-1584467541268-b040f83be3fd?q=80&w=640&auto=format&fit=crop"
    }
    
    success = True
    for name, url in samples.items():
        dest = os.path.join(sample_dir, name)
        if not download_file(url, dest):
            success = False
            
    if success:
        print("\nAll sample test images downloaded successfully!")
    else:
        print("\nSome sample images failed to download.", file=sys.stderr)

if __name__ == "__main__":
    main()
