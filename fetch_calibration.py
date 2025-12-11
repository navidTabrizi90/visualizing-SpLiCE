import os
import requests
from tqdm import tqdm

def download_calibration_set(target_folder="data/calibration", count=100):
    """
    Downloads random natural images to use as a baseline for the 'Image Mean'.
    """
    print(f"--- Downloading {count} calibration images ---")
    os.makedirs(target_folder, exist_ok=True)
    
    # URL that redirects to a random photo (Unsplash/Lorem Picsum)
    url = "https://picsum.photos/224/224"
    
    for i in tqdm(range(count)):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(f"{target_folder}/img_{i:03d}.jpg", "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"Skipped {i}: {e}")

    print(f"✅ Downloaded to {target_folder}/")

if __name__ == "__main__":
    download_calibration_set()