# File: datasets.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob

class SketchyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the hash folders.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Find all 'original_image.jpg' files inside the subfolders
        # Adjust pattern if you want the sketches (e.g. 'crop_*.png')
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "original_image.jpg"))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}. Check your folder structure.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), "error"

        if self.transform:
            image = self.transform(image)

        # Return the folder name as the label/ID
        label = os.path.basename(os.path.dirname(img_path))
        return image, label

# TEST CODE
if __name__ == "__main__":
    print("Testing Dataset...")
    # UPDATE THIS PATH TO YOUR REAL PATH
    dummy_path = r'F:\Thesis\SpLiCE\Sketchy\images'
    if os.path.exists(dummy_path):
        ds = SketchyDataset(dummy_path)
        print(f"Test Passed: Found {len(ds)} images.")
    else:
        print("Test Skipped: Path not found. Set 'dummy_path' to verify.")