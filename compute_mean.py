import torch
import os
import glob
from PIL import Image
from tqdm import tqdm
from splice.feature_extractor import CLIPFeatureExtractor

def compute_image_mean(image_folder, save_path="data/mscoco_mean.pt", batch_size=64):
    """
    Computes the average CLIP embedding for a large set of images.
    Paper Source: "estimated over MSCOCO train set"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Computing Image Cone Mean ---")
    print(f"Source: {image_folder}")
    print(f"Device: {device}")
    
    # 1. Load Model
    extractor = CLIPFeatureExtractor(device=device)
    
    # 2. Find Images (JPEGs, PNGs)
    # Adjust extensions if needed
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {image_folder}")
        return

    print(f"Found {len(image_paths)} images. Processing...")
    
    # 3. Compute Mean
    embedding_sum = torch.zeros(1, 512).to(device)
    count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        
        # Load and Preprocess
        valid_batch = False
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                # Preprocess returns (3, 224, 224)
                tensor = extractor.preprocess(img)
                batch_tensors.append(tensor)
                valid_batch = True
            except Exception as e:
                print(f"Skipping bad image: {p}")
        
        if not valid_batch:
            continue
            
        # Stack into (B, 3, 224, 224)
        input_batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            # Get features
            features = extractor.model.encode_image(input_batch)
            # Normalize per image FIRST (Critical per paper)
            features = features / features.norm(dim=-1, keepdim=True)
            
            # Add to sum
            embedding_sum += torch.sum(features, dim=0, keepdim=True)
            count += len(batch_tensors)
            
    # 4. Final Average
    if count > 0:
        mu_img = embedding_sum / count
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(mu_img, save_path)
        print(f"✅ Success! Mean vector computed from {count} images.")
        print(f"Saved to {save_path}")
    else:
        print("Failed to process any images.")

if __name__ == "__main__":
    # USER SETTING: Change this path to your folder of images!
    # If you don't have MSCOCO, use any folder with 1000+ diverse images.
    # Example: "C:/Datasets/coco/train2017"
    TARGET_FOLDER = "data/calibration"
    
    compute_image_mean(TARGET_FOLDER)   