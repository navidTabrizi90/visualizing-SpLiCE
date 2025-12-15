import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip

# --- CONFIGURATION ---
# limit the calculation to 20,000 images. This is statistically enough to find the "center" of the dataset 

LIMIT = 20000 
BATCH_SIZE = 64
TARGET_FOLDER = "Sketchy/images" 
OUTPUT_PATH = "data/mscoco_mean.pt" # Keep name consistent for other scripts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_image_mean():
    print(f"--- Computing Image Mean (Fast Mode: Limit {LIMIT}) ---")
    print(f"Source: {TARGET_FOLDER}")
    
    # 1. Load CLIP Model (for preprocessing only)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.to(DEVICE)
    model.eval()

    # 2. Setup Dataset
    try:
        dataset = datasets.ImageFolder(root=TARGET_FOLDER, transform=preprocess)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 3. Compute Mean
    total_embedding = torch.zeros(512).to(DEVICE)
    count = 0
    
    print("Processing...")
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(DEVICE)
            
            # Get Embeddings
            batch_embeddings = model.encode_image(images)
            # Normalize to unit sphere 
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            # Sum up
            total_embedding += batch_embeddings.sum(dim=0)
            count += len(images)
            
            # Stop if we hit the limit
            if count >= LIMIT:
                break
    
    # 4. Average and Save
    final_mean = total_embedding / count
    final_mean = final_mean / final_mean.norm() # Re-normalize the mean vector itself

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(final_mean.cpu(), OUTPUT_PATH)
    
    print(f"✅ Success! Computed mean from {count} images.")    
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    compute_image_mean()