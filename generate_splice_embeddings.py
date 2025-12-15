import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import your SpLiCE modules
from splice.feature_extractor import CLIPFeatureExtractor
from splice.dictionary import ConceptDictionary
from splice.aligner import ModalityAligner
from splice.solver import ADMMSpLICESolver

# --- CONFIGURATION ---
JSON_FILENAME = 'SKETCHY_train_nested_with_sketches_withHED.json'
IMAGE_FOLDER = os.path.join('Sketchy', 'images')
OUTPUT_FILE = 'splice_data_sketchy.npz'
VOCAB_PATH = 'data/vocab.txt' # Your generated vocabulary
MEAN_PATH = 'data/mscoco_mean.pt' # Optional: Path to image mean

MAX_IMAGES = 1500  
ALPHA = 0.1 # Sparsity penalty
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_json_map(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        full_path = item.get('original_filename', '')
        caption = item.get('original_caption', '')
        if full_path and caption:
            folder = full_path.split('/')[-2]
            mapping[folder] = caption
    return mapping

def main():
    print(f"--- Generating SpLiCE Embeddings on {DEVICE} ---")

    # 1. Load SpLiCE Components
    extractor = CLIPFeatureExtractor(device=DEVICE)
    aligner = ModalityAligner(device=DEVICE)
    
    # Load Vocab
    if not os.path.exists(VOCAB_PATH):
        print("ERROR: vocab.txt not found. Run 'fetch_real_vocab.py' first.")
        return
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    vocab = vocab[:10000] # Limit size for speed
    
    # Build Dictionary Matrix
    dict_builder = ConceptDictionary(extractor)
    C_matrix = dict_builder.build_vocabulary(vocab, do_pruning=False)
    
    # Set Image Mean (Load or Zero)
    if os.path.exists(MEAN_PATH):
        print("Loading Image Mean...")
        aligner.set_image_mean(torch.load(MEAN_PATH, map_location=DEVICE))
    else:
        print("Warning: Using Zero-Mean (Approximate).")
        aligner.mu_img = torch.zeros(1, 512).to(DEVICE)

    # Align Dictionary
    C_centered = aligner.align_dictionary(C_matrix)
    
    # Initialize Solver
    solver = ADMMSpLICESolver(C_centered, alpha=ALPHA, device=DEVICE)

    # 2. Process Images
    folder_to_caption = load_json_map(JSON_FILENAME)
    filenames = []
    captions = []
    sparse_weights_list = []

    print("Processing images...")
    count = 0
    
    # We will process one by one for simplicity (or batches if you prefer)
    for root, _, files in os.walk(IMAGE_FOLDER):
        if count >= MAX_IMAGES: break
        if 'original_image.jpg' not in files: continue

        folder = os.path.basename(root)
        if folder not in folder_to_caption: continue

        path = os.path.join(root, 'original_image.jpg')
        
        try:
            # Load and Embed
            z_img = extractor.get_image_embedding(path) # (1, 512)
            
            # SpLiCE Pipeline
            z_centered = aligner.align_image(z_img)
            weights = solver.solve(z_centered) # (1, V)
            
            # Save data (move to CPU numpy)
            sparse_weights_list.append(weights.detach().cpu().numpy())
            captions.append(folder_to_caption[folder])
            filenames.append(os.path.relpath(path, IMAGE_FOLDER).replace("\\", "/"))
            
            count += 1
            if count % 50 == 0: print(f"Processed {count}...")
                
        except Exception as e:
            print(f"Error on {path}: {e}")
            continue

    # 3. Stack and Save
    # Result shape: (N, Vocab_Size) e.g., (1500, 10000)
    all_weights = np.vstack(sparse_weights_list)
    
    print(f"Saving SpLiCE data to {OUTPUT_FILE}...")
    np.savez(
        OUTPUT_FILE,
        weights=all_weights,      # The SpLiCE embeddings
        vocab=np.array(vocab),    # Needed to decode the weights
        filenames=np.array(filenames),
        captions=np.array(captions)
    )
    print("Done!")

if __name__ == "__main__":
    main()