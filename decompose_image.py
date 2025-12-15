import torch
import argparse
import os
import numpy as np
from PIL import Image

# Import your SpLiCE package
from splice.feature_extractor import CLIPFeatureExtractor
from splice.dictionary import ConceptDictionary
from splice.aligner import ModalityAligner
from splice.solver import ADMMSpLICESolver

# --- CONFIGURATION ---
VOCAB_PATH = 'data/vocab.txt'
MEAN_PATH = 'data/mscoco_mean.pt' # This now contains your Sketchy mean
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def decompose_single_image(image_path):
    print(f"--- SpLiCE Decomposition ---")
    print(f"Target: {image_path}")
    print(f"Device: {DEVICE}")

    # 1. Initialize Components
    print("Loading CLIP and SpLiCE modules...")
    extractor = CLIPFeatureExtractor(device=DEVICE)
    aligner = ModalityAligner(device=DEVICE)
    
    # 2. Load Vocabulary
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found. Run fetch_real_vocab_cc3m.py first.")
        return
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    
    # Build Dictionary Matrix
    dict_builder = ConceptDictionary(extractor)
    # Note: For single image inference, we can skip complex pruning for speed
    C_matrix = dict_builder.build_vocabulary(vocab, do_pruning=False)

    # 3. Load & Set Image Mean (Calibration)
    if os.path.exists(MEAN_PATH):
        aligner.set_image_mean(torch.load(MEAN_PATH, map_location=DEVICE))
    else:
        print("Warning: Mean file not found. Results might be inaccurate.")
        aligner.mu_img = torch.zeros(1, 512).to(DEVICE)

    # 4. Align Dictionary
    # (Center the vocabulary concepts relative to the image mean)
    C_centered = aligner.align_dictionary(C_matrix)
    
    # 5. Initialize Solver
    solver = ADMMSpLICESolver(C_centered, alpha=0.1, device=DEVICE)

    # --- INFERENCE ---
    try:
        # A. Embed Image
        z_img = extractor.get_image_embedding(image_path)
        
        # B. Align Image
        z_centered = aligner.align_image(z_img)
        
        # C. Solve for Sparse Weights
        weights = solver.solve(z_centered)
        
        # D. Interpret Results
        weights_np = weights.detach().cpu().numpy().flatten()
        
        # Sort by weight value (highest first)
        top_indices = np.argsort(weights_np)[::-1]
        
        print("\n--- RESULTS ---")
        print(f"Found {np.count_nonzero(weights_np > 0.001)} active concepts.\n")
        
        print(f"{'WEIGHT':<10} | {'CONCEPT'}")
        print("-" * 30)
        
        for idx in top_indices:
            val = weights_np[idx]
            if val < 0.01: break # Stop showing weak concepts
            print(f"{val:<10.4f} | {vocab[idx]}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose a single image into semantic concepts.")
    parser.add_argument("image", type=str, help="Path to the input image file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File '{args.image}' does not exist.")
    else:
        decompose_single_image(args.image)