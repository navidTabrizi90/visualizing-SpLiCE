import torch
import os
import sys
import numpy as np
from PIL import Image

from splice.feature_extractor import CLIPFeatureExtractor
from splice.dictionary import ConceptDictionary
from splice.aligner import ModalityAligner
from splice.solver import ADMMSpLICESolver, CpuSpliceSolver
from splice.interpreter import SpLiCEInterpreter

def main():
    # --- CONFIG ---
    # Paper uses l0 norms of 5-20. Increase Alpha to increase sparsity.
    ALPHA = 0.1 
    VOCAB_PATH = "data/vocab.txt"
    MEAN_PATH = "data/mscoco_mean.pt"
    
    # Get image from args or default
    IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- SpLiCE Decomposition ---")
    print(f"Target: {IMAGE_PATH} | Device: {DEVICE}")

    # 1. Load Model
    extractor = CLIPFeatureExtractor(device=DEVICE)

    # 2. Load Vocabulary
    if not os.path.exists(VOCAB_PATH):
        print("Error: Run 'fetch_real_vocab.py' first.")
        return
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    
    # Use only first 10k to match paper scale & save memory
    vocab = vocab[:10000]
    
    dict_builder = ConceptDictionary(extractor)
    # do_pruning=False because our fetched vocab is already clean
    C_matrix = dict_builder.build_vocabulary(vocab, do_pruning=False)

    # 3. Align Modalities
    aligner = ModalityAligner(device=DEVICE)
    if os.path.exists(MEAN_PATH):
        print(f"Loading Image Mean from {MEAN_PATH}...")
        aligner.set_image_mean(torch.load(MEAN_PATH, map_location=DEVICE))
    else:
        print("Warning: Using Zero-Mean (Approximate). Run 'fetch_calibration.py' + 'compute_mean.py' for better fidelity.")
        aligner.mu_img = torch.zeros(1, 512).to(DEVICE)
    
    C_centered = aligner.align_dictionary(C_matrix)

    # 4. Solve
    if not os.path.exists(IMAGE_PATH):
        print(f"Image {IMAGE_PATH} not found.")
        return

    z_img = extractor.get_image_embedding(IMAGE_PATH)
    z_centered = aligner.align_image(z_img)

    print("Solving sparse decomposition...")
    if DEVICE == "cuda":
        solver = ADMMSpLICESolver(C_centered, alpha=ALPHA, device=DEVICE)
    else:
        solver = CpuSpliceSolver(alpha=ALPHA)
        
    weights = solver.solve(z_centered) if isinstance(solver, ADMMSpLICESolver) else solver.solve(C_centered, z_centered)

    # 5. Interpret (with Blocklist Cleaning)
    interpreter = SpLiCEInterpreter(aligner, dict_builder.concept_texts)
    
    # Block web artifacts common in LAION
    BLOCKLIST = ["download", "image", "photo", "picture", "wallpaper", "jpg", "jpeg", "stock"]
    
    w_cpu = weights.flatten().cpu().numpy()
    for i, text in enumerate(dict_builder.concept_texts):
        if text.lower() in BLOCKLIST:
            w_cpu[i] = 0.0
            
    df = interpreter.get_explanation(torch.from_numpy(w_cpu), top_k=15)
    print("\n--- Concepts ---")
    print(df)
    
    z_rec = interpreter.reconstruct_embedding(C_centered, weights)
    fid = interpreter.compute_fidelity(z_img, z_rec)
    print(f"\nFidelity: {fid:.4f}")

if __name__ == "__main__":
    main()