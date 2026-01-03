# File: main.py
import torch
import open_clip
import os
import glob
from PIL import Image
from model import SPLICE

def run_splice_demo(sketchy_path, vocab_path, mean_path, device="cuda"):
    # ------------------------------------------------------------------
    # 1. SETUP & LOADING
    # ------------------------------------------------------------------
    print(f"Loading Backbone (ViT-B-32)...")
    # Must match the model used in calculate_mean.py
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # ------------------------------------------------------------------
    # 2. PREPARE VOCABULARY (THE "DICTIONARY")
    # ------------------------------------------------------------------
    print(f"Loading Vocabulary from {vocab_path}...")
    if not os.path.exists(vocab_path):
        print(f"Error: Vocab file not found at {vocab_path}")
        return

    # Load words from text file
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        concepts_text = [line.strip() for line in lines[-10000:]]


    print(f"Embedding {len(concepts_text)} concepts (this may take a moment)...")
    
    # We batch the tokens to avoid OutOfMemory errors
    batch_size = 1000
    concept_embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(concepts_text), batch_size):
            batch_text = concepts_text[i : i + batch_size]
            tokens = tokenizer(batch_text).to(device)
            emb = model.encode_text(tokens)
            concept_embeddings_list.append(emb)
            
    concept_embeddings = torch.cat(concept_embeddings_list, dim=0)

    # ALIGNMENT: Center the concepts (Paper Section 4.1)
    # 1. Normalize
    concept_embeddings = torch.nn.functional.normalize(concept_embeddings, dim=1)
    # 2. Calculate Concept Mean
    concept_mean = torch.mean(concept_embeddings, dim=0)
    # 3. Center and Re-normalize
    concept_embeddings = torch.nn.functional.normalize(concept_embeddings - concept_mean, dim=1)

    # ------------------------------------------------------------------
    # 3. LOAD YOUR CUSTOM SKETCH MEAN
    # ------------------------------------------------------------------
    print(f"Loading Image Mean from {mean_path}...")
    if not os.path.exists(mean_path):
        print("Error: Mean file not found. Please check the name.")
        return
    image_mean = torch.load(mean_path, map_location=device)

    # ------------------------------------------------------------------
    # 4. INITIALIZE SPLICE
    # ------------------------------------------------------------------
    print("Initializing SpLiCE Model...")
    # This creates the solver with your custom mean and dictionary
    splice_model = SPLICE(image_mean, concept_embeddings, model, device=device)

    # ------------------------------------------------------------------
    # 5. RUN ON A SAMPLE SKETCH
    # ------------------------------------------------------------------
    # Find a random image to test
    all_images = glob.glob(os.path.join(sketchy_path, "*", "original_image.jpg"))
    if not all_images:
        print("No images found! Check path.")
        return

    test_image_path = all_images[0] # Pick the first one found
    print(f"\nAnalyzing Sketch: {test_image_path}")
    
    # Load and Preprocess
    raw_image = Image.open(test_image_path).convert("RGB")
    image_input = preprocess(raw_image).unsqueeze(0).to(device)

    # Get Sparse Weights
    # This runs the ADMM optimization
    weights = splice_model.encode_image(image_input)

    # ------------------------------------------------------------------
    # 6. PRINT RESULTS
    # ------------------------------------------------------------------
    # Find indices with the highest weights
    top_vals, top_inds = torch.topk(weights.squeeze(), k=10)

    print("\n--- DECOMPOSITION RESULTS ---")
    print(f"{'WEIGHT':<10} | {'CONCEPT'}")
    print("-" * 30)
    for score, idx in zip(top_vals, top_inds):
        if score > 0: # Only print active concepts
            word = concepts_text[idx]
            print(f"{score.item():.4f}     | {word}")

if __name__ == "__main__":
    # === CONFIGURATION ===
    # Update these paths to match your folder structure exactly
    
    # 1. Path to the folder containing your sketch subfolders
    SKETCHY_IMAGES_PATH = r"F:\Thesis\SpLiCE\Sketchy\images"
    
    # 2. Path to the LAION text file you showed in the screenshot
    VOCAB_PATH = r"F:\Thesis\SpLiCE\vocab\laion_bigrams.txt"
    
    # 3. The mean file you just calculated
    MEAN_PATH = r"means/open_clip_ViT-B-32_image.pt"

    run_splice_demo(SKETCHY_IMAGES_PATH, VOCAB_PATH, MEAN_PATH)