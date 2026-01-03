import torch
import open_clip
from PIL import Image
import os
import glob
from model import SPLICE
import numpy as np


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASET_ROOT = "Sketchy_original_only/"
MEAN_PATH = "means/open_clip_ViT-B-32_image.pt"
VOCAB_PATH = "vocab/laion.txt"
VOCAB_SIZE = 10000
DEVICE = "cuda"
MAX_IMAGES = 100      # <-- limit here
BATCH_SIZE = 16       # safe default


# --------------------------------------------------
# LOAD CLIP
# --------------------------------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# --------------------------------------------------
# BUILD DICTIONARY (EXACT SpLiCE LOGIC)
# --------------------------------------------------
print("Building SpLiCE dictionary...")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [l.strip() for l in f.readlines()]

if VOCAB_SIZE > 0:
    vocab = vocab[-VOCAB_SIZE:]

concept_embeddings = []
batch_size = 512

with torch.no_grad():
    for i in range(0, len(vocab), batch_size):
        tokens = tokenizer(vocab[i:i + batch_size]).to(DEVICE)
        emb = model.encode_text(tokens)
        concept_embeddings.append(emb)

dictionary = torch.cat(concept_embeddings, dim=0)

# SpLiCE alignment (Section 4.1 of paper)
dictionary = torch.nn.functional.normalize(dictionary, dim=1)
dictionary = torch.nn.functional.normalize(
    dictionary - dictionary.mean(dim=0),
    dim=1
)

print(f"Dictionary shape: {dictionary.shape}")

# --------------------------------------------------
# LOAD IMAGE MEAN (DO NOT RECOMPUTE)
# --------------------------------------------------
image_mean = torch.load(MEAN_PATH, map_location=DEVICE)

# --------------------------------------------------
# INIT SPLICE
# --------------------------------------------------
splice = SPLICE(
    image_mean=image_mean,
    dictionary=dictionary,
    clip_model=model,
    device=DEVICE
)
splice.eval()

# --------------------------------------------------
# LOOP OVER DATASET (RECURSIVE)
# --------------------------------------------------
splice_sparse = []
splice_dense = []

image_paths = glob.glob(
    os.path.join(DATASET_ROOT, "**", "original_image.jpg"),
    recursive=True
)

print(f"Found {len(image_paths)} images.")

from tqdm import tqdm
import math

# --------------------------------------------------
# LIMIT DATASET TO 100 IMAGES
# --------------------------------------------------
image_paths = image_paths[:MAX_IMAGES]
print(f"Processing {len(image_paths)} images.")

num_batches = math.ceil(len(image_paths) / BATCH_SIZE)

splice_sparse = []
splice_dense = []

with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Extracting SpLiCE embeddings"):
        batch_paths = image_paths[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))

        image_tensor = torch.stack(images).to(DEVICE)  # (B, 3, 224, 224)

        # 1️⃣ CLIP encode (batched)
        img_emb = model.encode_image(image_tensor)
        img_emb = torch.nn.functional.normalize(img_emb, dim=1)

        # 2️⃣ Mean-centering (reuse precomputed mean)
        centered = torch.nn.functional.normalize(img_emb - image_mean, dim=1)

        # 3️⃣ SpLiCE sparse decomposition (batched)
        weights = splice.decompose(centered)           # (B, num_concepts)

        # 4️⃣ SpLiCE dense reconstruction (batched)
        recon = splice.recompose_image(weights)        # (B, 512)

        splice_sparse.append(weights.cpu())
        splice_dense.append(recon.cpu())


# --------------------------------------------------
# SAVE EMBEDDINGS AS .npz
# --------------------------------------------------
splice_sparse_np = torch.cat(splice_sparse).numpy()
splice_dense_np = torch.cat(splice_dense).numpy()

np.savez(
    "splice_embeddings_100.npz",
    sparse=splice_sparse_np,
    dense=splice_dense_np,
    image_paths=np.array(image_paths[:len(splice_sparse_np)], dtype=object)
)

print("Saved SpLiCE embeddings to splice_embeddings_100.npz")
