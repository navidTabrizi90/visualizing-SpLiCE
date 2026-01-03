# SpLiCE — Applying Sparse Linear Concept Explanations to New Datasets

This repository provides a **clean, minimal, and explicit implementation of SpLiCE** (Sparse Linear Concept Explanations) for **analyzing CLIP embeddings on new datasets**.

The code applies **post-hoc sparse decomposition** to **frozen CLIP embeddings** in order to obtain **interpretable concept-based representations**, without any training or fine-tuning.

---

## 🔍 What This Repository Does

- Loads a **pretrained CLIP / OpenCLIP model**
- Builds a **concept dictionary** from a text vocabulary
- Reuses a **precomputed image mean** (no retraining)
- Applies **ADMM-based sparse optimization** at inference time
- Produces:
  - **Sparse SpLiCE embeddings** (concept weights)
  - **Dense SpLiCE embeddings** (reconstructed CLIP-space vectors)
- Supports applying SpLiCE to **new datasets** for analysis and visualization

---

## 🚫 What This Repository Does NOT Do

- ❌ No training
- ❌ No fine-tuning
- ❌ No gradient updates
- ❌ No parameter learning
- ❌ No modification of CLIP weights

SpLiCE here is used strictly as a **post-hoc interpretability method**.

---

## 📌 Method Overview (High Level)

Given an image:

1. Encode the image using a **frozen CLIP encoder**
2. Normalize and **mean-center** the embedding
3. Solve the sparse optimization problem:

\[
\min_{w \ge 0} \;\|Cw - z\|_2^2 + 2\lambda \|w\|_1
\]

where:
- \( z \) is the image embedding
- \( C \) is the concept dictionary
- \( w \) is the sparse concept weight vector

4. Use \( w \) as:
   - an **interpretable embedding**
   - or reconstruct a dense embedding in CLIP space

---

## 📂 Repository Structure

├── admm.py # ADMM solver for sparse non-negative optimization
├── model.py # SPLICE model (decompose, recompose, encode)
├── main.py # CLI-style experiment runner
├── predict_one.py # Single-image inference and explanation
├── extract_dataset_embeddings.py # Apply SpLiCE to a full dataset
├── vocab/
│ └── laion.txt # Concept vocabulary
├── means/
│ └── open_clip_ViT-B-32_image.pt # Precomputed image mean
└── README.md


---

## ⚙️ Requirements

- Python ≥ 3.8
- PyTorch
- open_clip
- Pillow
- NumPy

Install dependencies (example):

```bash
pip install torch torchvision open-clip-torch pillow numpy


## 🚀 Quick Start — Single Image

Run SpLiCE on a single image to obtain an interpretable concept-based explanation.

```bash
python predict_one.py \
  -path path/to/image.jpg \
  -mean_path means/open_clip_ViT-B-32_image.pt \
  -vocab_path vocab/laion.txt \
  -vocab_size 10000 \
  -l1_penalty 0.25 \
  -device cuda

