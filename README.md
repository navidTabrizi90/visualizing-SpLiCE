# Visualizing SpLiCE
**Sparse Concept-Based Decomposition and Visualization of CLIP Embeddings**

This repository provides an end-to-end implementation of **SpLiCE (Sparse Linear Concept Explanations)** applied to CLIP image embeddings. It allows users to decompose high-dimensional black-box embeddings into human-readable semantic concepts.

---

## 1. Method Overview

SpLiCE represents each image embedding as a **sparse, non-negative linear combination of semantic concepts** from a predefined dictionary (e.g., words from LAION).

### The Pipeline
1.  **Encode:** Extract image embedding using a CLIP image encoder.
2.  **Align:** Normalize and **mean-center** the embedding using a precomputed dataset mean for modality alignment.
3.  **Decompose:** Solve a sparse coding optimization problem using **ADMM (Alternating Direction Method of Multipliers)**.
4.  **Interpret:** Analyze the resulting sparse concept weights and reconstructed embeddings.

---

## 2. Repository Structure

```text
.
├── vocab/
│   └── laion.txt                # Vocabulary used as concept dictionary
├── means/
│   └── open_clip_ViT-B-32_image.pt # Precomputed CLIP image mean
├── admm.py                      # ADMM solver for sparse optimization
├── model.py                     # SpLiCE model logic (decompose / recompose)
├── main.py                      # Demo script for a single image
├── predict_one.py               # CLI tool for single-image analysis
├── extract_dataset_embeddings.py # Batch processing for large datasets
├── visualize_splice.py          # Interactive Dash visualization app
├── image.py                     # Image loading and utility functions
├── requirements.txt
└── README.md

## 3. Installation

### Requirements
- **Python:** ≥ 3.9  
- **Hardware:** CUDA-enabled GPU is highly recommended for ADMM optimization.

```bash
# Clone the repository
git clone [https://github.com/your-username/visualizing-splice.git](https://github.com/your-username/visualizing-splice.git)
cd visualizing-splice

# Install dependencies
pip install -r requirements.txt
```
