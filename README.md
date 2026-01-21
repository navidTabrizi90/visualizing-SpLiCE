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
```

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

## 4. Data Preparation

### 4.1 Vocabulary (Concept Dictionary)

The vocabulary file (`vocab/laion.txt`) must contain **one concept per line**.  
Only the **last `VOCAB_SIZE` entries** are used (default: 10,000), following the SpLiCE setup with a large semantic dictionary.

The quality and coverage of this vocabulary directly affect interpretability and sparsity.

---

### 4.2 Image Mean (Required)

SpLiCE requires a **precomputed CLIP image mean** for modality alignment:

```bash
means/open_clip_ViT-B-32_image.pt
```

Important constraints:
- Must be computed using the **same CLIP backbone**
- Must use the **same preprocessing**
- Must reflect the **same data distribution**

This repository **reuses** the mean and never recomputes it.

---

## 5. Running the Demo (Single Image)

```bash
python main.py
```

What this does:
- Loads CLIP ViT-B/32
- Builds concept dictionary from vocab
- Loads image mean
- Decomposes one sketch image
- Prints top active concepts and weights

## 6. Single Image via CLI

```bash
python predict_one.py \
  -path path/to/image.jpg \
  -mean_path means/open_clip_ViT-B-32_image.pt \
  -vocab_path vocab/laion.txt \
  -vocab_size 10000 \
  -l1_penalty 0.25
```
Outputs:

- Top contributing concepts
- L0 norm (sparsity level)
- Cosine similarity between original CLIP embedding and SpLiCE reconstruction

## 7. Dataset-Level Embedding Extraction

```bash
python extract_dataset_embeddings.py
```
This script:
- Processes images in batches
- Applies SpLiCE sparse decomposition

Saves results to:
```bash
splice_embeddings_100.npz
The file contains:
- `sparse`: sparse concept weights (N × num_concepts)
- `dense`: reconstructed CLIP embeddings (N × 512)
- `image_paths`: image file paths
```
You can control:
- `MAX_IMAGES`
- `BATCH_SIZE`
- `VOCAB_SIZE`

---

## 8. Interactive Visualization

```bash
python visualize_splice.py
```
Starts a Dash app at:
```cpp
http://127.0.0.1:8050/
```

### Features
- PCA / t-SNE / UMAP projections  
- K-Means clustering  
- Hover tooltips with:
  - Image preview
  - Top SpLiCE concepts
  - Cluster membership
- Dynamic cluster highlighting

This enables qualitative inspection of the semantic structure induced by SpLiCE.

---

## 9. Optimization Details (ADMM)

The sparse coding problem solved is:

\[\min_{\mathbf{z}} \; \|\mathbf{C}\mathbf{z} - \mathbf{v}\|_2^2 + \lambda \|\mathbf{z}\|_1\quad \text{s.t.} \quad \mathbf{z} \ge 0\]

Where:
- `C` is the concept dictionary  
- `v` is the centered image embedding  

Key ADMM parameters:
- `rho`: penalty parameter  
- `l1_penalty`: sparsity strength  
- `tol`: convergence tolerance  
- `max_iter`: maximum iterations  

The Cholesky factorization is **precomputed once** for efficiency.

---

## 10. Design Choices and Trade-offs

- Large vocabularies (10k) → richer semantics, slower optimization  
- Non-negativity constraint → improved interpretability  
- Mean-centering → modality alignment (SpLiCE paper, Section 4.1)  
- No finetuning → pure inference, fast experimentation  

---

## 11. Limitations

- ADMM becomes slow for extremely large vocabularies  
- Concept quality depends heavily on vocabulary quality  
- Incorrect image mean degrades reconstruction quality  
- CLIP biases propagate into concept explanations  

---

## 12. Intended Use Cases

- Interpretable vision–language research  
- Dataset exploration and semantic clustering  
- Concept-based analysis of CLIP embeddings  
- Educational demonstrations of sparse explanations  

---

## 13. Attribution

This implementation is inspired by:

**SpLiCE: Sparse Linear Concept Explanations for Image Classifiers**

CLIP backbone provided by **OpenCLIP**.

This repository focuses on practical sparse decomposition, batching, and interactive visualization, extending the original SpLiCE method to dataset-level analysis.


