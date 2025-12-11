import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Union

class SpLiCEInterpreter:
    """
    Handles the translation of sparse weights into human-readable concepts
    and the mathematical reconstruction of dense embeddings.
    
    Paper Reference:
    - Reconstruction: z_hat = sigma(C w + mu_img) [cite: 323, 334]
    - Interpretation: Maps non-zero weights to vocabulary strings[cite: 325].
    - Metric: Uses Cosine Similarity to measure fidelity[cite: 327].
    """
    
    def __init__(self, aligner, vocabulary_list: List[str]):
        """
        Args:
            aligner: Instance of ModalityAligner (must have mu_img set).
            vocabulary_list: List of strings corresponding to the dictionary rows.
        """
        self.aligner = aligner
        self.vocab = vocabulary_list

    def reconstruct_embedding(self, 
                              centered_dictionary: torch.Tensor, 
                              sparse_weights: torch.Tensor) -> torch.Tensor:
        """
        Mathematically reconstructs the dense CLIP embedding from sparse weights.
        
        Formula: z_hat = Normalize(C_centered * w + mu_img)
        
        Args:
            centered_dictionary: (V, 512) tensor (C_centered).
            sparse_weights: (B, V) or (V,) tensor of weights.
        
        Returns:
            z_hat: (B, 512) tensor in the original CLIP space.
        """
        # Ensure weights are (V, B) for matrix multiplication
        # If input is 1D (V,), turn it into (V, 1)
        if sparse_weights.dim() == 1:
            w = sparse_weights.unsqueeze(1) 
        else:
            w = sparse_weights.T # Expecting (B, V) -> Transpose to (V, B)
            
        C = centered_dictionary.T # (512, V)
        
        # 1. Compute weighted sum in the CENTERED space
        # z_rec_centered = C * w
        z_rec_centered = torch.mm(C, w).T # Result is (B, 512)
        
        # 2. Un-align (Add image mean and re-normalize)
        # This calls the helper from Step 3 to reverse the modality shift
        z_hat = self.aligner.unalign_reconstruction(z_rec_centered)
        
        return z_hat

    def get_explanation(self, 
                        sparse_weights: Union[torch.Tensor, np.ndarray], 
                        top_k: int = 10) -> pd.DataFrame:
        """
        Converts a single weight vector into a readable DataFrame.
        
        Args:
            sparse_weights: (V,) tensor or array for a single image.
            top_k: Number of top concepts to return.
            
        Returns:
            DataFrame with columns ['concept', 'weight'].
        """
        # Ensure input is a flat CPU numpy array
        if isinstance(sparse_weights, torch.Tensor):
            w = sparse_weights.flatten().detach().cpu().numpy()
        else:
            w = sparse_weights.flatten()
            
        # Find non-zero indices
        active_indices = w.nonzero()[0]
        
        results = []
        for idx in active_indices:
            weight_val = w[idx]
            # Optional: Filter tiny numerical noise (e.g. < 1e-5)
            if weight_val > 1e-5:
                results.append({
                    "concept": self.vocab[idx],
                    "weight": weight_val
                })
            
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Sort by weight descending
            df = df.sort_values(by="weight", ascending=False).head(top_k)
            # Reset index for clean display
            df = df.reset_index(drop=True)
        else:
            # Return empty structure if no concepts found (e.g., lambda too high)
            df = pd.DataFrame(columns=["concept", "weight"])
        
        return df

    def compute_fidelity(self, 
                         original_z: torch.Tensor, 
                         reconstructed_z: torch.Tensor) -> float:
        """
        Computes Cosine Similarity between the original and reconstructed embeddings.
        High fidelity (e.g., >0.75) indicates the decomposition is accurate[cite: 175].
        """
        # Ensure shapes match for calculation
        if original_z.shape != reconstructed_z.shape:
            # Attempt to broadcast if one dimension is 1
            pass
            
        sim = F.cosine_similarity(original_z, reconstructed_z, dim=-1)
        return sim.mean().item()