import torch
import torch.nn.functional as F
from typing import Optional

class ModalityAligner:
    """
    Handles geometric alignment to fix the 'Modality Gap'.
    CLIP image and text embeddings exist on two different cones
    Solution: Center both using their respective means (mu_img, mu_con)
    Vectors must be re-normalized after centering
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu_img: Optional[torch.Tensor] = None
        self.mu_con: Optional[torch.Tensor] = None
        
    def set_image_mean(self, mean_tensor: torch.Tensor):
        self.mu_img = mean_tensor.to(self.device)

    def align_dictionary(self, concept_matrix: torch.Tensor) -> torch.Tensor:
        """
        Centers the dictionary C using the Concept Mean (mu_con).
        C_centered = sigma(g(x) - mu_con)
        """
        self.mu_con = torch.mean(concept_matrix, dim=0, keepdim=True).to(self.device)
        C_centered = concept_matrix.to(self.device) - self.mu_con
        # Re-normalize to unit sphere
        C_centered = F.normalize(C_centered, p=2, dim=-1)
        return C_centered

    def align_image(self, z_img: torch.Tensor) -> torch.Tensor:
        """
        Centers the image using the Image Mean (mu_img).
        z_centered = sigma(z^img - mu_img)
        """
        if self.mu_img is None:
            raise ValueError("Image mean (mu_img) must be set before aligning images.")
        
        z_centered = z_img.to(self.device) - self.mu_img
        # Re-normalize to unit sphere
        z_centered = F.normalize(z_centered, p=2, dim=-1)
        return z_centered

    def unalign_reconstruction(self, z_rec_centered: torch.Tensor) -> torch.Tensor:
        """
        Reverses alignment for reconstruction.
        z_hat = sigma(C w + mu_img)
        """
        if self.mu_img is None:
            raise ValueError("Image mean is missing.")
            
        z_original_space = z_rec_centered + self.mu_img
        return F.normalize(z_original_space, p=2, dim=-1)