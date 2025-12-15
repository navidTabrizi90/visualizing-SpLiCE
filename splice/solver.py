import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Lasso

class CpuSpliceSolver:
    """
    Standard Lasso solver using Scikit-Learn.
    Recommended for single-image inference or small batches on CPU.
    
    Mathematical Formulation:
    min_w ||Cw - z||^2 + 2*alpha*||w||_1  s.t. w >= 0
    """
    
    def __init__(self, alpha: float = 0.03):
        """
        Args:
            alpha: Sparsity regularization strength (lambda). 
                   Paper suggests aiming for l0 norm of 5-20.
                   Typical range: 0.01 - 0.05.
        """
        self.alpha = alpha

    def solve(self, dictionary_matrix: torch.Tensor, target_image_vector: torch.Tensor) -> torch.Tensor:
        """
        Solves for w using coordinate descent (sklearn).
        
        Args:
            dictionary_matrix: (V, 512) tensor (C_centered).
            target_image_vector: (1, 512) tensor (z_centered).
            
        Returns:
            weights: (V,) tensor of sparse coefficients.
        """
        # Move to CPU / Numpy
        X = dictionary_matrix.detach().cpu().numpy().T  # Shape (512, V)
        y = target_image_vector.detach().cpu().numpy().flatten() # Shape (512,)
        
        # Configure Lasso
        # positive=True enforces w >= 0 (Non-negativity constraint)
        # fit_intercept=False because data is already centered
        model = Lasso(
            alpha=self.alpha, 
            positive=True, 
            fit_intercept=False, 
            max_iter=5000,
            tol=1e-4
        )
        
        model.fit(X, y)
        
        return torch.from_numpy(model.coef_).float()


class ADMMSpLICESolver:
    """
    GPU-accelerated solver using ADMM (Alternating Direction Method of Multipliers).
    Recommended for batched processing (e.g., decomposing entire datasets).
    
    - Optimizes: min ||Cw - z||^2 + 2*lambda*||w||_1
    - Uses Woodbury Matrix Identity for fast inversion of (V,V) matrix.
    """
    
    def __init__(self, dictionary_matrix: torch.Tensor, alpha: float = 0.03, rho: float = 5.0, device: str = 'cuda'):
        """
        Initializes solver and pre-computes the large matrix inversion.
        
        Args:
            dictionary_matrix: (V, 512) tensor (C).
            alpha: L1 regularization strength.
            rho: ADMM penalty parameter (rho=5.0).
            device: 'cuda' or 'cpu'.
        """
        self.device = device
        self.alpha = alpha
        self.rho = rho
        
        # notation: C is (512, V). Input is (V, 512), so we transpose.
        # self.C shape: (512, 15000)
        self.C = dictionary_matrix.T.to(device)
        self.D, self.V = self.C.shape
        
        
        print("Pre-computing ADMM matrices (Woodbury)...")
        with torch.no_grad():
            C_CT = torch.mm(self.C, self.C.T) # (512, 512)
            
            # Core term to invert: (I + (2/rho) * C * C^T)
            core_matrix = torch.eye(self.D, device=device) + (2.0 / rho) * C_CT
            
            self.core_inv = torch.linalg.inv(core_matrix)
            self.C_T = self.C.T # (15000, 512)

    def soft_threshold(self, x: torch.Tensor, kappa: float) -> torch.Tensor:
        """
        Proximal operator for L1 norm: S_k(a) = sign(a) * max(|a| - k, 0)
        """
        return torch.sign(x) * torch.maximum(torch.abs(x) - kappa, torch.tensor(0.0, device=self.device))

    def solve(self, target_images: torch.Tensor, max_iter: int = 1000, tol: int = 1e-4) -> torch.Tensor:
        """
        Batched solve.
        
        Args:
            target_images: (B, 512) tensor of centered images.
            
        Returns:
            weights: (B, V) tensor of sparse coefficients.
        """
        B = target_images.shape[0]
        targets = target_images.T.to(self.device) # (512, B)
        
        # Initialize variables
        w_k = torch.zeros(self.V, B, device=self.device)
        z_k = torch.zeros(self.V, B, device=self.device)
        u_k = torch.zeros(self.V, B, device=self.device)
        
        for k in range(max_iter):
            prev_z = z_k.clone()
            
            # --- 1. w-update (Quadratic Min via Woodbury) ---
            # RHS = rho(z - u) + 2*C^T*y
            rhs = self.rho * (z_k - u_k) + 2.0 * torch.mm(self.C_T, targets)
            
            # Apply inverse: w = (1/rho) * (rhs - 2/rho * C^T * core_inv * C * rhs)
            c_rhs = torch.mm(self.C, rhs)
            inv_c_rhs = torch.mm(self.core_inv, c_rhs)
            term_2 = torch.mm(self.C_T, inv_c_rhs)
            
            w_next = (rhs - (2.0 / self.rho) * term_2) / self.rho
            
            # --- 2. z-update (Proximal + Constraints) ---
            # Soft thresholding for Sparsity (L1)
            # ReLU for Non-negativity
            threshold = self.alpha / self.rho
            z_next = self.soft_threshold(w_next + u_k, threshold)
            z_next = F.relu(z_next)
            
            # --- 3. u-update (Dual) ---
            u_next = u_k + w_next - z_next
            
            # --- Convergence Check ---
            primal_res = torch.norm(w_next - z_next)
            dual_res = torch.norm(self.rho * (z_next - prev_z))
            
            w_k, z_k, u_k = w_next, z_next, u_next
            
            if primal_res < tol and dual_res < tol:
                break
                
        return z_k.T # Return (B, V)