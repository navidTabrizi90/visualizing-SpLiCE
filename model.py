import torch
import torch.nn as nn
from admm import ADMM


class SPLICE(nn.Module):
    def __init__(self, image_mean, dictionary, clip_model, device="cuda"):
        super().__init__()
        self.device = device
        self.clip = clip_model.to(device)
        self.image_mean = image_mean.to(device)
        self.dictionary = dictionary.to(device)

        self.admm = ADMM(
            rho=5,
            l1_penalty=0.25,
            tol=1e-6,
            max_iter=2000,
            device=device
        )

        # Precompute once (important for speed)
        self.admm.precompute(self.dictionary)

    # --------------------------------------------------
    # Core decomposition
    # --------------------------------------------------
    def decompose(self, embedding):
        """
        embedding: (batch x CLIP_dim), centered + normalized
        returns:   (batch x num_concepts) sparse weights
        """
        return self.admm.fit(self.dictionary, embedding)

    # --------------------------------------------------
    # Reconstruction
    # --------------------------------------------------
    def recompose_image(self, weights):
        """
        weights: (batch x num_concepts)
        returns: (batch x CLIP_dim) reconstructed embedding
        """
        recon = weights @ self.dictionary
        recon = torch.nn.functional.normalize(recon, dim=1)
        recon = torch.nn.functional.normalize(recon + self.image_mean, dim=1)
        return recon

    # --------------------------------------------------
    # Encode image with optional cosine similarity
    # --------------------------------------------------
    def encode_image(self, image, return_weights=False, return_cosine=False):
        """
        image: input images
        return_weights: if True, return sparse weights
        return_cosine:  if True, also return cosine similarity
        """

        # 1) CLIP encoding
        with torch.no_grad():
            img = self.clip.encode_image(image)

        img = torch.nn.functional.normalize(img, dim=1)

        # 2) Center + normalize (modality alignment)
        centered = torch.nn.functional.normalize(img - self.image_mean, dim=1)

        # 3) Decompose
        weights = self.decompose(centered)

        if return_weights and not return_cosine:
            return weights

        # 4) Recompose
        recon = self.recompose_image(weights)

        if return_cosine:
            # cosine similarity per image
            cosine = torch.sum(recon * img, dim=1)  # shape: (batch,)
            if return_weights:
                return weights, cosine
            return recon, cosine

        return recon
