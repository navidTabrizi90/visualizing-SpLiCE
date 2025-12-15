import torch
import open_clip
from PIL import Image
from typing import List, Union
from tqdm import tqdm

class CLIPFeatureExtractor:
    """
    Wrapper for OpenCLIP to extract dense image and text embeddings.
    Uses OpenCLIP ViT-B/32 model
    Normalizes embeddings to the unit sphere (sigma operation)
    """
    
    def __init__(self, 
                 model_name: str = 'ViT-B-32', 
                 pretrained: str = 'laion2b_s34b_b79k', 
                 device: str = None):
        """
        Initializes the CLIP model.
        Args:
            device: 'cuda' or 'cpu'. If None, auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} ({pretrained}) on {self.device}...")
        
        # Load model and preprocessing transform
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained, 
            device=self.device
        )
        self.model.eval()

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """
        Extracts and normalizes dense image embedding (z^img).
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input
            
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            #SpLiCE requires unit-norm embeddings 
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features

    def get_text_embedding(self, text_list: List[str], batch_size: int = 256) -> torch.Tensor:
        """
        Extracts and normalizes dense text embeddings (z^txt) for dictionary C.
        """
        all_features = []
        
        # Batch processing to handle large vocabularies (10k+ words)
        for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding Concepts"):
            batch = text_list[i : i + batch_size]
            tokens = open_clip.tokenize(batch).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)   
                
        return torch.cat(all_features, dim=0)