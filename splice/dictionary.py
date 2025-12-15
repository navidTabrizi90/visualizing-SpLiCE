import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

class ConceptDictionary:
    """
    Manages the concept vocabulary (C).
    Constructs an overcomplete dictionary
    Pruning logic removes redundant concepts (cosine sim > 0.9)
    """
    
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.concept_texts = []
        self.concept_matrix = None 

    def prune_compositionality(self, texts: List[str], embeddings: torch.Tensor, threshold: float = 0.9) -> Tuple[List[str], torch.Tensor]:
        """
        Pruning Rule 2: Remove bigrams highly similar to the average of their words.e.g., if 'red car' ~ mean('red', 'car'), drop 'red car'
        """
        keep_indices = []
        print("Pruning bigrams based on compositionality...")
        
        for i, text in enumerate(tqdm(texts, desc="Checking Compositionality")):
            words = text.split()
            # Keep single words
            if len(words) == 1:
                keep_indices.append(i)
                continue
                
            if len(words) >= 2:
                z_phrase = embeddings[i]
                z_parts = self.extractor.get_text_embedding(words, batch_size=len(words))
                z_avg = torch.mean(z_parts, dim=0)
                z_avg = z_avg / z_avg.norm(dim=-1, keepdim=True)
                
                sim = torch.dot(z_phrase, z_avg).item()
                if sim <= threshold:
                    keep_indices.append(i)
        
        keep_indices_tensor = torch.tensor(keep_indices, device=embeddings.device)
        return [texts[i] for i in keep_indices], embeddings[keep_indices_tensor]

    def prune_pairwise_similarity(self, texts: List[str], embeddings: torch.Tensor, threshold: float = 0.9) -> Tuple[List[str], torch.Tensor]:
        """
        Pruning Rule 1: Ensure no two concepts have cosine sim > 0.9
        """
        print(f"Pruning pairwise redundancy (Input: {len(texts)})...")
        
        with tqdm(total=1, desc="Calculating Similarity Matrix") as pbar:
            sim_matrix = embeddings @ embeddings.T
            triu_sim = torch.triu(sim_matrix, diagonal=1)
            high_sim_indices = torch.where(triu_sim > threshold)
            indices_to_remove = set(high_sim_indices[1].cpu().numpy().tolist())
            pbar.update(1)
        
        keep_indices = [i for i in range(len(texts)) if i not in indices_to_remove]
        keep_tensor = torch.tensor(keep_indices, device=embeddings.device)
        
        return [texts[i] for i in keep_indices], embeddings[keep_tensor]

    def build_vocabulary(self, raw_concept_list: List[str], do_pruning: bool = True) -> torch.Tensor:
        """
        Main pipeline to convert strings into matrix C.
        """
        print(f"Processing {len(raw_concept_list)} candidate concepts...")
        embeddings = self.extractor.get_text_embedding(raw_concept_list)
        
        curr_texts = raw_concept_list
        curr_embeddings = embeddings
        
        if do_pruning:
            curr_texts, curr_embeddings = self.prune_compositionality(curr_texts, curr_embeddings)
            print(f"Concepts remaining after composition check: {len(curr_texts)}")
            curr_texts, curr_embeddings = self.prune_pairwise_similarity(curr_texts, curr_embeddings)
            print(f"Concepts remaining after pairwise check: {len(curr_texts)}")
            
        self.concept_texts = curr_texts
        self.concept_matrix = curr_embeddings
        return self.concept_matrix