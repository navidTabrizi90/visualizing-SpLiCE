import re
import pandas as pd
from collections import Counter
import torch
from tqdm import tqdm

class ConceptBuilder:
    def __init__(self, encoder, min_freq=50, max_concepts=15000):
        """
        encoder: CLIPEncoder instance from clip_encoder.py
        """
        self.encoder = encoder
        self.min_freq = min_freq
        self.max_concepts = max_concepts

    # ------------------------------
    # 1. Clean captions
    # ------------------------------
    def clean_caption(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------
    # 2. Extract unigrams + bigrams
    # ------------------------------
    def extract_terms(self, captions):
        unigram_counter = Counter()
        bigram_counter = Counter()

        for cap in captions:
            words = cap.split()
            unigram_counter.update(words)

            bigrams = [" ".join([words[i], words[i+1]]) 
                       for i in range(len(words)-1)]
            bigram_counter.update(bigrams)

        unigrams = [w for w, c in unigram_counter.items() if c >= self.min_freq]
        bigrams = [w for w, c in bigram_counter.items() if c >= self.min_freq]

        return unigrams, bigrams

    # ------------------------------
    # 3. Filter terms (NSFW, duplicates)
    # ------------------------------
    def filter_terms(self, terms):
        banned = ["nsfw", "nude", "porn", "sex", "erotic"]
        new_terms = []

        for t in terms:
            if not any(b in t for b in banned):
                new_terms.append(t)

        # Remove duplicates, keep first
        return list(dict.fromkeys(new_terms))

    # ------------------------------
    # 4. Encode concepts with CLIP
    # ------------------------------
    def encode_concept_matrix(self, concept_list, batch_size=256):
        C = []
        for i in tqdm(range(0, len(concept_list), batch_size)):
            batch = concept_list[i : i + batch_size]
            emb = self.encoder.encode_text(batch).cpu()  # normalized
            C.append(emb)

        C = torch.cat(C, dim=0)
        return C  # shape [num_concepts, dim]

    # ------------------------------
    # MAIN FUNCTION: Build Vocabulary
    # ------------------------------
    def build_from_laion(self, laion_tsv):
        df = pd.read_csv(laion_tsv, sep="\t")
        captions = df["caption"].astype(str).apply(self.clean_caption)

        unigrams, bigrams = self.extract_terms(captions)

        # Filter
        unigrams = self.filter_terms(unigrams)
        bigrams = self.filter_terms(bigrams)

        # Combine
        concepts = unigrams + bigrams
        concepts = concepts[: self.max_concepts]

        print(f"Final vocabulary size: {len(concepts)}")

        # Encode with CLIP
        C = self.encode_concept_matrix(concepts)

        return concepts, C
