import os
import nltk
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

# We scan 500,000 captions to get a rich, paper-quality vocabulary.
LIMIT = 500_000 
SAVE_PATH = "data/vocab.txt"

def main():
    print("--- 1. Setup NLTK ---")
    # We download both to ensure tokenization works perfectly
    nltk.download('punkt')
    nltk.download('punkt_tab')

    print("--- 2. Connecting to Conceptual Captions ---")
    try:
        dataset = load_dataset("conceptual_captions", split="train", streaming=True)
    except Exception as e:
        print(f"Error connecting to dataset: {e}")
        return

    print(f"--- 3. Scanning {LIMIT} captions for vocabulary ---")
    word_counts = Counter()
    count = 0

    # Iterate through the stream
    for sample in tqdm(dataset, total=LIMIT):
        caption = sample.get("caption")
        
        if caption and isinstance(caption, str):
            try:
                # Tokenize and Clean
                text = caption.lower()
                words = nltk.word_tokenize(text)
                # Keep only alphabet words > 2 chars 
                clean_words = [w for w in words if len(w) > 2 and w.isalpha()]
                word_counts.update(clean_words)
            except Exception:
                pass # Skip bad lines

        count += 1
        if count >= LIMIT:
            break

    print("--- 4. Saving Top Concepts ---")
    # Take the top 15,000 most frequent words (Standard SpLiCE size)
    most_common = [word for word, _ in word_counts.most_common(15000)]
    most_common.sort()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(most_common))

    print(f"✅ Success! Generated vocabulary from {LIMIT} real web captions.")
    print(f"Saved {len(most_common)} words to: {SAVE_PATH}")

if __name__ == "__main__":
    main()