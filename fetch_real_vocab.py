import requests
import os

def download_file(url):
    """Helper to download text from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.splitlines()
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return []

def build_real_vocab(save_path="data/vocab.txt"):
    print("--- Building 'Real' Visual Vocabulary ---")
    
    # 1. Google's 10,000 Most Common English Words (No Swears)
    # This serves as the frequency baseline (similar to LAION top 10k)
    print("Downloading Google-10k common words...")
    url_google = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
    common_words = download_file(url_google)
    
    # 2. ImageNet 1000 Class Labels
    # These are highly specific visual objects (e.g., 'tabby', 'Egyptian cat')
    # This fixes the "missing specific object" problem.
    print("Downloading ImageNet class labels...")
    url_imagenet = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    # This source is JSON, so we treat it differently or just fetch a raw text list
    # Let's use a simpler raw text source for ImageNet classes
    url_imagenet_txt = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    imagenet_classes = download_file(url_imagenet_txt)
    
    # 3. Clean and Merge
    print("Merging and cleaning lists...")
    final_set = set()
    
    # Add Common Words (filter out tiny words)
    for w in common_words:
        w = w.strip().lower()
        if len(w) > 2 and w.isalpha():
            final_set.add(w)
            
    # Add ImageNet Classes (handle multi-word classes like 'great white shark')
    for label in imagenet_classes:
        # Split "great white shark" into ["great", "white", "shark"] and the full phrase
        clean_label = label.strip().lower()
        final_set.add(clean_label) # Add the full phrase (SpLiCE supports bigrams/trigrams!)
        
    # 4. Save
    sorted_vocab = sorted(list(final_set))
    
    # Limit to ~15k to match paper's scale (Paper uses 15k total [cite: 347])
    # But usually, just keeping them all is fine for modern GPUs.
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_vocab))
        
    print(f"✅ Success! Saved {len(sorted_vocab)} concepts to {save_path}")
    print("You can now run 'decompose_image.py' with this new vocabulary.")

if __name__ == "__main__":
    build_real_vocab()