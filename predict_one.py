import argparse
import torch
import open_clip
from PIL import Image
from model import SPLICE

def build_dictionary(model, tokenizer, vocab_path, vocab_size, device):
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    if vocab_size > 0:
        lines = lines[-vocab_size:]
    vocab = lines

    batch_size = 500
    embs = []
    with torch.no_grad():
        for i in range(0, len(vocab), batch_size):
            tokens = tokenizer(vocab[i:i+batch_size]).to(device)
            embs.append(model.encode_text(tokens))
    dictionary = torch.cat(embs, dim=0)

    dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    dictionary = dictionary - dictionary.mean(dim=0)
    dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    return vocab, dictionary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", required=True, type=str)
    parser.add_argument("-mean_path", default="means/open_clip_ViT-B-32_image.pt", type=str)
    parser.add_argument("-vocab_path", default="vocab/laion.txt", type=str)
    parser.add_argument("-vocab_size", default=10000, type=int)
    parser.add_argument("-l1_penalty", default=0.25, type=float)
    parser.add_argument("-device", default="cuda", type=str)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    print(f"1. Loading Backbone on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print("2. Building dictionary...")
    vocab, dictionary = build_dictionary(model, tokenizer, args.vocab_path, args.vocab_size, device)

    print("3. Loading mean...")
    image_mean = torch.load(args.mean_path, map_location=device)
    print("mean norm:", image_mean.norm().item())

    print("4. Init SpLiCE...")
    splice = SPLICE(image_mean=image_mean, dictionary=dictionary, clip_model=model, device=device)
    # IMPORTANT: override solver penalty to match CLI
    splice.admm.l1_penalty = args.l1_penalty

    print(f"5. Processing image: {args.path}")
    image = Image.open(args.path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = torch.nn.functional.normalize(model.encode_image(image_tensor), dim=1)
        weights = splice.encode_image(image_tensor, return_weights=True)
        recon = splice.recompose_image(weights)
        cosine = (recon @ img_emb.T).diag().item()

    weights = weights[0].cpu()
    top_vals, top_inds = torch.topk(weights, 10)

    print("\n--- RESULTS ---")
    print(f"{'CONCEPT':<25} | WEIGHT")
    print("-" * 42)
    for i, idx in enumerate(top_inds):
        if top_vals[i] <= 0:
            break
        print(f"{vocab[idx]:<25} | {top_vals[i].item():.4f}")

    print("\nDecomposition L0 Norm:", torch.linalg.vector_norm(weights, ord=0).item())
    print("CLIP, SpLiCE Cosine Sim:", round(cosine, 4))

if __name__ == "__main__":
    main()
