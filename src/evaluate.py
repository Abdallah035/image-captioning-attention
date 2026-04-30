"""
Evaluate the trained model with BLEU-1/2/3/4 on the validation set.

Run from the project root:
    python -m src.evaluate

What BLEU measures:
  How many n-grams of the generated caption appear in any of the
  5 reference captions. BLEU-1 = unigrams, BLEU-4 = 4-grams.
  Each is in [0, 1]. Higher is better.

Saves the result to checkpoints/bleu.json.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from .vocab import Vocabulary
from .model import EncoderCNN, DecoderWithAttention
from .inference import beam_search
from .dataset import default_transform, split_image_filenames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions-csv", default="data/captions.txt")
    p.add_argument("--image-dir", default="data/Images")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--vocab", default="checkpoints/vocab.pth")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="checkpoints/bleu.json")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate on at most this many images (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load the saved vocab and model
    vocab = Vocabulary.load(args.vocab)
    print(f"vocab size: {len(vocab)}")

    encoder = EncoderCNN().to(device).eval()
    decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)
    decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    decoder.eval()
    print(f"loaded {args.checkpoint}")

    # Get the val image list (must use same seed/ratio as training)
    _, val_imgs = split_image_filenames(
        args.captions_csv, val_ratio=args.val_ratio, seed=args.seed
    )

    # Group all 5 reference captions per image
    df = pd.read_csv(args.captions_csv)
    val_set = set(val_imgs)
    refs_by_image = defaultdict(list)
    for _, row in df.iterrows():
        if row["image"] in val_set:
            # Tokenize references the SAME way the model produces tokens
            ref_tokens = vocab.tokenize(row["caption"])
            refs_by_image[row["image"]].append(ref_tokens)

    image_list = sorted(refs_by_image.keys())
    if args.limit > 0:
        image_list = image_list[: args.limit]
    print(f"evaluating on {len(image_list)} images")

    transform = default_transform()
    references = []   # list of (list of reference token-lists)
    hypotheses = []   # list of generated token-lists

    for fname in tqdm(image_list, desc="captioning"):
        # Generate a caption with beam search
        img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        result = beam_search(encoder, decoder, img_t, vocab, beam=args.beam)

        # Drop special tokens for fair comparison
        words = [vocab.itos[t] for t in result["token_ids"]
                 if vocab.itos[t] not in ("<pad>", "<start>", "<end>")]
        hypotheses.append(words)
        references.append(refs_by_image[fname])

    # Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 with smoothing
    smooth = SmoothingFunction().method1
    weights_per_metric = [
        (1.0, 0, 0, 0),       # BLEU-1: only unigrams
        (0.5, 0.5, 0, 0),     # BLEU-2
        (1/3, 1/3, 1/3, 0),   # BLEU-3
        (0.25, 0.25, 0.25, 0.25),  # BLEU-4
    ]
    bleu_scores = [
        corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth)
        for w in weights_per_metric
    ]

    out = {
        "bleu_1": round(bleu_scores[0], 4),
        "bleu_2": round(bleu_scores[1], 4),
        "bleu_3": round(bleu_scores[2], 4),
        "bleu_4": round(bleu_scores[3], 4),
        "n_images": len(image_list),
        "checkpoint": args.checkpoint,
        "beam": args.beam,
    }
    print()
    print(json.dumps(out, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
