"""
Train the captioning model on Flickr8k.

Run from the project root:
    python -m src.train --epochs 10 --batch-size 32

Saves three files in checkpoints/:
    vocab.pth   the vocabulary (so inference uses the same word -> id mapping)
    best.pth    decoder weights from the epoch with the lowest validation loss
    last.pth    decoder weights from the most recent epoch
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .vocab import Vocabulary
from .dataset import Flickr8kDataset, collate_fn, split_image_filenames
from .model import EncoderCNN, DecoderWithAttention


def parse_args():
    """Read command-line options. All have sensible defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--captions-csv", default="data/captions.txt")
    p.add_argument("--image-dir", default="data/Images")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--lambda-ds", type=float, default=1.0,
                   help="weight of the doubly-stochastic regularization term")
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip", type=float, default=5.0,
                   help="max gradient norm (prevents exploding gradients)")
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint to continue training from")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="epoch number to start counting from when resuming")
    return p.parse_args()


def run_one_epoch(encoder, decoder, loader, criterion, optimizer,
                  device, lambda_ds, clip, description):
    """Run one full pass over the data.

    If optimizer is None, we are validating (no weight updates).
    Otherwise we are training.

    Returns the average total loss across all batches.
    """
    is_training = optimizer is not None
    decoder.train(is_training)   # turns dropout on for training, off for val

    total_loss = 0.0
    n_batches = 0
    progress = tqdm(loader, desc=description, leave=False)

    for images, captions, lengths in progress:
        # Move tensors to GPU (or stay on CPU if no GPU)
        images = images.to(device)
        captions = captions.to(device)

        # Teacher forcing: input is captions[:, :-1], target is captions[:, 1:]
        # (for each input position, predict the next caption token)
        inp = captions[:, :-1]
        tgt = captions[:, 1:]

        # Forward pass.
        # Encoder runs in no_grad because it is frozen, this saves memory.
        # Decoder runs with gradients enabled when training, off when validating.
        with torch.set_grad_enabled(is_training):
            with torch.no_grad():
                features = encoder(images)
            logits, alphas = decoder(features, inp)

            # Cross-entropy loss across (batch * timesteps).
            # We reshape so each token prediction is a separate "sample".
            loss_xe = criterion(
                logits.reshape(-1, logits.size(-1)),  # (B*T, vocab_size)
                tgt.reshape(-1),                      # (B*T,)
            )

            # Doubly-stochastic regularization on attention weights.
            # alphas.sum(dim=1) sums attention across timesteps -> (B, 49).
            # We want each region's total attention to be ~1.
            loss_ds = ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            loss = loss_xe + lambda_ds * loss_ds

        # Backward + optimizer step (only when training)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients so they don't explode in LSTMs
            nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(xe=f"{loss_xe.item():.3f}",
                             ds=f"{loss_ds.item():.3f}")

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------- Step 1: train/val split (by image, not by caption row) ----------
    print("Splitting images...")
    train_imgs, val_imgs = split_image_filenames(
        args.captions_csv, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"  train: {len(train_imgs)} images,  val: {len(val_imgs)} images")

    # ---------- Step 2: build vocabulary from training captions only ----------
    # IMPORTANT: never include val captions when building the vocab,
    # otherwise the model has unfair access to val-only words.
    print("Building vocab from training captions only...")
    df = pd.read_csv(args.captions_csv)
    train_captions = df[df["image"].isin(set(train_imgs))]["caption"].tolist()

    vocab = Vocabulary()
    vocab.build(train_captions, min_freq=args.min_freq)
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.pth")
    vocab.save(vocab_path)
    print(f"  vocab size: {len(vocab)}  (saved to {vocab_path})")

    # ---------- Step 3: data loaders ----------
    print("Building DataLoaders...")
    train_dataset = Flickr8kDataset(
        args.image_dir, args.captions_csv, vocab, image_filenames=train_imgs
    )
    val_dataset = Flickr8kDataset(
        args.image_dir, args.captions_csv, vocab, image_filenames=val_imgs
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"  train batches: {len(train_loader)},  val batches: {len(val_loader)}")

    # ---------- Step 4: build the model ----------
    print("Building model...")
    encoder = EncoderCNN().to(device).eval()        # frozen, always in eval mode
    decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  decoder trainable params: {n_params:,}")

    # If resuming, load a previously-saved checkpoint
    if args.resume is not None:
        decoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"  resumed decoder weights from {args.resume}")

    # ---------- Step 5: loss + optimizer ----------
    pad_id = vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # ---------- Step 6: the main training loop ----------
    best_val_loss = float("inf")
    end_epoch = args.start_epoch + args.epochs

    for epoch in range(args.start_epoch, end_epoch):
        t0 = time.time()

        train_loss = run_one_epoch(
            encoder, decoder, train_loader, criterion, optimizer,
            device, args.lambda_ds, args.clip,
            description=f"train ep {epoch}",
        )
        val_loss = run_one_epoch(
            encoder, decoder, val_loader, criterion, None,   # None -> validation
            device, args.lambda_ds, args.clip,
            description=f"val   ep {epoch}",
        )

        elapsed = (time.time() - t0) / 60.0
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f} min")

        # Always save the latest weights
        torch.save(decoder.state_dict(),
                   os.path.join(args.checkpoint_dir, "last.pth"))

        # Save best.pth only if val loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(),
                       os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> new best val loss saved to checkpoints/best.pth")


if __name__ == "__main__":
    main()
