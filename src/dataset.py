"""
Dataset for Flickr8k: gives PyTorch (image, caption) pairs to train on.

Three jobs:
  1. Load each image and prepare it for ResNet-50 (resize, normalize).
  2. Encode each caption as a list of integer IDs.
  3. Pad short captions in a batch so they fit a rectangular tensor.

Usage in train.py:
    train_dataset = Flickr8kDataset(
        image_dir="data/Images",
        captions_csv="data/captions.txt",
        vocab=vocab,
        image_filenames=train_image_list,
    )
    loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    for images, captions, lengths in loader:
        ...
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image


# These numbers come from ImageNet (mean/std of all training pixels per channel).
# ResNet-50 expects inputs normalized this way, so we apply the SAME normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform():
    """The image preprocessing pipeline.

    A transform is a series of operations applied to each image:
      1. Resize to 224 x 224  (ResNet-50 expects this size)
      2. Convert PIL image to tensor, scale pixels to [0, 1]
      3. Normalize using ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kDataset(Dataset):

    def __init__(self, image_dir, captions_csv, vocab, image_filenames=None):
        """
        Args:
            image_dir:        path to folder with .jpg files
            captions_csv:     path to captions.txt (CSV with columns "image", "caption")
            vocab:            a built Vocabulary object
            image_filenames:  optional list to filter rows (used for train/val split)
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = default_transform()

        # Load all (image_filename, caption) rows from the CSV
        df = pd.read_csv(captions_csv)

        # If a filename list was provided, only keep rows for those images
        # (this is how we get train-only or val-only samples)
        if image_filenames is not None:
            allowed = set(image_filenames)
            df = df[df["image"].isin(allowed)].reset_index(drop=True)

        self.df = df

    def __len__(self):
        """Total number of (image, caption) pairs."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return one sample: (image_tensor, caption_tensor)."""
        row = self.df.iloc[idx]
        image_filename = row["image"]
        caption_text = row["caption"]

        # --- prepare the image ---
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")  # ensure 3 channels
        image = self.transform(image)                  # -> tensor (3, 224, 224)

        # --- prepare the caption ---
        # We add <start> at the beginning and <end> at the end.
        # The decoder uses <start> as its first input,
        # and learns to predict <end> when the caption is finished.
        start_id = self.vocab.stoi["<start>"]
        end_id = self.vocab.stoi["<end>"]
        token_ids = [start_id] + self.vocab.encode(caption_text) + [end_id]
        caption = torch.tensor(token_ids, dtype=torch.long)

        return image, caption


def collate_fn(batch):
    """Combine a list of samples into one batched tensor.

    Why we need a custom collate_fn:
      The default one stacks tensors directly, but our captions have
      different lengths so they can't be stacked. We pad them first.

    Input:
      batch = [(image_0, caption_0), (image_1, caption_1), ...]
              where each caption has a different length.

    Output:
      images: tensor (B, 3, 224, 224)
      captions: tensor (B, T) padded with 0 (the <pad> id)
      lengths: tensor (B,) actual length of each caption (before padding)
    """
    # split the list of pairs into a list of images and a list of captions
    images, captions = zip(*batch)

    # all images are already 224 x 224, so we can stack them directly
    images = torch.stack(images, dim=0)

    # captions have different lengths -> save real lengths first
    lengths = torch.tensor([c.size(0) for c in captions], dtype=torch.long)

    # pad shorter captions with 0 (which is the <pad> token's id)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions_padded, lengths


def split_image_filenames(captions_csv, val_ratio=0.1, seed=42):
    """Split the dataset by IMAGE (not by row).

    Why split by image, not by row?
      Each image has 5 captions in the CSV. If we split by row,
      the same image could land in both train and val, leaking info.
      By splitting on the unique image list, every image is fully
      in one split and the model never sees val images during training.

    Returns:
        (train_filenames, val_filenames)
    """
    df = pd.read_csv(captions_csv)
    all_images = sorted(df["image"].unique())   # all unique image filenames

    # use a fixed seed so the split is the same every time we run
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(all_images), generator=generator).tolist()

    n_val = int(len(all_images) * val_ratio)
    val_indices = set(shuffled_indices[:n_val])

    train = [all_images[i] for i in range(len(all_images)) if i not in val_indices]
    val = [all_images[i] for i in val_indices]
    return train, val
