"""Flickr8kDataset + collate_fn for batching variable-length captions.

Used by:
- train.py:    creates train/val DataLoaders that yield (images, captions, lengths)
- evaluate.py: iterates the test split to compute BLEU scores
"""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image

from .vocab import Vocabulary


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        captions_csv: str,
        vocab: Vocabulary,
        transform: transforms.Compose | None = None,
        image_filenames: list[str] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.transform = transform or default_transform()

        df = pd.read_csv(captions_csv)
        if image_filenames is not None:
            allowed = set(image_filenames)
            df = df[df["image"].isin(allowed)].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row["image"]).convert("RGB")
        img = self.transform(img)

        start_id = self.vocab.stoi[Vocabulary.START]
        end_id = self.vocab.stoi[Vocabulary.END]
        token_ids = [start_id] + self.vocab.encode(row["caption"]) + [end_id]
        caption = torch.tensor(token_ids, dtype=torch.long)
        return img, caption


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor([c.size(0) for c in captions], dtype=torch.long)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded, lengths


def split_image_filenames(
    captions_csv: str, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    df = pd.read_csv(captions_csv)
    unique_images = sorted(df["image"].unique())
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_images), generator=rng).tolist()
    n_val = int(len(unique_images) * val_ratio)
    val_idx = set(perm[:n_val])
    train = [unique_images[i] for i in range(len(unique_images)) if i not in val_idx]
    val = [unique_images[i] for i in val_idx]
    return train, val
