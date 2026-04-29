"""Vocabulary: word <-> integer index for Flickr8k captions.

Used by:
- train.py:        builds vocab from training captions, saves to disk
- dataset.py:      encodes captions to integer tensors during data loading
- inference.py:    decodes predicted token IDs back to human-readable text
- streamlit_app.py: loads the saved vocab so IDs match the trained model
"""
import re
import pickle
from collections import Counter


class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self):
        self.itos: dict[int, str] = {}
        self.stoi: dict[str, int] = {}
        for tok in [self.PAD, self.START, self.END, self.UNK]:
            self._add(tok)

    def _add(self, tok: str) -> None:
        idx = len(self.itos)
        self.itos[idx] = tok
        self.stoi[tok] = idx

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z']+", text.lower())

    def build(self, captions: list[str], min_freq: int = 5) -> None:
        counter: Counter = Counter()
        for c in captions:
            counter.update(self.tokenize(c))
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                self._add(word)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        words = [self.itos.get(i, self.UNK) for i in ids]
        if strip_special:
            words = [w for w in words if w not in (self.PAD, self.START, self.END)]
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"itos": self.itos, "stoi": self.stoi}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        v = cls.__new__(cls)
        v.itos, v.stoi = data["itos"], data["stoi"]
        return v

    def __len__(self) -> int:
        return len(self.itos)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"
