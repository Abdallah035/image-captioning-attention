"""
Vocabulary: converts words to integers and back.

Why we need this:
  Neural networks work with numbers, not text.
  So "a dog runs" must become something like [4, 5, 6] before the model sees it.
  And the model's output [4, 5, 6] must become "a dog runs" before we display it.

Two dictionaries do the job:
  stoi: word -> integer  (for encoding text)
  itos: integer -> word  (for decoding model output)

We also reserve 4 special words at fixed positions:
  <pad>  (id 0): filler so all captions in a batch have the same length
  <start>(id 1): tells the decoder "begin generating"
  <end>  (id 2): tells the decoder "stop generating"
  <unk>  (id 3): replaces rare words we didn't include in our vocabulary
"""

import re
import pickle
from collections import Counter


class Vocabulary:

    def __init__(self):
        # the four special tokens, always at IDs 0, 1, 2, 3
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

    def tokenize(self, text):
        """Split a sentence into clean lowercase words.

        Example:
            "A Dog's running!"  ->  ["a", "dog's", "running"]

        We lowercase everything (so "Dog" and "dog" are the same word)
        and drop punctuation/numbers (they aren't useful for captions).
        We keep apostrophes so "don't" stays one word.
        """
        return re.findall(r"[a-z']+", text.lower())

    def build(self, captions, min_freq=5):
        """Build the vocabulary from a list of training captions.

        Args:
            captions: list of strings, e.g. ["a dog runs", "a cat sleeps", ...]
            min_freq: only keep words that appear at least this many times.
                      Rare words become <unk> later.
        """
        # Step 1: count how many times each word appears across all captions
        counts = Counter()
        for caption in captions:
            words = self.tokenize(caption)
            counts.update(words)

        # Step 2: add words that pass the frequency threshold
        # (the 4 special tokens are already there from __init__)
        next_id = 4
        for word, count in counts.items():
            if count >= min_freq:
                self.stoi[word] = next_id
                self.itos[next_id] = word
                next_id += 1

    def encode(self, text):
        """Turn a sentence into a list of integer IDs.

        Example (assuming "a"=4, "dog"=5):
            "a dog flies"  ->  [4, 5, 3]    # "flies" -> <unk> (3) if rare
        """
        unk_id = self.stoi["<unk>"]
        ids = []
        for word in self.tokenize(text):
            # if the word isn't in our vocab, fall back to <unk>
            ids.append(self.stoi.get(word, unk_id))
        return ids

    def decode(self, ids, strip_special=True):
        """Turn a list of integer IDs back into a sentence.

        Example:
            [1, 4, 5, 6, 2]  ->  "a dog runs"   # strips <start> and <end>
        """
        words = [self.itos[i] for i in ids]
        if strip_special:
            words = [w for w in words if w not in ("<pad>", "<start>", "<end>")]
        return " ".join(words)

    def __len__(self):
        """Vocab size (used by the decoder's output layer)."""
        return len(self.itos)

    def save(self, path):
        """Save vocabulary to disk so we can reuse it at inference time."""
        with open(path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        """Load a previously saved vocabulary."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        v = cls()
        v.stoi = data["stoi"]
        v.itos = data["itos"]
        return v
