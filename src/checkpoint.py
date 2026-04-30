"""
Auto-download model + vocab from Hugging Face Hub on first run.

The trained checkpoint (best.pth) and vocabulary (vocab.pth) are too large
to commit to GitHub, so we host them on the Hub. This module ensures the
files exist locally before the app or training script tries to use them.

Set HF_REPO_ID below to your own HF repo once it is created.
Example: "your-username/image-captioning-attention-flickr8k"

If the env var IMG_CAP_HF_REPO is set, it overrides the default.
"""

import os
import shutil

CHECKPOINTS_DIR = "checkpoints"
BEST_PATH = os.path.join(CHECKPOINTS_DIR, "best.pth")
VOCAB_PATH = os.path.join(CHECKPOINTS_DIR, "vocab.pth")

DEFAULT_HF_REPO = "abdallah-03/image-captioning-attention-flickr8k"
HF_REPO_ID = os.environ.get("IMG_CAP_HF_REPO", DEFAULT_HF_REPO)


def ensure_checkpoint():
    """Download best.pth + vocab.pth from HF Hub if they're missing locally."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    need_best = not os.path.exists(BEST_PATH)
    need_vocab = not os.path.exists(VOCAB_PATH)
    if not (need_best or need_vocab):
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download the model. "
            "Install it with: pip install huggingface_hub"
        ) from e

    if need_best:
        print(f"[checkpoint] downloading best.pth from {HF_REPO_ID}...")
        path = hf_hub_download(repo_id=HF_REPO_ID, filename="best.pth")
        shutil.copy(path, BEST_PATH)

    if need_vocab:
        print(f"[checkpoint] downloading vocab.pth from {HF_REPO_ID}...")
        path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab.pth")
        shutil.copy(path, VOCAB_PATH)
