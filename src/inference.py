"""
Generate captions for new images, using the trained model.

Two functions:
  - greedy_search:  fast, picks the most likely word at each step
  - beam_search:    better quality, keeps the top-K candidates alive

Used by:
  - evaluate.py:      generates captions for the val set to compute BLEU
  - streamlit_app.py: generates captions for user-uploaded images
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .vocab import Vocabulary
from .dataset import default_transform


@torch.no_grad()
def greedy_search(encoder, decoder, image, vocab, max_len=20):
    """Generate a caption by always picking the highest-probability next word.

    image: (1, 3, 224, 224) preprocessed tensor on the same device as the models.

    Returns: (token_ids, alphas)
        token_ids: list of integers like [<start>, w1, w2, ..., <end>]
        alphas:    list of (49,) numpy arrays, one per generated word
    """
    encoder.eval()
    decoder.eval()
    device = image.device

    # Encode the image once
    features = encoder(image)                    # (1, 49, 2048)
    h, c = decoder.init_hidden(features)         # both (1, 512)

    start_id = vocab.stoi["<start>"]
    end_id = vocab.stoi["<end>"]

    tokens = [start_id]
    alphas = []

    for _ in range(max_len):
        # Embedding of the previous word
        prev_word = torch.tensor([tokens[-1]], device=device)
        emb = decoder.embedding(prev_word)                 # (1, 256)

        # Attention over image regions
        context, alpha = decoder.attention(features, h)    # (1, 2048), (1, 49)

        # One LSTM step
        lstm_input = torch.cat([emb, context], dim=1)
        h, c = decoder.lstm(lstm_input, (h, c))

        # Score every word, pick the best
        logits = decoder.fc(h).squeeze(0)                  # (vocab_size,)
        next_id = int(logits.argmax().item())

        tokens.append(next_id)
        alphas.append(alpha.squeeze(0).cpu().numpy())

        if next_id == end_id:
            break

    return tokens, alphas


@torch.no_grad()
def beam_search(encoder, decoder, image, vocab, beam=5, max_len=20):
    """Generate a caption with beam search.

    Idea: at every step, keep the top `beam` partial captions instead of
    committing to one. At the end, pick the one with the best total score.

    Returns a dict with:
        token_ids: best caption as a list of ints
        alphas:    list of (49,) numpy arrays for the best caption
        log_prob:  total log-probability (sum across timesteps)
    """
    encoder.eval()
    decoder.eval()
    device = image.device

    features = encoder(image)                    # (1, 49, 2048)
    h, c = decoder.init_hidden(features)
    start_id = vocab.stoi["<start>"]
    end_id = vocab.stoi["<end>"]

    # A "beam" is a candidate caption.  We track 5 of them.
    # Each beam stores: (log_prob, tokens, h, c, alphas_list)
    beams = [(0.0, [start_id], h, c, [])]
    finished = []

    for _ in range(max_len):
        new_beams = []
        for log_prob, tokens, h, c, alphas in beams:
            # If this beam already produced <end>, save and skip
            if tokens[-1] == end_id:
                finished.append((log_prob, tokens, alphas))
                continue

            # One LSTM step (same as greedy, but kept per beam)
            prev_word = torch.tensor([tokens[-1]], device=device)
            emb = decoder.embedding(prev_word)
            context, alpha = decoder.attention(features, h)
            new_h, new_c = decoder.lstm(torch.cat([emb, context], dim=1), (h, c))
            log_probs = F.log_softmax(decoder.fc(new_h), dim=-1).squeeze(0)

            # Pick the top `beam` continuations from this beam
            top_log_probs, top_ids = log_probs.topk(beam)
            for k in range(beam):
                new_beams.append((
                    log_prob + top_log_probs[k].item(),
                    tokens + [top_ids[k].item()],
                    new_h, new_c,
                    alphas + [alpha.squeeze(0).cpu().numpy()],
                ))

        if not new_beams:
            break
        # Across all (beam * beam) candidates, keep only the top `beam`
        new_beams.sort(key=lambda b: b[0], reverse=True)
        beams = new_beams[:beam]

    # Beams that never produced <end> are still candidates
    for log_prob, tokens, h, c, alphas in beams:
        finished.append((log_prob, tokens, alphas))

    # Length-normalize: longer sequences accumulate more negative log-probs.
    # Dividing by length makes the comparison fair.
    finished.sort(key=lambda b: b[0] / max(len(b[1]), 1), reverse=True)
    best_log_prob, best_tokens, best_alphas = finished[0]

    return {
        "token_ids": best_tokens,
        "alphas": best_alphas,
        "log_prob": best_log_prob,
    }


def load_image(path, device):
    """Load and preprocess one image into a (1, 3, 224, 224) tensor."""
    img = Image.open(path).convert("RGB")
    return default_transform()(img).unsqueeze(0).to(device)


def caption_image(encoder, decoder, image_path, vocab, device, beam=5):
    """Convenience wrapper: load image -> beam search -> decoded text."""
    image = load_image(image_path, device)
    result = beam_search(encoder, decoder, image, vocab, beam=beam)
    text = vocab.decode(result["token_ids"])
    return text, result
