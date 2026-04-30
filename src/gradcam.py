"""
Grad-CAM (Selvaraju et al. 2017) for the captioning model.

What it does:
  After the model generates a caption, Grad-CAM produces a heatmap
  showing which image regions most influenced the prediction.

How:
  1. Pick a target conv layer (encoder's last conv: outputs 2048 x 7 x 7).
  2. Forward pass: capture activations A from that layer.
  3. Pick a target score (sentence-level: sum of chosen-word logits).
  4. Backward pass: capture gradients dy/dA.
  5. Per-channel weights = mean of gradients over the spatial dims.
  6. Heatmap = ReLU(sum over channels of weight * activation).
  7. Upsample 7x7 to 224x224 and normalize to [0, 1].

Used by:
  streamlit_app.py to show the heatmap next to the caption.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Captures activations + gradients on a target conv layer."""

    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # Register two hooks to capture forward output and backward grads
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out  # keep in graph so backward can flow

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        """Remove the hooks (so they don't fire on later forward passes)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self, target_scalar, image_size=(224, 224)):
        """Backprop from target_scalar, return a (H, W) numpy heatmap in [0, 1]."""
        target_scalar.backward(retain_graph=True)

        # weights = mean gradient per channel  -> shape (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # weighted sum over channels, then ReLU
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # upsample 7x7 -> 224x224
        cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)

        # normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy()


def _enable_encoder_grad(encoder):
    """Temporarily turn ON requires_grad for the frozen encoder."""
    for p in encoder.cnn.parameters():
        p.requires_grad_(True)


def _disable_encoder_grad(encoder):
    """Turn it OFF again so future training/inference is unaffected."""
    for p in encoder.cnn.parameters():
        p.requires_grad_(False)


def _generate_with_grad(encoder, decoder, image, vocab, max_len=20):
    """Greedy decode that runs WITH gradients enabled (needed for Grad-CAM).

    Returns (tokens, list of chosen-word logits per timestep).
    """
    features = encoder(image)
    h, c = decoder.init_hidden(features)
    start_id = vocab.stoi["<start>"]
    end_id = vocab.stoi["<end>"]

    tokens = [start_id]
    chosen_logits = []

    for _ in range(max_len):
        prev_word = torch.tensor([tokens[-1]], device=image.device)
        emb = decoder.embedding(prev_word)
        context, _ = decoder.attention(features, h)
        h, c = decoder.lstm(torch.cat([emb, context], dim=1), (h, c))
        step_logits = decoder.fc(h).squeeze(0)        # (vocab_size,)
        next_id = int(step_logits.argmax().item())
        tokens.append(next_id)
        chosen_logits.append(step_logits[next_id])
        if next_id == end_id:
            break

    return tokens, chosen_logits


def gradcam_for_caption(encoder, decoder, image, vocab,
                        mode="sentence", word_step=None,
                        image_size=(224, 224)):
    """Generate a caption and compute a Grad-CAM heatmap.

    mode:
        "sentence": sum of chosen-token logits across all timesteps
        "word":     logit of the word at index `word_step` (excluding <start>)

    Returns (heatmap, token_ids).
    """
    target_layer = encoder.cnn[-1]  # last conv block of ResNet
    cam = GradCAM(target_layer)
    _enable_encoder_grad(encoder)
    encoder.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)

    try:
        tokens, chosen_logits = _generate_with_grad(encoder, decoder, image, vocab)
        if mode == "sentence":
            target_scalar = torch.stack(chosen_logits).sum()
        elif mode == "word":
            if word_step is None:
                raise ValueError("word_step is required when mode='word'")
            target_scalar = chosen_logits[word_step]
        else:
            raise ValueError(f"unknown mode: {mode}")

        heatmap = cam.compute(target_scalar, image_size=image_size)
    finally:
        cam.remove()
        _disable_encoder_grad(encoder)

    return heatmap, tokens
