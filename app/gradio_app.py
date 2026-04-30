"""
Gradio web app: upload an image, see the generated caption,
per-word attention heatmaps, and a Grad-CAM heatmap.

Run from the project root:
    python app/gradio_app.py
"""

import os
import sys

# Make 'src' importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import io
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
import torch
from PIL import Image

from src.vocab import Vocabulary
from src.model import EncoderCNN, DecoderWithAttention
from src.inference import beam_search
from src.gradcam import gradcam_for_caption
from src.dataset import default_transform
from src.checkpoint import ensure_checkpoint, BEST_PATH, VOCAB_PATH


# Use non-GUI backend (required when running on a server)
matplotlib.use("Agg")


# Load model once at startup. Auto-downloads from HF Hub on first run.
ensure_checkpoint()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"loading on device: {device}")
vocab = Vocabulary.load(VOCAB_PATH)
encoder = EncoderCNN().to(device).eval()
decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)
decoder.load_state_dict(torch.load(BEST_PATH, map_location=device))
decoder.eval()
transform = default_transform()
print(f"model ready (vocab={len(vocab)})")


def overlay_heatmap(image_np, heatmap, title=""):
    """Return a PIL image: original image with the heatmap overlaid.

    Uses a single warm colormap (hot) and makes low-attention regions
    transparent so the original image shows through clearly.
    """
    # Sharpen contrast: power > 1 makes hot regions hotter, cool regions cooler
    heatmap = heatmap ** 1.5

    # Per-pixel alpha: low values fully transparent, high values opaque
    alpha = np.clip(heatmap, 0, 1) * 0.7

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np)
    ax.imshow(heatmap, cmap="hot", alpha=alpha)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def caption_image(pil_image, beam_size):
    """Main function called when the user clicks Generate."""
    if pil_image is None:
        return "(no image)", [], None

    # 1. Preprocess
    image = pil_image.convert("RGB")
    img_np = np.array(image.resize((224, 224))) / 255.0
    img_t = transform(image).unsqueeze(0).to(device)

    # 2. Generate the caption with beam search
    result = beam_search(encoder, decoder, img_t, vocab, beam=int(beam_size))
    words = [vocab.itos[t] for t in result["token_ids"]
             if vocab.itos[t] not in ("<pad>", "<start>", "<end>")]
    caption_text = " ".join(words)

    # 3. Per-word attention heatmaps
    attention_images = []
    n_words = min(len(words), len(result["alphas"]))
    for i in range(n_words):
        alpha = result["alphas"][i].reshape(7, 7)
        # upsample 7x7 -> 224x224 by block expansion
        alpha = np.kron(alpha, np.ones((32, 32)))
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
        attn_img = overlay_heatmap(img_np, alpha, title=words[i])
        attention_images.append(attn_img)

    # 4. Grad-CAM (sentence-level)
    cam_heatmap, _ = gradcam_for_caption(encoder, decoder, img_t, vocab, mode="sentence")
    gradcam_img = overlay_heatmap(img_np, cam_heatmap, title="Grad-CAM (sentence)")

    return caption_text, attention_images, gradcam_img


# ---------- Gradio UI ----------
with gr.Blocks(title="Image Captioning + Grad-CAM") as demo:
    gr.Markdown(
        """
        # Image Captioning with Attention + Grad-CAM
        Upload an image to see the generated caption,
        per-word attention heatmaps, and a Grad-CAM heatmap.

        **Papers:** Bahdanau et al. 2015 · Xu et al. 2015 (Show, Attend & Tell) · Selvaraju et al. 2017 (Grad-CAM).
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input image")
            beam_slider = gr.Slider(1, 10, value=5, step=1, label="Beam size")
            generate_btn = gr.Button("Generate caption", variant="primary")
        with gr.Column():
            caption_output = gr.Textbox(label="Generated caption", lines=2)
            gradcam_output = gr.Image(label="Grad-CAM (sentence-level)")

    gr.Markdown("### Per-word attention heatmaps")
    attention_gallery = gr.Gallery(
        label="Per-word attention",
        columns=6, rows=2, height="auto", show_label=False,
    )

    generate_btn.click(
        fn=caption_image,
        inputs=[image_input, beam_slider],
        outputs=[caption_output, attention_gallery, gradcam_output],
    )


if __name__ == "__main__":
    demo.launch()
