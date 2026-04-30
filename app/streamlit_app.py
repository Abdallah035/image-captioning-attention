"""
Streamlit web app: upload an image, see the generated caption,
per-word attention heatmaps, and a Grad-CAM heatmap.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys

# Make 'src' importable when running with `streamlit run`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.vocab import Vocabulary
from src.model import EncoderCNN, DecoderWithAttention
from src.inference import beam_search
from src.gradcam import gradcam_for_caption
from src.dataset import default_transform


CHECKPOINT_PATH = "checkpoints/best.pth"
VOCAB_PATH = "checkpoints/vocab.pth"

st.set_page_config(page_title="Image Captioning + Grad-CAM", layout="wide")


@st.cache_resource
def load_model():
    """Load model + vocab once and reuse on every interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_PATH)
    encoder = EncoderCNN().to(device).eval()
    decoder = DecoderWithAttention(vocab_size=len(vocab)).to(device)
    decoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    decoder.eval()
    return encoder, decoder, vocab, device


def attention_overlay(image_pil, alpha_49, size=224):
    """Convert a 49-vector attention map into a (size, size) overlay-ready array."""
    a = alpha_49.reshape(7, 7)
    # Upsample 7x7 -> size x size via Kronecker product (block expansion)
    a = np.kron(a, np.ones((size // 7, size // 7)))
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a


def plot_overlay(image_np, heatmap, title=""):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return fig


# ---- UI ----
encoder, decoder, vocab, device = load_model()
transform = default_transform()

st.title("Image Captioning with Attention + Grad-CAM")
st.caption("Bahdanau attention (2015) + Grad-CAM (Selvaraju 2017) on Flickr8k")

with st.sidebar:
    st.header("Settings")
    beam = st.slider("Beam size", 1, 10, 5)
    st.markdown("---")
    st.markdown("**Papers**")
    st.markdown("- [Bahdanau et al. 2015](https://arxiv.org/abs/1409.0473)")
    st.markdown("- [Xu et al. 2015 (Show, Attend & Tell)](https://arxiv.org/abs/1502.03044)")
    st.markdown("- [Selvaraju et al. 2017 (Grad-CAM)](https://arxiv.org/abs/1610.02391)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to get a generated caption with attention and Grad-CAM heatmaps.")
    st.stop()

# Preprocess
img = Image.open(uploaded).convert("RGB")
img_t = transform(img).unsqueeze(0).to(device)

col1, col2 = st.columns([1, 1])
with col1:
    st.image(img, caption="Input image", use_container_width=True)

# Beam search caption
with st.spinner("Generating caption..."):
    result = beam_search(encoder, decoder, img_t, vocab, beam=beam)

words = [vocab.itos[t] for t in result["token_ids"]
         if vocab.itos[t] not in ("<pad>", "<start>", "<end>")]
caption_text = " ".join(words)

with col2:
    st.subheader("Generated caption")
    st.markdown(f"### *{caption_text}*")
    st.caption(f"log-prob: {result['log_prob']:.2f}  beam: {beam}")

# Per-word attention heatmaps
st.divider()
st.subheader("Per-word attention heatmaps")
img_np = np.array(img.resize((224, 224))) / 255.0

# alphas length matches token_ids length minus the leading <start>
# but our beam search includes the alpha for the very first generation step.
n_words = min(len(words), len(result["alphas"]))
cols = st.columns(min(n_words, 6) or 1)

for i in range(n_words):
    word = words[i]
    alpha = result["alphas"][i]
    with cols[i % len(cols)]:
        heat = attention_overlay(img, alpha)
        st.pyplot(plot_overlay(img_np, heat, title=word), clear_figure=True)

# Grad-CAM
st.divider()
st.subheader("Grad-CAM (sentence-level)")
with st.spinner("Computing Grad-CAM..."):
    cam_heatmap, _ = gradcam_for_caption(encoder, decoder, img_t, vocab, mode="sentence")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.image(img, caption="Original", use_container_width=True)
with col_b:
    st.pyplot(plot_overlay(img_np, cam_heatmap, title="Grad-CAM"), clear_figure=True)

st.caption("Heatmap shows which image regions most influenced the prediction.")
