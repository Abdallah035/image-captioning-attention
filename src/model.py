"""
The neural network for image captioning.

Three pieces:
  1. EncoderCNN         - extracts features from the image (ResNet-50, frozen)
  2. BahdanauAttention  - decides which image regions to look at for each word
  3. DecoderWithAttention - LSTM that writes the caption, using attention each step

Based on:
  - Bahdanau, Cho, Bengio (2015): attention mechanism
  - Xu et al. (2015) "Show, Attend and Tell": this whole architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1. The Encoder: ResNet-50 with the last two layers removed
# ============================================================
class EncoderCNN(nn.Module):
    """
    Takes an image (B, 3, 224, 224) and returns features (B, 49, 2048).
    The 49 = 7 x 7 spatial regions, each described by a 2048-dim feature vector.

    We freeze ResNet (don't train it) because:
      - It's pretrained on 1.2M ImageNet images and already knows good features
      - We only have 8000 images; fine-tuning would ruin its features
      - Freezing saves a lot of GPU memory (no gradients through ResNet)
    """

    def __init__(self):
        super().__init__()

        # Load ResNet-50 with ImageNet pretrained weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ResNet's children list ends with: ..., AvgPool, Linear
        # We drop the last 2 (AvgPool + Linear) to keep the spatial feature map.
        all_layers = list(resnet.children())
        useful_layers = all_layers[:-2]
        self.cnn = nn.Sequential(*useful_layers)

        # Freeze all weights so they don't get updated during training
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 49, 2048)
        """
        features = self.cnn(images)                  # (B, 2048, 7, 7)
        # Rearrange so the 49 spatial positions are along the 2nd dimension:
        #   (B, 2048, 7, 7)  -> permute -> (B, 7, 7, 2048)  -> flatten -> (B, 49, 2048)
        features = features.permute(0, 2, 3, 1)      # (B, 7, 7, 2048)
        features = features.flatten(start_dim=1, end_dim=2)  # (B, 49, 2048)
        return features


# ============================================================
# 2. Bahdanau Attention
# ============================================================
class BahdanauAttention(nn.Module):
    """
    For one decoder hidden state, computes:
      - 49 attention weights (one per image region)
      - a "context vector" = weighted sum of region features

    This is the Bahdanau (2015) formula:
        score_i  = v_a . tanh( W_a . h  +  U_a . F[i] )
        alpha    = softmax(scores)
        context  = sum over i of alpha_i * F[i]
    """

    def __init__(self, encoder_dim=2048, decoder_dim=512, attn_dim=512):
        super().__init__()
        # Three learnable linear layers (the matrices in the formula).
        # bias=False because the paper uses no bias here.
        self.W_a = nn.Linear(decoder_dim, attn_dim, bias=False)  # for the hidden state
        self.U_a = nn.Linear(encoder_dim, attn_dim, bias=False)  # for each region feature
        self.v_a = nn.Linear(attn_dim, 1, bias=False)            # collapses to a single score

    def forward(self, features, hidden):
        """
        features: (B, 49, encoder_dim)  -- image regions
        hidden:   (B, decoder_dim)      -- decoder's current state

        returns:
            context: (B, encoder_dim)
            alpha:   (B, 49)
        """
        # Project the decoder hidden state and each region feature into the same space.
        proj_h = self.W_a(hidden)            # (B, attn_dim)
        proj_F = self.U_a(features)          # (B, 49, attn_dim)

        # Add them. We use unsqueeze(1) to broadcast hidden across all 49 regions.
        combined = torch.tanh(proj_F + proj_h.unsqueeze(1))   # (B, 49, attn_dim)

        # Collapse attn_dim to 1 number per region -> these are the scores
        scores = self.v_a(combined).squeeze(-1)               # (B, 49)

        # Softmax converts scores to weights summing to 1 (over regions)
        alpha = F.softmax(scores, dim=1)                      # (B, 49)

        # Weighted sum of region features, weighted by alpha.
        # alpha.unsqueeze(-1) has shape (B, 49, 1); features (B, 49, encoder_dim)
        # multiplying then summing over regions (dim=1) gives (B, encoder_dim)
        context = (alpha.unsqueeze(-1) * features).sum(dim=1)

        return context, alpha


# ============================================================
# 3. The Decoder: LSTM that writes the caption word-by-word
# ============================================================
class DecoderWithAttention(nn.Module):
    """
    At each timestep t, this decoder:
      1. Uses attention to make a context vector (which regions to look at)
      2. Combines [embedding(prev_word), context] -> input to LSTM
      3. LSTMCell step -> new hidden state h
      4. Linear(h) -> logits over the whole vocabulary -> predict next word

    Inputs to the forward method (training, with teacher forcing):
      features: (B, 49, encoder_dim)  the image, already encoded
      captions: (B, T)                ground-truth caption tokens

    Outputs:
      logits: (B, T, vocab_size)   predictions per timestep
      alphas: (B, T, 49)           attention weights per timestep
    """

    def __init__(self, vocab_size, embedding_dim=256,
                 decoder_dim=512, encoder_dim=2048, attn_dim=512, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        # Word embedding: vocabulary integer -> embedding_dim vector
        # (each word gets a 256-dim "meaning" vector that the model learns)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The attention module
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attn_dim)

        # LSTMCell: takes ONE timestep at a time
        # Its input is [word_embedding, context_vector] concatenated
        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)

        # Two small layers to convert mean image feature -> initial h_0 and c_0
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Final layer: hidden state -> score for each word in the vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Dropout for regularization (turns off some neurons during training)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, features):
        """Initialize LSTM hidden state from the average image feature."""
        mean_feature = features.mean(dim=1)        # (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_feature))  # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_feature))  # (B, decoder_dim)
        return h, c

    def forward(self, features, captions):
        """
        Teacher-forced training pass.

        features: (B, 49, encoder_dim)
        captions: (B, T)               -- input tokens (we will predict the NEXT token at each step)
        """
        batch_size, T = captions.shape

        # Initialize LSTM state from the image
        h, c = self.init_hidden(features)

        # Look up embeddings for ALL input tokens at once
        embeddings = self.embedding(captions)         # (B, T, embedding_dim)

        # Pre-allocate output tensors
        logits = features.new_zeros(batch_size, T, self.vocab_size)
        alphas = features.new_zeros(batch_size, T, features.size(1))  # (B, T, 49)

        # Loop over timesteps: at each step, predict the next word
        for t in range(T):
            # 1. Attention: which regions to look at, given current h
            context, alpha = self.attention(features, h)   # (B, encoder_dim), (B, 49)

            # 2. LSTM input = [embedding of previous true word, context vector]
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            # lstm_input shape: (B, embedding_dim + encoder_dim)

            # 3. One LSTM step
            h, c = self.lstm(lstm_input, (h, c))           # (B, decoder_dim) each

            # 4. Project h to vocabulary scores
            logits[:, t] = self.fc(self.dropout(h))         # (B, vocab_size)
            alphas[:, t] = alpha                            # save attention weights

        return logits, alphas
