"""
Multimodal stock prediction model.

Architecture:
  MultimodalStockTransformer
    ├── PriceEncoder   (4 × TransformerBlock, reused from dl.models)
    ├── NewsEncoder    (2 × TransformerBlock, lightweight)
    ├── GatedCrossAttention  (MSGCA-style fusion)
    └── ClassifierHead (3-class: Down / Flat / Up)

Key design choices:
  - GatedCrossAttention: price sequence as Query, news as Key/Value.
    A sigmoid gate controls how much news signal is mixed in.
  - Last-3 price timesteps are used as Query to align with the 3-day
    news window (T-2, T-1, T), avoiding degenerate cross-attention over
    27 historically-irrelevant positions.
  - BERT (MacBERTEncoder) is intentionally NOT a submodule here so that
    Phase 1 training can use a single optimizer over this module only,
    and Phase 2 can add BERT as a second param group with a lower LR.
  - Flash Attention via F.scaled_dot_product_attention (same as dl/).
  - Pre-norm residuals (same as dl/).
  - Weight init: N(0, 0.02) following GPT-2 / Karpathy convention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Direct reuse — no code duplication
from dl.models import TransformerBlock, PositionalEncoding

from .config import MM_NUM_CLASSES, BERT_HIDDEN_DIM


class GatedCrossAttention(nn.Module):
    """
    MSGCA-style gated cross-attention (Multimodal Stable Fusion).

    Price sequence is the Query; news sequence is Key and Value.
    A learned sigmoid gate at each position controls how much of the
    cross-attention output replaces the price query — preventing
    degenerate fusion when news is uninformative.

    Reference: https://arxiv.org/html/2406.06594v1

    Args:
        d_model: Embedding dimension (must equal price encoder output dim).
        nhead:   Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.dropout  = dropout

        # Separate Q / K / V projections (no fused QKV: Q and KV come from
        # different sources, so fusing would add unnecessary complexity).
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Gate: learned per-position, per-channel sigmoid gate conditioned
        # on the price query (not on the attention output — avoids a feedback
        # loop and keeps the gate causal w.r.t. the price stream).
        self.gate_fc = nn.Linear(d_model, d_model)

        # Pre-norm for each input stream (separate norms — streams differ in scale)
        self.ln_q  = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        price_q: torch.Tensor,   # (B, T_q, d_model)  — price queries
        news_kv: torch.Tensor,   # (B, T_kv, d_model) — news keys / values
    ) -> torch.Tensor:
        """Returns gated fusion output of shape (B, T_q, d_model)."""
        B, T_q, _ = price_q.shape

        # Pre-norm each stream before projection
        q_norm  = self.ln_q(price_q)
        kv_norm = self.ln_kv(news_kv)

        Q = self.q_proj(q_norm)    # (B, T_q,  d_model)
        K = self.k_proj(kv_norm)   # (B, T_kv, d_model)
        V = self.v_proj(kv_norm)   # (B, T_kv, d_model)

        # Reshape for multi-head attention: (B, nhead, T, head_dim)
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            B_, T_, C = x.shape
            return x.view(B_, T_, self.nhead, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Flash Attention — same kernel as dl/models.SelfAttention
        dropout_p = self.dropout if self.training else 0.0
        attn_out  = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p)
        # (B, nhead, T_q, head_dim) → (B, T_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, -1)
        attn_out = self.out_proj(attn_out)

        # Sigmoid gate conditioned on raw price query (pre-normalisation value)
        gate = torch.sigmoid(self.gate_fc(price_q))   # (B, T_q, d_model)

        # Gated residual: interpolate between news signal and price query
        return gate * attn_out + (1.0 - gate) * price_q


class MultimodalStockTransformer(nn.Module):
    """
    Transformer that fuses price/technical sequences with daily news embeddings.

    Inputs
    ------
    price_seq : (B, 30, 106)  — 30 trading days × 106 technical features
    news_seq  : (B,  3, 768)  — 3-day rolling news window (pre-computed MacBERT CLS)

    Output
    ------
    logits    : (B, 3)        — 3-class (Down / Flat / Up) log-probabilities

    Args:
        price_input_dim : Number of price/technical features (default 106).
        bert_dim        : MacBERT hidden size (default 768).
        num_classes     : Number of output classes (default 3).
        d_model         : Internal embedding dimension.
        nhead           : Attention heads.
        num_price_layers: Transformer blocks in the price encoder.
        num_news_layers : Transformer blocks in the news temporal encoder.
        dim_feedforward : FFN hidden dimension.
        dropout         : Dropout probability.
        news_window     : Length of the news window (must match news_seq T dim).
    """

    def __init__(
        self,
        price_input_dim:  int   = 106,
        bert_dim:         int   = BERT_HIDDEN_DIM,
        num_classes:      int   = MM_NUM_CLASSES,
        d_model:          int   = 256,
        nhead:            int   = 8,
        num_price_layers: int   = 4,
        num_news_layers:  int   = 2,
        dim_feedforward:  int   = 1024,
        dropout:          float = 0.1,
        news_window:      int   = 3,
    ):
        super().__init__()
        self.news_window = news_window

        # ── Price encoder ──────────────────────────────────────────────────
        self.price_proj    = nn.Linear(price_input_dim, d_model)
        self.price_pos_enc = PositionalEncoding(d_model, dropout)
        self.price_blocks  = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_price_layers)
        ])
        self.price_ln      = nn.LayerNorm(d_model)

        # ── News temporal encoder ──────────────────────────────────────────
        # Lightweight (2 blocks): news_window=3 is a very short sequence.
        self.news_proj    = nn.Linear(bert_dim, d_model)
        self.news_pos_enc = PositionalEncoding(d_model, dropout)
        self.news_blocks  = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_news_layers)
        ])
        self.news_ln      = nn.LayerNorm(d_model)

        # ── Gated cross-attention ──────────────────────────────────────────
        self.cross_attn = GatedCrossAttention(d_model, nhead, dropout)

        # ── Classifier ─────────────────────────────────────────────────────
        # Input: concat of global price mean-pool + fused (news-aligned) mean-pool
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # GPT-2 / Karpathy N(0, 0.02) weight init — same as dl/models.py
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        price_seq: torch.Tensor,  # (B, 30, 106)
        news_seq:  torch.Tensor,  # (B,  3, 768)
    ) -> torch.Tensor:
        # ── Price encoder ──────────────────────────────────────────────────
        p = self.price_proj(price_seq)    # (B, 30, d_model)
        p = self.price_pos_enc(p)
        for block in self.price_blocks:
            p = block(p)
        p = self.price_ln(p)              # (B, 30, d_model)

        # ── News temporal encoder ──────────────────────────────────────────
        n = self.news_proj(news_seq)      # (B,  3, d_model)
        n = self.news_pos_enc(n)
        for block in self.news_blocks:
            n = block(n)
        n = self.news_ln(n)               # (B,  3, d_model)

        # ── Gated cross-attention ──────────────────────────────────────────
        # Align the last `news_window` price timesteps with the 3-day news
        # window: T-2, T-1, T all correspond to the same trading days.
        # Using only these recent price steps avoids spreading 3 news vectors
        # across 27 historically-irrelevant positions.
        price_recent = p[:, -self.news_window:, :]       # (B, 3, d_model)
        fused = self.cross_attn(price_recent, n)          # (B, 3, d_model)

        # ── Aggregation ────────────────────────────────────────────────────
        price_global = p.mean(dim=1)      # (B, d_model) — full 30-day trend
        fused_global = fused.mean(dim=1)  # (B, d_model) — news-enriched signal

        combined = torch.cat([price_global, fused_global], dim=-1)  # (B, 2·d_model)
        return self.classifier(combined)   # (B, 3)


def create_multimodal_model(config: dict) -> MultimodalStockTransformer:
    """Factory: instantiate model from config dict.

    price_input_dim is taken from config['n_features'] when present (set by
    run_train from the cache metadata.json so the model always matches the
    data actually on disk), falling back to len(FEATURE_COLUMNS) otherwise.
    """
    from dl.config import FEATURE_COLUMNS
    return MultimodalStockTransformer(
        price_input_dim  = config.get('n_features', len(FEATURE_COLUMNS)),
        bert_dim         = config.get('bert_hidden_dim',   BERT_HIDDEN_DIM),
        num_classes      = config.get('num_classes',       MM_NUM_CLASSES),
        d_model          = config.get('d_model',           256),
        nhead            = config.get('nhead',             8),
        num_price_layers = config.get('num_layers',        4),
        num_news_layers  = config.get('news_num_layers',   2),
        dim_feedforward  = config.get('dim_feedforward',   1024),
        dropout          = config.get('dropout',           0.1),
        news_window      = config.get('news_window',       3),
    )
