"""
Neural network model architectures for stock prediction.

Architecture improvements over vanilla TransformerEncoderLayer:
  1. Flash Attention 2   — via F.scaled_dot_product_attention; on Ampere+ GPUs
                           (RTX 4070 Super = sm89) PyTorch dispatches this to the
                           FA-2 CUDA kernel automatically when dtype=float16.
  2. Pre-norm residuals  — LayerNorm BEFORE attention/FFN (Karpathy / GPT-2 style).
                           More stable gradients, faster convergence than post-norm.
  3. GELU activations    — smoother than ReLU; standard in modern Transformers.
  4. Fused QKV           — single Linear(d, 3d) instead of three separate projections;
                           reduces kernel launches by 2 per layer.
  5. No bias on Q/K/V    — saves params, no accuracy cost (Karpathy style).
  6. Weight init N(0,02) — GPT-2 / Karpathy standard; zero-initialise biases.
  7. Multi-horizon heads — one independent MLP head per prediction horizon (day 3/4/5).
                           Shared backbone amortises compute; separate heads learn
                           horizon-specific class boundaries.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SelfAttention(nn.Module):
    """
    Multi-head self-attention using Flash Attention.

    F.scaled_dot_product_attention dispatches to Flash Attention 2 on CUDA
    sm>=8.0 when the input dtype is float16 or bfloat16 (AMP path).
    Falls back to an efficient math kernel on CPU or older GPUs — no code
    change needed.

    Bidirectional (is_causal=False): all 30 timesteps attend to each other,
    which is correct for an encoder predicting from a complete history window.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.dropout  = dropout

        # Fused Q/K/V: one call instead of three — fewer GPU kernel launches
        # bias=False following Karpathy (saves params, no accuracy cost)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Fused projection then split
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) → (B, nhead, T, head_dim)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Flash Attention: handles 1/√d scaling, softmax, and dropout internally.
        # With AMP enabled PyTorch picks the FA-2 kernel on Ampere+ GPUs.
        dropout_p = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )

        # (B, nhead, T, head_dim) → (B, T, C)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm residuals (Karpathy / GPT-2 style).

    Pre-norm applies LayerNorm BEFORE each sub-layer:
        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Compared to original post-norm (LayerNorm after residual add), pre-norm:
      - Avoids vanishing gradients at initialisation
      - Allows higher learning rates
      - Converges faster and more stably
    """

    def __init__(
        self,
        d_model:        int,
        nhead:          int,
        dim_feedforward: int,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.attn = SelfAttention(d_model, nhead, dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),                           # GELU > ReLU for Transformers
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm attention
        x = x + self.ff(self.ln2(x))     # pre-norm FFN
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, batch-first layout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)  # (1, T, d_model) — broadcast over batch
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]    # (B, T, d_model) + (1, T, d_model)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Multi-horizon Transformer classifier for stock price change prediction.

    Architecture:
      - Shared Transformer encoder (input_projection + positional_encoding + N blocks)
      - Mean-pool over the sequence → context vector
      - Optional sector embedding concatenated to context vector
      - num_horizons independent MLP heads, one per prediction horizon

    Forward output: (batch_size, num_horizons, num_classes)
    """

    def __init__(
        self,
        input_dim:            int,
        num_classes:          int,
        num_horizons:         int   = 3,
        d_model:              int   = 256,
        nhead:                int   = 8,
        num_layers:           int   = 4,
        dim_feedforward:      int   = 1024,
        dropout:              float = 0.1,
        num_sectors:          int   = 0,
        use_sector:           bool  = False,
        num_industries:       int   = 0,
        use_relative_head:    bool  = False,
        num_relative_classes: int   = 5,
    ):
        super().__init__()
        self.use_sector       = use_sector and num_sectors > 0
        self.use_industry     = num_industries > 0
        self.num_horizons     = num_horizons
        self.use_relative_head = use_relative_head

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder      = PositionalEncoding(d_model, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Final LayerNorm after all blocks (pre-norm arch requires this)
        self.ln_final = nn.LayerNorm(d_model)

        classifier_input_dim = d_model
        if self.use_sector:
            self.sector_embedding    = nn.Embedding(num_sectors + 1,   d_model // 4)
            classifier_input_dim    += d_model // 4
        if self.use_industry:
            self.industry_embedding  = nn.Embedding(num_industries + 1, d_model // 4)
            classifier_input_dim    += d_model // 4

        # One independent two-layer MLP head per horizon.
        # Shared backbone → common representation; separate heads → per-horizon
        # decision boundaries.  All heads have the same architecture.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(classifier_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
            for _ in range(num_horizons)
        ])

        # Optional auxiliary relative-return classification heads (one per horizon).
        # Predicts stock return − CSI300 return (5 symmetric classes).
        # Smaller hidden layer (d_model//2) since this is an auxiliary signal.
        if use_relative_head:
            self.relative_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(classifier_input_dim, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_relative_classes),
                )
                for _ in range(num_horizons)
            ])

        # GPT-2 / Karpathy weight init: N(0, 0.02) for all Linear/Embedding weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x:            torch.Tensor,
        sector_ids:   torch.Tensor = None,
        industry_ids: torch.Tensor = None,
    ):
        """
        Args:
            x:            (batch_size, seq_len, input_dim)
            sector_ids:   (batch_size,) — optional coarse sector indices
            industry_ids: (batch_size,) — optional fine-grained industry indices

        Returns:
            Without relative head: logits (B, H, C)
            With relative head: (logits (B, H, C), rel_logits (B, H, C_rel))
        """
        x = self.input_projection(x)   # (B, T, d_model)
        x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        x = x.mean(dim=1)              # mean-pool over sequence → (B, d_model)

        if self.use_sector and sector_ids is not None:
            x = torch.cat([x, self.sector_embedding(sector_ids)], dim=-1)
        if self.use_industry and industry_ids is not None:
            x = torch.cat([x, self.industry_embedding(industry_ids)], dim=-1)

        # Stack per-horizon classification logits → (B, H, C)
        cls_logits = torch.stack([head(x) for head in self.heads], dim=1)

        if self.use_relative_head:
            rel_logits = torch.stack([head(x) for head in self.relative_heads], dim=1)
            return cls_logits, rel_logits   # tuple

        return cls_logits


class TemperatureScaler(nn.Module):
    """
    Per-horizon temperature scaling for probability calibration.

    Guo et al. (2017) "On Calibration of Modern Neural Networks".
    One learned scalar T per horizon, initialised to 1.0 (identity).
    T > 1 → softer / more uncertain; T < 1 → sharper / more confident.

    Fitted post-training on the validation set via LBFGS NLL minimisation
    (see training.fit_temperature).  Only the temperatures are updated —
    the backbone remains frozen.
    """

    def __init__(self, num_horizons: int = 3):
        super().__init__()
        # Initialise to 1.0 so the scaler is a no-op before fitting
        self.temperatures = nn.Parameter(torch.ones(num_horizons))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, H, C)
        Returns:
            scaled logits (B, H, C)
        """
        T = self.temperatures.to(logits.device).clamp(min=0.05).view(1, -1, 1)
        return logits / T

    def extra_repr(self) -> str:
        if self.temperatures.requires_grad or not self.training:
            vals = self.temperatures.detach().tolist()
            return '  '.join(f'T_day{i+3}={t:.4f}' for i, t in enumerate(vals))
        return ''


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(
        self,
        sequences: 'np.ndarray',
        labels:    'np.ndarray',   # (N,) or (N, H) int64
        sectors:   'np.ndarray',
    ):
        # torch.as_tensor shares memory with the numpy array (zero-copy).
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels    = torch.as_tensor(labels,    dtype=torch.long)
        self.sectors   = torch.as_tensor(sectors,   dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], self.sectors[idx]


def create_model(
    config:         dict,
    input_dim:      int,
    num_sectors:    int = 0,
    num_industries: int = 0,
) -> TransformerClassifier:
    """Factory function — creates model from config dict."""
    from .config import NUM_CLASSES, NUM_HORIZONS, NUM_RELATIVE_CLASSES

    return TransformerClassifier(
        input_dim             = input_dim,
        num_classes           = NUM_CLASSES,
        num_horizons          = NUM_HORIZONS,
        d_model               = config['d_model'],
        nhead                 = config['nhead'],
        num_layers            = config['num_layers'],
        dim_feedforward       = config['dim_feedforward'],
        dropout               = config['dropout'],
        num_sectors           = num_sectors,
        use_sector            = (num_sectors > 0),
        num_industries        = num_industries,
        use_relative_head     = config.get('use_relative_head', False),
        num_relative_classes  = NUM_RELATIVE_CLASSES,
    )


def create_tft_model(
    config:         dict,
    num_sectors:    int = 0,
    num_industries: int = 0,
):
    """Delegate to tft_model.create_tft_model."""
    from .tft_model import create_tft_model as _create_tft
    return _create_tft(config, num_sectors, num_industries)
