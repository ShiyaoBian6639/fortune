"""
Temporal Fusion Transformer for multi-horizon stock classification.

Implements the TFT architecture from:
  "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
  Lim et al., 2021 (https://arxiv.org/abs/1912.09363)

Adapted for:
  - 7-class relative-return classification (stock − CSI300) at horizons day3/4/5
  - Chinese A-share features: 186 observed-past + 27 known-future (calendar/holiday)
  - Sector/industry static categorical covariates
  - Interpretability outputs: VSN weights + attention heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    NUM_CLASSES, NUM_HORIZONS, NUM_RELATIVE_CLASSES,
    FORWARD_WINDOWS,
    NUM_KNOWN_FUTURE_FEATURES, NUM_OBSERVED_PAST_FEATURES,
    _OBS_PAST_FEAT_IDX,
)

# Decoder positions (0-indexed) corresponding to FORWARD_WINDOWS=[3,4,5]
_HORIZON_DEC_POSITIONS = [fw - 1 for fw in FORWARD_WINDOWS]   # [2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    """GLU: σ(gate(x)) ⊗ fc(x)  (dimension-preserving)."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc   = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc(x)) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN(a, c) = LayerNorm(a + GLU(ELU(W1·[a; W_c·c] + b1)))

    Paper eq. (2)–(4). Context c is optional; if context_dim > 0, a linear
    maps c into the hidden layer before ELU activation.
    A skip projection is added when input_dim ≠ output_dim.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        output_dim:  int,
        context_dim: int   = 0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.fc1  = nn.Linear(input_dim,  hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.elu  = nn.ELU()
        self.glu  = GatedLinearUnit(output_dim, output_dim, dropout)
        self.ln   = nn.LayerNorm(output_dim)
        self.skip = (nn.Linear(input_dim, output_dim, bias=False)
                     if input_dim != output_dim else nn.Identity())
        self.ctx_proj = (nn.Linear(context_dim, hidden_dim, bias=False)
                         if context_dim > 0 else None)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        h = self.fc1(x)
        if c is not None and self.ctx_proj is not None:
            h = h + self.ctx_proj(c)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.ln(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: per-variable GRNs + softmax selection → weighted combination.

    Paper section 4.2. Each variable is treated as a scalar (var_dim=1) so
    every feature is independently embedded before selection.

    Handles both:
      static:   x (B, num_vars, var_dim),   c (B, context_dim)
      temporal: x (B, T, num_vars, var_dim), c (B, T, context_dim)
    """

    def __init__(
        self,
        num_vars:    int,
        var_dim:     int,
        hidden_dim:  int,
        context_dim: int   = 0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(var_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])
        self.wt_grn = GatedResidualNetwork(
            input_dim   = num_vars * var_dim,
            hidden_dim  = hidden_dim,
            output_dim  = num_vars,
            context_dim = context_dim,
            dropout     = dropout,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        Returns:
            output  (B, [T,] hidden)   — weighted combination of per-var embeddings
            weights (B, [T,] num_vars) — softmax selection weights (interpretability)
        """
        temporal = (x.dim() == 4)
        if temporal:
            B, T, V, D = x.shape
            xf = x.reshape(B * T, V, D)
            cf = c.reshape(B * T, -1) if c is not None else None
        else:
            B, V, D = x.shape
            xf, cf = x, c

        # Per-variable GRN embeddings
        var_out = torch.stack(
            [self.var_grns[v](xf[:, v, :]) for v in range(self.num_vars)],
            dim=1,
        )   # (B[*T], V, hidden)

        # Softmax selection weights (conditioned on static context)
        flat = xf.reshape(xf.size(0), -1)                        # (B[*T], V*D)
        wts  = torch.softmax(self.wt_grn(flat, cf), dim=-1)      # (B[*T], V)

        out = (var_out * wts.unsqueeze(-1)).sum(dim=1)            # (B[*T], hidden)

        if temporal:
            out = out.reshape(B, T, -1)
            wts = wts.reshape(B, T, V)

        return out, wts


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable MHA: shared V projection, per-head Q/K, average attention weights.

    Paper section 4.3. Averaging attention weights across heads produces a single
    interpretable (T_q, T_k) map showing which historical timesteps are most attended.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_q  = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(nhead)])
        self.W_k  = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(nhead)])
        self.out  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            q, k, v: (B, T, d_model)
            mask:    bool tensor (B, T_q, T_k) — True positions are masked out

        Returns:
            output       (B, T_q, d_model)
            attn_weights (B, T_q, T_k) — averaged over heads
        """
        scale  = self.head_dim ** -0.5
        V_proj = self.W_v(v)   # (B, T_k, d_model) — shared across heads

        head_outs, head_attn = [], []
        for h in range(self.nhead):
            Q_h = self.W_q[h](q)                                            # (B, T_q, head_dim)
            K_h = self.W_k[h](k)                                            # (B, T_k, head_dim)
            scores = torch.bmm(Q_h, K_h.transpose(1, 2)) * scale           # (B, T_q, T_k)
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = self.drop(attn)
            head_outs.append(torch.bmm(attn, V_proj))                       # (B, T_q, d_model)
            head_attn.append(attn)

        out  = torch.stack(head_outs, dim=0).mean(dim=0)   # (B, T_q, d_model)
        wts  = torch.stack(head_attn, dim=0).mean(dim=0)   # (B, T_q, T_k)
        return self.out(out), wts


# ─────────────────────────────────────────────────────────────────────────────
# Full TFT model
# ─────────────────────────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    TFT for multi-horizon stock movement classification.

    Forward inputs:
        past_seq      (B, 30, 213)  — full FEATURE_COLUMNS sequence; obs-past split done internally
        future_inputs (B,  5,  27)  — KNOWN_FUTURE_FEATURE_COLUMNS for decoder horizon steps
        sector_ids    (B,)          — sector categorical index
        industry_ids  (B,)          — industry categorical index

    Forward output:
        logits        (B, 3,  7)    — per-horizon classification logits (day3/4/5)

    Interpretability attributes set after each forward pass:
        _enc_vsn_weights  (B, 30, 186) — encoder VSN selection weights per timestep
        _dec_vsn_weights  (B,  5,  27) — decoder VSN selection weights per timestep
        _attn_weights     (B, 35, 35)  — averaged multi-head attention weights
    """

    def __init__(
        self,
        num_past_features:    int   = NUM_OBSERVED_PAST_FEATURES,
        num_future_features:  int   = NUM_KNOWN_FUTURE_FEATURES,
        num_classes:          int   = NUM_CLASSES,
        num_horizons:         int   = NUM_HORIZONS,
        hidden_dim:           int   = 160,
        num_heads:            int   = 4,
        lstm_layers:          int   = 2,
        dropout:              float = 0.1,
        num_sectors:          int   = 0,
        num_industries:       int   = 0,
        use_relative_head:    bool  = False,
        num_relative_classes: int   = NUM_RELATIVE_CLASSES,
    ):
        super().__init__()
        self._is_tft = True
        self.num_horizons         = num_horizons
        self.hidden_dim           = hidden_dim
        self.lstm_layers          = lstm_layers
        self.num_past_features    = num_past_features
        self.num_future_features  = num_future_features
        self.use_relative_head    = use_relative_head
        self.horizon_dec_positions = _HORIZON_DEC_POSITIONS   # [2, 3, 4]

        # ── Static covariate embeddings ──────────────────────────────────────
        emb_dim = hidden_dim // 2
        self.sector_emb   = (nn.Embedding(num_sectors   + 1, emb_dim)
                             if num_sectors   > 0 else None)
        self.industry_emb = (nn.Embedding(num_industries + 1, emb_dim)
                             if num_industries > 0 else None)

        static_dim = emb_dim * int(num_sectors > 0) + emb_dim * int(num_industries > 0)
        if static_dim == 0:
            static_dim = 1   # fallback: single constant input

        # 4 context vectors produced from static embedding (paper section 4.1)
        self.static_grn_s = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)  # c_s: VSN context
        self.static_grn_e = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)  # c_e: encoder h0
        self.static_grn_c = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)  # c_c: encoder cell
        self.static_grn_h = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)  # c_h: enrichment

        # ── Variable Selection Networks ───────────────────────────────────────
        # var_dim=1: each feature is a scalar; per-var GRN embeds it to hidden_dim
        self.vsn_encoder = VariableSelectionNetwork(
            num_vars    = num_past_features,
            var_dim     = 1,
            hidden_dim  = hidden_dim,
            context_dim = hidden_dim,
            dropout     = dropout,
        )
        self.vsn_decoder = VariableSelectionNetwork(
            num_vars    = num_future_features,
            var_dim     = 1,
            hidden_dim  = hidden_dim,
            context_dim = hidden_dim,
            dropout     = dropout,
        )

        # ── LSTM encoder + decoder ────────────────────────────────────────────
        lstm_kw = dict(batch_first=True,
                       dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, **lstm_kw)
        self.lstm_decoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, **lstm_kw)

        # ── Post-LSTM gating (skip connection: VSN output → LSTM output) ─────
        self.post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.post_lstm_ln  = nn.LayerNorm(hidden_dim)

        # ── Static enrichment ─────────────────────────────────────────────────
        self.static_enrich_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout
        )

        # ── Interpretable Multi-Head Attention ────────────────────────────────
        self.attn     = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.attn_ln  = nn.LayerNorm(hidden_dim)

        # ── Position-wise GRN + gating ────────────────────────────────────────
        self.pw_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.pw_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.pw_ln  = nn.LayerNorm(hidden_dim)

        # ── Classification heads (one per horizon) ────────────────────────────
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_horizons)
        ])
        if use_relative_head:
            self.relative_heads = nn.ModuleList([
                nn.Linear(hidden_dim, num_relative_classes) for _ in range(num_horizons)
            ])

        # Interpretability buffers (populated each forward pass)
        self._enc_vsn_weights: torch.Tensor = None
        self._dec_vsn_weights: torch.Tensor = None
        self._attn_weights:    torch.Tensor = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _static_context(self, sector_ids, industry_ids, B, device):
        """Build the single static embedding vector (B, static_dim)."""
        parts = []
        if self.sector_emb is not None and sector_ids is not None:
            parts.append(self.sector_emb(sector_ids))
        if self.industry_emb is not None and industry_ids is not None:
            parts.append(self.industry_emb(industry_ids))
        if parts:
            return torch.cat(parts, dim=-1)   # (B, static_dim)
        # Fallback: constant 1 — GRNs still produce meaningful projections
        return torch.ones(B, 1, device=device, dtype=torch.float32)

    def forward(
        self,
        past_seq:     torch.Tensor,
        future_inputs: torch.Tensor,
        sector_ids:   torch.Tensor = None,
        industry_ids: torch.Tensor = None,
    ):
        """
        Args:
            past_seq:      (B, 30, 213) — full feature sequence
            future_inputs: (B,  5,  27) — known-future calendar features for decoder
            sector_ids:    (B,) int64
            industry_ids:  (B,) int64

        Returns:
            Without relative head: logits (B, 3, 7)
            With relative head:    (cls_logits, rel_logits)
        """
        B, T, _ = past_seq.shape
        device   = past_seq.device

        # ── 1. Extract observed-past features from full sequence ──────────────
        obs_idx  = torch.tensor(_OBS_PAST_FEAT_IDX, device=device, dtype=torch.long)
        past_obs = past_seq[:, :, obs_idx]   # (B, 30, 186)

        # ── 2. Static encoder ─────────────────────────────────────────────────
        static_feat = self._static_context(sector_ids, industry_ids, B, device)
        c_s = self.static_grn_s(static_feat)   # (B, H) — VSN context
        c_e = self.static_grn_e(static_feat)   # (B, H) — LSTM encoder h0
        c_c = self.static_grn_c(static_feat)   # (B, H) — LSTM encoder cell
        c_h = self.static_grn_h(static_feat)   # (B, H) — static enrichment

        # ── 3. Encoder VSN ────────────────────────────────────────────────────
        # Each feature is a scalar: reshape (B, T, V) → (B, T, V, 1)
        enc_in  = past_obs.unsqueeze(-1)                        # (B, 30, 186, 1)
        c_s_enc = c_s.unsqueeze(1).expand(-1, T, -1)           # (B, 30, H)
        enc_vsn, enc_wts = self.vsn_encoder(enc_in, c_s_enc)   # (B, 30, H), (B, 30, 186)

        # ── 4. Decoder VSN ────────────────────────────────────────────────────
        dec_len = future_inputs.size(1)                         # 5
        dec_in  = future_inputs.unsqueeze(-1)                   # (B, 5, 27, 1)
        c_s_dec = c_s.unsqueeze(1).expand(-1, dec_len, -1)     # (B, 5, H)
        dec_vsn, dec_wts = self.vsn_decoder(dec_in, c_s_dec)   # (B, 5, H), (B, 5, 27)

        # ── 5. LSTM encoder (initialized from static context) ─────────────────
        h0 = c_e.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()  # (L, B, H)
        c0 = c_c.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(enc_vsn, (h0, c0))           # (B, 30, H)

        # ── 6. LSTM decoder (continues from encoder's final state) ────────────
        dec_out, _ = self.lstm_decoder(dec_vsn, (h_n, c_n))    # (B, 5, H)

        # ── 7. Post-LSTM GLU gating + skip from VSN ───────────────────────────
        full_vsn  = torch.cat([enc_vsn, dec_vsn], dim=1)       # (B, 35, H)
        full_lstm = torch.cat([enc_out, dec_out], dim=1)       # (B, 35, H)
        gated     = self.post_lstm_glu(full_lstm)
        full_seq  = self.post_lstm_ln(full_vsn + gated)        # (B, 35, H)

        # ── 8. Static enrichment ─────────────────────────────────────────────
        seq_len_total = T + dec_len                             # 35
        c_h_exp = c_h.unsqueeze(1).expand(-1, seq_len_total, -1)   # (B, 35, H)
        enriched = self.static_enrich_grn(full_seq, c_h_exp)   # (B, 35, H)

        # ── 9. Interpretable Multi-Head Attention ─────────────────────────────
        attn_out, attn_wts = self.attn(enriched, enriched, enriched)  # (B, 35, H), (B, 35, 35)
        gated2   = self.attn_glu(attn_out)
        attn_out = self.attn_ln(enriched + gated2)              # (B, 35, H)

        # ── 10. Position-wise GRN + gating ───────────────────────────────────
        pw = self.pw_grn(attn_out)
        pw = self.pw_ln(attn_out + self.pw_glu(pw))            # (B, 35, H)

        # ── 11. Classification heads at decoder horizon positions ─────────────
        dec_pw = pw[:, T:, :]                                   # (B, 5, H) — decoder portion
        cls_logits = torch.stack(
            [self.heads[h](dec_pw[:, self.horizon_dec_positions[h], :])
             for h in range(self.num_horizons)],
            dim=1,
        )                                                        # (B, 3, 7)

        # Store interpretability outputs for post-hoc analysis
        self._enc_vsn_weights = enc_wts.detach()    # (B, 30, 186)
        self._dec_vsn_weights = dec_wts.detach()    # (B, 5, 27)
        self._attn_weights    = attn_wts.detach()   # (B, 35, 35)

        if self.use_relative_head:
            rel_logits = torch.stack(
                [self.relative_heads[h](dec_pw[:, self.horizon_dec_positions[h], :])
                 for h in range(self.num_horizons)],
                dim=1,
            )
            return cls_logits, rel_logits

        return cls_logits


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_tft_model(
    config:         dict,
    num_sectors:    int = 0,
    num_industries: int = 0,
) -> TemporalFusionTransformer:
    return TemporalFusionTransformer(
        num_past_features    = NUM_OBSERVED_PAST_FEATURES,
        num_future_features  = NUM_KNOWN_FUTURE_FEATURES,
        num_classes          = NUM_CLASSES,
        num_horizons         = NUM_HORIZONS,
        hidden_dim           = config.get('tft_hidden',      160),
        num_heads            = config.get('tft_heads',         4),
        lstm_layers          = config.get('tft_lstm_layers',   2),
        dropout              = config.get('tft_dropout',      0.1),
        num_sectors          = num_sectors,
        num_industries       = num_industries,
        use_relative_head    = config.get('use_relative_head', False),
        num_relative_classes = NUM_RELATIVE_CLASSES,
    )
