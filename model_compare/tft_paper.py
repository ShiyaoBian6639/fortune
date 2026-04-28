"""
Faithful Temporal Fusion Transformer (Lim et al. 2019) — multi-horizon, no
modifications. No cross-stock attention, no extended feature merge — just the
paper architecture as specified.

Reference:
  Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister.
  "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
  Forecasting."  Int. J. of Forecasting 37(4), 2021. (preprint 2019.12.19)

Inputs (per sample):
  static       : (B, S)        per-entity time-invariant features
                                  e.g. sector_id, in_csi300, years_listed
  past_obs     : (B, T, P)     past observed inputs (only known up to t)
                                  e.g. all price/volume/macro features
  past_known   : (B, T, K)     past known inputs (known then AND in future)
                                  e.g. dow / month / holiday flags
  future_known : (B, H, K)     future known inputs for the forecast horizon
                                  e.g. dow for t+1..t+H

Output:
  pred         : (B, H)        per-horizon point prediction (no quantile heads)

Architecture (Lim 2019 §4):
  1.  Static covariate encoders (4 GRNs):
        c_s : context for VSNs (variable-selection)
        c_e : context for static enrichment layer
        c_c : LSTM cell-state initialiser
        c_h : LSTM hidden-state initialiser
  2.  Three Variable Selection Networks (per-timestep):
        VSN_static   : static    → s_emb        (B, d_model)
        VSN_past_obs : past_obs  → past_obs_emb (B, T, d_model) — uses c_s
        VSN_past_kn  : past_kn   → past_kn_emb  (B, T, d_model) — uses c_s
        VSN_fut_kn   : future_kn → fut_kn_emb   (B, H, d_model) — uses c_s
  3.  Locality enhancement (Seq2Seq):
        Encoder LSTM input  = past_obs_emb + past_kn_emb  (sum of the two)
        Decoder LSTM input  = fut_kn_emb                  (future-side)
        LSTM hidden/cell state initialised from c_h, c_c
        Add gated-skip residual back to inputs:  φ̃(t) = LayerNorm( φ(t) + GLU(LSTM(φ)(t)) )
  4.  Static enrichment via GRN with c_e context (per timestep)
  5.  Multi-head temporal self-attention with causal mask:
        Q,K,V ← enriched(t∈past+future);  decoder positions can attend to
        past + own (and prior decoder) positions only.
  6.  Position-wise feed-forward GRN
  7.  Output: per-horizon Linear over the H decoder positions only:
        pred[i] = head(decoder_output[T+i])
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building blocks (Lim §4) ───────────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit:  GLU(x) = a(x) ⊙ σ(b(x))."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.a = nn.Linear(d_in, d_out)
        self.b = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.a(x) * torch.sigmoid(self.b(x))


class GRN(nn.Module):
    """Gated Residual Network — paper Eq. 3.

       η_2 = ELU(W_2 a + W_3 c + b_2)              # context optional
       η_1 = W_1 η_2 + b_1
       GRN(a, c) = LayerNorm( a + GLU(η_1) )
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 d_context: int = 0, dropout: float = 0.1):
        super().__init__()
        self.skip   = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1    = nn.Linear(d_in, d_hidden)
        self.ctx_fc = nn.Linear(d_context, d_hidden, bias=False) if d_context > 0 else None
        self.fc2    = nn.Linear(d_hidden, d_out)
        self.drop   = nn.Dropout(dropout)
        self.glu    = GLU(d_out, d_out)
        self.ln     = nn.LayerNorm(d_out)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        h = self.fc1(x)
        if self.ctx_fc is not None and c is not None:
            ctx = self.ctx_fc(c)
            # broadcast static context (B, d) over time-axis if x is 3-D
            if h.dim() == 3 and ctx.dim() == 2:
                ctx = ctx.unsqueeze(1)
            h = h + ctx
        h = F.elu(h)
        h = self.fc2(h)
        h = self.drop(h)
        h = self.glu(h)
        return self.ln(self.skip(x) + h)


class VariableSelectionNetwork(nn.Module):
    """Per-timestep variable selection — paper §4.1, FAST.

    Each scalar feature x_i goes through its own GRN → emb_i ∈ R^d_model.
    Implemented via grouped 1×1 convolutions (cuDNN) so all n_vars per-feature
    Linear layers run as a single kernel — ~10× faster than a Python `for i in
    range(n_vars)` loop and ~5× faster than einsum (avoids materialising the
    full B×T×n_vars×d_hidden intermediate tensor).
    """
    def __init__(self, n_vars: int, d_model: int,
                 d_context: int = 0, dropout: float = 0.1):
        super().__init__()
        self.n_vars  = n_vars
        self.d_model = d_model
        # Each "group" has 1 input channel → d_model output channels.
        # n_vars groups → total out_channels = n_vars * d_model.
        # kernel_size=1 means we treat the time axis as the conv "length".
        self.fc1   = nn.Conv1d(n_vars, n_vars * d_model, 1, groups=n_vars)
        self.fc2   = nn.Conv1d(n_vars * d_model, n_vars * d_model, 1, groups=n_vars)
        self.glu_a = nn.Conv1d(n_vars * d_model, n_vars * d_model, 1, groups=n_vars)
        self.glu_b = nn.Conv1d(n_vars * d_model, n_vars * d_model, 1, groups=n_vars)
        self.skip  = nn.Conv1d(n_vars, n_vars * d_model, 1, groups=n_vars)
        # Per-feature LayerNorm — single tensor of (n_vars, d_model) γ/β
        self.ln_g  = nn.Parameter(torch.ones(n_vars, d_model))
        self.ln_b  = nn.Parameter(torch.zeros(n_vars, d_model))
        self.dropout = nn.Dropout(dropout)

        # Weight GRN — operates on the concatenated raw inputs (n_vars long)
        self.weight_grn = GRN(n_vars, d_model, n_vars,
                                d_context=d_context, dropout=dropout)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        # x: (..., n_vars). Flatten leading dims → (B', n_vars, T) for Conv1d.
        original_shape = x.shape                                  # (..., n_vars)
        n_vars, d_model = self.n_vars, self.d_model
        if x.dim() == 2:    # static input (B, n_vars), no time dim
            x_t = x.unsqueeze(-1)                                 # (B, n_vars, 1)
        elif x.dim() == 3:  # time-varying (B, T, n_vars)
            x_t = x.transpose(1, 2)                               # (B, n_vars, T)
        else:
            raise ValueError(f"VSN expects 2-D or 3-D input, got {x.shape}")

        h = self.fc1(x_t)                                         # (B, n_vars*d_model, T)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        a = self.glu_a(h)
        b = self.glu_b(h)
        glu = a * torch.sigmoid(b)
        skip_v = self.skip(x_t)                                   # (B, n_vars*d_model, T)
        z = skip_v + glu                                          # (B, n_vars*d_model, T)

        # Reshape to (B, n_vars, d_model, T) for per-feature LayerNorm
        B, _, T = z.shape
        z = z.view(B, n_vars, d_model, T)
        # LayerNorm across d_model dim
        mu = z.mean(dim=2, keepdim=True)
        sd = z.std (dim=2, keepdim=True, unbiased=False).clamp_min(1e-5)
        z = (z - mu) / sd * self.ln_g.unsqueeze(0).unsqueeze(-1) \
                       + self.ln_b.unsqueeze(0).unsqueeze(-1)
        # → emb shape (B, T, n_vars, d_model) for downstream weighted sum
        emb = z.permute(0, 3, 1, 2)                               # (B, T, n_vars, d_model)
        if x.dim() == 2:
            emb = emb.squeeze(1)                                  # (B, n_vars, d_model)

        # Variable weights from concatenated raw input + optional context
        w_logits = self.weight_grn(x, c)                          # (..., n_vars)
        weights  = torch.softmax(w_logits, dim=-1)
        out = (weights.unsqueeze(-1) * emb).sum(dim=-2)           # (..., d_model)
        return out, weights


# ─── Faithful multi-horizon TFT ─────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 n_static: int,
                 n_past_obs: int,
                 n_past_known: int,
                 n_future_known: int,
                 T: int = 30,           # past length
                 H: int = 5,            # forecast horizon
                 d_model: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.H = H
        self.d_model = d_model
        self.n_static = n_static

        # ── 1. Static covariate encoders → 4 context vectors ──
        if n_static > 0:
            self.static_vsn = VariableSelectionNetwork(n_static, d_model,
                                                         d_context=0, dropout=dropout)
            # 4 GRNs: c_s (VSN context), c_e (enrichment), c_c (LSTM cell), c_h (LSTM hidden)
            self.grn_s = GRN(d_model, d_model, d_model, dropout=dropout)
            self.grn_e = GRN(d_model, d_model, d_model, dropout=dropout)
            self.grn_c = GRN(d_model, d_model, d_model, dropout=dropout)
            self.grn_h = GRN(d_model, d_model, d_model, dropout=dropout)
            d_ctx = d_model
        else:
            self.static_vsn = None
            d_ctx = 0

        # ── 2. Three time-varying VSNs (past_obs, past_known, future_known) ──
        self.past_obs_vsn = (VariableSelectionNetwork(n_past_obs, d_model,
                                                         d_context=d_ctx, dropout=dropout)
                              if n_past_obs > 0 else None)
        self.past_kn_vsn  = (VariableSelectionNetwork(n_past_known, d_model,
                                                         d_context=d_ctx, dropout=dropout)
                              if n_past_known > 0 else None)
        self.fut_kn_vsn   = (VariableSelectionNetwork(n_future_known, d_model,
                                                         d_context=d_ctx, dropout=dropout)
                              if n_future_known > 0 else None)

        # ── 3. Seq2Seq locality enhancement ──
        # Encoder LSTM (past), Decoder LSTM (future)
        self.encoder_lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.decoder_lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        # Gated skip-connection wrapping the LSTM output
        self.lstm_glu = GLU(d_model, d_model)
        self.lstm_ln  = nn.LayerNorm(d_model)

        # ── 4. Static enrichment layer ──
        self.enrich_grn = GRN(d_model, d_model, d_model,
                                d_context=d_ctx, dropout=dropout)

        # ── 5. Temporal self-attention ──
        self.attn      = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                                  dropout=dropout, batch_first=True)
        self.attn_glu  = GLU(d_model, d_model)
        self.attn_ln   = nn.LayerNorm(d_model)

        # ── 6. Position-wise feed-forward GRN ──
        self.ff_grn = GRN(d_model, d_model, d_model, dropout=dropout)
        self.ff_glu = GLU(d_model, d_model)
        self.ff_ln  = nn.LayerNorm(d_model)

        # ── 7. Output head — per-horizon linear over the H decoder positions ──
        self.out_head = nn.Linear(d_model, 1)

        # Causal mask: position i in the joined past||future sequence can
        # only attend to positions ≤ i. Past timesteps cannot peek at future.
        L = T + H
        mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, static, past_obs, past_known, future_known):
        """
        static       : (B, n_static)               or None
        past_obs     : (B, T, n_past_obs)          or None
        past_known   : (B, T, n_past_known)        or None
        future_known : (B, H, n_future_known)      or None
        Returns: pred (B, H)
        """
        B = (past_obs if past_obs is not None
             else past_known if past_known is not None
             else future_known).size(0)

        # ── Static encoders → 4 context vectors ──
        if self.static_vsn is not None and static is not None:
            s_emb, _ = self.static_vsn(static, None)   # (B, d_model)
            c_s = self.grn_s(s_emb)
            c_e = self.grn_e(s_emb)
            c_c = self.grn_c(s_emb)
            c_h = self.grn_h(s_emb)
        else:
            c_s = c_e = c_c = c_h = None

        # ── Past VSNs ──
        past_pieces = []
        if self.past_obs_vsn is not None and past_obs is not None:
            past_obs_emb, _ = self.past_obs_vsn(past_obs, c_s)
            past_pieces.append(past_obs_emb)
        if self.past_kn_vsn is not None and past_known is not None:
            past_kn_emb, _ = self.past_kn_vsn(past_known, c_s)
            past_pieces.append(past_kn_emb)
        # Combine past sources (sum) per Lim §4 if both present
        if not past_pieces:
            raise ValueError("TFT requires at least one past input tensor.")
        past_in = sum(past_pieces) / len(past_pieces)            # (B, T, d_model)

        # ── Future VSN ──
        if self.fut_kn_vsn is not None and future_known is not None:
            fut_in, _ = self.fut_kn_vsn(future_known, c_s)       # (B, H, d_model)
        else:
            # If no future-known features, use a learned per-step bias broadcast.
            # The decoder still needs an input — we use zeros (same shape).
            fut_in = past_in.new_zeros(B, self.H, self.d_model)

        # ── Seq2Seq encoder (past) ──
        if c_h is not None:
            h0 = c_h.unsqueeze(0)                                 # (1, B, d_model)
            c0 = c_c.unsqueeze(0)
            past_lstm, (enc_h, enc_c) = self.encoder_lstm(past_in, (h0, c0))
        else:
            past_lstm, (enc_h, enc_c) = self.encoder_lstm(past_in)

        # ── Decoder (future) initialised by encoder's final state ──
        fut_lstm, _ = self.decoder_lstm(fut_in, (enc_h, enc_c))

        # ── Gated skip back to original VSN inputs ──
        joined_lstm = torch.cat([past_lstm, fut_lstm], dim=1)     # (B, T+H, d_model)
        joined_in   = torch.cat([past_in,   fut_in  ], dim=1)
        gated       = self.lstm_glu(joined_lstm)
        joined      = self.lstm_ln(gated + joined_in)

        # ── Static enrichment ──
        enriched = self.enrich_grn(joined, c_e)                   # (B, T+H, d_model)

        # ── Temporal self-attention with causal mask ──
        attn_out, _ = self.attn(enriched, enriched, enriched,
                                  attn_mask=self.causal_mask, need_weights=False)
        attn_out = self.attn_ln(self.attn_glu(attn_out) + enriched)

        # ── Position-wise FFN GRN ──
        ff = self.ff_grn(attn_out)
        ff = self.ff_ln(self.ff_glu(ff) + attn_out)

        # ── Output head — only the H future positions ──
        future_h = ff[:, self.T:, :]                              # (B, H, d_model)
        pred     = self.out_head(future_h).squeeze(-1)            # (B, H)
        return pred


def pseudo_huber(pred, target, slope=1.0):
    z = (pred - target) / slope
    return (slope ** 2 * (torch.sqrt(1 + z * z) - 1)).mean()
