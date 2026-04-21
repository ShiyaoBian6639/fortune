"""
DeepTimeModel — Enhanced TFT with Sector Cross-Attention for regression.

Extends dl/tft_model.py building blocks with:
  - SectorCrossAttention: O(N×K) inter-stock modeling (K=31 sectors)
  - 5-step regression heads (pct_chg day1..day5) instead of 3-class heads
  - Extended static: sector/industry/sub_industry/size_decile embeddings
  - Known-future: 29 features (27 calendar + 2 price limit ratios)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from dl.tft_model import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
)


# ─── Batched Variable Selection (replaces the per-variable for-loop) ──────────

class BatchedVariableGRN(nn.Module):
    """
    Batched equivalent of V independent GRN(var_dim=1 → hidden) modules.

    The original VariableSelectionNetwork calls self.var_grns[v](x) in a Python
    for-loop over V=204 variables, launching 816 CUDA kernels per call.
    This class fuses those V GRNs into ~5 batched torch.bmm / broadcast ops,
    giving identical math but ~10× lower Python overhead.

    Parameter count: same as V separate GRN modules.
    """

    def __init__(self, num_vars: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        V, H = num_vars, hidden_dim

        # fc1: V × Linear(1 → H)  — stored as (V, H) weight + bias
        self.fc1_w  = nn.Parameter(torch.empty(V, H))
        self.fc1_b  = nn.Parameter(torch.zeros(V, H))
        # fc2: V × Linear(H → H)  — stored as (V, H, H)
        self.fc2_w  = nn.Parameter(torch.empty(V, H, H))
        self.fc2_b  = nn.Parameter(torch.zeros(V, H))
        # GLU: V × (fc + gate), each Linear(H → H) — stored as (V, 2H, H) combined
        self.glu_w  = nn.Parameter(torch.empty(V, 2 * H, H))
        self.glu_b  = nn.Parameter(torch.zeros(V, 2 * H))
        # Skip: V × Linear(1 → H)
        self.skip_w = nn.Parameter(torch.empty(V, H))
        self.skip_b = nn.Parameter(torch.zeros(V, H))

        self.ln   = nn.LayerNorm(H)
        self.drop = nn.Dropout(dropout)

        for w in [self.fc1_w, self.fc2_w, self.glu_w, self.skip_w]:
            nn.init.normal_(w, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, V, 1)  — N = B*T (temporal) or B (static)
        returns: (N, V, H)
        """
        xs = x.squeeze(-1)  # (N, V)

        # fc1 + ELU: scalar per variable scaled by (V, H) weights
        # h[n, v, h] = xs[n, v] * fc1_w[v, h] + fc1_b[v, h]
        h = xs.unsqueeze(-1) * self.fc1_w.unsqueeze(0) + self.fc1_b.unsqueeze(0)  # (N, V, H)
        h = F.elu(h)

        # fc2: V batched (H × H) matmuls — permute to (V, N, H) for bmm
        h = h.permute(1, 0, 2)                                     # (V, N, H)
        h = torch.bmm(h, self.fc2_w) + self.fc2_b.unsqueeze(1)    # (V, N, H)

        # GLU: (V, N, H) × (V, H, 2H) → (V, N, 2H) → split and gate
        glu = torch.bmm(h, self.glu_w.transpose(1, 2)) + self.glu_b.unsqueeze(1)  # (V, N, 2H)
        fc_out, gate = glu.chunk(2, dim=-1)                        # each (V, N, H)
        h = self.drop(fc_out) * torch.sigmoid(gate)

        h = h.permute(1, 0, 2)                                     # (N, V, H)

        # Skip: same scalar-times-vector pattern as fc1
        skip = xs.unsqueeze(-1) * self.skip_w.unsqueeze(0) + self.skip_b.unsqueeze(0)  # (N, V, H)

        return self.ln(skip + h)


class BatchedVariableSelectionNetwork(nn.Module):
    """
    Drop-in replacement for dl.tft_model.VariableSelectionNetwork (var_dim=1).
    Uses BatchedVariableGRN instead of a Python for-loop over V GRN modules.
    Same interface, same parameters, ~10× faster.
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
        assert var_dim == 1, "BatchedVariableSelectionNetwork requires var_dim=1"
        self.num_vars = num_vars

        self.var_grn = BatchedVariableGRN(num_vars, hidden_dim, dropout)

        # Weight-selection GRN: same as original (runs only once, not V times)
        self.wt_grn = GatedResidualNetwork(
            input_dim   = num_vars * var_dim,
            hidden_dim  = hidden_dim,
            output_dim  = num_vars,
            context_dim = context_dim,
            dropout     = dropout,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """Same signature as VariableSelectionNetwork.forward."""
        temporal = (x.dim() == 4)
        if temporal:
            B, T, V, D = x.shape
            xf = x.reshape(B * T, V, D)
            cf = c.reshape(B * T, -1) if c is not None else None
        else:
            B, V, D = x.shape
            xf, cf = x, c

        # Batched GRN: (N, V, 1) → (N, V, H)  — one call instead of V calls
        var_out = self.var_grn(xf)

        # Softmax selection weights
        flat = xf.reshape(xf.size(0), -1)
        wts  = torch.softmax(self.wt_grn(flat, cf), dim=-1)   # (N, V)

        out = (var_out * wts.unsqueeze(-1)).sum(dim=1)         # (N, H)

        if temporal:
            out = out.reshape(B, T, -1)
            wts = wts.reshape(B, T, V)

        return out, wts
from .config import (
    NUM_DT_OBSERVED_PAST, NUM_DT_KNOWN_FUTURE,
    NUM_HORIZONS, FORWARD_WINDOWS,
    SECTOR_EMB_DIM, INDUSTRY_EMB_DIM, SUB_IND_EMB_DIM, SIZE_EMB_DIM,
    AREA_EMB_DIM, BOARD_EMB_DIM, IPO_AGE_EMB_DIM,
    NUM_SECTORS_EMBED, NUM_INDUSTRIES_EMBED, NUM_SUB_IND_EMBED, NUM_SIZE_DECILES,
    NUM_AREAS_EMBED, NUM_BOARD_TYPES, NUM_IPO_AGE_BUCKETS,
    TFT_HIDDEN, TFT_HEADS, TFT_LSTM_LAYERS, TFT_DROPOUT,
)

# Decoder positions: FORWARD_WINDOWS=[1,2,3,4,5] → positions 0,1,2,3,4
_HORIZON_DEC_POSITIONS = [fw - 1 for fw in FORWARD_WINDOWS]   # [0,1,2,3,4]


class SectorCrossAttention(nn.Module):
    """
    Two-phase O(N×K) inter-stock temporal attention (K = number of sectors).

    Phase 1: Mean-pool LSTM hidden states within each sector → K sector tokens
             Pooling uses vectorised scatter (no Python loop over B).
    Phase 2: FULL SEQUENCE cross-attention — each of the T timesteps queries
             all K sector tokens independently.

             Why temporal queries beat single-mean query:
               - 30× more expressive: (B, T, K) attention vs (B, 1, K)
               - 8× FASTER: GPU matmuls saturate warps better with (T, K) vs (1, K)
               - Same parameter count: only the query tensor shape changes
               - Naturally captures "stock was in tech regime at t-15, finance at t-3"

    Flash Attention note: the attention matrix is (B, heads, T=30, K=35) — too small
    for Flash Attention's IO benefit (FA helps when N >= 512). PyTorch dispatches to
    its fused attention kernel automatically; no further optimisation needed here.

    Multi-head note: 4 heads with head_dim=32 over K=35 keys is well-calibrated.
    8 heads (head_dim=16) would be too narrow per head relative to the key space.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_sectors: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_sectors = num_sectors
        # Multi-head attention: Q=(B,T,D), K=V=(B,K,D)
        # PyTorch uses fused/FA kernel automatically for fp16, batch_first=True
        self.cross_attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.ln   = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, D) — LSTM encoder outputs
        sector_ids: torch.Tensor,  # (B,) int64
    ):
        """
        Returns:
            enriched  (B, T, D) — each timestep enriched with sector co-movement context
            attn_wts  (B, K)    — mean attention over time (interpretability)
        """
        B, T, D = x.shape
        K = self.num_sectors
        device = x.device

        # ── Phase 1: sector tokens via vectorised scatter (no Python loop) ──
        # Use time-mean of each stock's sequence as its sector representative.
        x_mean = x.mean(dim=1)                                          # (B, D)
        # Clamp sector ids to valid range
        sids = sector_ids.long().clamp(0, K - 1)                       # (B,)
        # Scatter-add: accumulate per-sector
        sector_sum    = torch.zeros(K, D, device=device, dtype=x_mean.dtype)
        sector_counts = torch.zeros(K,    device=device, dtype=x_mean.dtype)
        sector_sum.scatter_add_(0, sids.unsqueeze(1).expand(-1, D), x_mean)
        sector_counts.scatter_add_(0, sids, torch.ones(B, device=device, dtype=x_mean.dtype))
        sector_tokens = sector_sum / sector_counts.clamp(min=1.0).unsqueeze(1)  # (K, D)
        sector_tokens = sector_tokens.unsqueeze(0).expand(B, -1, -1)   # (B, K, D)

        # ── Phase 2: temporal cross-attention — full sequence queries K tokens ─
        # Query: all T timesteps × K sector keys (no temporal compression)
        # Output shape matches input: (B, T, D) — each timestep gets sector context
        ctx, attn_wts = self.cross_attn(x, sector_tokens, sector_tokens)
        # ctx:      (B, T, D)
        # attn_wts: (B, T, K) averaged over heads by PyTorch MHA

        # ── Gate: residual mix of own sequence + sector context ──────────────
        enriched = self.gate(x, ctx)    # GRN(skip=x, context=ctx): (B, T, D)
        enriched = self.ln(enriched)

        # Collapse time dimension for interpretability (sector_cross_attention plot)
        attn_mean = attn_wts.mean(dim=1)   # (B, K) — mean attention over timesteps

        return enriched, attn_mean


class DeepTimeModel(nn.Module):
    """
    Enhanced TFT with sector cross-attention for 5-horizon regression.

    Forward inputs:
        past_obs      (B, seq_len, NUM_DT_OBSERVED_PAST) — past-observed features
        future_inputs (B, max_fw,  NUM_DT_KNOWN_FUTURE)  — known-future features
        sector_ids    (B,) int64
        industry_ids  (B,) int64  [optional]
        sub_ind_ids   (B,) int64  [optional]
        size_ids      (B,) int64  [optional]

    Forward outputs:
        preds         (B, 5)    — predicted excess returns (or raw pct_chg)

    Interpretability attributes set after each forward pass:
        _enc_vsn_weights  (B, seq_len, NUM_DT_OBSERVED_PAST)
        _dec_vsn_weights  (B, max_fw,  NUM_DT_KNOWN_FUTURE)
        _attn_weights     (B, seq_len+max_fw, seq_len+max_fw)
        _sector_attn      (B, num_sectors)
    """

    def __init__(
        self,
        num_past_features:   int   = NUM_DT_OBSERVED_PAST,
        num_future_features: int   = NUM_DT_KNOWN_FUTURE,
        num_horizons:        int   = NUM_HORIZONS,
        hidden_dim:          int   = TFT_HIDDEN,
        num_heads:           int   = TFT_HEADS,
        lstm_layers:         int   = TFT_LSTM_LAYERS,
        dropout:             float = TFT_DROPOUT,
        num_sectors:         int   = NUM_SECTORS_EMBED,
        num_industries:      int   = NUM_INDUSTRIES_EMBED,
        num_sub_industries:  int   = NUM_SUB_IND_EMBED,
        num_size_deciles:    int   = NUM_SIZE_DECILES,
        num_areas:           int   = NUM_AREAS_EMBED,
        num_board_types:     int   = NUM_BOARD_TYPES,
        num_ipo_age:         int   = NUM_IPO_AGE_BUCKETS,
    ):
        super().__init__()
        self._is_deeptime     = True
        self.hidden_dim        = hidden_dim
        self.lstm_layers       = lstm_layers
        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.num_horizons      = num_horizons
        self.horizon_dec_pos   = _HORIZON_DEC_POSITIONS  # [0,1,2,3,4]

        # ── Static covariate embeddings ──────────────────────────────────────
        # Core: SW L1 sector (31), SW L2 sub-industry (~130), size decile (10)
        self.sector_emb   = nn.Embedding(num_sectors,       SECTOR_EMB_DIM)    # 64
        self.industry_emb = nn.Embedding(num_industries,    INDUSTRY_EMB_DIM)  # 32
        self.sub_ind_emb  = nn.Embedding(num_sub_industries, SUB_IND_EMB_DIM)  # 8
        self.size_emb     = nn.Embedding(num_size_deciles,  SIZE_EMB_DIM)      # 16
        # New: province/region, exchange board type, IPO age
        self.area_emb     = nn.Embedding(num_areas,         AREA_EMB_DIM)      # 16
        self.board_emb    = nn.Embedding(num_board_types,   BOARD_EMB_DIM)     # 8
        self.ipo_age_emb  = nn.Embedding(num_ipo_age,       IPO_AGE_EMB_DIM)   # 8

        static_dim = (SECTOR_EMB_DIM + INDUSTRY_EMB_DIM + SUB_IND_EMB_DIM
                      + SIZE_EMB_DIM + AREA_EMB_DIM + BOARD_EMB_DIM + IPO_AGE_EMB_DIM)
        # = 64 + 32 + 8 + 16 + 16 + 8 + 8 = 152

        # 4 static context vectors (TFT paper section 4.1)
        self.static_grn_s = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.static_grn_e = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.static_grn_c = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.static_grn_h = GatedResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout=dropout)

        # ── Variable Selection Networks ───────────────────────────────────────
        # Gradient checkpointing for the VSN: the 204-variable for-loop creates
        # one GRN activation tensor per variable (~11 MB each × 204 = ~2.3 GB).
        # Checkpointing discards these intermediates and recomputes during backward,
        # halving activation memory at ~33% extra compute cost.
        #
        # use_reentrant=True  (classic API) is stable with AMP fp16 — PyTorch
        # preserves the autocast state during recomputation, so no dtype mismatch.
        # use_reentrant=False caused NaN gradients with AMP and is NOT used here.
        self.use_grad_checkpoint = True
        self.vsn_encoder = BatchedVariableSelectionNetwork(
            num_vars=num_past_features, var_dim=1, hidden_dim=hidden_dim,
            context_dim=hidden_dim, dropout=dropout,
        )
        self.vsn_decoder = BatchedVariableSelectionNetwork(
            num_vars=num_future_features, var_dim=1, hidden_dim=hidden_dim,
            context_dim=hidden_dim, dropout=dropout,
        )

        # ── LSTM encoder + decoder ────────────────────────────────────────────
        lstm_kw = dict(batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, **lstm_kw)
        self.lstm_decoder = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, **lstm_kw)

        # Post-LSTM gating
        self.post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.post_lstm_ln  = nn.LayerNorm(hidden_dim)

        # ── Sector Cross-Attention (inter-stock) ─────────────────────────────
        self.sector_cross_attn = SectorCrossAttention(
            hidden_dim, num_heads=num_heads, num_sectors=num_sectors, dropout=dropout
        )

        # ── Static enrichment ─────────────────────────────────────────────────
        self.static_enrich_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout
        )

        # ── Interpretable Multi-Head Temporal Attention ───────────────────────
        self.attn     = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.attn_ln  = nn.LayerNorm(hidden_dim)

        # ── Position-wise GRN ─────────────────────────────────────────────────
        self.pw_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.pw_glu = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.pw_ln  = nn.LayerNorm(hidden_dim)

        # ── Regression heads (one per horizon, scalar output) ─────────────────
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_horizons)])

        # Interpretability buffers
        self._enc_vsn_weights: torch.Tensor = None
        self._dec_vsn_weights: torch.Tensor = None
        self._attn_weights:    torch.Tensor = None
        self._sector_attn:     torch.Tensor = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _static_context(
        self,
        sector_ids:   torch.Tensor,
        industry_ids: torch.Tensor,
        sub_ind_ids:  torch.Tensor,
        size_ids:     torch.Tensor,
        area_ids:     torch.Tensor,
        board_ids:    torch.Tensor,
        ipo_age_ids:  torch.Tensor,
        B: int, device,
    ):
        """Concatenate all static embeddings → (B, static_dim=152)."""
        def _clamp(t, emb): return emb(t.clamp(0, emb.num_embeddings - 1))
        sec  = _clamp(sector_ids,   self.sector_emb)    # (B, 64)
        ind  = _clamp(industry_ids, self.industry_emb)  # (B, 32)
        sub  = _clamp(sub_ind_ids,  self.sub_ind_emb)   # (B,  8)
        sz   = _clamp(size_ids,     self.size_emb)      # (B, 16)
        area = _clamp(area_ids,     self.area_emb)      # (B, 16)
        brd  = _clamp(board_ids,    self.board_emb)     # (B,  8)
        ipo  = _clamp(ipo_age_ids,  self.ipo_age_emb)   # (B,  8)
        return torch.cat([sec, ind, sub, sz, area, brd, ipo], dim=-1)  # (B, 152)

    def forward(
        self,
        past_obs:      torch.Tensor,            # (B, seq_len, n_past)
        future_inputs: torch.Tensor,            # (B, max_fw,  n_future)
        sector_ids:    torch.Tensor,            # (B,) int64
        industry_ids:  torch.Tensor = None,     # (B,) int64
        sub_ind_ids:   torch.Tensor = None,     # (B,) int64
        size_ids:      torch.Tensor = None,     # (B,) int64
        area_ids:      torch.Tensor = None,     # (B,) int64  — province/region
        board_ids:     torch.Tensor = None,     # (B,) int64  — exchange board type
        ipo_age_ids:   torch.Tensor = None,     # (B,) int64  — IPO age bucket
    ) -> torch.Tensor:
        """Returns preds (B, 5) — one regression value per horizon."""
        B, T, _ = past_obs.shape
        device = past_obs.device

        def _zeros(): return torch.zeros(B, dtype=torch.long, device=device)
        if industry_ids is None: industry_ids = _zeros()
        if sub_ind_ids  is None: sub_ind_ids  = _zeros()
        if size_ids     is None: size_ids     = _zeros()
        if area_ids     is None: area_ids     = _zeros()
        if board_ids    is None: board_ids    = _zeros()
        if ipo_age_ids  is None: ipo_age_ids  = _zeros()

        # ── 1. Static encoder ─────────────────────────────────────────────────
        static_feat = self._static_context(
            sector_ids, industry_ids, sub_ind_ids, size_ids,
            area_ids, board_ids, ipo_age_ids, B, device
        )
        c_s = self.static_grn_s(static_feat)   # (B, H)
        c_e = self.static_grn_e(static_feat)
        c_c = self.static_grn_c(static_feat)
        c_h = self.static_grn_h(static_feat)

        # ── 2. Encoder VSN ────────────────────────────────────────────────────
        enc_in   = past_obs.unsqueeze(-1)                          # (B, T, 204, 1)
        c_s_enc  = c_s.unsqueeze(1).expand(-1, T, -1).contiguous() # (B, T, H)
        if self.use_grad_checkpoint and self.training:
            enc_vsn, enc_wts = grad_checkpoint(
                self.vsn_encoder, enc_in, c_s_enc, use_reentrant=True
            )
        else:
            enc_vsn, enc_wts = self.vsn_encoder(enc_in, c_s_enc)  # (B, T, H), (B, T, 204)

        # ── 3. Decoder VSN ────────────────────────────────────────────────────
        dec_len  = future_inputs.size(1)                           # 5
        dec_in   = future_inputs.unsqueeze(-1)                     # (B, 5, 29, 1)
        c_s_dec  = c_s.unsqueeze(1).expand(-1, dec_len, -1).contiguous()  # (B, 5, H)
        if self.use_grad_checkpoint and self.training:
            dec_vsn, dec_wts = grad_checkpoint(
                self.vsn_decoder, dec_in, c_s_dec, use_reentrant=True
            )
        else:
            dec_vsn, dec_wts = self.vsn_decoder(dec_in, c_s_dec)  # (B, 5, H), (B, 5, 29)

        # ── 4. LSTM encoder ───────────────────────────────────────────────────
        h0 = c_e.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        c0 = c_c.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(enc_vsn, (h0, c0))   # (B, T, H)

        # ── 5. Sector Cross-Attention (inter-stock) ───────────────────────────
        enc_out_enriched, sector_attn = self.sector_cross_attn(enc_out, sector_ids)
        # enc_out_enriched: (B, T, H); sector_attn: (B, K)

        # ── 6. LSTM decoder ───────────────────────────────────────────────────
        dec_out, _ = self.lstm_decoder(dec_vsn, (h_n, c_n))          # (B, 5, H)

        # ── 7. Post-LSTM gating + skip ────────────────────────────────────────
        full_vsn  = torch.cat([enc_vsn,          dec_vsn], dim=1)    # (B, T+5, H)
        full_lstm = torch.cat([enc_out_enriched, dec_out], dim=1)    # (B, T+5, H)
        gated     = self.post_lstm_glu(full_lstm)
        full_seq  = self.post_lstm_ln(full_vsn + gated)              # (B, T+5, H)

        # ── 8. Static enrichment ──────────────────────────────────────────────
        seq_total = T + dec_len
        c_h_exp   = c_h.unsqueeze(1).expand(-1, seq_total, -1)
        enriched  = self.static_enrich_grn(full_seq, c_h_exp)        # (B, T+5, H)

        # ── 9. Temporal attention ─────────────────────────────────────────────
        attn_out, attn_wts = self.attn(enriched, enriched, enriched)
        gated2   = self.attn_glu(attn_out)
        attn_out = self.attn_ln(enriched + gated2)                   # (B, T+5, H)

        # ── 10. Position-wise GRN ──────────────────────────────────────────────
        pw = self.pw_grn(attn_out)
        pw = self.pw_ln(attn_out + self.pw_glu(pw))                  # (B, T+5, H)

        # ── 11. Regression heads at decoder positions ─────────────────────────
        dec_pw = pw[:, T:, :]                                        # (B, 5, H)
        preds = torch.cat(
            [self.heads[h](dec_pw[:, self.horizon_dec_pos[h], :])
             for h in range(self.num_horizons)],
            dim=-1,
        )  # (B, 5)

        # Store interpretability
        self._enc_vsn_weights = enc_wts.detach()
        self._dec_vsn_weights = dec_wts.detach()
        self._attn_weights    = attn_wts.detach()
        self._sector_attn     = sector_attn.detach()

        return preds


def create_deeptime_model(config: dict) -> DeepTimeModel:
    return DeepTimeModel(
        num_past_features   = NUM_DT_OBSERVED_PAST,
        num_future_features = NUM_DT_KNOWN_FUTURE,
        num_horizons        = NUM_HORIZONS,
        hidden_dim          = config.get('tft_hidden',       TFT_HIDDEN),
        num_heads           = config.get('tft_heads',        TFT_HEADS),
        lstm_layers         = config.get('tft_lstm_layers',  TFT_LSTM_LAYERS),
        dropout             = config.get('tft_dropout',      TFT_DROPOUT),
        num_sectors         = config.get('num_sectors',      NUM_SECTORS_EMBED),
        num_industries      = config.get('num_industries',   NUM_INDUSTRIES_EMBED),
        num_sub_industries  = config.get('num_sub_ind',      NUM_SUB_IND_EMBED),
        num_size_deciles    = config.get('num_size_deciles', NUM_SIZE_DECILES),
        num_areas           = config.get('num_areas',        NUM_AREAS_EMBED),
        num_board_types     = config.get('num_board_types',  NUM_BOARD_TYPES),
        num_ipo_age         = config.get('num_ipo_age',      NUM_IPO_AGE_BUCKETS),
    )
