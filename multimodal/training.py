"""
Two-phase training loop for the MultimodalStockTransformer.

Phase 1 — frozen BERT (static cache):
  Only MultimodalStockTransformer parameters are trained.
  MacBERTEncoder is not part of the model's parameter tree, so it is
  automatically excluded from the optimiser — no special handling needed.

Phase 2 — unfrozen BERT (cache-rebuild cycle):
  After Phase 1, rebuild the news embedding cache with the now-unfrozen
  BERT (run build_daily_news_cache again), then call train_phase2() which
  adds BERT's parameters as a separate group with its own (lower) LR.

Training utilities reused from dl/:
  - dl.training.compute_metrics  (accuracy / precision / recall / F1)
  - dl.training.set_seed
  - dl.losses.create_loss_function (FocalLoss)
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl.training import compute_metrics, set_seed
from dl.losses import create_loss_function

from .config import MM_NUM_CLASSES, MM_CLASS_NAMES
from .models import MultimodalStockTransformer
from .text_encoder import MacBERTEncoder


# ─── Single epoch ─────────────────────────────────────────────────────────────

def _train_epoch(
    model:     MultimodalStockTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    str,
    use_amp:   bool = False,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    device_type = device.split(':')[0]

    for price_seq, news_seq, labels in loader:
        price_seq = price_seq.to(device, non_blocking=True)
        news_seq  = news_seq.to(device,  non_blocking=True)
        labels    = labels.to(device,    non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=use_amp):
            logits = model(price_seq, news_seq)
            loss   = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(logits, 1)
        total   += labels.size(0)
        correct += (pred == labels).sum().item()

    return total_loss / len(loader), correct / total


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_multimodal(
    model:     MultimodalStockTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
    use_amp:   bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model.

    Returns: (avg_loss, predictions, labels, probabilities)
    """
    model.eval()
    total_loss   = 0.0
    all_preds    = []
    all_labels   = []
    all_probs    = []
    device_type  = device.split(':')[0]

    with torch.no_grad():
        for price_seq, news_seq, labels in loader:
            price_seq = price_seq.to(device, non_blocking=True)
            news_seq  = news_seq.to(device,  non_blocking=True)
            labels    = labels.to(device,    non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits = model(price_seq, news_seq)
                loss   = criterion(logits, labels)

            probs = torch.softmax(logits.float(), dim=1)
            _, pred = torch.max(logits, 1)

            total_loss  += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return (
        avg_loss,
        np.array(all_preds),
        np.array(all_labels),
        np.vstack(all_probs),
    )


# ─── Phase 1 ──────────────────────────────────────────────────────────────────

def train_phase1(
    model:        MultimodalStockTransformer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion:    nn.Module,
    config:       dict,
    device:       str,
) -> Tuple[MultimodalStockTransformer, Dict]:
    """
    Phase 1: train with frozen BERT (static embeddings from cache).

    MacBERTEncoder is NOT passed in — its parameters are never part of
    the optimiser here, so BERT gradients are never computed.
    """
    use_amp   = config.get('use_amp', True) and device.startswith('cuda')
    epochs    = config.get('phase1_epochs', 10)
    lr        = config.get('phase1_lr',    5e-4)
    patience  = config.get('phase1_patience', 5)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    best_val_f1     = -1.0
    best_state      = None
    patience_ctr    = 0
    history: Dict   = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    print("\n" + "─" * 50)
    print("Phase 1 — frozen BERT")
    print("─" * 50)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_loss, preds, labels, _ = evaluate_multimodal(model, val_loader, criterion, device, use_amp)
        val_metrics = compute_metrics(labels, preds)
        val_acc = val_metrics['accuracy']
        val_f1  = val_metrics['f1']

        # LR scheduler still tracks loss (convex, always meaningful).
        # Best-model selection uses F1: focal loss with class weights doesn't
        # correlate monotonically with accuracy, so loss-based selection picks
        # the wrong checkpoint when the two metrics diverge.
        scheduler.step(val_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        marker = ' *' if val_f1 > best_val_f1 else ''
        elapsed = time.perf_counter() - t0
        print(
            f"  epoch {epoch+1:3d}/{epochs}  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  {elapsed:.1f}s{marker}"
        )

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [early stop] No F1 improvement for {patience} epochs.")
                break

    print(f"  [phase 1] Best val_f1={best_val_f1:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─── Phase 2 ──────────────────────────────────────────────────────────────────

def _encode_news_inline(
    bert_encoder: MacBERTEncoder,
    input_ids_win:  torch.Tensor,   # (B, W, A, L)
    attn_mask_win:  torch.Tensor,   # (B, W, A, L)
    n_articles_win: torch.Tensor,   # (B, W)  int
    device: str,
) -> torch.Tensor:                  # (B, W, 768)
    """
    Run BERT inline over a batched news window and mean-pool valid articles.

    Flattens the batch × window × article dimensions, runs one BERT forward,
    then mean-pools each (window-day, batch) slot over its valid articles.
    Zero-padded article slots (beyond n_articles for that day) are excluded.

    Short-circuit: if the entire batch has no valid articles (n_articles_win
    all-zero), return a zero tensor without calling BERT at all.  This avoids
    a ~10× slowdown caused by BERT + gradient-checkpointing processing
    all-zero attention masks (which hits a degenerate attention code path).
    This happens in smoke tests with tiny news caches and can also occur in
    production for batches that fall entirely outside the news coverage window.
    """
    B, W, A, L = input_ids_win.shape

    # Fast path: no news in this batch at all
    if n_articles_win.sum().item() == 0:
        return torch.zeros(B, W, 768, dtype=torch.float32, device=device)

    # Build validity mask before flattening so we can pack only real articles.
    # art_idx < n_articles_win[b,w]  →  True for valid slots, False for padding.
    art_idx = torch.arange(A, device=device).view(1, 1, A)
    valid   = art_idx < n_articles_win.to(device).unsqueeze(-1)   # (B, W, A) bool

    flat_ids   = input_ids_win.reshape(B * W * A, L).to(device)
    flat_mask  = attn_mask_win.reshape(B * W * A, L).to(device)
    flat_valid = valid.reshape(B * W * A)                          # (B*W*A,) bool

    # Pack: run BERT only on valid article slots (skip all-padding slots).
    # With A=16 articles and typical fill-rate of 1–8 real articles per day,
    # this is 2–16× fewer BERT sequences → proportional wall-clock speedup.
    valid_idx   = flat_valid.nonzero(as_tuple=False).squeeze(1)  # (n_valid,)
    packed_ids  = flat_ids[valid_idx]     # (n_valid, L)
    packed_mask = flat_mask[valid_idx]    # (n_valid, L)
    packed_cls  = bert_encoder(packed_ids, packed_mask)  # (n_valid, 768)

    # Scatter back via non-in-place index_put so gradients flow to packed_cls.
    # In-place cls_flat[mask]=... on a zeros tensor would break autograd because
    # the zeros leaf has requires_grad=False.
    cls_zeros = torch.zeros(B * W * A, 768, dtype=packed_cls.dtype, device=device)
    cls_flat  = cls_zeros.index_put((valid_idx,), packed_cls)   # new tensor, grad-tracked
    cls_all   = cls_flat.reshape(B, W, A, 768)                  # (B, W, A, 768)

    # Mean-pool valid articles per (batch, window-day) slot
    valid_f  = valid.float().unsqueeze(-1)                 # (B, W, A, 1)
    news_sum = (cls_all * valid_f).sum(dim=2)              # (B, W, 768)
    n_valid  = valid_f.sum(dim=2).clamp(min=1.0)           # (B, W, 1)
    return (news_sum / n_valid).float()                    # (B, W, 768)


def _evaluate_phase2(
    model:        MultimodalStockTransformer,
    bert_encoder: MacBERTEncoder,
    loader:       DataLoader,
    criterion:    nn.Module,
    device:       str,
    use_amp:      bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate using inline BERT (Phase2Dataset batches)."""
    model.eval()
    bert_encoder.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    all_probs   = []
    device_type = device.split(':')[0]

    # Make eval visible: without this, val can take longer than train and
    # the user just sees the screen go quiet.
    pbar = tqdm(loader, desc='  p2 eval', leave=False, unit='step', dynamic_ncols=True)
    with torch.no_grad():
        for price_seq, ids_win, masks_win, n_arts, labels in pbar:
            price_seq = price_seq.to(device, non_blocking=True)
            labels    = labels.to(device,    non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_amp):
                news_seq = _encode_news_inline(bert_encoder, ids_win, masks_win, n_arts, device)
                logits   = model(price_seq, news_seq)
                loss     = criterion(logits, labels)

            probs = torch.softmax(logits.float(), dim=1)
            _, pred = torch.max(logits, 1)
            total_loss  += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    pbar.close()

    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_labels),
        np.vstack(all_probs),
    )


def train_phase2(
    model:        MultimodalStockTransformer,
    bert_encoder: MacBERTEncoder,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion:    nn.Module,
    config:       dict,
    device:       str,
) -> Tuple[MultimodalStockTransformer, MacBERTEncoder, Dict]:
    """
    Phase 2: full fine-tune with BERT called inline per-batch.

    ``train_loader`` and ``val_loader`` must come from ``create_phase2_dataloaders()``
    (Phase2Dataset), which serves raw token tensors rather than pre-computed
    embeddings.  BERT is run inside the training loop so its gradients are real
    and its weights actually update.

    Memory management
    -----------------
    Running BERT inline stores activations for every (batch × window × article) sequence
    until backward completes — on a 12GB GPU that overflows quickly.  Two strategies
    are applied together:

    1. Gradient checkpointing on BERT: activations are recomputed during backward
       instead of stored (~12× memory reduction for BERT).  Enabled here and disabled
       after training so Phase 1 inference is not affected.

    2. Gradient accumulation (phase2_accum_steps, default 2): uses a small micro-batch
       (phase2_batch_size=16) so only 16×3×16=768 BERT sequences live at once, while
       the effective gradient batch stays at 16×2=32.

    Mixed precision
    ---------------
    bfloat16 autocast is used when use_amp=True.  Unlike float16, bfloat16 has the same
    8-exponent-bit dynamic range as float32 — no overflow, no GradScaler needed.  The
    RTX 4070 Super supports bfloat16 at the same ~41 TFLOPS as float16.

    Optimiser has two param groups:
        - Main transformer model  →  lr = phase2_lr   (default 2e-5)
        - MacBERT encoder         →  lr = phase2_bert_lr (default 2e-5)

    Gradient clipping (max_norm=1.0) is applied jointly at each accumulated step.
    Early stopping is based on val F1 (same criterion as Phase 1).
    """
    bert_encoder.unfreeze_bert()
    bert_encoder.log_trainable_parameters()

    # Enable gradient checkpointing: recompute BERT activations during backward
    # instead of caching them.  This cuts activation memory ~12× at cost of ~40%
    # extra compute.  Applied only for Phase 2 training; disabled afterwards.
    # Uses MacBERTEncoder.enable_gradient_checkpointing() which handles the
    # PEFT/LoRA edge case (enable_input_require_grads must precede GC enable).
    bert_encoder.enable_gradient_checkpointing()

    use_amp     = config.get('use_amp', True) and device.startswith('cuda')
    epochs      = config.get('phase2_epochs',    20)
    lr          = config.get('phase2_lr',        2e-5)
    bert_lr     = config.get('phase2_bert_lr',   2e-5)
    patience    = config.get('phase2_patience',  10)
    accum_steps = config.get('phase2_accum_steps', 8)
    device_type = device.split(':')[0]
    all_params  = list(model.parameters()) + list(bert_encoder.parameters())

    optimizer = optim.AdamW(
        [
            {'params': model.parameters(),        'lr': lr},
            {'params': bert_encoder.parameters(), 'lr': bert_lr},
        ],
        betas=(0.9, 0.95), weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_f1      = -1.0
    best_model_state = None
    best_bert_state  = None
    patience_ctr     = 0
    history: Dict    = {'train_loss': [], 'val_loss': [], 'train_acc': [],
                        'val_acc': [], 'val_f1': []}

    # ── Resume from mid-training checkpoint if available ─────────────────────
    checkpoint_dir = config.get('checkpoint_dir', '')
    resume_meta_path = os.path.join(checkpoint_dir, 'phase2_resume.json')
    start_epoch = 0
    if os.path.exists(resume_meta_path):
        try:
            with open(resume_meta_path) as f:
                resume_meta = json.load(f)
            resume_epoch   = resume_meta.get('epoch', 0)
            resume_history = resume_meta.get('history', history)
            resume_best_f1 = resume_meta.get('best_val_f1', -1.0)
            resume_patience = resume_meta.get('patience_ctr', 0)
            # Load mid-training model weights
            mid_model_path = os.path.join(checkpoint_dir, 'phase2_resume_model.pth')
            mid_bert_path  = os.path.join(checkpoint_dir, 'phase2_resume_bert.pth')
            if os.path.exists(mid_model_path) and os.path.exists(mid_bert_path):
                model.load_state_dict(
                    torch.load(mid_model_path, map_location=device, weights_only=True)
                )
                bert_encoder.load_state_dict(
                    torch.load(mid_bert_path, map_location=device, weights_only=True)
                )
                start_epoch   = resume_epoch + 1
                history       = resume_history
                best_val_f1   = resume_best_f1
                patience_ctr  = resume_patience
                print(f"  [resume] Continuing Phase 2 from epoch {start_epoch + 1}/{epochs}  "
                      f"(best_f1={best_val_f1:.3f})")
        except Exception as e:
            print(f"  [resume] Could not load resume checkpoint: {e}  — starting fresh")

    print("\n" + "─" * 50)
    print(f"Phase 2 — full fine-tune (BERT inline, accum={accum_steps})")
    print("─" * 50)

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        model.train()
        bert_encoder.train()

        total_loss  = 0.0
        correct     = 0
        total       = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f'  p2 ep{epoch+1}', leave=False,
                    unit='step', dynamic_ncols=True)
        for batch_idx, (price_seq, ids_win, masks_win, n_arts, labels) in enumerate(pbar):
            price_seq = price_seq.to(device, non_blocking=True)
            labels    = labels.to(device,    non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_amp):
                news_seq = _encode_news_inline(
                    bert_encoder, ids_win, masks_win, n_arts, device
                )
                logits = model(price_seq, news_seq)
                # Scale loss by 1/accum_steps so accumulated gradients equal a
                # full-batch gradient (not accum_steps× too large).
                loss = criterion(logits, labels) / accum_steps

            loss.backward()

            total_loss += loss.item() * accum_steps   # log unscaled loss
            _, pred = torch.max(logits, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

            is_last_batch = (batch_idx + 1 == len(train_loader))
            if (batch_idx + 1) % accum_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        pbar.close()
        tr_loss = total_loss / len(train_loader)
        tr_acc  = correct / total

        val_loss, preds, lbls, _ = _evaluate_phase2(
            model, bert_encoder, val_loader, criterion, device, use_amp
        )
        val_metrics = compute_metrics(lbls, preds)
        val_acc = val_metrics['accuracy']
        val_f1  = val_metrics['f1']

        scheduler.step(val_loss)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        marker  = ' *' if val_f1 > best_val_f1 else ''
        elapsed = time.perf_counter() - t0
        print(
            f"  epoch {epoch+1:3d}/{epochs}  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  {elapsed:.1f}s{marker}"
        )

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_model_state = deepcopy(model.state_dict())
            best_bert_state  = deepcopy(bert_encoder.state_dict())
            patience_ctr     = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [early stop] No F1 improvement for {patience} epochs.")
                break

        # Save mid-training checkpoint so training can resume if interrupted.
        # Overwrites the previous epoch's checkpoint (only one slot kept).
        if checkpoint_dir:
            try:
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, 'phase2_resume_model.pth'))
                torch.save(bert_encoder.state_dict(),
                           os.path.join(checkpoint_dir, 'phase2_resume_bert.pth'))
                resume_snapshot = {
                    'epoch':       epoch,
                    'best_val_f1': best_val_f1,
                    'patience_ctr': patience_ctr,
                    'history':     history,
                }
                with open(resume_meta_path, 'w') as f:
                    json.dump(resume_snapshot, f)
            except Exception as e:
                print(f"  [checkpoint] Warning: could not save resume checkpoint: {e}")

    # Restore gradient checkpointing to disabled — inference and Phase 1 evaluation
    # do not need it and it adds unnecessary compute overhead.
    bert_encoder.disable_gradient_checkpointing()

    print(f"  [phase 2] Best val_f1={best_val_f1:.3f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_bert_state is not None:
        bert_encoder.load_state_dict(best_bert_state)

    # Training completed — remove mid-training resume checkpoint so a fresh
    # Phase 2 run (e.g. after changing config) starts from scratch.
    if checkpoint_dir:
        for fname in ('phase2_resume.json', 'phase2_resume_model.pth', 'phase2_resume_bert.pth'):
            p = os.path.join(checkpoint_dir, fname)
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    return model, bert_encoder, history


# ─── Checkpoint utilities ─────────────────────────────────────────────────────

def save_checkpoint(
    model:        MultimodalStockTransformer,
    bert_encoder: Optional[MacBERTEncoder],
    config:       dict,
    history:      dict,
    metrics:      dict,
    save_dir:     str,
    phase:        int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, f'phase{phase}_model.pth'))
    if bert_encoder is not None:
        torch.save(bert_encoder.state_dict(), os.path.join(save_dir, f'phase{phase}_bert.pth'))

    meta = {'config': config, 'history': history, 'metrics': metrics, 'phase': phase}
    with open(os.path.join(save_dir, f'phase{phase}_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[training] Checkpoint saved → {save_dir} (phase {phase})")


def load_checkpoint(
    model:        MultimodalStockTransformer,
    bert_encoder: Optional[MacBERTEncoder],
    save_dir:     str,
    phase:        int,
    device:       str,
) -> Tuple[MultimodalStockTransformer, Optional[MacBERTEncoder]]:
    model_path = os.path.join(save_dir, f'phase{phase}_model.pth')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    current = model.state_dict()
    shape_mismatches = [
        k for k, v in state_dict.items()
        if k in current and v.shape != current[k].shape
    ]
    if shape_mismatches:
        print(f"[training] Shape mismatch — dropping and reinitializing: {shape_mismatches}")
        for k in shape_mismatches:
            del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[training]   missing keys (kept as random init): {missing}")
    if unexpected:
        print(f"[training]   unexpected keys (ignored): {unexpected}")

    bert_path = os.path.join(save_dir, f'phase{phase}_bert.pth')
    if bert_encoder is not None and os.path.exists(bert_path):
        # When LoRA is active the saved keys use plain BERT naming
        # (bert.encoder.layer.X...) but the wrapped PeftModel expects
        # bert.base_model.model.encoder.layer.X...  The base BERT was frozen
        # during Phase 1 so its weights are unchanged from the pre-trained
        # init — skip loading to avoid the key mismatch.
        # Phase 2 LoRA checkpoints use the full LoRA state dict and load normally.
        if phase == 1 and getattr(bert_encoder, '_use_lora', False):
            print(f"[training] Skipping Phase 1 BERT weights (LoRA active; base weights unchanged from init)")
        else:
            bert_encoder.load_state_dict(torch.load(bert_path, map_location=device, weights_only=True))

    print(f"[training] Loaded phase {phase} checkpoint from {save_dir}")
    return model, bert_encoder


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_eval_report(
    val_loss:    float,
    predictions: np.ndarray,
    labels:      np.ndarray,
    split_name:  str = 'test',
) -> dict:
    metrics = compute_metrics(labels, predictions)
    print(f"\n{'─'*50}")
    print(f"Evaluation — {split_name}")
    print(f"{'─'*50}")
    print(f"  Loss      : {val_loss:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")

    # Per-class counts
    for i, name in enumerate(MM_CLASS_NAMES):
        n = int((labels == i).sum())
        c = int(((predictions == i) & (labels == i)).sum())
        print(f"  {name:5s}: {c}/{n} correct")

    return metrics
