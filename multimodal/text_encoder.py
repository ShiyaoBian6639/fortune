"""
Chinese text encoder for financial news.

Model: hfl/chinese-macbert-base  (Chinese MacBERT, 768-dim, ~392 MB)
       Runs entirely from a local snapshot — no HuggingFace network calls
       after the one-time setup step.

First run
---------
Call ``ensure_local_model()`` (or let MacBERTEncoder do it automatically).
The model is copied from the HF disk-cache (already present) to
``stock_data/models/chinese-macbert-base/`` using ``save_pretrained``.
All subsequent loads use ``local_files_only=True`` and make zero network
calls, which also suppresses the safetensors background conversion thread.

Public API
----------
  MacBERTEncoder        — nn.Module, wraps local BERT
  encode_articles_batch — list[str] → (N, 768) numpy array
  build_daily_news_cache — one-time YYYYMMDD.csv → .npz cache
  load_daily_news_cache  — .npz → {date_str: ndarray(768,)}
  ensure_local_model     — download/copy model to local directory once
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Silence transformers' own INFO-level logging (key mismatch reports, etc.)
# Must happen before the first transformers import.
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# ─── Local model directory ────────────────────────────────────────────────────

# Model is stored here after the one-time setup.  Loading from a filesystem
# path (instead of a HF Hub id) prevents the safetensors conversion background
# thread and all other network calls.
_PACKAGE_ROOT  = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = _PACKAGE_ROOT / 'stock_data' / 'models' / 'chinese-macbert-base'

HF_MODEL_ID = 'hfl/chinese-macbert-base'   # only used for the one-time download


# ─── One-time model setup ─────────────────────────────────────────────────────

def _find_hf_snapshot(hf_model_id: str) -> Optional[Path]:
    """
    Return the path to the local HuggingFace disk-cache snapshot for
    ``hf_model_id``, or None if it does not exist.

    Loading from a filesystem path (rather than a Hub model ID string)
    prevents transformers from spawning the ``Thread-auto_conversion``
    background thread that contacts the HF Discussions API — which returns
    403 when discussions are disabled for the repo.
    """
    try:
        import huggingface_hub
        path = huggingface_hub.snapshot_download(
            hf_model_id,
            local_files_only=True,   # never make a network call
        )
        return Path(path)
    except Exception:
        return None


def ensure_local_model(
    hf_model_id: str = HF_MODEL_ID,
    local_dir: Path  = LOCAL_MODEL_DIR,
) -> Path:
    """
    Ensure the model weights are saved in ``local_dir``.

    If ``local_dir/config.json`` already exists the function returns
    immediately (idempotent).  Otherwise it copies from the HuggingFace
    disk-cache snapshot (already present) into ``local_dir`` via
    ``save_pretrained``.

    Critically, both the read and write sides use filesystem paths, NOT
    Hub model ID strings.  This suppresses the ``Thread-auto_conversion``
    background thread (which GETs the HF Discussions API and crashes with
    403 when discussions are disabled for the repo).

    Args:
        hf_model_id: HuggingFace model identifier (used to locate the cache).
        local_dir:   Target directory for the local snapshot.

    Returns:
        Resolved Path to the local model directory.
    """
    local_dir = Path(local_dir)
    if (local_dir / 'config.json').exists():
        return local_dir  # already set up — zero network calls

    print(f"[text_encoder] One-time model setup: {hf_model_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Locate the HF disk-cache snapshot by its filesystem path.
    # Loading from a path (not a Hub ID) avoids the safetensors conversion thread.
    snapshot_path = _find_hf_snapshot(hf_model_id)
    load_src = str(snapshot_path) if snapshot_path else hf_model_id

    if snapshot_path:
        print(f"[text_encoder] Loading from local cache: {snapshot_path}")
    else:
        print(f"[text_encoder] Cache not found; downloading {hf_model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(load_src, local_files_only=bool(snapshot_path))
    model     = AutoModel.from_pretrained(load_src,     local_files_only=bool(snapshot_path))

    tokenizer.save_pretrained(str(local_dir))
    model.save_pretrained(str(local_dir))

    n_files = len(list(local_dir.iterdir()))
    print(f"[text_encoder] Saved {n_files} files to {local_dir}")
    return local_dir


# ─── Encoder module ───────────────────────────────────────────────────────────

class MacBERTEncoder(nn.Module):
    """
    Sentence encoder backed by the local chinese-macbert-base snapshot.

    ``forward()`` returns the CLS-token vector (B, 768) for each input.
    ``freeze_bert`` / ``unfreeze_bert`` control gradient flow for
    2-phase training.

    LoRA mode (use_lora=True)
    -------------------------
    Wraps the base BERT with PEFT LoRA adapters on the Q and V projections of
    every attention layer.  Only the adapter matrices (~295K params, rank=8)
    are trainable during Phase 2; the 110M base weights stay frozen.
    Benefits over full fine-tuning:
      - Adam optimizer states: ~3 MB vs ~880 MB  → frees RAM for larger batches
      - Gradients: only accumulated for LoRA params (~370× fewer)
      - Regularisation: strong inductive bias prevents BERT from forgetting
        general language knowledge on domain-specific news data

    Args:
        model_name:           HF model ID (used for one-time download only).
        max_length:           Tokeniser truncation length (default 128).
        freeze:               Freeze all BERT params at construction.
        local_dir:            Override local model directory.
        use_lora:             Wrap BERT with LoRA adapters.
        lora_r:               LoRA rank (default 8).
        lora_alpha:           LoRA scaling factor (default 16; effective scale = alpha/r = 2×).
        lora_target_modules:  List of module name suffixes to inject adapters into.
                              Default: ['query', 'value'] (standard for BERT).
        lora_dropout:         Dropout applied inside LoRA adapters.
    """

    hidden_size: int = 768   # MacBERT-base hidden dim — used by models.py

    def __init__(
        self,
        model_name:           str            = HF_MODEL_ID,
        max_length:           int            = 128,
        freeze:               bool           = True,
        local_dir:            Optional[Path] = None,
        use_lora:             bool           = False,
        lora_r:               int            = 8,
        lora_alpha:           int            = 16,
        lora_target_modules:  Optional[List[str]] = None,
        lora_dropout:         float          = 0.1,
    ):
        super().__init__()
        self.max_length = max_length
        self._use_lora  = use_lora

        effective_dir = ensure_local_model(
            hf_model_id=model_name,
            local_dir=local_dir or LOCAL_MODEL_DIR,
        )

        # local_files_only=True → zero network calls, no background threads
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(effective_dir), local_files_only=True
        )
        base_bert = AutoModel.from_pretrained(
            str(effective_dir), local_files_only=True
        )

        if use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "use_lora=True requires the 'peft' package: pip install peft"
                )
            lora_cfg = LoraConfig(
                task_type=PeftTaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ['query', 'value'],
                lora_dropout=lora_dropout,
                bias='none',
            )
            self.bert = get_peft_model(base_bert, lora_cfg)
        else:
            self.bert = base_bert

        if freeze:
            self.freeze_bert()

    # ── Gradient control ──────────────────────────────────────────────────────

    def freeze_bert(self):
        """Freeze all BERT parameters, including LoRA adapters (Phase 1)."""
        for p in self.bert.parameters():
            p.requires_grad_(False)

    def unfreeze_bert(self):
        """
        Phase 2: enable trainable parameters.

        LoRA mode   — only the LoRA adapter matrices are set to requires_grad.
                      The 110M base BERT weights stay frozen; only ~295K adapter
                      params are optimised.
        No-LoRA mode — all BERT parameters are unfrozen (full fine-tuning).
        """
        if self._use_lora:
            for name, p in self.bert.named_parameters():
                if 'lora_' in name:
                    p.requires_grad_(True)
        else:
            for p in self.bert.parameters():
                p.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing, handling the PEFT/LoRA edge case.

        With PEFT LoRA, ``enable_input_require_grads()`` must be called first so
        that the checkpoint function receives a tensor with a valid grad_fn.
        Without it, PyTorch silently skips checkpointing for frozen input
        embeddings and the gradient computation is incorrect.
        """
        if self._use_lora:
            self.bert.enable_input_require_grads()
            self.bert.base_model.model.gradient_checkpointing_enable()
        else:
            self.bert.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing (restore for inference / Phase 1 eval)."""
        if self._use_lora:
            self.bert.base_model.model.gradient_checkpointing_disable()
        else:
            self.bert.gradient_checkpointing_disable()

    def log_trainable_parameters(self):
        """Print trainable vs total parameter count."""
        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.bert.parameters())
        pct       = 100.0 * trainable / total if total else 0.0
        print(
            f"[MacBERTEncoder] trainable: {trainable:,} / {total:,} "
            f"({pct:.2f}%)  {'[LoRA]' if self._use_lora else '[full]'}"
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, seq_len)
        attention_mask: torch.Tensor,   # (B, seq_len)
    ) -> torch.Tensor:                  # (B, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]   # CLS token


# ─── Batch encoding utility ───────────────────────────────────────────────────

def encode_articles_batch(
    texts:          List[str],
    tokenizer,
    model:          MacBERTEncoder,
    device:         str,
    max_length:     int = 128,
    sub_batch_size: int = 32,
) -> np.ndarray:
    """
    Encode a list of article strings → numpy array (N, 768).

    Processes in sub-batches to keep GPU memory bounded.
    Runs under ``torch.no_grad()``.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    all_vecs  = []
    model.eval()
    device_obj = torch.device(device)

    with torch.no_grad():
        for start in range(0, len(texts), sub_batch_size):
            batch = texts[start: start + sub_batch_size]
            enc   = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )
            ids  = enc['input_ids'].to(device_obj)
            mask = enc['attention_mask'].to(device_obj)
            vecs = model(ids, mask)           # (sub_B, 768)
            all_vecs.append(vecs.cpu().float().numpy())

    return np.concatenate(all_vecs, axis=0).astype(np.float32)


# ─── News cache builder ───────────────────────────────────────────────────────

def _collect_date_files(news_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan ``news_dir`` for YYYYMMDD.csv files and return a dict mapping each
    date string to all CSV files (from any source) available for that day.

    Supports two layouts — both can coexist:
      Single-source : news_dir/YYYYMMDD.csv
      Multi-source  : news_dir/<src>/YYYYMMDD.csv   (e.g. sina/, eastmoney/, …)

    Subdirectories that do not contain date-named CSVs are silently skipped
    (e.g. ``sh/``, ``sz/`` which hold stock-specific news in a different format).
    """
    date_to_files: Dict[str, List[Path]] = {}
    for f in sorted(news_dir.glob('????????.csv')):
        date_to_files.setdefault(f.stem, []).append(f)
    for f in sorted(news_dir.glob('*/????????.csv')):
        date_to_files.setdefault(f.stem, []).append(f)
    return date_to_files


def _texts_from_csvs(csv_paths: List[Path]) -> List[str]:
    """
    Read one or more news CSVs, return deduplicated ``title + content`` strings.

    Schema expected: columns ``datetime``, ``title``, ``content`` (all sources).

    Cleaning applied:
    - NaN fields are treated as empty (pandas NaN is truthy, so naive
      ``str(nan or '')`` yields the literal string ``'nan'`` — we use
      ``pd.isna()`` instead).
    - Exact-duplicate articles (same content body) are dropped within the
      combined day batch.  Eastmoney, for example, can have 60–70 % of rows
      duplicated in a single daily CSV; sina titles are always NaN so only
      the content column carries signal.
    - Empty rows (both title and content missing / blank) are skipped.
    """
    frames: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception:
            continue
        # Accept any file that has at least a content or title column
        if 'content' not in df.columns and 'title' not in df.columns:
            continue
        frames.append(df)

    if not frames:
        return []

    combined = pd.concat(frames, ignore_index=True)

    # Normalise: replace NaN with empty string (pd.isna-safe)
    title_col   = combined['title'].where(~combined['title'].isna(),   '') if 'title'   in combined.columns else pd.Series([''] * len(combined))
    content_col = combined['content'].where(~combined['content'].isna(), '') if 'content' in combined.columns else pd.Series([''] * len(combined))

    title_col   = title_col.astype(str).str.strip()
    content_col = content_col.astype(str).str.strip()

    # Build combined text: "title content" when both present, else whichever exists
    texts_raw = []
    for title, content in zip(title_col, content_col):
        if title and content:
            texts_raw.append(title + ' ' + content)
        elif content:
            texts_raw.append(content)
        elif title:
            texts_raw.append(title)
        # both empty → skip

    # Deduplicate: keep first occurrence of each unique text (preserves temporal order).
    # Using full text as the key handles all cases: same article appearing multiple
    # times in one source's CSV, and cross-source exact reposts.
    seen: set = set()
    texts: List[str] = []
    for text in texts_raw:
        if text not in seen:
            seen.add(text)
            texts.append(text)

    return texts


def build_daily_news_cache(
    news_dir:             str,
    cache_path:           str,
    model_name:           str           = HF_MODEL_ID,
    device:               str           = 'cuda',
    max_articles_per_day: int           = 16,
    max_length:           int           = 128,
    sub_batch_size:       int           = 32,
    max_days:             Optional[int] = None,
    local_dir:            Optional[Path] = None,
) -> None:
    """
    One-time preprocessing: encode all YYYYMMDD.csv files found under
    ``news_dir`` with MacBERT, mean-pool articles per day, save to
    ``cache_path`` (.npz).

    ``news_dir`` may point to a single-source directory (CSVs directly
    inside) or to a parent directory whose subdirectories each hold
    per-source CSVs (e.g. ``stock_data/news/`` containing ``sina/``,
    ``eastmoney/``, ``wallstreetcn/``, etc.).  Articles from all sources
    found for the same trading day are pooled together before encoding.

    Article selection when N > max_articles_per_day:
        Sort by ``len(content)`` descending → take top max_articles_per_day.
        Rationale: longer articles tend to carry more information.

    Cache format (compressed .npz):
        dates:   str array  (D,)        — YYYYMMDD strings
        vectors: float32    (D, 768)    — mean-pooled CLS per day

    Days with < 2 articles (across all sources) get a zero vector.
    """
    news_path = Path(news_dir)
    date_to_files = _collect_date_files(news_path)

    if not date_to_files:
        print(f"[text_encoder] No news CSV files found in {news_dir}")
        return

    all_dates = sorted(date_to_files.keys())
    if max_days is not None:
        all_dates = all_dates[:max_days]

    source_dirs = {f.parent.name for files in date_to_files.values() for f in files}
    print(
        f"[text_encoder] Processing {len(all_dates)} news days "
        f"across {len(source_dirs)} source(s): {sorted(source_dirs)}"
    )

    use_cuda      = (device == 'cuda') and torch.cuda.is_available()
    actual_device = 'cuda' if use_cuda else 'cpu'

    print(f"[text_encoder] Loading MacBERT on {actual_device} ...")
    encoder = MacBERTEncoder(
        model_name=model_name,
        max_length=max_length,
        freeze=True,
        local_dir=local_dir,
    )
    encoder.to(actual_device)
    encoder.eval()
    tokenizer = encoder.tokenizer

    date_list: List[str]        = []
    vec_list:  List[np.ndarray] = []

    for date_str in tqdm(all_dates, desc='Encoding news days'):
        csv_paths = date_to_files[date_str]

        # Gather articles from ALL sources available for this day
        texts = _texts_from_csvs(csv_paths)

        if len(texts) < 2:
            date_list.append(date_str)
            vec_list.append(np.zeros(768, dtype=np.float32))
            continue

        if len(texts) > max_articles_per_day:
            texts = sorted(texts, key=len, reverse=True)[:max_articles_per_day]

        vecs    = encode_articles_batch(
            texts, tokenizer, encoder, actual_device, max_length, sub_batch_size
        )
        day_vec = vecs.mean(axis=0)   # (768,)

        date_list.append(date_str)
        vec_list.append(day_vec)

    if not date_list:
        print("[text_encoder] No valid days to cache.")
        return

    dates_arr   = np.array(date_list)
    vectors_arr = np.stack(vec_list).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, dates=dates_arr, vectors=vectors_arr)
    print(f"[text_encoder] Cache saved: {len(date_list)} days → {cache_path}")


# ─── Token cache builder (Phase 2) ───────────────────────────────────────────

def build_daily_token_cache(
    news_dir:             str,
    cache_path:           str,
    model_name:           str           = HF_MODEL_ID,
    max_articles_per_day: int           = 16,
    max_length:           int           = 128,
    max_days:             Optional[int] = None,
    local_dir:            Optional[Path] = None,
) -> None:
    """
    Build a per-day token cache for Phase 2 inline BERT fine-tuning.

    Unlike ``build_daily_news_cache()`` which pre-computes fixed BERT embeddings,
    this stores raw tokenized inputs (input_ids + attention_mask) so BERT can be
    called inline per-batch and gradients flow back through it.

    Cache format (compressed .npz):
        dates:      str   (D,)                 — YYYYMMDD strings
        input_ids:  int32 (D, A, max_length)   — A = max_articles_per_day
        attn_masks: int32 (D, A, max_length)   — zero-padded for days with < A articles
        n_articles: int32 (D,)                 — actual article count per day (for mean-pool mask)

    Rows beyond ``n_articles[d]`` in ``input_ids[d]`` are zero-padded and must be
    excluded from the mean-pool (``n_articles`` provides the mask boundary).
    """
    news_path = Path(news_dir)
    date_to_files = _collect_date_files(news_path)

    if not date_to_files:
        print(f"[text_encoder] No news CSV files found in {news_dir}")
        return

    all_dates = sorted(date_to_files.keys())
    if max_days is not None:
        all_dates = all_dates[:max_days]

    print(f"[text_encoder] Tokenizing {len(all_dates)} news days for Phase 2 cache ...")

    effective_dir = ensure_local_model(model_name, local_dir or LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(effective_dir), local_files_only=True)

    A = max_articles_per_day
    L = max_length

    dates_out  : List[str]        = []
    ids_out    : List[np.ndarray] = []
    masks_out  : List[np.ndarray] = []
    n_arts_out : List[int]        = []

    for date_str in tqdm(all_dates, desc='Tokenizing news days'):
        texts = _texts_from_csvs(date_to_files[date_str])
        if len(texts) > A:
            texts = sorted(texts, key=len, reverse=True)[:A]

        n = len(texts)
        day_ids   = np.zeros((A, L), dtype=np.int32)
        day_masks = np.zeros((A, L), dtype=np.int32)

        if n > 0:
            enc = tokenizer(
                texts,
                max_length=L,
                truncation=True,
                padding='max_length',
                return_tensors='np',
            )
            day_ids[:n]   = enc['input_ids'][:n].astype(np.int32)
            day_masks[:n] = enc['attention_mask'][:n].astype(np.int32)

        dates_out.append(date_str)
        ids_out.append(day_ids)
        masks_out.append(day_masks)
        n_arts_out.append(n)

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.savez_compressed(
        cache_path,
        dates=np.array(dates_out),
        input_ids=np.stack(ids_out).astype(np.int32),
        attn_masks=np.stack(masks_out).astype(np.int32),
        n_articles=np.array(n_arts_out, dtype=np.int32),
    )
    print(f"[text_encoder] Token cache saved: {len(dates_out)} days → {cache_path}")


def load_daily_token_cache(cache_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load per-day token cache from .npz.

    Returns ``{date_str: {'input_ids': (A, L), 'attn_mask': (A, L), 'n_articles': int}}``
    where A = max_articles_per_day, L = max_length.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(cache_path):
        return {}

    data      = np.load(cache_path, allow_pickle=False)
    dates     = data['dates']        # (D,)
    input_ids = data['input_ids']    # (D, A, L)
    attn_masks = data['attn_masks']  # (D, A, L)
    n_articles = data['n_articles']  # (D,)

    return {
        str(d): {
            'input_ids':  input_ids[i],   # (A, L) int32
            'attn_mask':  attn_masks[i],  # (A, L) int32
            'n_articles': int(n_articles[i]),
        }
        for i, d in enumerate(dates)
    }


# ─── Cache loader ─────────────────────────────────────────────────────────────

def load_daily_news_cache(cache_path: str) -> Dict[str, np.ndarray]:
    """
    Load ``{date_str → embedding (768,)}`` from a .npz file.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(cache_path):
        return {}

    data    = np.load(cache_path, allow_pickle=False)
    dates   = data['dates']     # (D,) str
    vectors = data['vectors']   # (D, 768) float32
    return {str(d): vectors[i] for i, d in enumerate(dates)}
