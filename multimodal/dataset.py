"""
Datasets and loaders for the compact multimodal cache.

Cache layout — see ``multimodal/data_pipeline.py``.  The dataset classes here
read:

  * ``price_matrix.npy`` — concatenated per-stock daily features
  * ``sample_end.npy`` / ``sample_date_idx.npy`` / ``sample_labels.npy``
    — tiny per-sample index arrays
  * ``trading_calendar.npy`` + the per-day news cache (``news_embeddings.npz``)
    — news embeddings looked up at __getitem__ time, never materialised
    per-sample.

Phase 1 (frozen BERT)        — MultimodalStockDataset / MultimodalChunkedLoader
Phase 2 (LoRA fine-tune)     — Phase2Dataset (raw token tensors)

Memmap handles are deferred via __getstate__ / __setstate__ to play nicely
with DataLoader workers on Windows (same pattern as ``dl/memmap_dataset.py``).
"""

from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import MM_NUM_CLASSES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_news_per_date(
    news_cache_path: str,
    trading_calendar: np.ndarray,
) -> np.ndarray:
    """
    Load the per-day news embedding table aligned to the cache's trading
    calendar.  Days missing from the news cache get a zero vector.

    Returns (D, 768) float32 where D = len(trading_calendar).
    """
    if not news_cache_path or not os.path.exists(news_cache_path):
        raise FileNotFoundError(
            f"news cache not found at {news_cache_path!r}.  "
            "Pass news_cache_path explicitly or run --mode preprocess first."
        )
    npz          = np.load(news_cache_path, allow_pickle=False)
    news_dates   = npz['dates']
    news_vectors = npz['vectors'].astype(np.float32)
    news_dim     = news_vectors.shape[1]
    D            = len(trading_calendar)

    table  = np.zeros((D, news_dim), dtype=np.float32)
    lookup = {str(d): i for i, d in enumerate(news_dates)}
    for i, d in enumerate(trading_calendar):
        j = lookup.get(str(d))
        if j is not None:
            table[i] = news_vectors[j]
    return table


def _resolve_news_cache_path(meta: dict, override: Optional[str]) -> str:
    """Prefer explicit override, fall back to the path stored at build time."""
    if override:
        return override
    p = meta.get('news_cache_path')
    if not p:
        raise ValueError(
            "news_cache_path is missing from metadata.json and was not "
            "provided.  Pass it via the dataset constructor."
        )
    return p


# ─── Phase 1 dataset ──────────────────────────────────────────────────────────

class MultimodalStockDataset(Dataset):
    """
    Random-access dataset backed by the compact cache.

    __getitem__ returns ``(price_seq, news_seq, label)`` as
    ``(F32[seq_len, F], F32[news_window, 768], i64)``.  Used for val/test
    where sequential access is preferred over the chunked random shuffle.
    """

    def __init__(
        self,
        cache_dir:        str,
        split:            str           = 'train',
        news_cache_path:  Optional[str] = None,
        news_window:      Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.split     = split

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)

        self.seq_len     = int(meta['sequence_length'])
        self.n_features  = int(meta['n_features'])
        self.news_window = int(news_window if news_window is not None
                               else meta.get('news_window', 3))

        split_info       = meta['splits'][split]
        self.split_start = int(split_info['start'])
        self.split_end   = int(split_info['end'])
        self.n_samples   = int(split_info['n_samples'])

        # Tiny per-sample index — load fully in RAM
        self._sample_end      = np.load(os.path.join(cache_dir, 'sample_end.npy'))
        self._sample_date_idx = np.load(os.path.join(cache_dir, 'sample_date_idx.npy'))
        self._sample_labels   = np.load(os.path.join(cache_dir, 'sample_labels.npy'))

        # Per-day news embeddings — ~10 MB, in RAM
        trading_calendar  = np.load(os.path.join(cache_dir, 'trading_calendar.npy'))
        news_path         = _resolve_news_cache_path(meta, news_cache_path)
        self._news_per_date = _load_news_per_date(news_path, trading_calendar)
        self._news_dim    = self._news_per_date.shape[1]

        self._price_mm = None
        self._open_memmaps()

    def _open_memmaps(self):
        self._price_mm = np.load(
            os.path.join(self.cache_dir, 'price_matrix.npy'), mmap_mode='r',
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_price_mm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gi       = self.split_start + i
        end      = int(self._sample_end[gi])
        date_idx = int(self._sample_date_idx[gi])
        label    = int(self._sample_labels[gi])

        L = self.seq_len
        # `np.array(...)` forces a copy out of the memmap so the tensor below
        # owns its memory (avoids dangling-mmap issues across worker forks).
        price = np.array(self._price_mm[end - L + 1: end + 1], dtype=np.float32)

        W = self.news_window
        news = np.zeros((W, self._news_dim), dtype=np.float32)
        for w in range(W):
            d = date_idx - (W - 1 - w)
            if d >= 0:
                news[w] = self._news_per_date[d]

        return (
            torch.from_numpy(price),
            torch.from_numpy(news),
            torch.tensor(label, dtype=torch.long),
        )

    def get_labels(self) -> np.ndarray:
        return self._sample_labels[self.split_start:self.split_end].copy()


# ─── Phase 1 chunked loader ───────────────────────────────────────────────────

class MultimodalChunkedLoader:
    """
    Background-prefetched chunked loader for Phase 1 training.

    Each chunk is a randomly-permuted slice of the split's sample indices.
    Gathering uses fancy indexing into the price memmap (the OS page cache
    keeps recently-touched rows hot — the matrix is only ~6 GB).  News is
    looked up from a small in-RAM table.

    Two prefetch threads stay one chunk ahead of training so GPU never waits
    on disk.

    Peak RAM ≈ 3 × chunk_samples × (seq_len × n_feat + news_window × 768) × 4
    plus the (D, 768) news table (~10 MB).
    """

    def __init__(
        self,
        cache_dir:       str,
        split:           str           = 'train',
        batch_size:      int           = 1024,
        chunk_samples:   int           = 50_000,
        seed:            int           = 42,
        news_cache_path: Optional[str] = None,
        news_window:     Optional[int] = None,
    ):
        self.cache_dir     = cache_dir
        self.split         = split
        self.batch_size    = batch_size
        self.chunk_samples = chunk_samples
        self._seed         = seed

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.seq_len     = int(meta['sequence_length'])
        self.n_features  = int(meta['n_features'])
        self.news_window = int(news_window if news_window is not None
                               else meta.get('news_window', 3))

        split_info       = meta['splits'][split]
        self.split_start = int(split_info['start'])
        self.split_end   = int(split_info['end'])
        self.n_samples   = int(split_info['n_samples'])

        # Sliced per-sample arrays (small)
        self._sample_end      = np.load(os.path.join(cache_dir, 'sample_end.npy'))[
            self.split_start:self.split_end
        ]
        self._sample_date_idx = np.load(os.path.join(cache_dir, 'sample_date_idx.npy'))[
            self.split_start:self.split_end
        ]
        self._sample_labels   = np.load(os.path.join(cache_dir, 'sample_labels.npy'))[
            self.split_start:self.split_end
        ]

        # News embedding table aligned to trading_calendar
        trading_calendar    = np.load(os.path.join(cache_dir, 'trading_calendar.npy'))
        news_path           = _resolve_news_cache_path(meta, news_cache_path)
        self._news_per_date = _load_news_per_date(news_path, trading_calendar)
        self._news_dim      = self._news_per_date.shape[1]

        # Read-only memmap (numpy memmap is thread-safe for reads)
        self._price_mm = np.load(
            os.path.join(cache_dir, 'price_matrix.npy'), mmap_mode='r',
        )

    def __len__(self) -> int:
        """Number of complete batches per epoch."""
        return self.n_samples // self.batch_size

    # ── Chunk gather ────────────────────────────────────────────────────────

    def _gather_chunk(
        self, indices: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather ``len(indices)`` samples from the cache.

        Built so consecutive batch-of-bs slices of the returned tensors are
        already in shuffled order — the caller just splits along dim 0.
        """
        L = self.seq_len
        n = len(indices)

        ends   = self._sample_end[indices].astype(np.int64)
        starts = ends - L + 1
        # row_idx[i, t] = starts[i] + t
        row_idx = (starts[:, None] + np.arange(L, dtype=np.int64)[None, :]).ravel()
        prices  = np.asarray(self._price_mm[row_idx]).reshape(n, L, self.n_features)

        date_idx = self._sample_date_idx[indices].astype(np.int64)
        W        = self.news_window
        news     = np.zeros((n, W, self._news_dim), dtype=np.float32)
        for w in range(W):
            d     = date_idx - (W - 1 - w)
            valid = d >= 0
            if valid.any():
                news[valid, w] = self._news_per_date[d[valid]]

        labels = self._sample_labels[indices].astype(np.int64)

        return (
            torch.from_numpy(np.ascontiguousarray(prices)),
            torch.from_numpy(news),
            torch.from_numpy(labels),
        )

    def _yield_batches(
        self, chunk: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        price, news, labels = chunk
        bs = self.batch_size
        n  = len(labels)
        for b in range(0, n - bs + 1, bs):
            yield price[b:b + bs], news[b:b + bs], labels[b:b + bs]

    # ── Iteration ───────────────────────────────────────────────────────────

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        # Advance epoch seed
        self._seed = int(rng.integers(0, 2**31))

        chunk_starts = list(range(0, self.n_samples, self.chunk_samples))
        rng.shuffle(chunk_starts)

        def _make_indices(cs: int) -> np.ndarray:
            ce  = min(cs + self.chunk_samples, self.n_samples)
            sub = np.arange(cs, ce, dtype=np.int64)
            np.random.default_rng(int(rng.integers(0, 2**31))).shuffle(sub)
            return sub

        # Synchronously load the first chunk
        current = self._gather_chunk(_make_indices(chunk_starts[0]))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: deque = deque()
            for cs in chunk_starts[1:3]:
                futures.append(pool.submit(self._gather_chunk, _make_indices(cs)))

            for cs in chunk_starts[3:]:
                futures.append(pool.submit(self._gather_chunk, _make_indices(cs)))
                yield from self._yield_batches(current)
                current = futures.popleft().result()

            yield from self._yield_batches(current)
            while futures:
                current = futures.popleft().result()
                yield from self._yield_batches(current)

    def get_labels(self) -> np.ndarray:
        return self._sample_labels.copy()


# ─── Phase 2 dataset (BERT inline) ────────────────────────────────────────────

class Phase2Dataset(Dataset):
    """
    Dataset for Phase 2 inline-BERT fine-tuning.

    __getitem__ returns:
        price_seq      : float32 (seq_len, F)
        input_ids_win  : int64   (W, A, max_length)
        attn_mask_win  : int64   (W, A, max_length)
        n_articles_win : int64   (W,)
        label          : int64

    Token cache is the per-day dict from
    ``text_encoder.load_daily_token_cache``.  Walking the news window uses the
    token cache's own trading calendar so that gaps between the build-time
    news calendar and the token cache (rare) cannot map to the wrong day.
    """

    def __init__(
        self,
        cache_dir:        str,
        token_cache:      Dict[str, Dict[str, np.ndarray]],
        trading_calendar: List[str],
        split:            str            = 'train',
        news_window:      int            = 3,
        max_samples:      Optional[int]  = None,
        random_seed:      int            = 42,
    ):
        self.cache_dir        = cache_dir
        self.split            = split
        self.news_window      = news_window
        self.trading_calendar = list(trading_calendar)

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.seq_len    = int(meta['sequence_length'])
        self.n_features = int(meta['n_features'])

        split_info       = meta['splits'][split]
        self.split_start = int(split_info['start'])
        self.split_end   = int(split_info['end'])
        n_total          = int(split_info['n_samples'])

        # Sliced per-sample arrays
        self._sample_end      = np.load(os.path.join(cache_dir, 'sample_end.npy'))[
            self.split_start:self.split_end
        ]
        self._sample_date_idx = np.load(os.path.join(cache_dir, 'sample_date_idx.npy'))[
            self.split_start:self.split_end
        ]
        self._sample_labels   = np.load(os.path.join(cache_dir, 'sample_labels.npy'))[
            self.split_start:self.split_end
        ]

        # date_idx → date_str using the cache's build-time calendar
        cache_calendar       = np.load(os.path.join(cache_dir, 'trading_calendar.npy'))
        self._cache_calendar = [str(d) for d in cache_calendar]

        # Phase 2 sub-sampling for practical epoch length
        if max_samples is not None and max_samples < n_total:
            rng = np.random.default_rng(random_seed)
            self._indices = rng.choice(n_total, max_samples, replace=False).astype(np.int64)
            self._indices.sort()
            self.n_samples = max_samples
        else:
            self._indices  = None
            self.n_samples = n_total

        self._token_cache = token_cache
        self._tok_cal_idx = {d: i for i, d in enumerate(self.trading_calendar)}
        sample_entry      = next(iter(token_cache.values())) if token_cache else None
        self._A           = sample_entry['input_ids'].shape[0] if sample_entry else 16
        self._L           = sample_entry['input_ids'].shape[1] if sample_entry else 128

        self._price_mm = None
        self._open_memmaps()

    def _open_memmaps(self):
        self._price_mm = np.load(
            os.path.join(self.cache_dir, 'price_matrix.npy'), mmap_mode='r',
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_price_mm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gi = int(self._indices[idx]) if self._indices is not None else idx

        end          = int(self._sample_end[gi])
        date_idx_c   = int(self._sample_date_idx[gi])
        label        = int(self._sample_labels[gi])

        L     = self.seq_len
        price = np.array(self._price_mm[end - L + 1: end + 1], dtype=np.float32)

        W, A, Lt  = self.news_window, self._A, self._L
        ids_win   = np.zeros((W, A, Lt), dtype=np.int32)
        masks_win = np.zeros((W, A, Lt), dtype=np.int32)
        n_arts    = np.zeros((W,),       dtype=np.int32)

        if 0 <= date_idx_c < len(self._cache_calendar):
            date_str = self._cache_calendar[date_idx_c]
            t_idx    = self._tok_cal_idx.get(date_str, -1)
            if t_idx >= 0:
                for w in range(W):
                    d_idx = t_idx - (W - 1 - w)
                    if d_idx < 0:
                        continue
                    day   = self.trading_calendar[d_idx]
                    entry = self._token_cache.get(day)
                    if entry is None:
                        continue
                    ids_win[w]   = entry['input_ids']
                    masks_win[w] = entry['attn_mask']
                    n_arts[w]    = entry['n_articles']

        return (
            torch.from_numpy(price),
            torch.as_tensor(ids_win,   dtype=torch.long),
            torch.as_tensor(masks_win, dtype=torch.long),
            torch.as_tensor(n_arts,    dtype=torch.long),
            torch.tensor(label,        dtype=torch.long),
        )

    def get_labels(self) -> np.ndarray:
        if self._indices is not None:
            return self._sample_labels[self._indices].copy()
        return self._sample_labels.copy()


# ─── Public factory functions ────────────────────────────────────────────────

def create_phase2_dataloaders(
    cache_dir:        str,
    token_cache:      Dict[str, Dict[str, np.ndarray]],
    trading_calendar: List[str],
    config:           dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Phase 2: inline-BERT loaders served from raw token tensors."""
    device      = config.get('device', 'cpu')
    pin         = device.startswith('cuda')
    batch_sz    = config.get('batch_size', 64)
    win         = config.get('news_window', 3)
    max_samples = config.get('phase2_max_samples')
    seed        = config.get('random_seed', 42)

    train_ds = Phase2Dataset(cache_dir, token_cache, trading_calendar,
                             'train', win, max_samples=max_samples, random_seed=seed)
    val_ds   = Phase2Dataset(cache_dir, token_cache, trading_calendar, 'val',  win)
    test_ds  = Phase2Dataset(cache_dir, token_cache, trading_calendar, 'test', win)

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False,
                              num_workers=0, pin_memory=pin)

    print(
        f"[dataset/p2] Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
        f"Test: {len(test_ds):,}  batch_size: {batch_sz}"
    )
    return train_loader, val_loader, test_loader


def create_val_test_dataloaders(
    cache_dir: str,
    config:    dict,
    splits:    Tuple[str, ...] = ('val', 'test'),
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Phase 1 val/test: sequential DataLoader over MultimodalStockDataset.

    News cache path is read from metadata.json and can be overridden via
    ``config['news_cache_path']``.
    """
    device   = config.get('device', 'cpu')
    pin      = device.startswith('cuda')
    batch_sz = config.get('batch_size', 1024)
    news_cp  = config.get('news_cache_path')
    win      = config.get('news_window')

    def _make(split: str) -> DataLoader:
        ds = MultimodalStockDataset(
            cache_dir, split=split, news_cache_path=news_cp, news_window=win,
        )
        return DataLoader(ds, batch_size=batch_sz, shuffle=False,
                          num_workers=0, pin_memory=pin)

    val_loader  = _make('val')  if 'val'  in splits else None
    test_loader = _make('test') if 'test' in splits else None

    parts: List[str] = []
    if val_loader:
        parts.append(f"Val: {len(val_loader.dataset):,}")
    if test_loader:
        parts.append(f"Test: {len(test_loader.dataset):,}")
    if parts:
        print(f"[dataset] {'  '.join(parts)}  batch_size: {batch_sz}")
    return val_loader, test_loader
