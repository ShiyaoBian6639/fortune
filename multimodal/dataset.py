"""
PyTorch Dataset and DataLoader utilities for the multimodal pipeline.

Phase 1 — MultimodalStockDataset
    Wraps {split}_price.npy, {split}_news.npy, {split}_labels.npy.
    Returns pre-computed BERT embeddings; BERT is never called during training.

Phase 2 — Phase2Dataset
    Wraps {split}_price.npy, {split}_labels.npy, {split}_dates.npy, and
    the per-day token cache (input_ids + attn_mask per article).
    Returns raw token tensors so BERT can be called inline per-batch,
    allowing gradients to flow back through the encoder.

Both follow the Windows-safe pickling pattern from dl/memmap_dataset.py
so DataLoader workers do not attempt to serialise memory-mapped arrays.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MultimodalStockDataset(Dataset):
    """
    Dataset backed by numpy memmap files for memory-efficient loading.

    __getitem__ returns (price_seq, news_seq, label) as float32/int64 tensors.
    """

    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = cache_dir
        self.split     = split

        meta_path = os.path.join(cache_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)

        split_info     = meta['splits'][split]
        self.n_samples = split_info['n_samples']

        # Defer opening memmaps until _open_memmaps() — avoids issues with
        # multiprocessing fork on Windows (same pattern as dl/memmap_dataset).
        self._price_mm = None
        self._news_mm  = None
        self._label_mm = None
        self._open_memmaps()

    def _open_memmaps(self):
        self._price_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_price.npy'),
            mmap_mode='r',
        )
        self._news_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_news.npy'),
            mmap_mode='r',
        )
        self._label_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_labels.npy'),
            mmap_mode='r',
        )

    # Windows-safe pickling for DataLoader with num_workers > 0
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_price_mm'] = None
        state['_news_mm']  = None
        state['_label_mm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        price = torch.as_tensor(self._price_mm[idx].copy(), dtype=torch.float32)
        news  = torch.as_tensor(self._news_mm[idx].copy(),  dtype=torch.float32)
        label = torch.tensor(int(self._label_mm[idx]),      dtype=torch.long)
        return price, news, label

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array — used for class-weight computation."""
        return np.array(self._label_mm)


class Phase2Dataset(Dataset):
    """
    Dataset for Phase 2 training where BERT is called inline per-batch.

    __getitem__ returns:
        price_seq      : float32  (30, 106)
        input_ids_win  : int32    (news_window, A, max_length)
        attn_mask_win  : int32    (news_window, A, max_length)
        n_articles_win : int32    (news_window,)
        label          : int64

    where A = max_articles_per_day and news_window = 3 by default.

    The ``token_cache`` dict maps YYYYMMDD date strings to per-day token dicts
    (see text_encoder.load_daily_token_cache).  ``trading_calendar`` is a sorted
    list of YYYYMMDD strings used to walk backwards through trading days for the
    rolling news window.
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
        self.trading_calendar = trading_calendar

        # Load token cache arrays and build O(1) date → index lookup
        self._token_cache = token_cache
        self._cal_index: Dict[str, int] = {d: i for i, d in enumerate(trading_calendar)}

        meta_path = os.path.join(cache_dir, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        n_total = meta['splits'][split]['n_samples']

        # Optionally subsample to keep Phase 2 epoch time practical.
        # The full dataset is chronologically ordered; random subsampling
        # preserves the date distribution across epochs via shuffle=True in
        # the DataLoader (no temporal bias introduced).
        if max_samples is not None and max_samples < n_total:
            rng = np.random.default_rng(random_seed)
            self._indices: Optional[np.ndarray] = rng.choice(
                n_total, max_samples, replace=False
            )
            self._indices.sort()   # keep memmap access roughly sequential
            self.n_samples = max_samples
        else:
            self._indices = None
            self.n_samples = n_total

        # Infer A and L from the first cache entry
        sample_entry = next(iter(token_cache.values())) if token_cache else None
        self._A = sample_entry['input_ids'].shape[0] if sample_entry else 16
        self._L = sample_entry['input_ids'].shape[1] if sample_entry else 128

        self._price_mm = None
        self._label_mm = None
        self._dates_mm = None
        self._open_memmaps()

    def _open_memmaps(self):
        self._price_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_price.npy'),
            mmap_mode='r',
        )
        self._label_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_labels.npy'),
            mmap_mode='r',
        )
        self._dates_mm = np.load(
            os.path.join(self.cache_dir, f'{self.split}_dates.npy'),
            mmap_mode='r',
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_price_mm'] = None
        state['_label_mm'] = None
        state['_dates_mm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx  = int(self._indices[idx]) if self._indices is not None else idx
        price     = torch.as_tensor(self._price_mm[real_idx].copy(), dtype=torch.float32)
        label     = torch.tensor(int(self._label_mm[real_idx]),      dtype=torch.long)
        date_str  = str(self._dates_mm[real_idx])

        W = self.news_window
        A = self._A
        L = self._L
        zero_entry = {'input_ids': np.zeros((A, L), np.int32),
                      'attn_mask': np.zeros((A, L), np.int32),
                      'n_articles': 0}

        cal_idx = self._cal_index.get(date_str, -1)
        ids_win   = np.zeros((W, A, L), dtype=np.int32)
        masks_win = np.zeros((W, A, L), dtype=np.int32)
        n_arts    = np.zeros((W,),      dtype=np.int32)

        for w in range(W):
            offset   = W - 1 - w           # w=0 → T-(W-1), w=W-1 → T
            day_idx  = cal_idx - offset
            if day_idx < 0:
                continue
            day = self.trading_calendar[day_idx]
            entry = self._token_cache.get(day, zero_entry)
            ids_win[w]   = entry['input_ids']
            masks_win[w] = entry['attn_mask']
            n_arts[w]    = entry['n_articles']

        return (
            price,
            torch.as_tensor(ids_win,   dtype=torch.long),
            torch.as_tensor(masks_win, dtype=torch.long),
            torch.as_tensor(n_arts,    dtype=torch.long),
            label,
        )

    def get_labels(self) -> np.ndarray:
        if self._indices is not None:
            # Index the memmap directly — avoids loading the full array (4M×8 bytes)
            # just to select the subset (e.g., 30K samples when max_samples is set).
            return self._label_mm[self._indices].copy()
        return np.array(self._label_mm)


def create_phase2_dataloaders(
    cache_dir:        str,
    token_cache:      Dict[str, Dict[str, np.ndarray]],
    trading_calendar: List[str],
    config:           dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Phase 2 DataLoaders that serve raw token tensors for inline BERT.
    """
    device       = config.get('device', 'cpu')
    pin          = device.startswith('cuda')
    batch_sz     = config.get('batch_size', 64)
    win          = config.get('news_window', 3)
    max_samples  = config.get('phase2_max_samples')
    seed         = config.get('random_seed', 42)

    train_ds = Phase2Dataset(cache_dir, token_cache, trading_calendar, 'train', win,
                             max_samples=max_samples, random_seed=seed)
    val_ds   = Phase2Dataset(cache_dir, token_cache, trading_calendar, 'val',   win)
    test_ds  = Phase2Dataset(cache_dir, token_cache, trading_calendar, 'test',  win)

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


class MultimodalChunkedLoader:
    """
    Background-thread chunked loader for Phase 1 training.

    Eliminates 100% disk usage caused by DataLoader workers making concurrent
    random seeks into 57 GB + 35 GB memmap files.

    Strategy (mirrors dl/memmap_dataset.py::ChunkedMemmapLoader):
      1. Divide N samples into sequential chunks of `chunk_samples` each.
      2. Shuffle chunk ORDER each epoch (epoch-level diversity).
      3. Load each chunk as one bulk sequential read → cheap for any disk type.
      4. Within each chunk, shuffle BATCH order (not individual samples) to
         avoid a costly ~1.2 GB gather that would stall the GPU for seconds.
      5. Within-batch shuffle via torch.randperm gives per-batch randomness
         over a ~13 MB slice — fits in CPU L3 cache, ~2 ms overhead.
      6. Background thread (depth-2) pre-loads the next chunk while the GPU
         trains on the current one — hides I/O latency completely.

    Peak RAM ≈ 3 × chunk_samples × (30×n_features + news_window×bert_dim) × 4 bytes
    Default 50K-sample chunk: 3 × ~1.2 GB ≈ 3.6 GB.  Tune via config['chunk_samples'].
    """

    def __init__(
        self,
        cache_dir:     str,
        split:         str   = 'train',
        batch_size:    int   = 1024,
        chunk_samples: int   = 50_000,
        seed:          int   = 42,
    ):
        self.cache_dir     = cache_dir
        self.split         = split
        self.batch_size    = batch_size
        self.chunk_samples = chunk_samples
        self._seed         = seed

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.n_samples = meta['splits'][split]['n_samples']

        # Open memmaps once — background thread shares these handles.
        # numpy memmap read-only access is thread-safe.
        self._price_mm = np.load(
            os.path.join(cache_dir, f'{split}_price.npy'), mmap_mode='r'
        )
        self._news_mm = np.load(
            os.path.join(cache_dir, f'{split}_news.npy'), mmap_mode='r'
        )
        self._label_mm = np.load(
            os.path.join(cache_dir, f'{split}_labels.npy'), mmap_mode='r'
        )

    def __len__(self) -> int:
        """Number of complete batches per epoch (used by training loop for loss avg)."""
        return self.n_samples // self.batch_size

    def _load_chunk(
        self, start: int, end: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Bulk sequential read of one contiguous memmap slice into RAM.

        np.array() forces a single contiguous read (not page-by-page),
        giving the OS a chance to issue an efficient sequential I/O request.
        Called from background thread — uses a local rng to avoid thread-safety issues.
        """
        rng = np.random.default_rng(seed)
        n   = end - start
        price  = torch.from_numpy(np.array(self._price_mm[start:end]))   # (N, 30, F)
        news   = torch.from_numpy(np.array(self._news_mm[start:end]))    # (N, W, 768)
        labels = torch.from_numpy(np.array(self._label_mm[start:end]))   # (N,)
        # Shuffle only batch order — permuting ~(N/bs) integers is negligible
        batch_order = rng.permutation(n // self.batch_size)
        return price, news, labels, batch_order

    def _yield_batches(
        self,
        chunk: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray],
    ):
        """Yield batches from an in-memory chunk with within-batch shuffle."""
        price, news, labels, batch_order = chunk
        bs = self.batch_size
        for b in batch_order:
            i, j = int(b) * bs, int(b) * bs + bs
            perm = torch.randperm(bs)   # within-batch shuffle — ~13 MB, L3-cached
            yield price[i:j][perm], news[i:j][perm], labels[i:j][perm]

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        # Advance seed so each epoch has a different chunk order
        self._seed = int(rng.integers(0, 2**31))

        chunk_starts = list(range(0, self.n_samples, self.chunk_samples))
        rng.shuffle(chunk_starts)

        # Synchronously pre-load first chunk
        s0 = chunk_starts[0]
        e0 = min(s0 + self.chunk_samples, self.n_samples)
        current = self._load_chunk(s0, e0, int(rng.integers(0, 2**31)))

        # Depth-2 prefetch: two threads load chunks N+1, N+2 while GPU trains on N.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: deque = deque()

            for s in chunk_starts[1:3]:
                e = min(s + self.chunk_samples, self.n_samples)
                futures.append(
                    pool.submit(self._load_chunk, s, e, int(rng.integers(0, 2**31)))
                )

            for s in chunk_starts[3:]:
                e = min(s + self.chunk_samples, self.n_samples)
                futures.append(
                    pool.submit(self._load_chunk, s, e, int(rng.integers(0, 2**31)))
                )
                yield from self._yield_batches(current)
                current = futures.popleft().result()

            yield from self._yield_batches(current)
            while futures:
                current = futures.popleft().result()
                yield from self._yield_batches(current)

    def get_labels(self) -> np.ndarray:
        """Return all labels (used for class-weight computation if needed)."""
        return np.array(self._label_mm)


def create_val_test_dataloaders(
    cache_dir: str,
    config:    dict,
    splits:    Tuple[str, ...] = ('val', 'test'),
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for val and/or test splits only.

    Phase 1 training uses MultimodalChunkedLoader for the train split;
    val/test use a plain DataLoader with num_workers=0 and no shuffle —
    sequential access is efficient for evaluation.
    """
    device   = config.get('device', 'cpu')
    pin      = device.startswith('cuda')
    batch_sz = config.get('batch_size', 1024)

    def _make(split: str) -> DataLoader:
        ds = MultimodalStockDataset(cache_dir, split)
        return DataLoader(ds, batch_size=batch_sz, shuffle=False,
                          num_workers=0, pin_memory=pin)

    val_loader  = _make('val')  if 'val'  in splits else None
    test_loader = _make('test') if 'test' in splits else None

    parts = []
    if val_loader:
        parts.append(f"Val: {len(val_loader.dataset):,}")
    if test_loader:
        parts.append(f"Test: {len(test_loader.dataset):,}")
    if parts:
        print(f"[dataset] {'  '.join(parts)}  batch_size: {batch_sz}")
    return val_loader, test_loader
