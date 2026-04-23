"""
Memory-mapped dataset for the deeptime regression pipeline.

Key differences from dl/memmap_dataset.py:
  - Labels are float32 (regression targets), shape (N, 5)
  - Stores anchor_dates for date-stratified batching
  - Stores sub_industry_ids and size_ids (extended static)
  - Separate observed-past and known-future arrays (not full feature array)
  - RegressionDataWriter builds rolling-window split inline (no post-process step)
"""

import gc
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .config import (
    SEQUENCE_LENGTH, MAX_FORWARD_WINDOW, NUM_HORIZONS,
    NUM_DT_OBSERVED_PAST, NUM_DT_KNOWN_FUTURE, NUM_DT_FEATURES,
    ROLLING_TRAIN_MONTHS, ROLLING_VAL_MONTHS, ROLLING_TEST_MONTHS,
    ROLLING_STEP_MONTHS, INTERLEAVED_TEST_START, PURGE_GAP_DAYS,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RegressionMemmapDataset(Dataset):
    """
    PyTorch Dataset backed by memory-mapped numpy arrays.

    Returns per-sample (10-tuple):
        obs_seq       (seq_len, n_past)   float32
        future_inputs (max_fw,  n_future) float32
        targets       (num_horizons,)     float32
        sector_id     ()  int64  — SW L1 sector
        industry_id   ()  int64  — SW L2 sub-industry
        sub_ind_id    ()  int64  — placeholder (zeros if SW L3 not available)
        size_id       ()  int64  — market-cap decile
        area_id       ()  int64  — province/region
        board_id      ()  int64  — exchange board type
        ipo_age_id    ()  int64  — IPO age bucket
        anchor_date   ()  int32
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = data_dir
        self.split    = split

        with open(os.path.join(data_dir, 'metadata.json')) as f:
            self.metadata = json.load(f)

        si = self.metadata['splits'][split]
        self.n_samples    = si['n_samples']
        self.seq_length   = self.metadata['seq_length']
        self.n_past       = self.metadata['n_past']
        self.n_future     = self.metadata['n_future']
        self.num_horizons = self.metadata['num_horizons']
        self.max_fw       = self.metadata['max_fw']

        self._open_memmaps()

    def _open_memmaps(self):
        s = self.split; n = self.n_samples; d = self.data_dir

        def _mm(fname, dtype, shape):
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return np.memmap(path, dtype=dtype, mode='r', shape=shape)
            return np.zeros(shape, dtype=dtype)

        # Use plain .npy memmap files (float32)
        self.obs_seqs      = _mm(f'{s}_obs.npy',        'float32', (n, self.seq_length, self.n_past))
        self.future_inputs = _mm(f'{s}_future.npy',     'float32', (n, self.max_fw, self.n_future))
        self.targets       = _mm(f'{s}_targets.npy',    'float32', (n, self.num_horizons))
        self.sector_ids    = _mm(f'{s}_sectors.npy',    'int64',   (n,))
        self.industry_ids  = _mm(f'{s}_industries.npy', 'int64',   (n,))
        self.sub_ind_ids   = _mm(f'{s}_sub_inds.npy',   'int64',   (n,))
        self.size_ids      = _mm(f'{s}_sizes.npy',      'int64',   (n,))
        self.area_ids      = _mm(f'{s}_areas.npy',      'int64',   (n,))
        self.board_ids     = _mm(f'{s}_boards.npy',     'int64',   (n,))
        self.ipo_age_ids   = _mm(f'{s}_ipo_ages.npy',   'int64',   (n,))
        self.anchor_dates  = _mm(f'{s}_dates.npy',      'int32',   (n,))

    _STATIC_KEYS = ('obs_seqs', 'future_inputs', 'targets',
                    'sector_ids', 'industry_ids', 'sub_ind_ids', 'size_ids',
                    'area_ids', 'board_ids', 'ipo_age_ids', 'anchor_dates')

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._STATIC_KEYS:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs_seqs[idx].copy()),
            torch.from_numpy(self.future_inputs[idx].copy()),
            torch.from_numpy(self.targets[idx].copy()),
            int(self.sector_ids[idx]),
            int(self.industry_ids[idx]),
            int(self.sub_ind_ids[idx]),
            int(self.size_ids[idx]),
            int(self.area_ids[idx]),
            int(self.board_ids[idx]),
            int(self.ipo_age_ids[idx]),
            int(self.anchor_dates[idx]),
        )


class DateStratifiedSampler(Sampler):
    """
    Samples batches where all items share the same anchor_date block.
    Ensures sector pooling sees a real cross-section (same trading day).
    Shuffles date order between epochs, but keeps within-date order.
    """

    def __init__(self, dataset: RegressionMemmapDataset, batch_size: int):
        self.dataset    = dataset
        self.batch_size = batch_size
        dates = dataset.anchor_dates[:]
        unique_dates = np.unique(dates)
        # {date → list of indices}
        self.date_to_idx = {d: np.where(dates == d)[0] for d in unique_dates}
        self.unique_dates = unique_dates

    def __iter__(self):
        perm = np.random.permutation(len(self.unique_dates))
        for d in self.unique_dates[perm]:
            idxs = self.date_to_idx[d]
            # Shuffle within date
            idxs = idxs[np.random.permutation(len(idxs))]
            # Yield batch-size chunks
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                for idx in batch:
                    yield int(idx)

    def __len__(self):
        return len(self.dataset)


# ─── RAM-cached loader (fastest if you have enough RAM) ───────────────────────

class RegressionRAMLoader:
    """
    Loads entire dataset into RAM at startup for maximum throughput.

    Use this if you have enough RAM (dataset size + ~50% overhead).
    For seq_len=60, n_past=204, n_samples=4.8M: ~250 GB RAM needed.

    For smaller datasets or subsets, this gives near 100% GPU utilization.
    """

    def __init__(
        self,
        data_dir:   str,
        split:      str,
        batch_size: int,
        device:     str = 'cpu',
        shuffle:    bool = True,
    ):
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle

        with open(os.path.join(data_dir, 'metadata.json')) as f:
            meta = json.load(f)

        si = meta['splits'][split]
        self.n_samples = si['n_samples']
        seq_len  = meta['seq_length']
        n_past   = meta['n_past']
        n_future = meta['n_future']
        max_fw   = meta['max_fw']
        n_horiz  = meta['num_horizons']

        # Estimate RAM needed
        bytes_total = (
            self.n_samples * seq_len * n_past * 4 +      # obs
            self.n_samples * max_fw * n_future * 4 +     # future
            self.n_samples * n_horiz * 4 +               # targets
            self.n_samples * 7 * 8                        # int64 static IDs
        )
        gb_needed = bytes_total / 1024**3
        print(f"  [RAM Loader] Loading {self.n_samples:,} samples ({gb_needed:.1f} GB) into RAM...")

        import time
        t0 = time.time()

        # Load ALL data into RAM (contiguous numpy arrays)
        def _load(fname, dtype, shape):
            path = os.path.join(data_dir, fname)
            mm = np.memmap(path, dtype=dtype, mode='r', shape=shape)
            return np.array(mm)  # copy to RAM

        self._obs     = _load(f'{split}_obs.npy',        'float32', (self.n_samples, seq_len, n_past))
        self._future  = _load(f'{split}_future.npy',     'float32', (self.n_samples, max_fw, n_future))
        self._targets = _load(f'{split}_targets.npy',    'float32', (self.n_samples, n_horiz))
        self._sectors = _load(f'{split}_sectors.npy',    'int64',   (self.n_samples,))
        self._inds    = _load(f'{split}_industries.npy', 'int64',   (self.n_samples,))
        self._subs    = _load(f'{split}_sub_inds.npy',   'int64',   (self.n_samples,))
        self._sizes   = _load(f'{split}_sizes.npy',      'int64',   (self.n_samples,))
        self._areas   = _load(f'{split}_areas.npy',      'int64',   (self.n_samples,))
        self._boards  = _load(f'{split}_boards.npy',     'int64',   (self.n_samples,))
        self._ipo     = _load(f'{split}_ipo_ages.npy',   'int64',   (self.n_samples,))

        print(f"  [RAM Loader] Loaded in {time.time()-t0:.1f}s")

        self._n_batches = (self.n_samples + batch_size - 1) // batch_size
        self._use_pinned = (device != 'cpu' and torch.cuda.is_available())

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for b_start in range(0, self.n_samples, self.batch_size):
            b_end = min(b_start + self.batch_size, self.n_samples)
            idx = indices[b_start:b_end]

            # Direct RAM access (no disk I/O)
            obs_t    = torch.from_numpy(self._obs[idx])
            future_t = torch.from_numpy(self._future[idx])
            tgt_t    = torch.from_numpy(self._targets[idx])
            sec_t    = torch.from_numpy(self._sectors[idx])
            ind_t    = torch.from_numpy(self._inds[idx])
            sub_t    = torch.from_numpy(self._subs[idx])
            sz_t     = torch.from_numpy(self._sizes[idx])
            area_t   = torch.from_numpy(self._areas[idx])
            board_t  = torch.from_numpy(self._boards[idx])
            ipo_t    = torch.from_numpy(self._ipo[idx])

            if self._use_pinned:
                obs_t = obs_t.pin_memory()
                future_t = future_t.pin_memory()
                tgt_t = tgt_t.pin_memory()

            yield (
                obs_t.to(self.device, non_blocking=True),
                future_t.to(self.device, non_blocking=True),
                tgt_t.to(self.device, non_blocking=True),
                sec_t.to(self.device, non_blocking=True),
                ind_t.to(self.device, non_blocking=True),
                sub_t.to(self.device, non_blocking=True),
                sz_t.to(self.device, non_blocking=True),
                area_t.to(self.device, non_blocking=True),
                board_t.to(self.device, non_blocking=True),
                ipo_t.to(self.device, non_blocking=True),
            )


# ─── Chunked background loader (optimized for RTX 5090) ───────────────────────

class RegressionChunkedLoader:
    """
    Background-thread prefetching loader with async GPU transfer.

    Optimizations for high-end GPUs (RTX 5090 etc):
      1. Auto-scaling chunk size: max(100K, batch_size * 50) to amortize I/O
      2. Double-buffering: 2 threads prefetch chunks in parallel
      3. Pinned memory: faster CPU→GPU DMA transfers
      4. Async transfer: non_blocking=True overlaps compute and transfer
      5. Pre-allocated pinned buffers: avoid per-batch allocation overhead

    With batch=4096 on RTX 5090, this achieves ~80-95% GPU utilization.
    """

    def __init__(
        self,
        data_dir:   str,
        split:      str,
        batch_size: int,
        chunk_samples: int = 40_000,
        device:     str = 'cpu',
        shuffle:    bool = True,
        prefetch_factor: int = 2,   # number of chunks to prefetch
        max_chunk_gb: float = None, # manual override for max chunk size in GB
    ):
        self.data_dir   = data_dir
        self.split      = split
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle
        self.prefetch_factor = prefetch_factor

        with open(os.path.join(data_dir, 'metadata.json')) as f:
            meta = json.load(f)

        si = meta['splits'][split]
        self.n_samples    = si['n_samples']
        self.seq_length   = meta['seq_length']
        self.n_past       = meta['n_past']
        self.n_future     = meta['n_future']
        self.num_horizons = meta['num_horizons']
        self.max_fw       = meta['max_fw']

        # Auto-scale chunk size based on AVAILABLE system RAM
        # obs array: chunk_samples × seq_len × n_past × 4 bytes
        bytes_per_sample = self.seq_length * self.n_past * 4 + self.max_fw * self.n_future * 4 + 100

        # Detect available RAM - check container cgroup limits first
        avail_ram = None
        cgroup_limit = None

        # Check cgroup memory limit (Docker/K8s containers)
        for cgroup_path in [
            '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
            '/sys/fs/cgroup/memory.max',                     # cgroup v2
        ]:
            try:
                with open(cgroup_path) as f:
                    val = f.read().strip()
                    if val != 'max' and int(val) < 500 * 1024**3:  # ignore if >500GB (no limit)
                        cgroup_limit = int(val)
                        break
            except:
                pass

        # Get system available RAM
        try:
            import psutil
            sys_avail = psutil.virtual_memory().available
        except ImportError:
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            sys_avail = int(line.split()[1]) * 1024
                            break
                    else:
                        sys_avail = 64 * 1024**3
            except:
                sys_avail = 64 * 1024**3

        # Use container limit if set, otherwise system available
        if cgroup_limit is not None:
            avail_ram = int(cgroup_limit * 0.8)  # 80% of container limit
            print(f"  [Container] cgroup limit: {cgroup_limit/1024**3:.0f}GB, using {avail_ram/1024**3:.0f}GB")
        else:
            avail_ram = sys_avail

        # Memory budget: prefetch buffers + active chunk + shuffle temp copy
        # Peak usage = (prefetch_factor + 2) × chunk_size
        if max_chunk_gb is not None:
            # User override - trust their memory knowledge
            target_chunk_bytes = int(max_chunk_gb * 1024**3)
            print(f"  [Manual] max_chunk_gb={max_chunk_gb} → {target_chunk_bytes/1024**3:.1f}GB per chunk")
        else:
            # Auto-detect: Use 60% of available RAM for safety margin
            usable_ram = int(avail_ram * 0.6)
            n_buffers = prefetch_factor + 2  # prefetch + active + shuffle copy
            target_chunk_bytes = usable_ram // n_buffers
            # Safety bounds: 2GB min, 8GB max per chunk (conservative for containers)
            target_chunk_bytes = max(target_chunk_bytes, 2 * 1024**3)
            target_chunk_bytes = min(target_chunk_bytes, 8 * 1024**3)  # cap at 8GB
        mem_limited_chunk = int(target_chunk_bytes / bytes_per_sample)

        # Also ensure at least 20 batches per chunk for GPU efficiency
        min_chunk = batch_size * 20

        # Honor user request if it fits in memory, otherwise cap
        if chunk_samples <= mem_limited_chunk:
            effective = max(min_chunk, chunk_samples)
        else:
            effective = max(min_chunk, mem_limited_chunk)
        self.chunk_samples = min(effective, self.n_samples)

        # Log effective chunk size
        chunk_gb = self.chunk_samples * bytes_per_sample / 1024**3
        avail_gb = avail_ram / 1024**3
        print(f"  Chunk: {self.chunk_samples:,} samples ({chunk_gb:.1f} GB) × {prefetch_factor} prefetch | {avail_gb:.0f}GB RAM avail")

        self._open_memmaps()
        self._n_batches = max(1, (self.n_samples + batch_size - 1) // batch_size)

        # Pinned memory for faster GPU transfers (only on CUDA)
        self._use_pinned = (device != 'cpu' and torch.cuda.is_available())

    def _open_memmaps(self):
        s = self.split; n = self.n_samples; d = self.data_dir

        def _mm(fname, dtype, shape):
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return np.memmap(path, dtype=dtype, mode='r', shape=shape)
            return np.zeros(shape, dtype=dtype)   # zero-fill for old caches

        # Use plain .npy memmap files (float32)
        self._obs     = _mm(f'{s}_obs.npy',        'float32', (n, self.seq_length, self.n_past))
        self._future  = _mm(f'{s}_future.npy',     'float32', (n, self.max_fw, self.n_future))
        self._targets = _mm(f'{s}_targets.npy',    'float32', (n, self.num_horizons))
        self._sectors = _mm(f'{s}_sectors.npy',    'int64',   (n,))
        self._inds    = _mm(f'{s}_industries.npy', 'int64',   (n,))
        self._subs    = _mm(f'{s}_sub_inds.npy',   'int64',   (n,))
        self._sizes   = _mm(f'{s}_sizes.npy',      'int64',   (n,))
        self._areas   = _mm(f'{s}_areas.npy',      'int64',   (n,))
        self._boards  = _mm(f'{s}_boards.npy',     'int64',   (n,))
        self._ipo     = _mm(f'{s}_ipo_ages.npy',   'int64',   (n,))
        self._dates   = _mm(f'{s}_dates.npy',      'int32',   (n,))

    def __len__(self):
        return self._n_batches

    def _to_pinned(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to pinned memory tensor for fast GPU transfer."""
        t = torch.from_numpy(arr)
        if self._use_pinned:
            return t.pin_memory()
        return t

    def __iter__(self):
        # Shuffle CHUNK ORDER (not sample indices) so each chunk is a contiguous
        # memmap slice — sequential I/O is 5-10× faster than fancy indexing on
        # a multi-hundred-GB memmap. Global randomness comes from (a) shuffled
        # chunk order, (b) a per-epoch random byte offset that shifts chunk
        # boundaries (wrap-around at the end), and (c) a single intra-chunk
        # permutation of samples so each batch sees a wide cross-section of
        # stocks (the writer stores stocks contiguously — without per-sample
        # shuffle, a batch would contain only 4-5 stocks).
        n_chunks = (self.n_samples + self.chunk_samples - 1) // self.chunk_samples
        chunk_order = np.arange(n_chunks)
        epoch_offset = 0
        if self.shuffle:
            np.random.shuffle(chunk_order)
            epoch_offset = int(np.random.randint(0, max(self.chunk_samples, 1)))

        executor = ThreadPoolExecutor(max_workers=self.prefetch_factor)

        def _read_range(arr, start, end):
            """Contiguous memmap slice with wrap-around when start+size > N."""
            N = self.n_samples
            if end <= N:
                return arr[start:end].copy()
            a = arr[start:N].copy()
            b = arr[:end - N].copy()
            return np.concatenate([a, b], axis=0)

        def _load_chunk(chunk_idx_in_order):
            actual_chunk = int(chunk_order[chunk_idx_in_order])
            start = (actual_chunk * self.chunk_samples + epoch_offset) % self.n_samples
            end   = start + self.chunk_samples  # may overshoot; _read_range wraps

            obs     = _read_range(self._obs,     start, end)
            future  = _read_range(self._future,  start, end)
            targets = _read_range(self._targets, start, end)
            sectors = _read_range(self._sectors, start, end)
            inds    = _read_range(self._inds,    start, end)
            subs    = _read_range(self._subs,    start, end)
            sizes   = _read_range(self._sizes,   start, end)
            areas   = _read_range(self._areas,   start, end)
            boards  = _read_range(self._boards,  start, end)
            ipo     = _read_range(self._ipo,     start, end)

            # Intra-chunk sample shuffle so each batch has diverse stocks
            n = len(obs)
            if self.shuffle and n > 1:
                perm    = np.random.permutation(n)
                obs     = obs[perm];     future = future[perm]; targets = targets[perm]
                sectors = sectors[perm]; inds   = inds[perm];   subs    = subs[perm]
                sizes   = sizes[perm];   areas  = areas[perm];  boards  = boards[perm]
                ipo     = ipo[perm]

            # Pin memory in the prefetch thread so the ~0.4-0.6 s pinning cost
            # overlaps with the GPU consuming the previous chunk. pin_memory()
            # is thread-safe after CUDA init.
            return (
                self._to_pinned(obs),
                self._to_pinned(future),
                self._to_pinned(targets),
                self._to_pinned(sectors),
                self._to_pinned(inds),
                self._to_pinned(subs),
                self._to_pinned(sizes),
                self._to_pinned(areas),
                self._to_pinned(boards),
                self._to_pinned(ipo),
            )

        n_actual = len(chunk_order)

        futures = []
        for i in range(min(self.prefetch_factor, n_actual)):
            futures.append(executor.submit(_load_chunk, i))

        for chunk_i in range(n_actual):
            chunk = futures[0].result()
            futures.pop(0)

            next_chunk_i = chunk_i + self.prefetch_factor
            if next_chunk_i < n_actual:
                futures.append(executor.submit(_load_chunk, next_chunk_i))

            obs_t, future_t, tgt_t, sec_t, ind_t, sub_t, sz_t, area_t, board_t, ipo_t = chunk
            n = obs_t.shape[0]

            for b_start in range(0, n, self.batch_size):
                b_end = min(b_start + self.batch_size, n)
                yield (
                    obs_t[b_start:b_end].to(self.device, non_blocking=True),
                    future_t[b_start:b_end].to(self.device, non_blocking=True),
                    tgt_t[b_start:b_end].to(self.device, non_blocking=True),
                    sec_t[b_start:b_end].to(self.device, non_blocking=True),
                    ind_t[b_start:b_end].to(self.device, non_blocking=True),
                    sub_t[b_start:b_end].to(self.device, non_blocking=True),
                    sz_t[b_start:b_end].to(self.device, non_blocking=True),
                    area_t[b_start:b_end].to(self.device, non_blocking=True),
                    board_t[b_start:b_end].to(self.device, non_blocking=True),
                    ipo_t[b_start:b_end].to(self.device, non_blocking=True),
                )

        executor.shutdown(wait=False)


# ─── Data writer (streaming — no in-memory accumulation) ─────────────────────

class RegressionDataWriter:
    """
    Two-pass streaming writer. Peak RAM ≈ O(1 stock at a time).

    Pass 1 (external): caller runs `scan_dates()` across all stocks → builds
    split map + sequence counts per split → calls `setup(split_counts)` to
    pre-allocate memmap files on disk.

    Pass 2 (external): caller runs full feature engineering per stock →
    calls `write_batch()` for each stock's sequences → written directly to
    the pre-allocated split memmaps. No concatenation step needed.
    """

    def __init__(
        self,
        output_dir:   str,
        seq_length:   int,
        n_features:   int,
        n_past:       int,
        n_future:     int,
        num_horizons: int,
        max_fw:       int,
    ):
        self.output_dir   = output_dir
        self.seq_length   = seq_length
        self.n_features   = n_features
        self.n_past       = n_past
        self.n_future     = n_future
        self.num_horizons = num_horizons
        self.max_fw       = max_fw

        os.makedirs(output_dir, exist_ok=True)

        # Populated by setup()
        self._mms:        Dict[str, Dict[str, np.memmap]] = {}
        self._write_idx:  Dict[str, int] = {}
        self._capacities: Dict[str, int] = {}
        self.date_split_map: Dict[int, str] = {}
        self.total_written: int = 0

    # ── Pass-1 helper: build split map from collected anchor dates ────────────

    def build_split_map(self, all_anchor_dates: np.ndarray,
                        split_mode: str, config: dict) -> None:
        """Build date→split mapping and store in self.date_split_map."""
        self.date_split_map = self._build_split_map(all_anchor_dates, split_mode, config)

    # ── Pass-1 helper: pre-allocate memmap files of known capacity ────────────

    def setup(self, split_counts: Dict[str, int]) -> None:
        """
        Pre-allocate per-split memmap files. Must be called after build_split_map().

        split_counts: {split_name: n_sequences}  (can be over-estimates)
        """
        d = self.output_dir
        for split, n in split_counts.items():
            if n == 0:
                continue
            self._capacities[split] = n
            self._write_idx[split]  = 0
            self._mms[split] = {
                'obs':        np.memmap(os.path.join(d, f'{split}_obs.npy'),        dtype='float32', mode='w+', shape=(n, self.seq_length, self.n_past)),
                'future':     np.memmap(os.path.join(d, f'{split}_future.npy'),     dtype='float32', mode='w+', shape=(n, self.max_fw,       self.n_future)),
                'targets':    np.memmap(os.path.join(d, f'{split}_targets.npy'),    dtype='float32', mode='w+', shape=(n, self.num_horizons)),
                'sectors':    np.memmap(os.path.join(d, f'{split}_sectors.npy'),    dtype='int64',   mode='w+', shape=(n,)),
                'industries': np.memmap(os.path.join(d, f'{split}_industries.npy'), dtype='int64',   mode='w+', shape=(n,)),
                'sub_inds':   np.memmap(os.path.join(d, f'{split}_sub_inds.npy'),   dtype='int64',   mode='w+', shape=(n,)),
                'sizes':      np.memmap(os.path.join(d, f'{split}_sizes.npy'),      dtype='int64',   mode='w+', shape=(n,)),
                'areas':      np.memmap(os.path.join(d, f'{split}_areas.npy'),      dtype='int64',   mode='w+', shape=(n,)),
                'boards':     np.memmap(os.path.join(d, f'{split}_boards.npy'),     dtype='int64',   mode='w+', shape=(n,)),
                'ipo_ages':   np.memmap(os.path.join(d, f'{split}_ipo_ages.npy'),   dtype='int64',   mode='w+', shape=(n,)),
                'dates':      np.memmap(os.path.join(d, f'{split}_dates.npy'),      dtype='int32',   mode='w+', shape=(n,)),
            }

    # ── Pass-2: stream-write one stock's sequences ────────────────────────────

    def write_batch(
        self,
        obs_seqs:     np.ndarray,  # (N, seq_len, n_past)
        future_seqs:  np.ndarray,  # (N, max_fw, n_future)
        targets:      np.ndarray,  # (N, num_horizons)
        sector_ids:   np.ndarray,  # (N,)
        ind_ids:      np.ndarray,  # (N,)
        sub_ids:      np.ndarray,  # (N,)
        size_ids:     np.ndarray,  # (N,)
        anchor_dates: np.ndarray,  # (N,)
        area_ids:     np.ndarray = None,  # (N,)
        board_ids:    np.ndarray = None,  # (N,)
        ipo_age_ids:  np.ndarray = None,  # (N,)
    ) -> None:
        """Write sequences directly to the appropriate pre-allocated split memmaps."""
        N = len(obs_seqs)
        _zeros = lambda: np.zeros(N, dtype=np.int64)
        if area_ids    is None: area_ids    = _zeros()
        if board_ids   is None: board_ids   = _zeros()
        if ipo_age_ids is None: ipo_age_ids = _zeros()

        # Route each sequence to its split
        split_masks: Dict[str, np.ndarray] = {}
        for split in ('train', 'val', 'test'):
            mask = np.array([
                self.date_split_map.get(int(d), 'gap') == split
                for d in anchor_dates
            ], dtype=bool)
            split_masks[split] = mask

        for split, mask in split_masks.items():
            n = int(mask.sum())
            if n == 0 or split not in self._mms:
                continue
            idx = self._write_idx[split]
            cap = self._capacities[split]
            if idx + n > cap:
                n = max(0, cap - idx)
                mask_idx = np.where(mask)[0][:n]
                mask = np.zeros(len(anchor_dates), dtype=bool)
                mask[mask_idx] = True

            mms = self._mms[split]
            mms['obs'][idx:idx+n]        = obs_seqs[mask]
            mms['future'][idx:idx+n]     = future_seqs[mask]
            mms['targets'][idx:idx+n]    = targets[mask]
            mms['sectors'][idx:idx+n]    = sector_ids[mask]
            mms['industries'][idx:idx+n] = ind_ids[mask]
            mms['sub_inds'][idx:idx+n]   = sub_ids[mask]
            mms['sizes'][idx:idx+n]      = size_ids[mask]
            mms['areas'][idx:idx+n]      = area_ids[mask]
            mms['boards'][idx:idx+n]     = board_ids[mask]
            mms['ipo_ages'][idx:idx+n]   = ipo_age_ids[mask]
            mms['dates'][idx:idx+n]      = anchor_dates[mask]
            self._write_idx[split] += n
            self.total_written += n

    # ── Finalize: flush, write metadata with ACTUAL counts ────────────────────

    def close(self, split_mode: str, config: dict) -> dict:
        """Flush all memmaps and write metadata with actual written counts."""
        for split, mms in self._mms.items():
            for mm in mms.values():
                mm.flush()
                del mm
        self._mms.clear()
        gc.collect()

        actual_counts = {split: self._write_idx.get(split, 0)
                         for split in ('train', 'val', 'test')}

        for split, n in actual_counts.items():
            if n > 0:
                print(f"  {split}: {n:,} samples")
                if n < self._capacities.get(split, n):
                    pct = 100.0 * n / self._capacities[split]
                    print(f"    (used {pct:.0f}% of pre-allocated capacity)")

        metadata = {
            'seq_length':    self.seq_length,
            'n_features':    self.n_features,
            'n_past':        self.n_past,
            'n_future':      self.n_future,
            'num_horizons':  self.num_horizons,
            'max_fw':        self.max_fw,
            'total_samples': self.total_written,
            'split_mode':    split_mode,
            'splits': {name: {'n_samples': n} for name, n in actual_counts.items()},
        }
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata written to {self.output_dir}/metadata.json")
        return metadata

    # ── Legacy shim: kept for backward compat (not used in new two-pass flow) ─

    def add_batch(self, *args, **kwargs):
        raise RuntimeError(
            "RegressionDataWriter.add_batch() is deprecated. "
            "Use the two-pass flow: build_split_map() → setup() → write_batch() → close()."
        )

    def finalize(self, split_mode: str = 'rolling_window', config: dict = None) -> dict:
        return self.close(split_mode, config or {})

    def _build_split_map(
        self,
        all_dates:  np.ndarray,
        split_mode: str,
        config:     dict,
    ) -> Dict[int, str]:
        """Build {date_int → 'train'|'val'|'test'|'gap'} for rolling_window split."""
        unique_dates = np.unique(all_dates.astype(np.int32))
        date_map     = {int(d): 'train' for d in unique_dates}

        test_start = config.get('interleaved_test_start', INTERLEAVED_TEST_START)
        purge      = config.get('purge_gap_days', PURGE_GAP_DAYS)

        if split_mode == 'temporal':
            # Chronological 70/15/15 split with purge gaps at both boundaries.
            # Guarantees balanced val≈test and no sequence-level contamination.
            sorted_dates = sorted(date_map.keys())
            n = len(sorted_dates)
            # Boundary indices (strict chronological)
            tv_boundary = int(n * 0.70)   # train→val
            vt_boundary = int(n * 0.85)   # val→test
            for i, d in enumerate(sorted_dates):
                if i < tv_boundary:
                    date_map[d] = 'train'
                elif i < vt_boundary:
                    date_map[d] = 'val'
                else:
                    date_map[d] = 'test'
            # Purge gap at train→val boundary
            for i in range(max(0, tv_boundary - purge), tv_boundary):
                if date_map[sorted_dates[i]] == 'train':
                    date_map[sorted_dates[i]] = 'gap'
            # Purge gap at val→test boundary
            for i in range(max(0, vt_boundary - purge), vt_boundary):
                if date_map[sorted_dates[i]] == 'val':
                    date_map[sorted_dates[i]] = 'gap'
            counts = {}
            for v in date_map.values():
                counts[v] = counts.get(v, 0) + 1
            total = len(sorted_dates)
            print(f"\n  Temporal 70/15/15 split (purge={purge} days at each boundary):")
            for sp in ('train','val','test','gap'):
                n_sp = counts.get(sp, 0)
                print(f"    {sp:6s}: {n_sp:5d} days ({100.*n_sp/total:.1f}%)")
            return date_map

        # Always mark global holdout as test (rolling_window mode only)
        for d in unique_dates:
            if int(d) >= test_start:
                date_map[int(d)] = 'test'

        if split_mode not in ('rolling_window',):
            # Unknown mode — fall back to temporal
            return self._build_split_map(all_dates, 'temporal', config)

        # Rolling window split
        first_date = pd.Timestamp(str(int(unique_dates[0])))
        cursor     = first_date.replace(day=1)
        test_ts    = pd.Timestamp(str(test_start))

        train_m = config.get('rolling_train_months', ROLLING_TRAIN_MONTHS)
        val_m   = config.get('rolling_val_months',   ROLLING_VAL_MONTHS)
        test_m  = config.get('rolling_test_months',  ROLLING_TEST_MONTHS)
        step_m  = config.get('rolling_step_months',  ROLLING_STEP_MONTHS)

        folds = []
        while True:
            val_s  = cursor + pd.DateOffset(months=train_m)
            val_e  = val_s   + pd.DateOffset(months=val_m)
            test_s = val_e
            test_e = test_s  + pd.DateOffset(months=test_m)
            if test_e > test_ts:
                break
            folds.append((val_s, val_e, test_s, test_e))
            cursor += pd.DateOffset(months=step_m)

        def _mark(start_ts, end_ts, label):
            s = int(start_ts.strftime('%Y%m%d'))
            e = int(end_ts.strftime('%Y%m%d'))
            si = int(np.searchsorted(unique_dates, s, 'left'))
            ei = int(np.searchsorted(unique_dates, e, 'left'))
            for i in range(si, ei):
                if date_map.get(int(unique_dates[i])) == 'train':
                    date_map[int(unique_dates[i])] = label
            return si

        def _purge_before(boundary_idx, n):
            for i in range(max(0, boundary_idx - n), boundary_idx):
                if date_map.get(int(unique_dates[i])) == 'train':
                    date_map[int(unique_dates[i])] = 'gap'

        # Pass 1: mark test windows
        for (_, _, test_s, test_e) in folds:
            _mark(test_s, test_e, 'test')

        # Pass 2: mark val windows + purge before each val start
        for (val_s, val_e, _, _) in folds:
            si = _mark(val_s, val_e, 'val')
            _purge_before(si, purge)

        # Purge before global holdout
        holdout_idx = int(np.searchsorted(unique_dates, test_start, 'left'))
        _purge_before(holdout_idx, purge)

        counts = {}
        for v in date_map.values():
            counts[v] = counts.get(v, 0) + 1
        total = len(date_map)
        print(f"\n  Rolling split ({train_m}m train / {val_m}m val / {test_m}m test, "
              f"step={step_m}m, {len(folds)} folds, purge={purge} days):")
        for sp in ('train', 'val', 'test', 'gap'):
            n = counts.get(sp, 0)
            print(f"    {sp:6s}: {n:5d} days ({100.*n/total:.1f}%)")

        return date_map


# ─── Post-processing feature normalization ────────────────────────────────────

def normalize_cache(
    cache_dir:  str,
    chunk_size: int = 2_000,
    reservoir_size: int = 50_000,
    force: bool = False,
) -> None:
    """
    Post-process an existing cache with two-tier feature normalization.

    Fits ALL statistics on the TRAIN split only to prevent leakage.
    Applies in-place to train/val/test obs memmaps (modifies files on disk).

    Tier 1 — fina indicators + market MACD/MTM (percentile clip + standardize):
        These are unbounded and raw. Clip to [p5, p95] fitted on train,
        then rescale to zero-mean unit-variance.  Handles ROE=300%, MTM spikes.

    Tier 2 — all other features (safety clip at ±5σ):
        After CS normalization most features are ~N(0,1). This clips any
        residual outliers (gap stocks, ST warnings, penny-stock extremes)
        without altering the distribution for well-behaved features.

    Idempotent: if metadata shows 'normalized=true' the function returns early
    unless force=True.
    """
    meta_path = os.path.join(cache_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get('normalized') and not force:
        print("  Cache already normalized (use force=True to reapply).")
        return

    from .config import (
        DT_OBSERVED_PAST_COLUMNS, FINA_INDICATOR_COLUMNS, EXTRA_MONEYFLOW_COLUMNS
    )

    n_train = meta['splits']['train']['n_samples']
    T       = meta['seq_length']
    n_past  = meta['n_past']

    if n_train == 0:
        print("  [skip] normalize_cache: empty train split")
        return

    # ── Identify feature tiers ────────────────────────────────────────────────
    # Tier 1: fina ratios + unbounded market indicators (MACD, MTM, pe_ttm)
    TIER1_KWS = ('_macd', '_mtm', 'sse_pe_ttm', 'sse50_pe_ttm')
    tier1_set = set(FINA_INDICATOR_COLUMNS)   # fina always Tier 1
    tier1_set.add('has_fina_data')            # binary flag: treat like fina

    tier1_idx = []
    for i, feat in enumerate(DT_OBSERVED_PAST_COLUMNS[:n_past]):
        if feat in tier1_set or any(kw in feat for kw in TIER1_KWS):
            tier1_idx.append(i)
    tier1_idx = np.array(tier1_idx, dtype=np.intp)
    print(f"  Tier 1 (clip+standardize): {len(tier1_idx)} features")

    # ── Pass 1: compute statistics from train split (chunked, low RAM) ────────
    print(f"  Computing feature stats from {n_train:,} train sequences...")
    train_obs, _ = _open_zarr_or_memmap(
        os.path.join(cache_dir, 'train_obs'), 'float32', (n_train, T, n_past)
    )

    feat_sum    = np.zeros(n_past, dtype='float64')
    feat_sq_sum = np.zeros(n_past, dtype='float64')
    total_rows  = 0
    reservoir   = np.zeros((reservoir_size, n_past), dtype='float32')
    res_n       = 0   # total reservoir candidates seen

    for start in range(0, n_train, chunk_size):
        end   = min(start + chunk_size, n_train)
        # Flatten time dim: (chunk, T, F) → (chunk*T, F)
        chunk = train_obs[start:end].reshape(-1, n_past).astype('float64')
        # Replace NaN/Inf with 0 for stats computation
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        feat_sum    += chunk.sum(axis=0)
        feat_sq_sum += (chunk ** 2).sum(axis=0)
        total_rows  += len(chunk)

        # Reservoir sampling for percentile estimation (last timestep per seq)
        last = train_obs[start:end, -1, :].astype('float32')  # (chunk, F)
        for i in range(len(last)):
            res_n += 1
            if res_n <= reservoir_size:
                reservoir[res_n - 1] = last[i]
            else:
                j = np.random.randint(0, res_n)
                if j < reservoir_size:
                    reservoir[j] = last[i]

    del train_obs

    mean = feat_sum / max(total_rows, 1)
    var  = feat_sq_sum / max(total_rows, 1) - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-10))

    res_valid = min(res_n, reservoir_size)
    p5  = np.percentile(reservoir[:res_valid], 5,  axis=0).astype('float64')
    p95 = np.percentile(reservoir[:res_valid], 95, axis=0).astype('float64')

    # ── Build per-feature clip bounds ──────────────────────────────────────────
    clip_lo = mean - 5.0 * std    # Tier 2 global safety bounds
    clip_hi = mean + 5.0 * std

    # Tier 1: use percentile bounds (tighter, handles heavy tails)
    clip_lo[tier1_idx] = p5[tier1_idx]
    clip_hi[tier1_idx] = p95[tier1_idx]

    # Tier 1 standardize params: shift/scale so Tier-1 features are ~N(0,1) after clip
    tier1_mean = mean.copy()
    tier1_std  = std.copy()
    tier1_mean[tier1_idx] = (p5[tier1_idx] + p95[tier1_idx]) / 2.0  # midpoint of clip range
    tier1_std[tier1_idx]  = np.maximum(
        (p95[tier1_idx] - p5[tier1_idx]) / (2.0 * 1.96),   # ≈ σ from IQR
        1e-8
    )

    # ── Save scaler ────────────────────────────────────────────────────────────
    scaler_path = os.path.join(cache_dir, 'feature_scaler.npz')
    np.savez(scaler_path,
             mean=mean, std=std, p5=p5, p95=p95,
             clip_lo=clip_lo, clip_hi=clip_hi,
             tier1_idx=tier1_idx,
             tier1_mean=tier1_mean, tier1_std=tier1_std)
    print(f"  Scaler saved: {scaler_path}")

    # ── Pass 2: apply normalization to all splits ──────────────────────────────
    clip_lo32 = clip_lo.astype('float32')
    clip_hi32 = clip_hi.astype('float32')

    for split in ('train', 'val', 'test'):
        n = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0:
            continue

        zarr_path = os.path.join(cache_dir, f'{split}_obs.zarr')
        npy_path  = os.path.join(cache_dir, f'{split}_obs.npy')
        if os.path.isdir(zarr_path):
            import zarr as _zarr
            obs      = _zarr.open(zarr_path, mode='r+')
            obs_kind = 'zarr'
        elif os.path.exists(npy_path):
            obs      = np.memmap(npy_path, dtype='float32', mode='r+', shape=(n, T, n_past))
            obs_kind = 'memmap'
        else:
            print(f"  [skip] {split}_obs not found")
            continue
        print(f"  Normalizing {split} ({n:,} seqs, {obs_kind})...", end=' ', flush=True)

        for start in range(0, n, chunk_size):
            end   = min(start + chunk_size, n)
            chunk = np.asarray(obs[start:end], dtype='float64')   # (B, T, F)

            # Tier 2: clip all features to ±5σ
            np.clip(chunk, clip_lo[np.newaxis, np.newaxis, :],
                    clip_hi[np.newaxis, np.newaxis, :], out=chunk)

            # Tier 1: standardize fina + market MACD/MTM features
            for fi in tier1_idx:
                s = float(tier1_std[fi])
                if s > 1e-8:
                    chunk[:, :, fi] = (chunk[:, :, fi] - tier1_mean[fi]) / s

            if obs_kind == 'memmap':
                obs[start:end] = chunk.astype('float32')
            else:
                obs[start:end] = chunk.astype('float16')   # zarr stores float16

        if obs_kind == 'memmap':
            obs.flush()
        del obs
        print("done")

    # ── Mark cache as normalized ───────────────────────────────────────────────
    meta['normalized']     = True
    meta['scaler_path']    = scaler_path
    meta['tier1_features'] = [DT_OBSERVED_PAST_COLUMNS[i] for i in tier1_idx
                               if i < len(DT_OBSERVED_PAST_COLUMNS)]
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print("  Cache normalization complete.")


# ─── Cache compression (float16 zarr) ────────────────────────────────────────

def compress_cache(
    cache_dir:   str,
    chunk_size:  int   = 20_000,
    cname:       str   = 'zstd',
    clevel:      int   = 3,
    force:       bool  = False,
) -> None:
    """
    Convert obs and future memmaps → float16 zarr with blosc compression.

    Must be called AFTER normalize_cache() so features are already in [-5,5].
    Float16 precision is sufficient for z-scored/clipped financial features.

    Compression target (5.5M seqs, 5190 stocks):
      Float32 memmap (current): ~135 GB obs
      Float16 zarr  zstd-3:     ~47 GB obs  → total ≤ 50 GB ✓
      Float16 zarr  zstd-5:     ~32 GB obs  → total ≤ 35 GB ✓

    chunk_size: zarr chunk rows — set equal to ChunkedLoader.chunk_samples
                so one loader read = one zarr decompression (minimal overhead).
    cname/clevel: 'zstd'/3 is the default (good speed/ratio). Use 'zstd'/5
                  for maximum compression (2-3× slower decompress).

    After successful conversion, original .npy files are deleted.
    Idempotent: re-run safe (checks for .zarr directory first).
    """
    try:
        import zarr
        from zarr.codecs import BytesCodec, BloscCodec
    except ImportError:
        print("  [warn] zarr not installed — skipping compression. "
              "pip install zarr to enable.")
        return

    meta_path = os.path.join(cache_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get('compressed') and not force:
        print("  Cache already compressed (use force=True to reapply).")
        return

    if not meta.get('normalized'):
        print("  [warn] normalize_cache() should be run before compress_cache().")

    codecs = [
        BytesCodec(),
        BloscCodec(cname=cname, clevel=clevel, shuffle='bitshuffle'),
    ]
    T      = meta['seq_length']
    n_past = meta['n_past']
    n_fut  = meta['n_future']
    max_fw = meta['max_fw']

    total_saved = 0.0

    for split in ('train', 'val', 'test'):
        n = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0:
            continue

        # ── obs: (N, T, n_past) float32 → float16 zarr ───────────────────────
        npy_path  = os.path.join(cache_dir, f'{split}_obs.npy')
        zarr_path = os.path.join(cache_dir, f'{split}_obs.zarr')

        if os.path.isdir(zarr_path) and not force:
            print(f"  {split}_obs.zarr already exists — skipping")
        elif os.path.exists(npy_path):
            print(f"  Compressing {split}_obs ({n:,} seqs)...", end=' ', flush=True)
            src = np.memmap(npy_path, dtype='float32', mode='r', shape=(n, T, n_past))
            z   = zarr.open(zarr_path, mode='w', shape=(n, T, n_past), dtype='float16',
                            chunks=(min(chunk_size, n), T, n_past), codecs=codecs)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                z[start:end] = src[start:end].astype('float16')
            del src
            orig_gb  = os.path.getsize(npy_path) / 1e9
            z_sz     = sum(os.path.getsize(os.path.join(r, f))
                           for r, d, fs in os.walk(zarr_path) for f in fs)
            comp_gb  = z_sz / 1e9
            ratio    = orig_gb / comp_gb if comp_gb > 0 else 1.0
            saved    = orig_gb - comp_gb
            total_saved += saved
            print(f"{orig_gb:.1f}GB → {comp_gb:.1f}GB ({ratio:.2f}x)")
            os.remove(npy_path)

        # ── future: (N, max_fw, n_future) float32 → float16 zarr ─────────────
        fut_npy  = os.path.join(cache_dir, f'{split}_future.npy')
        fut_zarr = os.path.join(cache_dir, f'{split}_future.zarr')
        if os.path.isdir(fut_zarr) and not force:
            pass
        elif os.path.exists(fut_npy):
            src = np.memmap(fut_npy, dtype='float32', mode='r', shape=(n, max_fw, n_fut))
            z   = zarr.open(fut_zarr, mode='w', shape=(n, max_fw, n_fut), dtype='float16',
                            chunks=(min(chunk_size, n), max_fw, n_fut), codecs=codecs)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                z[start:end] = src[start:end].astype('float16')
            del src
            orig_gb = os.path.getsize(fut_npy) / 1e9
            z_sz = sum(os.path.getsize(os.path.join(r, f))
                       for r, d, fs in os.walk(fut_zarr) for f in fs)
            total_saved += orig_gb - z_sz / 1e9
            os.remove(fut_npy)

    meta['compressed']   = True
    meta['zarr_cname']   = cname
    meta['zarr_clevel']  = clevel
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Compression complete. Saved {total_saved:.1f} GB.")


def _open_zarr_or_memmap(path_base: str, dtype, shape):
    """Open zarr if available, fall back to float32 memmap (backward compat)."""
    zarr_path = path_base + '.zarr'
    npy_path  = path_base + '.npy'
    if os.path.isdir(zarr_path):
        try:
            import zarr as _zarr
            z = _zarr.open(zarr_path, mode='r')
            return z, 'zarr'
        except Exception:
            pass
    if os.path.exists(npy_path):
        arr = np.memmap(npy_path, dtype=dtype, mode='r', shape=shape)
        return arr, 'memmap'
    return np.zeros(shape, dtype=dtype), 'zeros'


# ─── Cache helpers ─────────────────────────────────────────────────────────────

def cache_exists(cache_dir: str) -> bool:
    meta_path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        for split in ('train', 'val', 'test'):
            if meta.get('splits', {}).get(split, {}).get('n_samples', 0) == 0:
                return False
            if not os.path.exists(os.path.join(cache_dir, f'{split}_obs.npy')):
                return False
        return True
    except Exception:
        return False


def get_cache_info(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, 'metadata.json')) as f:
        return json.load(f)


def load_regression_datasets(
    cache_dir:  str,
    batch_size: int,
    device:     str = 'cpu',
    chunk_samples: int = 100_000,
    prefetch_factor: int = 2,
    num_workers: int = 0,
    preload: bool = False,
    max_chunk_gb: float = None,
    use_chunked: bool = True,
):
    """
    Load all three splits as loaders.

    Args:
        chunk_samples: Samples per I/O chunk. Auto-scaled based on memory budget.
        prefetch_factor: Number of chunks to prefetch (2=double-buffer, 3=triple-buffer).
        num_workers: Number of DataLoader workers. If >0, uses PyTorch DataLoader instead
                     of RegressionChunkedLoader for better parallelism.
        preload: If True, loads entire dataset into RAM for maximum GPU throughput.
                 Requires sufficient RAM (dataset_size * 1.5).
    """
    meta = get_cache_info(cache_dir)
    loaders = {}

    # WARNING: DataLoader with num_workers > 0 is SLOWER with memmap files!
    # Each worker opens separate file handles → no page cache sharing → random seeks
    # Use chunked loader (num_workers=0) for memmap-backed datasets
    use_dataloader = (num_workers > 0)

    if use_dataloader:
        print(f"\n  [WARN] num_workers={num_workers} with memmap is often SLOWER than chunked loader!")
        print(f"         Each worker does random disk seeks. Try --num_workers 0 if slow.\n")

    for split in ('train', 'val', 'test'):
        n = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0:
            loaders[split] = None
            continue

        if preload and split == 'train':
            # RAM Loader: preload entire dataset for maximum GPU throughput
            # Best for datasets that fit in RAM (~50GB for 4.8M samples)
            loaders[split] = RegressionRAMLoader(
                cache_dir, split, batch_size, device, shuffle=True,
            )
            print(f"  [Preload] Train data loaded to RAM for max GPU throughput")
        elif use_dataloader:
            # PyTorch DataLoader with multiprocessing workers
            # NOTE: This is often slower due to memmap + multiprocessing conflict
            from torch.utils.data import DataLoader
            ds = RegressionMemmapDataset(cache_dir, split)
            nw = num_workers if split == 'train' else min(2, num_workers)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=nw, pin_memory=(device != 'cpu'),
                persistent_workers=(nw > 0),
                prefetch_factor=2 if nw > 0 else None,
            )
            if split == 'train':
                print(f"  DataLoader: {nw} workers, batch={batch_size}, pin_memory=True")
        elif use_chunked and split == 'train':
            loaders[split] = RegressionChunkedLoader(
                cache_dir, split, batch_size, chunk_samples, device,
                shuffle=True, prefetch_factor=prefetch_factor,
                max_chunk_gb=max_chunk_gb,
            )
        else:
            from torch.utils.data import DataLoader
            ds = RegressionMemmapDataset(cache_dir, split)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=0, pin_memory=(device != 'cpu'),
            )
    return loaders, meta
