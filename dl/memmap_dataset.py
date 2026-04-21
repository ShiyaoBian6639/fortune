"""
Memory-efficient dataset implementation using numpy memmap.

This module provides disk-based storage for large datasets, allowing training
on datasets larger than available RAM by loading data on-demand from disk.
"""

import gc
import os
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MemmapDataset(Dataset):
    """
    PyTorch Dataset that loads data from memory-mapped numpy files.

    Data is stored on disk and loaded on-demand, reducing RAM usage significantly.
    Only the requested batch is loaded into memory at any time.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        """
        Initialize the memmap dataset.

        Args:
            data_dir: Directory containing the memmap files
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to apply to sequences
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        split_info = self.metadata['splits'][split]
        self.n_samples   = split_info['n_samples']
        self.seq_length  = self.metadata['seq_length']
        self.n_features  = self.metadata['n_features']
        self.num_horizons = self.metadata.get('num_horizons', 1)

        self._open_memmaps()

    def _open_memmaps(self):
        """Open memmap file handles. Called in __init__ and after unpickling in workers."""
        H = self.num_horizons
        label_shape = (self.n_samples, H) if H > 1 else (self.n_samples,)

        self.sequences = np.memmap(
            os.path.join(self.data_dir, f'{self.split}_sequences.npy'),
            dtype='float32', mode='r',
            shape=(self.n_samples, self.seq_length, self.n_features)
        )
        self.labels = np.memmap(
            os.path.join(self.data_dir, f'{self.split}_labels.npy'),
            dtype='int64', mode='r',
            shape=label_shape
        )
        self.sectors = np.memmap(
            os.path.join(self.data_dir, f'{self.split}_sectors.npy'),
            dtype='int64', mode='r',
            shape=(self.n_samples,)
        )
        ind_path = os.path.join(self.data_dir, f'{self.split}_industries.npy')
        self.industries = np.memmap(ind_path, dtype='int64', mode='r',
                                    shape=(self.n_samples,)) if os.path.exists(ind_path) else None
        rel_path = os.path.join(self.data_dir, f'{self.split}_relative_labels.npy')
        self.relative_labels = np.memmap(rel_path, dtype='int64', mode='r',
                                         shape=label_shape) if os.path.exists(rel_path) else None
        fut_path = os.path.join(self.data_dir, f'{self.split}_future_inputs.npy')
        n_fut = self.metadata.get('n_future_features', 0)
        max_fw = self.metadata.get('max_fw', 5)
        self.future_inputs = (
            np.memmap(fut_path, dtype='float32', mode='r',
                      shape=(self.n_samples, max_fw, n_fut))
            if os.path.exists(fut_path) and n_fut > 0 else None
        )

    def __getstate__(self):
        """Pickle only metadata — not the memmap arrays themselves.
        Without this, numpy serialises the full array into the pickle stream,
        sending gigabytes to every DataLoader worker process."""
        state = self.__dict__.copy()
        del state['sequences']
        del state['labels']
        del state['sectors']
        state.pop('industries', None)
        state.pop('relative_labels', None)
        state.pop('future_inputs', None)
        return state

    def __setstate__(self, state):
        """Reopen memmap files in the worker process after unpickling."""
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx].copy()
        label    = self.labels[idx]
        sector   = self.sectors[idx]

        if self.transform:
            sequence = self.transform(sequence)

        seq_t = torch.tensor(sequence, dtype=torch.float32)
        lab_t = torch.tensor(label,    dtype=torch.long)
        sec_t = torch.tensor(sector,   dtype=torch.long)

        if self.industries is not None:
            ind_t = torch.tensor(self.industries[idx], dtype=torch.long)
            if self.relative_labels is not None:
                rel_t = torch.tensor(self.relative_labels[idx], dtype=torch.long)
                if self.future_inputs is not None:
                    fut_t = torch.tensor(self.future_inputs[idx].copy(), dtype=torch.float32)
                    return seq_t, lab_t, sec_t, ind_t, rel_t, fut_t
                return seq_t, lab_t, sec_t, ind_t, rel_t
            if self.future_inputs is not None:
                fut_t = torch.tensor(self.future_inputs[idx].copy(), dtype=torch.float32)
                return seq_t, lab_t, sec_t, ind_t, fut_t
            return seq_t, lab_t, sec_t, ind_t
        return seq_t, lab_t, sec_t

    def get_labels(self) -> np.ndarray:
        """Get all labels (for computing class weights)."""
        return np.array(self.labels)


def _preshuffle_split(
    output_dir: str,
    split: str,
    n_samples: int,
    seq_length: int,
    n_features: int,
    random_seed: int,
    chunk_size: int = 10000,
    num_horizons: int = 1,
    has_relative_labels: bool = False,
    has_future_inputs: bool = False,
    n_future_features: int = 0,
    max_fw: int = 5,
):
    """
    Rewrite a split's memmap files in a randomly shuffled order.

    After this, the DataLoader can use shuffle=False and still get random-order
    samples via cheap sequential disk reads instead of expensive random seeks.

    Strategy: write dst sequentially, read src at random positions.
    Sequential writes let the OS flush dirty pages continuously (write-behind),
    avoiding the dirty-page accumulation that hangs the system when writing
    randomly to a file larger than available RAM.  Random reads of clean pages
    are cheaper because evicted clean pages don't need to be written back.
    """
    # perm[j] = i  →  dst[j] = src[perm[j]]  (shuffle src into dst order)
    perm = np.random.RandomState(random_seed + 1).permutation(n_samples)

    label_shape = (n_samples, num_horizons) if num_horizons > 1 else (n_samples,)
    specs = [
        (f'{split}_sequences.npy',  'float32', (n_samples, seq_length, n_features)),
        (f'{split}_labels.npy',     'int64',   label_shape),
        (f'{split}_sectors.npy',    'int64',   (n_samples,)),
        (f'{split}_industries.npy', 'int64',   (n_samples,)),
    ]
    if has_relative_labels:
        specs.append((f'{split}_relative_labels.npy', 'int64', label_shape))
    if has_future_inputs and n_future_features > 0:
        specs.append((f'{split}_future_inputs.npy', 'float32',
                      (n_samples, max_fw, n_future_features)))

    for fname, dtype, shape in specs:
        src_path = os.path.join(output_dir, fname)
        if not os.path.exists(src_path):
            continue
        tmp_path = src_path + '.shuf'

        src = np.memmap(src_path, dtype=dtype, mode='r',  shape=shape)
        dst = np.memmap(tmp_path, dtype=dtype, mode='w+', shape=shape)

        # Write dst sequentially; read src at random (shuffled) positions.
        # Periodically flush dst so the OS can reclaim dirty pages and avoid
        # accumulating the entire file in the page cache before writing it out.
        flush_every = max(1, 200_000 // chunk_size)  # flush ~every 200 K samples
        for step, start in enumerate(range(0, n_samples, chunk_size)):
            end = min(start + chunk_size, n_samples)
            src_indices = perm[start:end]   # random read positions in src
            dst[start:end] = src[src_indices]  # sequential write in dst
            if (step + 1) % flush_every == 0:
                dst.flush()
                gc.collect()

        dst.flush()
        # Explicitly close the underlying mmap handles before renaming.
        # On Windows, file handles stay open until GC runs, causing PermissionError.
        src._mmap.close()
        dst._mmap.close()
        del src, dst
        gc.collect()
        os.replace(tmp_path, src_path)


class ChunkedMemmapLoader:
    """
    Drop-in DataLoader replacement for training that eliminates Windows
    multiprocessing overhead and random-seek bottlenecks.

    Shuffle strategy (two-level, avoids 4.8 GB random-gather CPU bottleneck):
      1. Batch-order shuffle  — permute ~390 batch indices (negligible cost)
      2. Within-batch shuffle — permute 1024 samples inside each batch (~13 MB
         fits in CPU L3 cache → cache-friendly, ~2 ms per batch)

    This is 100-1000x cheaper than shuffling the full 4.8 GB chunk at once,
    which causes 100% CPU for 10-30 s and starves the GPU.

    Background thread prefetch (depth=2):
      - Two threads load chunks N+1 and N+2 while GPU trains on chunk N.
      - Thread shares the memmap handle (read-only, thread-safe).
      - No multiprocessing / pickling / spawn overhead.
      - Depth-2 eliminates stalls even on SATA SSDs where one chunk takes
        longer to load than the GPU needs to process the current chunk.

    Peak RAM ≈ 3 × chunk_samples × seq_length × n_features × 4 bytes
    Default 200 K-sample chunk ≈ 2.4 GB → peak ~7.2 GB within 16 GB budget.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = 'train',
        batch_size: int = 512,
        chunk_samples: int = 400_000,
        seed: int = 42,
    ):
        self.cache_dir     = cache_dir
        self.split         = split
        self.batch_size    = batch_size
        self.chunk_samples = chunk_samples
        self.seed          = seed

        with open(os.path.join(cache_dir, 'metadata.json')) as f:
            meta = json.load(f)

        self.n_samples    = meta['splits'][split]['n_samples']
        self.seq_length   = meta['seq_length']
        self.n_features   = meta['n_features']
        self.num_horizons = meta.get('num_horizons', 1)

        H = self.num_horizons
        label_shape = (self.n_samples, H) if H > 1 else (self.n_samples,)

        # Open memmaps once — background thread shares these handles.
        # numpy memmap read-only access is thread-safe.
        self.sequences = np.memmap(
            os.path.join(cache_dir, f'{split}_sequences.npy'),
            dtype='float32', mode='r',
            shape=(self.n_samples, self.seq_length, self.n_features),
        )
        self.labels = np.memmap(
            os.path.join(cache_dir, f'{split}_labels.npy'),
            dtype='int64', mode='r', shape=label_shape,
        )
        self.sectors = np.memmap(
            os.path.join(cache_dir, f'{split}_sectors.npy'),
            dtype='int64', mode='r', shape=(self.n_samples,),
        )
        ind_path = os.path.join(cache_dir, f'{split}_industries.npy')
        self.industries = np.memmap(ind_path, dtype='int64', mode='r',
                                    shape=(self.n_samples,)) if os.path.exists(ind_path) else None
        rel_path = os.path.join(cache_dir, f'{split}_relative_labels.npy')
        H = self.num_horizons
        rel_label_shape = (self.n_samples, H) if H > 1 else (self.n_samples,)
        self.relative_labels = np.memmap(rel_path, dtype='int64', mode='r',
                                         shape=rel_label_shape) if os.path.exists(rel_path) else None
        fut_path = os.path.join(cache_dir, f'{split}_future_inputs.npy')
        n_fut  = meta.get('n_future_features', 0)
        max_fw = meta.get('max_fw', 5)
        self.future_inputs = (
            np.memmap(fut_path, dtype='float32', mode='r',
                      shape=(self.n_samples, max_fw, n_fut))
            if os.path.exists(fut_path) and n_fut > 0 else None
        )

    def __len__(self) -> int:
        """Number of complete batches per epoch."""
        return self.n_samples // self.batch_size

    def _load_chunk(
        self, start: int, end: int, seed: int
    ) -> tuple:
        """
        Bulk sequential read of one contiguous memmap slice into RAM.
        Returns tensors + a shuffled batch-index order.

        Called from background thread. Uses a local rng (not shared with main
        thread) to avoid Generator thread-safety issues.
        Does NOT shuffle samples — that would require a 4.8 GB random gather,
        which saturates the CPU and starves the GPU.
        """
        rng = np.random.default_rng(seed)
        n = end - start
        # np.array() forces a contiguous sequential read (fast SSD throughput)
        seq = torch.from_numpy(np.array(self.sequences[start:end]))
        lab = torch.from_numpy(np.array(self.labels[start:end]))
        sec = torch.from_numpy(np.array(self.sectors[start:end]))
        ind = torch.from_numpy(np.array(self.industries[start:end])) if self.industries is not None else None
        rel = torch.from_numpy(np.array(self.relative_labels[start:end])) if self.relative_labels is not None else None
        fut = torch.from_numpy(np.array(self.future_inputs[start:end])) if self.future_inputs is not None else None

        # Full within-chunk shuffle: permute all n samples while data is in RAM.
        # Needed when data is written stock-by-stock (consecutive samples = same stock).
        # Cost: O(n) index permutation; n=40K → negligible vs. the disk read above.
        perm = torch.from_numpy(rng.permutation(n))
        seq = seq[perm]
        lab = lab[perm]
        sec = sec[perm]
        if ind is not None:
            ind = ind[perm]
        if rel is not None:
            rel = rel[perm]
        if fut is not None:
            fut = fut[perm]

        # Batch-order permutation: permute ~80 batch indices
        batch_order = rng.permutation(n // self.batch_size)
        return seq, lab, sec, ind, rel, fut, batch_order

    def _yield_batches(self, chunk: tuple):
        """
        Yield all complete batches from a loaded chunk.
        Within-batch shuffle: permute batch_size samples inside a ~13 MB slice
        that fits in CPU L3 cache — ~2 ms per batch vs 10-30 s for full-chunk.
        """
        seq, lab, sec, ind, rel, fut, batch_order = chunk
        bs = self.batch_size
        for b in batch_order:
            i, j = int(b) * bs, int(b) * bs + bs
            perm = torch.randperm(bs)
            if ind is not None:
                if rel is not None:
                    if fut is not None:
                        yield seq[i:j][perm], lab[i:j][perm], sec[i:j][perm], ind[i:j][perm], rel[i:j][perm], fut[i:j][perm]
                    else:
                        yield seq[i:j][perm], lab[i:j][perm], sec[i:j][perm], ind[i:j][perm], rel[i:j][perm]
                else:
                    if fut is not None:
                        yield seq[i:j][perm], lab[i:j][perm], sec[i:j][perm], ind[i:j][perm], fut[i:j][perm]
                    else:
                        yield seq[i:j][perm], lab[i:j][perm], sec[i:j][perm], ind[i:j][perm]
            else:
                yield seq[i:j][perm], lab[i:j][perm], sec[i:j][perm]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)

        # Shuffle chunk order each epoch for epoch-level variety
        chunk_starts = list(range(0, self.n_samples, self.chunk_samples))
        rng.shuffle(chunk_starts)

        # Synchronously pre-load the first chunk
        s0 = chunk_starts[0]
        e0 = min(s0 + self.chunk_samples, self.n_samples)
        current = self._load_chunk(s0, e0, int(rng.integers(0, 2**31)))

        # Depth-2 prefetch: submit chunks N+1 and N+2 simultaneously so the
        # GPU never stalls even if one chunk takes longer than expected to load.
        # max_workers=2 means both can load concurrently (I/O bound, not CPU).
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: deque = deque()

            # Pre-submit up to 2 future chunks
            for s in chunk_starts[1:3]:
                e = min(s + self.chunk_samples, self.n_samples)
                futures.append(pool.submit(self._load_chunk, s, e, int(rng.integers(0, 2**31))))

            for s in chunk_starts[3:]:
                e = min(s + self.chunk_samples, self.n_samples)
                # Submit chunk N+2 before blocking on chunk N+1
                futures.append(pool.submit(self._load_chunk, s, e, int(rng.integers(0, 2**31))))
                yield from self._yield_batches(current)
                current = futures.popleft().result()

            # Drain remaining pre-submitted futures
            yield from self._yield_batches(current)
            while futures:
                current = futures.popleft().result()
                yield from self._yield_batches(current)


class MemmapDataWriter:
    """
    Writes processed data to disk in memmap format.

    Supports incremental writing to handle datasets larger than RAM.
    """

    def __init__(
        self,
        output_dir: str,
        seq_length: int,
        n_features: int,
        num_horizons: int = 1,
        expected_samples: int = 1000000  # Initial estimate, will resize if needed
    ):
        """
        Initialize the memmap writer.

        Args:
            output_dir: Directory to save memmap files
            seq_length: Sequence length
            n_features: Number of features per timestep
            num_horizons: Number of prediction horizons (labels shape (N, H))
            expected_samples: Expected total number of samples (can grow)
        """
        self.output_dir = output_dir
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_horizons = num_horizons
        self.expected_samples = expected_samples

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize memmap files for temporary storage
        self.temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Accumulate data in chunks
        self.sequences_chunks = []
        self.labels_chunks = []
        self.sectors_chunks = []
        self.industries_chunks = []
        self.total_samples = 0

        # Chunk size for flushing to disk (smaller = less memory)
        self.chunk_size = 5000
        self.current_chunk_sequences = []
        self.current_chunk_labels = []
        self.current_chunk_sectors = []
        self.current_chunk_industries = []
        self.current_chunk_dates = []           # YYYYMMDD int32 — used for time-based split
        self.current_chunk_rel_labels = []      # relative-return class labels (stock − CSI300)
        self.current_chunk_future_inputs = []   # TFT decoder inputs (N, max_fw, n_future_feat)
        self.n_future_features: Optional[int] = None  # inferred from first add_stock_data call

    def add_stock_data(
        self,
        sequences:       np.ndarray,
        labels:          np.ndarray,
        sectors:         np.ndarray,
        dates:           Optional[np.ndarray] = None,
        industries:      Optional[np.ndarray] = None,
        relative_labels: Optional[np.ndarray] = None,
        future_inputs:   Optional[np.ndarray] = None,
    ):
        """
        Add processed data for one stock.

        Args:
            sequences:       Array of shape (n_sequences, seq_length, n_features)
            labels:          Array of shape (n_sequences,)
            sectors:         Array of shape (n_sequences,)
            dates:           int32 array of shape (n_sequences,) — YYYYMMDD date of the
                             last day in each sequence window, used for time-based split.
            industries:      int64 array of shape (n_sequences,) — industry class indices.
            relative_labels: int64 array of shape (n_sequences, H) — relative-return
                             class labels (stock return − CSI300 return per horizon).
            future_inputs:   float32 array of shape (n_sequences, max_fw, n_future_feat) —
                             known-future calendar features for TFT decoder.
        """
        self.current_chunk_sequences.append(sequences)
        self.current_chunk_labels.append(labels)
        self.current_chunk_sectors.append(sectors)
        if industries is not None:
            self.current_chunk_industries.append(industries)
        if dates is not None:
            self.current_chunk_dates.append(dates)
        if relative_labels is not None:
            self.current_chunk_rel_labels.append(relative_labels)
        if future_inputs is not None:
            self.current_chunk_future_inputs.append(future_inputs)
            if self.n_future_features is None:
                self.n_future_features = future_inputs.shape[-1]
        self.total_samples += len(sequences)

        # Flush to disk if chunk is large enough
        current_size = sum(len(s) for s in self.current_chunk_sequences)
        if current_size >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        """Flush current chunk to disk."""
        if not self.current_chunk_sequences:
            return

        chunk_idx = len(self.sequences_chunks)

        # Concatenate current chunk
        sequences = np.concatenate(self.current_chunk_sequences, axis=0)
        labels    = np.concatenate(self.current_chunk_labels,    axis=0)
        sectors   = np.concatenate(self.current_chunk_sectors,   axis=0)

        # Save to temporary files
        np.save(os.path.join(self.temp_dir, f'seq_{chunk_idx}.npy'), sequences)
        np.save(os.path.join(self.temp_dir, f'lab_{chunk_idx}.npy'), labels)
        np.save(os.path.join(self.temp_dir, f'sec_{chunk_idx}.npy'), sectors)

        if self.current_chunk_industries:
            industries = np.concatenate(self.current_chunk_industries, axis=0)
            np.save(os.path.join(self.temp_dir, f'ind_{chunk_idx}.npy'), industries)

        # Save dates if available (used for time-based split in finalize)
        if self.current_chunk_dates:
            dates = np.concatenate(self.current_chunk_dates, axis=0)
            np.save(os.path.join(self.temp_dir, f'dates_{chunk_idx}.npy'), dates)

        if self.current_chunk_rel_labels:
            rel_labels = np.concatenate(self.current_chunk_rel_labels, axis=0)
            np.save(os.path.join(self.temp_dir, f'rel_lab_{chunk_idx}.npy'), rel_labels)

        if self.current_chunk_future_inputs:
            fut = np.concatenate(self.current_chunk_future_inputs, axis=0)
            np.save(os.path.join(self.temp_dir, f'fut_{chunk_idx}.npy'), fut)

        self.sequences_chunks.append(chunk_idx)

        # Clear current chunk
        self.current_chunk_sequences     = []
        self.current_chunk_labels        = []
        self.current_chunk_sectors       = []
        self.current_chunk_industries    = []
        self.current_chunk_dates         = []
        self.current_chunk_rel_labels    = []
        self.current_chunk_future_inputs = []

    def finalize(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler: Optional[StandardScaler] = None,
        random_seed: int = 42,
        split_mode: str = 'regime',
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Finalize the dataset: split, normalize, and save to memmap files.

        Uses streaming approach to avoid loading all data into memory.

        Args:
            train_ratio: Fraction of data for training (used only in 'random' mode)
            val_ratio: Fraction of data for validation (used only in 'random' mode)
            scaler: Optional pre-fitted scaler. If None, fits on training data.
            random_seed: Random seed for reproducibility
            split_mode: 'regime' for temporal regime-aware split, 'random' for
                        legacy random permutation split.
            data_dir: Path to stock_data root — required for regime mode to locate
                      the CSI300 signal file.

        Returns:
            Dictionary with metadata and scaler
        """
        # Flush remaining data
        self._flush_chunk()

        print(f"Finalizing dataset with {self.total_samples} total samples...")
        print("Using streaming approach to minimize memory usage...")

        # ── Build per-sample split assignment ─────────────────────────────────
        # regime mode: use CSI300 MA-250 signal to assign temporal blocks
        # random mode: legacy random permutation

        _has_date_files = any(
            os.path.exists(os.path.join(self.temp_dir, f'dates_{ci}.npy'))
            for ci in self.sequences_chunks
        )
        regime_ok = (
            split_mode in ('regime', 'temporal', 'interleaved_val', 'rolling_window')
            and _has_date_files
            and (split_mode in ('temporal', 'interleaved_val', 'rolling_window')
                 or (data_dir is not None
                     and os.path.exists(
                         os.path.join(data_dir, 'index', 'idx_factor_pro', '000300_SH.csv'))))
        )

        if regime_ok:
            # ── Regime-aware, temporal, or interleaved_val split ─────────────
            if split_mode == 'temporal':
                print(f"\nUsing TEMPORAL split (chronological {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%})...")
            elif split_mode == 'interleaved_val':
                print(f"\nUsing INTERLEAVED_VAL split (Q1 of 2018–2025 as val windows)...")
            elif split_mode == 'rolling_window':
                print(f"\nUsing ROLLING_WINDOW split (3m train / 1m val blocks across all years)...")
            else:
                print(f"\nUsing REGIME-AWARE split (CSI300 MA-250 signal)...")
            from .regime_split import (
                load_csi300_signal, detect_regime_blocks,
                assign_blocks_to_splits, build_date_split_map,
                print_split_stats,
            )
            from .config import (
                REGIME_MIN_BLOCK_DAYS, REGIME_PURGE_GAP_DAYS,
                REGIME_VAL_DAYS, REGIME_TEST_DAYS,
            )

            # Pass 0: collect all dates to find sample date range
            print("  Scanning sample dates for date range...")
            all_dates_list = []
            for chunk_idx in self.sequences_chunks:
                dates_path = os.path.join(self.temp_dir, f'dates_{chunk_idx}.npy')
                if os.path.exists(dates_path):
                    all_dates_list.append(np.load(dates_path))
            if all_dates_list:
                all_sample_dates = np.concatenate(all_dates_list)
                date_min_int = int(all_sample_dates.min())
                date_max_int = int(all_sample_dates.max())
            else:
                print("  WARNING: no date files found — falling back to random split")
                regime_ok = False

        if regime_ok:
            if split_mode == 'temporal':
                # ── Temporal split: simple chronological percentile split ─────
                unique_dates = np.unique(all_sample_dates.astype(np.int32))
                n_dates = len(unique_dates)
                val_idx  = int(n_dates * train_ratio)
                test_idx = int(n_dates * (train_ratio + val_ratio))

                date_split_map = {}
                for d in unique_dates:
                    di = int(d)
                    if d < unique_dates[val_idx]:
                        date_split_map[di] = 'train'
                    elif d < unique_dates[test_idx]:
                        date_split_map[di] = 'val'
                    else:
                        date_split_map[di] = 'test'

                # Purge gaps at both boundaries
                for i in range(max(0, val_idx - REGIME_PURGE_GAP_DAYS), val_idx):
                    date_split_map[int(unique_dates[i])] = 'gap'
                for i in range(max(0, test_idx - REGIME_PURGE_GAP_DAYS), test_idx):
                    date_split_map[int(unique_dates[i])] = 'gap'

                val_date_str  = str(unique_dates[val_idx])
                test_date_str = str(unique_dates[test_idx])
                print(f"  Temporal boundaries: train<{val_date_str} | val<{test_date_str} | test≥{test_date_str}")
                print(f"  Purge gap: {REGIME_PURGE_GAP_DAYS} trading days at each boundary")
                print_split_stats(date_split_map)

            elif split_mode == 'interleaved_val':
                # ── Interleaved val: Q1 of each year 2018-2025 ───────────────
                from .config import INTERLEAVED_VAL_WINDOWS, INTERLEAVED_TEST_START
                unique_dates = np.unique(all_sample_dates.astype(np.int32))

                # Default: everything is train
                date_split_map = {int(d): 'train' for d in unique_dates}

                # Mark test window (2025-07-01 onwards)
                for d in unique_dates:
                    if int(d) >= INTERLEAVED_TEST_START:
                        date_split_map[int(d)] = 'test'

                # Mark each val window + purge buffers
                for val_start, val_end in INTERLEAVED_VAL_WINDOWS:
                    start_idx = int(np.searchsorted(unique_dates, val_start, side='left'))
                    end_idx   = int(np.searchsorted(unique_dates, val_end,   side='left'))

                    # Val window itself
                    for i in range(start_idx, end_idx):
                        d = int(unique_dates[i])
                        if d < INTERLEAVED_TEST_START:
                            date_split_map[d] = 'val'

                    # Purge: 50 trading days before val start
                    for i in range(max(0, start_idx - REGIME_PURGE_GAP_DAYS), start_idx):
                        d = int(unique_dates[i])
                        if date_split_map.get(d) == 'train':
                            date_split_map[d] = 'gap'

                    # Purge: 50 trading days after val end (don't overwrite test)
                    for i in range(end_idx, min(len(unique_dates), end_idx + REGIME_PURGE_GAP_DAYS)):
                        d = int(unique_dates[i])
                        if d < INTERLEAVED_TEST_START and date_split_map.get(d) == 'train':
                            date_split_map[d] = 'gap'

                # Purge: 50 trading days before test start
                test_start_idx = int(np.searchsorted(unique_dates, INTERLEAVED_TEST_START, side='left'))
                for i in range(max(0, test_start_idx - REGIME_PURGE_GAP_DAYS), test_start_idx):
                    d = int(unique_dates[i])
                    if date_split_map.get(d) == 'train':
                        date_split_map[d] = 'gap'

                print(f"  Interleaved val: {len(INTERLEAVED_VAL_WINDOWS)} Q1 windows (2018–2025)")
                print(f"  Test start: {INTERLEAVED_TEST_START}, purge gap: {REGIME_PURGE_GAP_DAYS} trading days each boundary")
                print_split_stats(date_split_map)

            elif split_mode == 'rolling_window':
                # ── Rolling walk-forward: TRAIN → purge → VAL → TEST per fold ──────────
                # Boundary rules (no data leakage):
                #   1. Each fold: [cursor, cursor+T] train | purge | [T, T+V] val | [T+V, T+V+Te] test
                #   2. Purge gap (seq_len+max_fw = 35 days) applied only at train→val boundary.
                #      val→test needs no purge: val is not used for gradient updates, so
                #      test sequences that look back into val are safe.
                #   3. "Only overwrite 'train'" rule: once a date is labeled val/test in any
                #      fold, it can never be re-labeled train in a later fold → no leakage.
                #   4. Global holdout (>= INTERLEAVED_TEST_START) is always 'test'.
                import pandas as _pd
                from .config import (
                    ROLLING_TRAIN_MONTHS, ROLLING_VAL_MONTHS,
                    ROLLING_TEST_MONTHS, ROLLING_STEP_MONTHS,
                    INTERLEAVED_TEST_START,
                )

                unique_dates_arr = np.unique(all_sample_dates.astype(np.int32))
                date_split_map = {int(d): 'train' for d in unique_dates_arr}
                purge = REGIME_PURGE_GAP_DAYS

                # Global holdout: data from INTERLEAVED_TEST_START onwards is always test
                for d in unique_dates_arr:
                    if int(d) >= INTERLEAVED_TEST_START:
                        date_split_map[int(d)] = 'test'

                first_date = _pd.Timestamp(str(int(unique_dates_arr[0])))
                cursor     = first_date.replace(day=1)
                test_ts    = _pd.Timestamp(str(INTERLEAVED_TEST_START))

                # Build fold windows (val_start, val_end, test_start, test_end)
                folds = []
                while True:
                    val_start_ts  = cursor + _pd.DateOffset(months=ROLLING_TRAIN_MONTHS)
                    val_end_ts    = val_start_ts  + _pd.DateOffset(months=ROLLING_VAL_MONTHS)
                    test_start_ts = val_end_ts
                    test_end_ts   = test_start_ts + _pd.DateOffset(months=ROLLING_TEST_MONTHS)
                    if test_end_ts > test_ts:
                        break
                    folds.append((val_start_ts, val_end_ts, test_start_ts, test_end_ts))
                    cursor += _pd.DateOffset(months=ROLLING_STEP_MONTHS)

                def _mark_window(date_map, dates_arr, start_ts, end_ts, label):
                    """Mark dates in [start_ts, end_ts) as label (only if currently 'train')."""
                    s_i = int(np.searchsorted(dates_arr, int(start_ts.strftime('%Y%m%d')), 'left'))
                    e_i = int(np.searchsorted(dates_arr, int(end_ts.strftime('%Y%m%d')),   'left'))
                    for i in range(s_i, e_i):
                        if date_map.get(int(dates_arr[i])) == 'train':
                            date_map[int(dates_arr[i])] = label
                    return s_i  # returns start index for purge calculation

                def _purge_train_before(date_map, dates_arr, boundary_idx, n):
                    """Mark the last n 'train' dates before boundary as 'gap'."""
                    for i in range(max(0, boundary_idx - n), boundary_idx):
                        if date_map.get(int(dates_arr[i])) == 'train':
                            date_map[int(dates_arr[i])] = 'gap'

                # Pass 1: mark per-fold TEST windows (test > val > train priority)
                for (_, _, test_start_ts, test_end_ts) in folds:
                    _mark_window(date_split_map, unique_dates_arr, test_start_ts, test_end_ts, 'test')
                    # No purge between val and test: val is already the separation buffer.

                # Pass 2: mark per-fold VAL windows + purge train→val boundary
                for (val_start_ts, val_end_ts, _, _) in folds:
                    s_i = _mark_window(date_split_map, unique_dates_arr, val_start_ts, val_end_ts, 'val')
                    # Purge: last `purge` train days before val start → gap
                    _purge_train_before(date_split_map, unique_dates_arr, s_i, purge)

                # Purge train days before global holdout test
                holdout_idx = int(np.searchsorted(unique_dates_arr, INTERLEAVED_TEST_START, 'left'))
                _purge_train_before(date_split_map, unique_dates_arr, holdout_idx, purge)

                print(f"  Rolling walk-forward: {ROLLING_TRAIN_MONTHS}m train / "
                      f"{ROLLING_VAL_MONTHS}m val / {ROLLING_TEST_MONTHS}m test, "
                      f"step={ROLLING_STEP_MONTHS}m, {len(folds)} folds, "
                      f"purge={purge} train days before each val window")
                print(f"  Final holdout from: {INTERLEAVED_TEST_START}")
                print_split_stats(date_split_map)

            else:
                # ── Regime-aware split: CSI300 MA-250 signal ─────────────────
                signal_df = load_csi300_signal(data_dir)
                print(f"  CSI300 signal: {signal_df['trade_date'].min().date()} "
                      f"→ {signal_df['trade_date'].max().date()} "
                      f"({len(signal_df)} trading days)")

                # Filter signal to our sample window
                signal_df_window = signal_df[
                    (signal_df['trade_date'].dt.year  * 10000
                     + signal_df['trade_date'].dt.month * 100
                     + signal_df['trade_date'].dt.day).astype(int).between(date_min_int, date_max_int)
                ].copy()
                print(f"  Sample window: {date_min_int} → {date_max_int} "
                      f"({len(signal_df_window)} CSI300 trading days in window)")

                blocks = detect_regime_blocks(signal_df_window, min_block_days=REGIME_MIN_BLOCK_DAYS)
                print(f"\n  Regime blocks after merging (min={REGIME_MIN_BLOCK_DAYS} days):")
                for b in blocks:
                    print(f"    {b['regime']:4s} {str(b['start'].date())} → "
                          f"{str(b['end'].date())}  ({b['n_days']} days)")

                blocks = assign_blocks_to_splits(blocks, val_days=REGIME_VAL_DAYS, test_days=REGIME_TEST_DAYS)
                print(f"\n  Split assignment:")
                for b in blocks:
                    print(f"    {b['split']:20s}  {str(b['start'].date())} → "
                          f"{str(b['end'].date())}  ({b['n_days']} days)")

                date_split_map = build_date_split_map(
                    blocks, signal_df,
                    purge_gap_days=REGIME_PURGE_GAP_DAYS,
                    date_min=date_min_int,
                    date_max=date_max_int,
                )
                print_split_stats(date_split_map)

            # Pass 0b: build per-sample split assignment array (vectorised via
            # numpy integer array-indexing into a lookup table).
            print("\n  Assigning samples to splits by date...")

            # Build a compact integer lookup: date_int → split_code (0=train,1=val,2=test,3=gap)
            SPLIT_CODE = {'train': 0, 'val': 1, 'test': 2, 'gap': 3}
            CODE_SPLIT = {v: k for k, v in SPLIT_CODE.items()}

            all_map_dates = np.array(list(date_split_map.keys()), dtype=np.int32)
            all_map_codes = np.array(
                [SPLIT_CODE[date_split_map[d]] for d in all_map_dates], dtype=np.int8
            )
            # Sort for searchsorted lookups
            sort_order    = np.argsort(all_map_dates)
            sorted_dates  = all_map_dates[sort_order]
            sorted_codes  = all_map_codes[sort_order]

            sample_splits = np.full(self.total_samples, 'train', dtype='U5')
            global_idx = 0
            for chunk_idx in self.sequences_chunks:
                dates_path = os.path.join(self.temp_dir, f'dates_{chunk_idx}.npy')
                if os.path.exists(dates_path):
                    chunk_dates = np.load(dates_path).astype(np.int32)
                else:
                    n_chunk = len(np.load(os.path.join(self.temp_dir, f'lab_{chunk_idx}.npy')))
                    chunk_dates = np.zeros(n_chunk, dtype=np.int32)
                n = len(chunk_dates)

                # Vectorised lookup via searchsorted (O(n log m) instead of O(n) pure Python)
                pos  = np.searchsorted(sorted_dates, chunk_dates)
                pos  = np.clip(pos, 0, len(sorted_dates) - 1)
                hit  = sorted_dates[pos] == chunk_dates      # True where date found in map
                codes = np.where(hit, sorted_codes[pos], SPLIT_CODE['train'])
                # Map int codes back to string splits via np.select (no Python loop)
                sample_splits[global_idx:global_idx + n] = np.select(
                    [codes == 0, codes == 1, codes == 2, codes == 3],
                    ['train', 'val', 'test', 'gap'],
                    default='train',
                )
                global_idx += n

            n_train = int((sample_splits == 'train').sum())
            n_val   = int((sample_splits == 'val').sum())
            n_test  = int((sample_splits == 'test').sum())
            n_gap   = int((sample_splits == 'gap').sum())
            total_used = n_train + n_val + n_test

            print(f"\n  Sample counts (after date mapping):")
            print(f"    train: {n_train:,}")
            print(f"    val:   {n_val:,}")
            print(f"    test:  {n_test:,}")
            print(f"    gap:   {n_gap:,} (dropped)")
            print(f"    total used: {total_used:,} / {self.total_samples:,} "
                  f"({100.*total_used/self.total_samples:.1f}%)")

            if n_train == 0 or n_val == 0 or n_test == 0:
                print("  WARNING: one split is empty — falling back to random split")
                regime_ok = False

        if not regime_ok:
            # ── Random split (fallback / legacy) ─────────────────────────────
            if split_mode == 'regime':
                print("  (regime split unavailable — using random split)")
            np.random.seed(random_seed)
            all_indices = np.random.permutation(self.total_samples)
            n_train = int(self.total_samples * train_ratio)
            n_val   = int(self.total_samples * val_ratio)
            n_test  = self.total_samples - n_train - n_val

            sample_splits = np.empty(self.total_samples, dtype='U5')
            sample_splits[:] = 'gap'          # default (never used as gap here)
            sample_splits[all_indices[:n_train]] = 'train'
            sample_splits[all_indices[n_train:n_train + n_val]] = 'val'
            sample_splits[all_indices[n_train + n_val:]] = 'test'

            print(f"Random split (seed={random_seed}):")
            print(f"  train: {n_train:,} samples ({train_ratio:.0%})")
            print(f"  val:   {n_val:,} samples ({val_ratio:.0%})")
            print(f"  test:  {n_test:,} samples ({1 - train_ratio - val_ratio:.0%})")

        # Label shape: (n,) for single horizon, (n, H) for multi-horizon
        H = self.num_horizons
        label_shape_train = (n_train, H) if H > 1 else (n_train,)
        label_shape_val   = (n_val,   H) if H > 1 else (n_val,)
        label_shape_test  = (n_test,  H) if H > 1 else (n_test,)

        # Guard: reject if train_sequences.npy would exceed 32 GB — numpy memmap
        # on Windows fails with OSError(22) when the backing file is very large
        # (virtual address space / page-file exhaustion).  32 GB is a conservative
        # limit that leaves headroom for the 2× peak during pre-shuffle.
        train_bytes = int(n_train) * int(self.seq_length) * int(self.n_features) * 4
        if train_bytes > 32 * (1 << 30):
            raise RuntimeError(
                f"Projected train_sequences.npy is {train_bytes / 1e9:.1f} GB "
                f"({n_train:,} samples × {self.seq_length} × {self.n_features} × float32). "
                f"Reduce --max_stocks or lower max_sequences_per_stock in config to keep "
                f"the file under 32 GB."
            )

        # Create output memmap files
        print(f"Creating memmap files...")
        train_seq = np.memmap(
            os.path.join(self.output_dir, 'train_sequences.npy'),
            dtype='float32', mode='w+', shape=(n_train, self.seq_length, self.n_features)
        )
        train_lab = np.memmap(
            os.path.join(self.output_dir, 'train_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_train
        )
        train_sec = np.memmap(
            os.path.join(self.output_dir, 'train_sectors.npy'),
            dtype='int64', mode='w+', shape=(n_train,)
        )
        train_ind = np.memmap(
            os.path.join(self.output_dir, 'train_industries.npy'),
            dtype='int64', mode='w+', shape=(n_train,)
        )
        train_rel = np.memmap(
            os.path.join(self.output_dir, 'train_relative_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_train
        )

        val_seq = np.memmap(
            os.path.join(self.output_dir, 'val_sequences.npy'),
            dtype='float32', mode='w+', shape=(n_val, self.seq_length, self.n_features)
        )
        val_lab = np.memmap(
            os.path.join(self.output_dir, 'val_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_val
        )
        val_sec = np.memmap(
            os.path.join(self.output_dir, 'val_sectors.npy'),
            dtype='int64', mode='w+', shape=(n_val,)
        )
        val_ind = np.memmap(
            os.path.join(self.output_dir, 'val_industries.npy'),
            dtype='int64', mode='w+', shape=(n_val,)
        )
        val_rel = np.memmap(
            os.path.join(self.output_dir, 'val_relative_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_val
        )

        test_seq = np.memmap(
            os.path.join(self.output_dir, 'test_sequences.npy'),
            dtype='float32', mode='w+', shape=(n_test, self.seq_length, self.n_features)
        )
        test_lab = np.memmap(
            os.path.join(self.output_dir, 'test_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_test
        )
        test_sec = np.memmap(
            os.path.join(self.output_dir, 'test_sectors.npy'),
            dtype='int64', mode='w+', shape=(n_test,)
        )
        test_ind = np.memmap(
            os.path.join(self.output_dir, 'test_industries.npy'),
            dtype='int64', mode='w+', shape=(n_test,)
        )
        test_rel = np.memmap(
            os.path.join(self.output_dir, 'test_relative_labels.npy'),
            dtype='int64', mode='w+', shape=label_shape_test
        )

        # Track whether relative labels were actually provided
        _has_rel_labels = any(
            os.path.exists(os.path.join(self.temp_dir, f'rel_lab_{ci}.npy'))
            for ci in self.sequences_chunks
        )

        # Track whether future_inputs were provided (TFT mode)
        _has_future_inputs = any(
            os.path.exists(os.path.join(self.temp_dir, f'fut_{ci}.npy'))
            for ci in self.sequences_chunks
        )
        n_fut_feats = self.n_future_features or 0
        from .config import FORWARD_WINDOWS as _fw
        max_fw_val = max(_fw)  # 5

        # Create future_inputs memmaps if future data was provided (TFT mode)
        if _has_future_inputs and n_fut_feats > 0:
            train_fut = np.memmap(
                os.path.join(self.output_dir, 'train_future_inputs.npy'),
                dtype='float32', mode='w+', shape=(n_train, max_fw_val, n_fut_feats)
            )
            val_fut = np.memmap(
                os.path.join(self.output_dir, 'val_future_inputs.npy'),
                dtype='float32', mode='w+', shape=(n_val, max_fw_val, n_fut_feats)
            )
            test_fut = np.memmap(
                os.path.join(self.output_dir, 'test_future_inputs.npy'),
                dtype='float32', mode='w+', shape=(n_test, max_fw_val, n_fut_feats)
            )
        else:
            train_fut = val_fut = test_fut = None

        # First pass: fit scaler on training data (streaming, vectorised per chunk)
        if scaler is None:
            print("Fitting scaler on training data (streaming)...")
            scaler = StandardScaler()

            max_fit_samples = 50000
            fit_data_list = []
            samples_collected = 0
            global_idx = 0

            for chunk_idx in self.sequences_chunks:
                if samples_collected >= max_fit_samples:
                    break
                chunk_seq = np.load(os.path.join(self.temp_dir, f'seq_{chunk_idx}.npy'))
                n = len(chunk_seq)

                # Select training samples from this chunk
                chunk_sp = sample_splits[global_idx:global_idx + n]
                train_samples = chunk_seq[chunk_sp == 'train']

                remaining = max_fit_samples - samples_collected
                if len(train_samples) > remaining:
                    train_samples = train_samples[:remaining]

                if len(train_samples) > 0:
                    fit_data_list.append(train_samples.reshape(-1, self.n_features))
                    samples_collected += len(train_samples)

                global_idx += n
                del chunk_seq

            if fit_data_list:
                fit_data = np.vstack(fit_data_list)
                scaler.fit(fit_data)
                del fit_data, fit_data_list
            print(f"  Scaler fitted on {samples_collected} samples")

        # Second pass: vectorised per-chunk transform + bulk split writes
        print("Writing splits to disk (vectorised)...")
        global_idx = 0
        train_ptr, val_ptr, test_ptr = 0, 0, 0

        for chunk_idx in self.sequences_chunks:
            chunk_seq = np.load(os.path.join(self.temp_dir, f'seq_{chunk_idx}.npy'))
            chunk_lab = np.load(os.path.join(self.temp_dir, f'lab_{chunk_idx}.npy'))
            chunk_sec = np.load(os.path.join(self.temp_dir, f'sec_{chunk_idx}.npy'))
            ind_path  = os.path.join(self.temp_dir, f'ind_{chunk_idx}.npy')
            chunk_ind = np.load(ind_path) if os.path.exists(ind_path) else np.zeros(len(chunk_seq), dtype='int64')
            rel_path  = os.path.join(self.temp_dir, f'rel_lab_{chunk_idx}.npy')
            chunk_rel = np.load(rel_path) if os.path.exists(rel_path) else None
            fut_path  = os.path.join(self.temp_dir, f'fut_{chunk_idx}.npy')
            chunk_fut = np.load(fut_path) if os.path.exists(fut_path) else None
            n = len(chunk_seq)

            # Normalise entire chunk in one call
            chunk_norm = scaler.transform(
                chunk_seq.reshape(-1, self.n_features)
            ).reshape(n, self.seq_length, self.n_features).astype('float32')
            np.nan_to_num(chunk_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip to ±5σ: prevents non-stationary cumulative features (e.g. OBV,
            # PE/PB in regime shifts) from pushing test inputs far outside the
            # training distribution and causing erratic model predictions.
            np.clip(chunk_norm, -5.0, 5.0, out=chunk_norm)

            # Boolean masks from pre-built sample_splits
            chunk_sp   = sample_splits[global_idx:global_idx + n]
            mask_train = chunk_sp == 'train'
            mask_val   = chunk_sp == 'val'
            mask_test  = chunk_sp == 'test'

            n_tr = int(mask_train.sum())
            n_va = int(mask_val.sum())
            n_te = int(mask_test.sum())

            if n_tr:
                train_seq[train_ptr:train_ptr + n_tr] = chunk_norm[mask_train]
                train_lab[train_ptr:train_ptr + n_tr] = chunk_lab[mask_train]
                train_sec[train_ptr:train_ptr + n_tr] = chunk_sec[mask_train]
                train_ind[train_ptr:train_ptr + n_tr] = chunk_ind[mask_train]
                if chunk_rel is not None:
                    train_rel[train_ptr:train_ptr + n_tr] = chunk_rel[mask_train]
                if train_fut is not None and chunk_fut is not None:
                    train_fut[train_ptr:train_ptr + n_tr] = chunk_fut[mask_train]
                train_ptr += n_tr
            if n_va:
                val_seq[val_ptr:val_ptr + n_va] = chunk_norm[mask_val]
                val_lab[val_ptr:val_ptr + n_va] = chunk_lab[mask_val]
                val_sec[val_ptr:val_ptr + n_va] = chunk_sec[mask_val]
                val_ind[val_ptr:val_ptr + n_va] = chunk_ind[mask_val]
                if chunk_rel is not None:
                    val_rel[val_ptr:val_ptr + n_va] = chunk_rel[mask_val]
                if val_fut is not None and chunk_fut is not None:
                    val_fut[val_ptr:val_ptr + n_va] = chunk_fut[mask_val]
                val_ptr += n_va
            if n_te:
                test_seq[test_ptr:test_ptr + n_te] = chunk_norm[mask_test]
                test_lab[test_ptr:test_ptr + n_te] = chunk_lab[mask_test]
                test_sec[test_ptr:test_ptr + n_te] = chunk_sec[mask_test]
                test_ind[test_ptr:test_ptr + n_te] = chunk_ind[mask_test]
                if chunk_rel is not None:
                    test_rel[test_ptr:test_ptr + n_te] = chunk_rel[mask_test]
                if test_fut is not None and chunk_fut is not None:
                    test_fut[test_ptr:test_ptr + n_te] = chunk_fut[mask_test]
                test_ptr += n_te

            global_idx += n
            del chunk_seq, chunk_lab, chunk_sec, chunk_ind, chunk_norm, chunk_rel, chunk_fut

            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed {chunk_idx + 1}/{len(self.sequences_chunks)} chunks...")

            # Flush dirty memmap pages every 20 chunks to release RAM
            if (chunk_idx + 1) % 20 == 0:
                flush_list = [train_seq, train_lab, train_sec, train_ind, train_rel,
                              val_seq,   val_lab,   val_sec,   val_ind,   val_rel,
                              test_seq,  test_lab,  test_sec,  test_ind,  test_rel]
                if train_fut is not None:
                    flush_list += [train_fut, val_fut, test_fut]
                for m in flush_list:
                    m.flush()
                gc.collect()

        # Flush and close all memmaps before preshuffle.
        # On Windows, os.replace() fails with PermissionError if any handle is
        # still open — del inside a for-loop only removes the loop variable,
        # not the named references, so we must del each name explicitly.
        print("Flushing to disk...")
        for m in [train_seq, train_lab, train_sec, train_ind, train_rel,
                  val_seq,   val_lab,   val_sec,   val_ind,   val_rel,
                  test_seq,  test_lab,  test_sec,  test_ind,  test_rel]:
            m.flush()
        del train_seq, train_lab, train_sec, train_ind, train_rel
        del val_seq,   val_lab,   val_sec,   val_ind,   val_rel
        del test_seq,  test_lab,  test_sec,  test_ind,  test_rel
        if train_fut is not None:
            for m in [train_fut, val_fut, test_fut]:
                m.flush()
            del train_fut, val_fut, test_fut
        gc.collect()

        splits_info = {
            'train': {'n_samples': n_train},
            'val': {'n_samples': n_val},
            'test': {'n_samples': n_test}
        }

        # Save metadata
        metadata = {
            'total_samples': self.total_samples,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'num_horizons': self.num_horizons,
            'splits': splits_info,
            'split_mode': split_mode if regime_ok else 'random',
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'random_seed': random_seed,
            'has_relative_labels': _has_rel_labels,
            'has_future_inputs': _has_future_inputs and n_fut_feats > 0,
            'n_future_features': n_fut_feats if _has_future_inputs else 0,
            'max_fw': max_fw_val,
        }

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save scaler
        import joblib
        joblib.dump(scaler, os.path.join(self.output_dir, 'scaler.joblib'))

        # Cleanup temp directory
        shutil.rmtree(self.temp_dir)

        print(f"Dataset saved to {self.output_dir}")
        print(f"  Train: {splits_info['train']['n_samples']} samples")
        print(f"  Val: {splits_info['val']['n_samples']} samples")
        print(f"  Test: {splits_info['test']['n_samples']} samples")

        # Pre-shuffle train split on disk so DataLoader can use shuffle=False.
        # Skipped for files > 2 GB: preshuffle doubles peak disk/page-file usage,
        # and ChunkedMemmapLoader now performs a full within-chunk shuffle in RAM.
        seq_file_gb = n_train * self.seq_length * self.n_features * 4 / 1e9
        if seq_file_gb <= 2.0:
            print(f"\nPre-shuffling train split for sequential I/O during training...")
            print(f"  (train sequences file: {seq_file_gb:.1f} GB, {n_train:,} samples)")
            _preshuffle_split(
                self.output_dir, 'train', n_train,
                self.seq_length, self.n_features, random_seed,
                num_horizons=self.num_horizons,
                has_relative_labels=_has_rel_labels,
                has_future_inputs=_has_future_inputs and n_fut_feats > 0,
                n_future_features=n_fut_feats,
                max_fw=max_fw_val,
            )
            print("Pre-shuffle complete.")
        else:
            print(f"\nSkipping preshuffle (train file {seq_file_gb:.1f} GB > 2 GB threshold).")
            print(f"  ChunkedMemmapLoader will shuffle within each chunk in RAM.")

        # Mark cache as pre-shuffled in metadata
        metadata['train_preshuffled'] = seq_file_gb <= 2.0
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return {'metadata': metadata, 'scaler': scaler}


def load_memmap_datasets(
    data_dir: str
) -> Tuple[MemmapDataset, MemmapDataset, MemmapDataset, StandardScaler]:
    import joblib
    train_dataset = MemmapDataset(data_dir, split='train')
    val_dataset   = MemmapDataset(data_dir, split='val')
    test_dataset  = MemmapDataset(data_dir, split='test')
    scaler = joblib.load(os.path.join(data_dir, 'scaler.joblib'))
    return train_dataset, val_dataset, test_dataset, scaler


def load_into_ram(
    data_dir: str,
) -> Tuple[Tuple, Tuple, Tuple, StandardScaler]:
    """
    Load all split data from disk into RAM using sequential reads.

    The binary files written by finalize() are raw float32/int64 arrays with
    no numpy header, so np.fromfile is used instead of np.load.

    Returns:
        (train_seqs, train_labels, train_sectors, train_industries),
        (val_seqs,   val_labels,   val_sectors,   val_industries),
        (test_seqs,  test_labels,  test_sectors,  test_industries),
        scaler
        industries arrays are None if the file does not exist (pre-industry cache).
    """
    import joblib

    with open(os.path.join(data_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    seq_len       = metadata['seq_length']
    n_features    = metadata['n_features']
    num_horizons  = metadata.get('num_horizons', 1)
    results       = {}

    for split in ['train', 'val', 'test']:
        n       = metadata['splits'][split]['n_samples']
        size_gb = n * seq_len * n_features * 4 / 1e9
        print(f"  Loading {split}: {n:,} samples ({size_gb:.1f} GB) ...", flush=True)

        sequences = np.fromfile(
            os.path.join(data_dir, f'{split}_sequences.npy'), dtype='float32'
        ).reshape(n, seq_len, n_features)

        labels_flat = np.fromfile(os.path.join(data_dir, f'{split}_labels.npy'), dtype='int64')
        if num_horizons > 1:
            labels = labels_flat.reshape(n, num_horizons)
        else:
            labels = labels_flat

        sectors = np.fromfile(os.path.join(data_dir, f'{split}_sectors.npy'), dtype='int64')

        ind_path = os.path.join(data_dir, f'{split}_industries.npy')
        industries = np.fromfile(ind_path, dtype='int64') if os.path.exists(ind_path) else None

        results[split] = (sequences, labels, sectors, industries)

    scaler = joblib.load(os.path.join(data_dir, 'scaler.joblib'))
    return results['train'], results['val'], results['test'], scaler


def cache_exists(cache_dir: str) -> bool:
    """
    Check if a valid cache exists at the given directory.

    Args:
        cache_dir: Directory to check for cache

    Returns:
        True if valid cache exists, False otherwise
    """
    required_files = [
        'metadata.json',
        'scaler.joblib',
        'train_sequences.npy',
        'train_labels.npy',
        'train_sectors.npy',
        'val_sequences.npy',
        'val_labels.npy',
        'val_sectors.npy',
        'test_sequences.npy',
        'test_labels.npy',
        'test_sectors.npy',
        'train_industries.npy',
        'val_industries.npy',
        'test_industries.npy',
    ]

    if not os.path.exists(cache_dir):
        return False

    for f in required_files:
        if not os.path.exists(os.path.join(cache_dir, f)):
            return False

    return True


def get_cache_info(cache_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get information about cached dataset.

    Args:
        cache_dir: Directory containing the cache

    Returns:
        Metadata dictionary if cache exists, None otherwise
    """
    if not cache_exists(cache_dir):
        return None

    with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
        return json.load(f)


def preshuffle_cache(cache_dir: str, random_seed: int = 42):
    """
    Pre-shuffle the train split of an existing cache in-place.

    Run this once on an existing cache so DataLoader can use shuffle=False
    (sequential reads) instead of shuffle=True (random seeks), which is
    10-100x faster for large memmap files on Windows.
    """
    meta = get_cache_info(cache_dir)
    if meta is None:
        raise FileNotFoundError(f"No valid cache found at {cache_dir}")

    if meta.get('train_preshuffled'):
        print("Cache is already pre-shuffled. Nothing to do.")
        return

    n_train       = meta['splits']['train']['n_samples']
    seq_length    = meta['seq_length']
    n_features    = meta['n_features']
    num_horizons  = meta.get('num_horizons', 1)

    print(f"Pre-shuffling train split ({n_train:,} samples)...")
    _preshuffle_split(cache_dir, 'train', n_train, seq_length, n_features, random_seed,
                      num_horizons=num_horizons)

    meta['train_preshuffled'] = True
    with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done. Run training with shuffle=False (handled automatically).")
