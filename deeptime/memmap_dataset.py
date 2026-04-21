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
            return np.zeros(shape, dtype=dtype)   # graceful fallback for old caches

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


# ─── Chunked background loader (adapted from dl/memmap_dataset.py) ─────────────

class RegressionChunkedLoader:
    """
    Background-thread prefetching loader for large regression datasets.
    Reads sequential chunks from disk, overlaps I/O with GPU computation.
    """

    def __init__(
        self,
        data_dir:   str,
        split:      str,
        batch_size: int,
        chunk_samples: int = 40_000,
        device:     str = 'cpu',
        shuffle:    bool = True,
    ):
        self.data_dir   = data_dir
        self.split      = split
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle

        with open(os.path.join(data_dir, 'metadata.json')) as f:
            meta = json.load(f)

        si = meta['splits'][split]
        self.n_samples    = si['n_samples']
        self.seq_length   = meta['seq_length']
        self.n_past       = meta['n_past']
        self.n_future     = meta['n_future']
        self.num_horizons = meta['num_horizons']
        self.max_fw       = meta['max_fw']
        self.chunk_samples = min(chunk_samples, self.n_samples)

        self._open_memmaps()
        self._n_batches = max(1, (self.n_samples + batch_size - 1) // batch_size)

    def _open_memmaps(self):
        s = self.split; n = self.n_samples; d = self.data_dir

        def _mm(fname, dtype, shape):
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return np.memmap(path, dtype=dtype, mode='r', shape=shape)
            return np.zeros(shape, dtype=dtype)   # zero-fill for old caches

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

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        executor = ThreadPoolExecutor(max_workers=1)

        def _load_chunk(chunk_idx):
            start = chunk_idx * self.chunk_samples
            end   = min(start + self.chunk_samples, self.n_samples)
            idx   = indices[start:end]
            return (
                self._obs[idx].copy(),
                self._future[idx].copy(),
                self._targets[idx].copy(),
                self._sectors[idx].copy(),
                self._inds[idx].copy(),
                self._subs[idx].copy(),
                self._sizes[idx].copy(),
                self._areas[idx].copy(),
                self._boards[idx].copy(),
                self._ipo[idx].copy(),
                self._dates[idx].copy(),
            )

        n_chunks = (self.n_samples + self.chunk_samples - 1) // self.chunk_samples
        fut = executor.submit(_load_chunk, 0)

        for chunk_i in range(n_chunks):
            chunk = fut.result()
            if chunk_i + 1 < n_chunks:
                fut = executor.submit(_load_chunk, chunk_i + 1)

            obs, future, tgt, sec, ind, sub, sz, area, board, ipo, dates = chunk
            n = len(obs)
            if self.shuffle:
                perm = np.random.permutation(n)
                obs = obs[perm]; future = future[perm]; tgt   = tgt[perm]
                sec = sec[perm]; ind    = ind[perm];    sub   = sub[perm]
                sz  = sz[perm];  area   = area[perm];   board = board[perm]
                ipo = ipo[perm]; dates  = dates[perm]

            for b_start in range(0, n, self.batch_size):
                b_end = b_start + self.batch_size
                yield (
                    torch.from_numpy(obs[b_start:b_end]).to(self.device),
                    torch.from_numpy(future[b_start:b_end]).to(self.device),
                    torch.from_numpy(tgt[b_start:b_end]).to(self.device),
                    torch.from_numpy(sec[b_start:b_end]).to(self.device),
                    torch.from_numpy(ind[b_start:b_end]).to(self.device),
                    torch.from_numpy(sub[b_start:b_end]).to(self.device),
                    torch.from_numpy(sz[b_start:b_end]).to(self.device),
                    torch.from_numpy(area[b_start:b_end]).to(self.device),
                    torch.from_numpy(board[b_start:b_end]).to(self.device),
                    torch.from_numpy(ipo[b_start:b_end]).to(self.device),
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
    train_obs = np.memmap(
        os.path.join(cache_dir, 'train_obs.npy'),
        dtype='float32', mode='r', shape=(n_train, T, n_past)
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

        obs = np.memmap(
            os.path.join(cache_dir, f'{split}_obs.npy'),
            dtype='float32', mode='r+', shape=(n, T, n_past)
        )
        print(f"  Normalizing {split} ({n:,} seqs)...", end=' ', flush=True)

        for start in range(0, n, chunk_size):
            end   = min(start + chunk_size, n)
            chunk = obs[start:end].copy().astype('float64')   # (B, T, F)

            # Tier 2: clip all features to ±5σ (broadcast over B and T)
            np.clip(chunk, clip_lo[np.newaxis, np.newaxis, :],
                    clip_hi[np.newaxis, np.newaxis, :], out=chunk)

            # Tier 1: additionally standardize (in-place for tier1 features)
            for fi in tier1_idx:
                s = float(tier1_std[fi])
                if s > 1e-8:
                    chunk[:, :, fi] = (chunk[:, :, fi] - tier1_mean[fi]) / s

            obs[start:end] = chunk.astype('float32')

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
    chunk_samples: int = 40_000,
    use_chunked: bool = True,
):
    """Load all three splits as loaders."""
    meta = get_cache_info(cache_dir)
    loaders = {}
    for split in ('train', 'val', 'test'):
        n = meta['splits'].get(split, {}).get('n_samples', 0)
        if n == 0:
            loaders[split] = None
            continue
        if use_chunked and split == 'train':
            loaders[split] = RegressionChunkedLoader(
                cache_dir, split, batch_size, chunk_samples, device, shuffle=True
            )
        else:
            from torch.utils.data import DataLoader
            ds = RegressionMemmapDataset(cache_dir, split)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=0, pin_memory=(device != 'cpu'),
            )
    return loaders, meta
