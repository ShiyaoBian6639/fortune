"""
Torch profiler script to detect training bottlenecks.

Usage:
    python -m dl.profile_training --use_cache
    python -m dl.profile_training --use_cache --batch_size 512 --profile_steps 40

Outputs:
  1. Console table: CPU/CUDA time per op, sorted by total time
  2. Chrome trace: open in chrome://tracing  (dl_profile_trace.json)
  3. Loader benchmark: next() latency histogram, GPU-idle fraction
"""

import os
import time
import argparse
import json
import statistics
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from .config import get_config, NUM_CLASSES
from .data_processing import load_sector_data
from .models import create_model
from .losses import create_loss_function
from .memmap_dataset import (
    ChunkedMemmapLoader, load_memmap_datasets,
    cache_exists, get_cache_info,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _loader_benchmark(loader, device, n_batches=60):
    """
    Measure data-pipeline latency independent of the model.
    Returns a dict with statistics (all in ms).
    """
    it = iter(loader)
    next_times, to_times = [], []

    for _ in range(n_batches):
        t0 = time.perf_counter()
        try:
            seq, lab, sec = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        seq.to(device, non_blocking=True)
        lab.to(device, non_blocking=True)
        sec.to(device, non_blocking=True)
        torch.cuda.synchronize()          # wait for DMA to finish
        t2 = time.perf_counter()

        next_times.append((t1 - t0) * 1000)
        to_times.append((t2 - t1) * 1000)

    return {
        'next_avg':  statistics.mean(next_times),
        'next_p50':  statistics.median(next_times),
        'next_p95':  sorted(next_times)[int(0.95 * len(next_times))],
        'next_max':  max(next_times),
        'to_avg':    statistics.mean(to_times),
        'to_max':    max(to_times),
        'n':         len(next_times),
    }


def _forward_backward_benchmark(model, loader, criterion, device, n_batches=30):
    """Measure pure GPU forward+backward latency (no data loading)."""
    model.train()
    it = iter(loader)
    times = []
    # Pre-fetch all batches to RAM first so we isolate GPU time
    batches = []
    for _ in range(n_batches):
        try:
            seq, lab, sec = next(it)
        except StopIteration:
            break
        batches.append((
            seq.to(device, non_blocking=True),
            lab.to(device, non_blocking=True),
            sec.to(device, non_blocking=True),
        ))
    torch.cuda.synchronize()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for seq, lab, sec in batches:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq, sec)
        loss = criterion(logits, lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        'fwd_bwd_avg': statistics.mean(times),
        'fwd_bwd_p95': sorted(times)[int(0.95 * len(times))],
        'fwd_bwd_max': max(times),
        'n': len(times),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Torch profiler for training bottlenecks')
    parser.add_argument('--use_cache', action='store_true', required=True,
                        help='Use existing cached data (required)')
    parser.add_argument('--data_cache_dir', type=str, default=None)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--chunk_samples', type=int, default=200_000)
    parser.add_argument('--profile_steps', type=int, default=40,
                        help='Number of training steps to profile (default: 40)')
    parser.add_argument('--warmup_steps',  type=int, default=5,
                        help='Warmup steps before profiling starts')
    parser.add_argument('--trace_file',    type=str, default='dl_profile_trace.json',
                        help='Chrome trace output file')
    parser.add_argument('--skip_profiler', action='store_true',
                        help='Skip torch.profiler (faster loader-only benchmark)')
    args = parser.parse_args()

    config = get_config(batch_size=args.batch_size, chunk_samples=args.chunk_samples)
    device = config['device']
    print(f"Device: {device}")

    cache_dir = args.data_cache_dir or os.path.join(config['data_dir'], 'cache')
    if not cache_exists(cache_dir):
        print(f"ERROR: No cache at {cache_dir}. Run with --memory_efficient first.")
        return

    cache_info = get_cache_info(cache_dir)
    print(f"Cache: {cache_info['total_samples']:,} samples, "
          f"seq={cache_info['seq_length']}, feat={cache_info['n_features']}")

    # ── datasets ─────────────────────────────────────────────────────────────
    train_dataset, val_dataset, test_dataset, scaler = load_memmap_datasets(cache_dir)
    sector_data = load_sector_data(config['data_dir'])

    train_loader = ChunkedMemmapLoader(
        cache_dir     = cache_dir,
        split         = 'train',
        batch_size    = args.batch_size,
        chunk_samples = args.chunk_samples,
        seed          = config['random_seed'],
    )

    # ── model / loss ─────────────────────────────────────────────────────────
    input_dim   = cache_info['n_features']
    num_sectors = len(sector_data['sector'].unique()) if len(sector_data) > 0 else 0
    model       = create_model(config, input_dim, num_sectors).to(device)

    train_labels = train_dataset.get_labels()
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    criterion    = create_loss_function(
        loss_type        = 'focal',
        num_classes      = NUM_CLASSES,
        class_counts     = class_counts,
        device           = device,
        gamma            = config.get('focal_gamma', 2.0),
        beta             = config.get('cb_beta', 0.9999),
        label_smoothing  = config.get('label_smoothing', 0.0),
        use_class_weights= config.get('use_class_weights', True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # ══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 1: Loader latency (independent of model)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Data-loader latency")
    print("=" * 60)

    lb = _loader_benchmark(train_loader, device, n_batches=min(80, len(train_loader)))
    print(f"  next() avg:  {lb['next_avg']:7.1f} ms")
    print(f"  next() p50:  {lb['next_p50']:7.1f} ms")
    print(f"  next() p95:  {lb['next_p95']:7.1f} ms")
    print(f"  next() max:  {lb['next_max']:7.1f} ms  ← chunk-transition stall")
    print(f"  .to(device): {lb['to_avg']:7.1f} ms  avg")
    print(f"  .to(device): {lb['to_max']:7.1f} ms  max")
    print(f"  samples:     {lb['n']}")

    # ══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 2: Pure GPU forward+backward (no data loading)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BENCHMARK 2: GPU forward+backward latency (pre-fetched batches)")
    print("=" * 60)

    gb = _forward_backward_benchmark(model, train_loader, criterion, device,
                                     n_batches=min(40, len(train_loader)))
    gpu_ms     = gb['fwd_bwd_avg']
    loader_ms  = lb['next_avg'] + lb['to_avg']
    total_ms   = max(gpu_ms, loader_ms)   # pipelined: GPU and loader overlap
    gpu_util   = gpu_ms / (gpu_ms + max(0, loader_ms - gpu_ms)) * 100

    print(f"  fwd+bwd avg: {gpu_ms:7.1f} ms")
    print(f"  fwd+bwd p95: {gb['fwd_bwd_p95']:7.1f} ms")
    print(f"  fwd+bwd max: {gb['fwd_bwd_max']:7.1f} ms")

    print("\n" + "-" * 60)
    print("PIPELINE ANALYSIS")
    print("-" * 60)
    print(f"  GPU compute per batch:       {gpu_ms:6.1f} ms")
    print(f"  Data load+transfer per batch:{loader_ms:6.1f} ms  (avg next + to)")
    if loader_ms > gpu_ms:
        stall = loader_ms - gpu_ms
        print(f"  GPU stall per batch:         {stall:6.1f} ms  ← DATA BOTTLENECK")
        print(f"  Estimated GPU utilization:   {gpu_ms/loader_ms*100:5.1f}%")
        print("\n  >> Bottleneck: DATA LOADING. GPU idles waiting for batches.")
        print("     Fix: reduce chunk_samples, faster storage, or async prefetch.")
    else:
        idle = gpu_ms - loader_ms
        print(f"  Loader headroom per batch:   {idle:6.1f} ms  (loader is faster)")
        print(f"  Estimated GPU utilization:   ~100% (compute bound)")
        print("\n  >> Bottleneck: GPU COMPUTE. Data pipeline is keeping up.")

    if args.skip_profiler:
        print("\n(--skip_profiler set, skipping torch.profiler)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 3: torch.profiler — op-level breakdown
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"BENCHMARK 3: torch.profiler ({args.warmup_steps} warmup + {args.profile_steps} active steps)")
    print("=" * 60)
    print(f"Trace will be saved to: {args.trace_file}")
    print("Open with: chrome://tracing  or  https://ui.perfetto.dev")

    model.train()
    data_iter = iter(train_loader)

    prof_schedule = schedule(
        wait=0,
        warmup=args.warmup_steps,
        active=args.profile_steps,
        repeat=1,
    )

    activities = [ProfilerActivity.CPU]
    if device.startswith('cuda') or device == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.prof_log'),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for step in range(args.warmup_steps + args.profile_steps):
            with record_function("data_load"):
                try:
                    seq, lab, sec = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    seq, lab, sec = next(data_iter)

            with record_function("host_to_device"):
                seq = seq.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)
                sec = sec.to(device, non_blocking=True)

            with record_function("forward"):
                optimizer.zero_grad(set_to_none=True)
                logits = model(seq, sec)

            with record_function("loss"):
                loss = criterion(logits, lab)

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer_step"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            prof.step()

            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{args.warmup_steps + args.profile_steps}")

    # ── print key table ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TOP OPS by Self-CPU time  (excludes children):")
    print("─" * 60)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    ))

    if ProfilerActivity.CUDA in activities:
        print("\n" + "─" * 60)
        print("TOP OPS by Self-CUDA time:")
        print("─" * 60)
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        ))

    # ── per-region summary ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("REGION SUMMARY (user-annotated):")
    print("─" * 60)
    regions = ["data_load", "host_to_device", "forward", "loss", "backward", "optimizer_step"]
    key_avgs = prof.key_averages()

    header = f"{'Region':<20} {'CPU ms':>10} {'CUDA ms':>10} {'% of total':>12}"
    print(header)
    print("-" * len(header))

    region_times = {}
    for evt in key_avgs:
        if evt.key in regions:
            cpu_ms  = evt.cpu_time_total  / 1000 / args.profile_steps
            cuda_ms = evt.cuda_time_total / 1000 / args.profile_steps
            region_times[evt.key] = (cpu_ms, cuda_ms)

    total_cpu = sum(v[0] for v in region_times.values()) or 1
    for r in regions:
        if r in region_times:
            c, g = region_times[r]
            print(f"  {r:<18} {c:>10.2f} {g:>10.2f} {c/total_cpu*100:>11.1f}%")

    # ── export Chrome trace ───────────────────────────────────────────────────
    prof.export_chrome_trace(args.trace_file)
    print(f"\nChrome trace exported → {args.trace_file}")
    print("View at: https://ui.perfetto.dev  (drag & drop the file)")

    # ── diagnosis ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if "data_load" in region_times:
        dl_cpu, dl_cuda = region_times["data_load"]
        fwd_cpu, fwd_cuda = region_times.get("forward", (0, 0))
        bwd_cpu, bwd_cuda = region_times.get("backward", (0, 0))
        compute_cuda = fwd_cuda + bwd_cuda
        if dl_cpu > fwd_cuda + bwd_cuda:
            print(f"  DATA LOADING ({dl_cpu:.1f} ms CPU) > GPU COMPUTE ({compute_cuda:.1f} ms CUDA)")
            print("  >> GPU is starved. Increase prefetch depth or use faster storage.")
        else:
            print(f"  GPU COMPUTE ({compute_cuda:.1f} ms CUDA) >= DATA LOADING ({dl_cpu:.1f} ms CPU)")
            print("  >> Pipeline balanced. GPU is not being starved.")

    h2d_cpu = region_times.get("host_to_device", (0, 0))[0]
    if h2d_cpu > 3:
        print(f"\n  HOST-TO-DEVICE ({h2d_cpu:.1f} ms) is elevated.")
        print("  >> Verify pin_memory is enabled for val/test loaders.")

    print("\nDone.")


if __name__ == '__main__':
    main()
