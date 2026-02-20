#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Generate one synthetic dataset as a single file (no per-chunk files).

Random generation on GPU (CuPy); writes to disk in blocks via memmap.

Use with one_chunk_per_gpu.py --data-path <output_dir>.

Run:
    python python/cuvs/examples/generate_synthetic_dataset_single.py --output ./synthetic_64M_1024
"""
import argparse
import json
import os
import time

import cupy as cp
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate one synthetic dataset as a single file (GPU, memmap block writes)"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory (created if missing). Must contain data.bin and meta.json for one_chunk_per_gpu.py.",
    )
    parser.add_argument("--n-samples", type=int, default=56_000_000)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--block-rows", type=int, default=1_000_000, help="Rows per block when writing (memory bound)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data_path = os.path.join(args.output, "data.bin")
    meta_path = os.path.join(args.output, "meta.json")
    data_gib = args.n_samples * args.n_features * 4 / 2**30

    print("=== Generate synthetic dataset (single file) ===\n")
    print(f"Config: n_samples={args.n_samples:,}, n_features={args.n_features}")
    print(f"Output: {data_path} (~{data_gib:.1f} GiB)")
    print(f"Writing in blocks of {args.block_rows:,} rows (GPU)\n")

    t0 = time.perf_counter()

    arr = np.memmap(
        data_path,
        dtype=np.float32,
        mode="w+",
        shape=(args.n_samples, args.n_features),
    )
    bar_width = 40
    written = 0
    cp.random.seed(args.seed)
    while written < args.n_samples:
        n_rows = min(args.block_rows, args.n_samples - written)
        block_gpu = cp.random.randn(n_rows, args.n_features, dtype=cp.float32)  # GPU
        block = cp.asnumpy(block_gpu)  # GPU → host
        del block_gpu
        arr[written : written + n_rows] = block  # host → disk (memmap)
        written += n_rows
        if written % (10 * args.block_rows) == 0 or written == args.n_samples:
            arr.flush()
        # Progress bar: per-block time = GPU gen + transfer + disk write
        pct = written / args.n_samples
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}%  {written:,} / {args.n_samples:,} rows", end="", flush=True)
    print()
    del arr

    meta = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "dtype": "float32",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    t_elapsed = time.perf_counter() - t0
    print(f"\n  Saved to {args.output}/ in {t_elapsed:.2f} s")
    print(f"  Run: python python/cuvs/examples/one_chunk_per_gpu.py --data-path {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
