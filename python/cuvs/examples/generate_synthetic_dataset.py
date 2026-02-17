#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Generate a random synthetic dataset and save to disk in chunked format.

Use with example_fit_dask_mnmg.py --data-path <output_dir> to load and run K-Means.

Output format: directory with meta.json and chunk_00000.npy, chunk_00001.npy, ...
Path must be accessible from all machines (e.g. NFS) when using multi-node.

Run:
    python python/cuvs/examples/generate_synthetic_dataset.py --output ./synthetic_60M_1024
"""
import argparse
import json
import os
import time

import numpy as np


def _gen_and_save_chunk_on_worker(chunk_id, n_samples, n_features, chunk_rows, seed, output_dir):
    """Run on worker: generate on GPU, save to output_dir, return path."""
    import cupy as cp

    cp.random.seed(seed + chunk_id)
    row_start = chunk_id * chunk_rows
    row_end = min((chunk_id + 1) * chunk_rows, n_samples)
    n_r = row_end - row_start
    gpu = cp.random.random((n_r, n_features), dtype=cp.float32)
    cpu = cp.asnumpy(gpu)
    del gpu

    path = os.path.join(output_dir, f"chunk_{chunk_id:05d}.npy")
    np.save(path, cpu)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic random dataset (chunked)")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory (created if missing). Use with example_fit_dask_mnmg.py --data-path",
    )
    parser.add_argument("--n-samples", type=int, default=60_000_000)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    n_chunks = (args.n_samples + args.chunk_rows - 1) // args.chunk_rows
    data_gib = args.n_samples * args.n_features * 4 / 2**30

    print(f"=== Generate Synthetic Dataset ===\n")
    print(f"Config: n_samples={args.n_samples:,}, n_features={args.n_features}")
    print(f"chunk_rows={args.chunk_rows:,} → {n_chunks} chunks, ~{data_gib:.1f} GiB total\n")

    # Parallel generation on worker GPUs
    from dask.delayed import delayed
    from dask.distributed import Client, wait

    try:
        from dask_cuda import LocalCUDACluster
    except ImportError:
        raise RuntimeError("dask-cuda required: pip install dask-cuda")

    import cupy as cp

    n_gpus = cp.cuda.runtime.getDeviceCount()
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=list(range(n_gpus)),
        n_workers=n_gpus,
        threads_per_worker=1,
    )
    client = Client(cluster)

    print(f"Generating {n_chunks} chunks on {n_gpus} worker GPUs...")
    t0 = time.perf_counter()
    gen_delayed = [
        delayed(_gen_and_save_chunk_on_worker)(
            i, args.n_samples, args.n_features, args.chunk_rows, args.seed, args.output
        )
        for i in range(n_chunks)
    ]
    paths = client.compute(gen_delayed)
    wait(paths)
    t_gen = time.perf_counter() - t0
    client.close()
    cluster.close()

    # Write metadata
    meta = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "chunk_rows": args.chunk_rows,
        "seed": args.seed,
        "n_chunks": n_chunks,
    }
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {args.output}/ in {t_gen:.2f} s")
    print(f"  Run: python example_fit_dask_mnmg.py --data-path {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
