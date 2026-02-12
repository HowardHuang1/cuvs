#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Example: Multi-GPU KMeans with Dask.

Uses dask-cuda.LocalCUDACluster to create one worker per GPU, partitions
a large dataset across workers, and runs iterative Lloyd's algorithm with
map-reduce. Each partition runs cuVS predict + local reduction; results
are aggregated to update centroids.

Requires a regular (non-editable) cuvs install:
    pip install python/cuvs --no-build-isolation --no-deps
    pip install cuvs[dask]

Then run (adjust chunks/workers as needed):
    python python/cuvs/examples/example_fit_dask.py
"""
import os
import sys

import cupy as cp
import dask.array as da
from dask.distributed import Client

try:
    from cuvs.cluster.kmeans import KMeansParams, fit_dask
except ModuleNotFoundError as e:
    if "cydlpack" in str(e):
        print(
            "Error: cuvs is an editable install; examples need a regular install.\n"
            "Run: pip uninstall cuvs -y && pip install python/cuvs --no-build-isolation --no-deps"
        )
    raise

try:
    from dask_cuda import LocalCUDACluster
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False


def _print_gpu_memory():
    """Print used/total memory for each visible GPU."""
    n = cp.cuda.runtime.getDeviceCount()
    for i in range(n):
        free, total = cp.cuda.Device(i).mem_info
        used = total - free
        print(f"  GPU {i}: {used / 2**20:.1f} MiB used / {total / 2**20:.1f} MiB total")


def main():
    print("GPU memory (before cluster):")
    _print_gpu_memory()

    # Prefer LocalCUDACluster (one worker per GPU) when available
    if HAS_DASK_CUDA:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        cluster = LocalCUDACluster(n_workers=n_gpus)
    else:
        from dask.distributed import LocalCluster
        print("dask-cuda not installed; using LocalCluster (may use single GPU)")
        print("For multi-GPU, install: conda install -c rapidsai dask-cuda")
        cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    print("GPU memory (after cluster) / Dask overhead:")
    _print_gpu_memory()

    n_workers = len(client.scheduler_info()["workers"])
    _n_gpus = cp.cuda.runtime.getDeviceCount() if HAS_DASK_CUDA else "N/A"
    print(f"GPUs visible to CuPy: {_n_gpus}, Workers: {n_workers}")

    # Synthetic data: 200k samples, 50 features (CuPy-backed Dask array)
    n_samples = 200_000
    n_features = 50
    n_clusters = 5
    chunk_rows = 20_000  # Each chunk maps to a partition/worker

    X_cupy = cp.random.random((n_samples, n_features), dtype=cp.float32)
    X_dask = da.from_array(X_cupy, chunks=(chunk_rows, n_features))

    print("GPU memory (after data on driver):")
    _print_gpu_memory()

    params = KMeansParams(
        n_clusters=n_clusters,
        max_iter=100,
        tol=1e-4,
    )

    print("Running fit_dask...")
    centroids, inertia, n_iter = fit_dask(params, X_dask, client=client)

    print(f"Converged in {n_iter} iterations")
    print(f"Inertia: {inertia:.4f}")
    print(f"Centroids shape: {centroids.shape}")
    print("GPU memory (after fit; X_cupy/centroids still in driver memory):")
    _print_gpu_memory()

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
