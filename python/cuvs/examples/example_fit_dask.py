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

If dask-cuda sees fewer GPUs than CuPy, try unsetting CUDA_VISIBLE_DEVICES first:
    CUVS_DASK_UNSET_CUDA=1 python python/cuvs/examples/example_fit_dask.py

Data sources for large datasets:
  - Default: da.random (lazy, chunks on workers) - no OOM
  - CUVS_DASK_USE_FROM_ARRAY=1: da.from_array(numpy_array, name=False) - driver holds
    full array, chunks distributed on demand. For 100M+ rows ensure sufficient RAM.
  - From disk: da.from_zarr("path") or da.from_array(np.memmap(...)) for lazy reads.
"""
import os
import sys

import numpy as np
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


def _print_gpu_memory(max_devices=None):
    """Print used/total memory for each visible GPU.

    Limit to max_devices to avoid creating driver context on GPUs without workers
    (querying Device(i).mem_info creates context). Pass n_workers or n_gpus.
    """
    n = cp.cuda.runtime.getDeviceCount()
    if max_devices is not None:
        n = min(n, max_devices)
    for i in range(n):
        free, total = cp.cuda.Device(i).mem_info
        used = total - free
        print(f"  GPU {i}: {used / 2**20:.1f} MiB used / {total / 2**20:.1f} MiB total")


def main():
    n_gpus = cp.cuda.runtime.getDeviceCount() if HAS_DASK_CUDA else None

    print("Environment (before cluster):")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"  RAPIDS_NO_INITIALIZE: {os.environ.get('RAPIDS_NO_INITIALIZE', '(not set)')}")
    # Unset CUDA_VISIBLE_DEVICES so dask-cuda sees all GPUs (set CUVS_DASK_UNSET_CUDA=1 to try)
    if os.environ.get("CUVS_DASK_UNSET_CUDA") == "1":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print("  (CUVS_DASK_UNSET_CUDA=1: unset CUDA_VISIBLE_DEVICES)")

    # Force same overhead on all 8 GPUs: touch every device so driver creates context on all
    print("GPU memory (before cluster; touching all GPUs to force same baseline):")
    _print_gpu_memory(max_devices=None)  # query all visible

    # Prefer LocalCUDACluster (one worker per GPU) when available
    if HAS_DASK_CUDA:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        # Explicit CUDA_VISIBLE_DEVICES overrides env; try list [0,1,...,7]
        cuda_devices = list(range(n_gpus))
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=cuda_devices,
            n_workers=n_gpus,
        )
    else:
        from dask.distributed import LocalCluster
        print("dask-cuda not installed; using LocalCluster (may use single GPU)")
        print("For multi-GPU, install: conda install -c rapidsai dask-cuda")
        cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    n_workers = len(client.scheduler_info()["workers"])
    print("GPU memory (after cluster) / Dask overhead:")
    _print_gpu_memory(max_devices=None)  # show all GPUs to compare worker vs non-worker
    _n_gpus = cp.cuda.runtime.getDeviceCount() if HAS_DASK_CUDA else "N/A"
    print(f"GPUs visible to CuPy (driver): {_n_gpus}, Workers: {n_workers}")
    if HAS_DASK_CUDA and n_workers > 0:
        worker_cuda = client.run(lambda: os.environ.get("CUDA_VISIBLE_DEVICES", "?"))
        print("  CUDA_VISIBLE_DEVICES per worker:", worker_cuda)

    # Data source: lazy (da.random) vs pre-existing array (da.from_array)
    # Set CUVS_DASK_USE_FROM_ARRAY=1 to test with a numpy array (driver holds full data,
    # chunks sent to workers on demand). For CuPy arrays, from_array triggers a copy
    # and OOMs for large data; use numpy or load from disk (da.from_zarr, etc.) instead.
    n_samples = 200_000
    n_features = 50
    n_clusters = 5
    chunk_rows = 20_000  # Each chunk maps to a partition/worker

    if os.environ.get("CUVS_DASK_USE_FROM_ARRAY") == "1":
        # Pre-existing numpy array: driver holds full data; Dask chunks and distributes
        # on demand. name=False avoids hashing copy. For 100M+ rows ensure driver has RAM.
        X_np = np.random.random((n_samples, n_features)).astype(np.float32)
        X_dask = da.from_array(
            X_np,
            chunks=(chunk_rows, n_features),
            name=False,  # avoid hashing copy
        )
        print("GPU memory (after from_array; driver holds full numpy array):")
    else:
        # Lazy generation: chunks created on workers, no driver allocation
        X_dask = da.random.random(
            (n_samples, n_features),
            chunks=(chunk_rows, n_features),
            dtype=np.float32,
        )
        print("GPU memory (after data created lazily; no driver allocation):")
    _print_gpu_memory(max_devices=None)

    params = KMeansParams(
        n_clusters=n_clusters,
        max_iter=100,
        tol=1e-4,
    )

    n_chunks = (n_samples + chunk_rows - 1) // chunk_rows
    print(f"Running fit_dask ({n_chunks} chunks, ~{n_chunks * chunk_rows * n_features * 4 / 2**30:.1f} GiB total)...")
    print("  (Large datasets take several min/iteration; monitor with Dask dashboard if slow)")
    centroids, inertia, n_iter = fit_dask(params, X_dask, client=client)

    print(f"Converged in {n_iter} iterations")
    print(f"Inertia: {inertia:.4f}")
    print(f"Centroids shape: {centroids.shape}")
    print("GPU memory (after fit; centroids in driver memory):")
    _print_gpu_memory(max_devices=None)

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
