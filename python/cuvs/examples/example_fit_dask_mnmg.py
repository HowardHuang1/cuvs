#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
K-Means Multi-Node Multi-GPU (MNMG) with cuVS.

Similar to cuML's MNMG K-Means: uses Dask with one worker per GPU (OPG layout).
Data lives on CPU (numpy); Dask chunks and distributes to GPUs for parallel fit.
Uses cuVS fit_dask (Lloyd's algorithm with map-reduce across partitions).

Unlike cuML: we use cuVS KMeans (fit_dask) and da.from_array for CPU-held data.
cuML's make_blobs creates Dask cuPy arrays (data on GPU). Here we simulate the
common case: user has numpy array on CPU → Dask chunks it → sends to GPUs.

Memory: reports GPU total and driver RSS (if psutil installed).
For arrays > ~4 GiB, da.from_array embeds data in the task graph and can be slow;
consider da.random (lazy) or disk-backed sources for very large datasets.

Run:
    python python/cuvs/examples/example_fit_dask_mnmg.py

Optional: pip install psutil for driver (CPU) memory reporting.
"""
import os
import time

import numpy as np
import cupy as cp
import dask.array as da
from dask.distributed import Client

try:
    from cuvs.cluster.kmeans import KMeansParams, fit_dask
except ModuleNotFoundError as e:
    if "cydlpack" in str(e):
        print(
            "Error: cuvs is editable install; use regular install.\n"
            "Run: pip uninstall cuvs -y && pip install python/cuvs --no-build-isolation --no-deps"
        )
    raise

try:
    from dask_cuda import LocalCUDACluster
except ImportError:
    LocalCUDACluster = None


def _print_gpu_memory(max_devices=None):
    """Print used/total GiB for each visible GPU."""
    n = cp.cuda.runtime.getDeviceCount()
    if max_devices is not None:
        n = min(n, max_devices)
    for i in range(n):
        free, total = cp.cuda.Device(i).mem_info
        used = total - free
        print(f"  GPU {i}: {used / 2**30:.1f} GiB used / {total / 2**30:.1f} GiB total")


def _driver_memory():
    """Return (used_gib, total_gib) for driver process. used=RSS, total=system RAM. None if no psutil."""
    try:
        import psutil
        used = psutil.Process().memory_info().rss / (1024**3)
        vm = psutil.virtual_memory()
        total = vm.total / (1024**3)
        return used, total
    except ImportError:
        return None


def main():
    n_samples = 28_000_000
    n_features = 1024
    n_clusters = 6
    chunk_rows = 3_500_000  # Each chunk maps to a partition/worker

    print("=== cuVS K-Means MNMG ===\n")
    print(f"Config: n_samples={n_samples:,}, n_features={n_features}, n_clusters={n_clusters}")
    print(f"chunk_rows={chunk_rows:,} → chunk size ~{chunk_rows * n_features * 4 / 2**20:.0f} MiB")
    data_gib = n_samples * n_features * 4 / 2**30
    print(f"Total data: {data_gib:.2f} GiB (float32)\n")

    # --- Start cluster (OPG: one worker per GPU) ---
    if LocalCUDACluster is None:
        raise RuntimeError("dask-cuda required: pip install dask-cuda")
    n_gpus = cp.cuda.runtime.getDeviceCount()
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=list(range(n_gpus)),
        n_workers=n_gpus,
        threads_per_worker=1,
    )
    client = Client(cluster)
    n_workers = len(client.scheduler_info()["workers"])
    print(f"Dask: {n_workers} workers, {n_gpus} GPUs\n")

    # --- Memory: after cluster ---
    print("[Memory] After cluster:")
    _print_gpu_memory()
    driver_mem = _driver_memory()
    if driver_mem is not None:
        used, total = driver_mem
        print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)")

    # --- Generate data on CPU (user-held numpy array) ---
    print(f"\n[Data] Generating numpy array on CPU ({data_gib:.1f} GiB)...")
    print("  (Allocation + 28B randoms may take 1-3 min; monitor with htop/nvidia-smi)")
    t0 = time.perf_counter()
    np.random.seed(42)
    X_np = np.random.random((n_samples, n_features)).astype(np.float32)
    t_gen = time.perf_counter() - t0
    print(f"  Generated in {t_gen:.2f} s")

    driver_mem_after = _driver_memory()
    if driver_mem_after is not None:
        used, total = driver_mem_after
        print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)")

    # --- Chunk into Dask array (logical partitioning; data still on CPU) ---
    X_dask = da.from_array(X_np, chunks=(chunk_rows, n_features), name=False)
    n_chunks = (n_samples + chunk_rows - 1) // chunk_rows
    print(f"  Dask array: {n_chunks} chunks\n")

    # --- cuVS fit_dask ---
    params = KMeansParams(n_clusters=n_clusters, max_iter=100, tol=1e-4)
    print("[Fit] Running fit_dask (Lloyd iterations, cuVS predict per partition)...")
    t_fit0 = time.perf_counter()
    centroids, inertia, n_iter = fit_dask(params, X_dask, client=client)
    t_fit = time.perf_counter() - t_fit0
    print(f"  Converged in {n_iter} iterations, {t_fit:.2f} s")
    print(f"  Inertia: {inertia:.4f}")
    print(f"  Centroids: {centroids.shape}\n")

    # --- Memory: after fit ---
    print("[Memory] After fit:")
    _print_gpu_memory()
    driver_after = _driver_memory()
    if driver_after is not None:
        used, total = driver_after
        print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)\n")

    # --- Summary ---
    print("=== Summary ===")
    print(f"  Data: {n_samples:,} x {n_features} = {data_gib:.2f} GiB")
    print(f"  Workers: {n_workers}, Chunks: {n_chunks}")
    print(f"  Time: fit {t_fit:.2f} s, gen {t_gen:.2f} s")
    if driver_after is not None:
        print(f"  Driver peak: {driver_after[0]:.1f} GiB")

    # Release large arrays before shutdown so workers can clean up faster
    del X_np, X_dask
    client.close()
    try:
        cluster.close(timeout=15)
    except TimeoutError:
        pass  # Result already captured; shutdown is best-effort


if __name__ == "__main__":
    main()
