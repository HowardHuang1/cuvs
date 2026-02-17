#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
K-Means Multi-Node Multi-GPU (MNMG) with cuVS.

Similar to cuML's MNMG K-Means: uses Dask with one worker per GPU (OPG layout).
Data lives on CPU (numpy); Dask chunks and distributes to GPUs for parallel fit.
Uses cuVS fit_dask (Lloyd's algorithm with map-reduce across partitions).

Unlike cuML: we use cuVS KMeans (fit_dask). Data is generated or loaded in chunks
on workers (numpy), then sent to GPUs for fit. cuML's make_blobs creates Dask
cuPy arrays (data on GPU). Here we use numpy chunks on workers → CuPy per partition.

Memory: reports GPU used/total (pynvml > CuPy > nvidia-smi) and driver RSS (psutil).
Data paths:
  - Generate: chunks generated in parallel on worker GPUs (low driver memory).
  - Load: --data-path <dir> loads chunks from disk (from generate_synthetic_dataset.py).
    Chunks stay numpy on workers; _partition_fit converts to CuPy per call (freed after).

Run:
    python python/cuvs/examples/example_fit_dask_mnmg.py  # generate synthetic in parallel
    python example_fit_dask_mnmg.py --data-path ./synthetic_60M_1024  # load from disk

Script sets DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL=30m (default 5min causes worker restarts
and "Removing worker...scattered data" when GPU tasks take longer than 5min).
Override: DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL=2h python ... for very long runs.

Optional: pip install psutil for driver (CPU) memory, nvidia-ml-py for accurate GPU memory.
"""
import argparse
import json
import os
import time

# Set before any distributed imports (default 5min TTL causes worker removals during long fits)
os.environ.setdefault("DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL", "1h")

import numpy as np
import cupy as cp
import dask.array as da
from dask.delayed import delayed
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


def _worker_gpu_mem():
    """Run on each Dask worker: cudaMemGetInfo from the process that owns the GPU (like MPI)."""
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    dev = cp.cuda.runtime.getDevice()
    return {"device_id": dev, "free": free, "total": total, "used": total - free}


def _print_gpu_memory(max_devices=None, client=None):
    """Print used/total GiB for each visible GPU.

    Driver-side methods (pynvml, CuPy, nvidia-smi) query from the driver process.
    Worker-side (cudaMemGetInfo) queries from worker processes that own the GPUs - same as MPI.
    """
    n = cp.cuda.runtime.getDeviceCount()
    if max_devices is not None:
        n = min(n, max_devices)

    # 0. Worker-side cudaMemGetInfo - like MPI: query from process that owns the GPU
    if client is not None:
        try:
            results = client.run(_worker_gpu_mem)
            # Sort by worker address for stable ordering
            sorted_workers = sorted(results.keys())
            print("  [worker cudaMemGetInfo]")
            for i, addr in enumerate(sorted_workers):
                if i >= n:
                    break
                r = results[addr]
                used = r["used"] / (2**30)
                total = r["total"] / (2**30)
                print(f"    GPU {i}: {used:.1f} GiB used / {total:.1f} GiB total")
        except Exception as e:
            print(f"  [worker cudaMemGetInfo] failed: {e}")

    # 1. pynvml (nvidia-ml-py) - matches nvidia-smi, includes reserved memory
    try:
        import pynvml

        pynvml.nvmlInit()
        print("  [pynvml]")
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            used = info.used / (2**30)
            total = info.total / (2**30)
            print(f"    GPU {i}: {used:.1f} GiB used / {total:.1f} GiB total")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"  [pynvml] failed: {e}")

    # 2. CuPy mem_info (cudaMemGetInfo) - device-level free/total
    try:
        print("  [CuPy mem_info]")
        for i in range(n):
            free, total_bytes = cp.cuda.Device(i).mem_info
            used = (total_bytes - free) / (2**30)
            total_gib = total_bytes / (2**30)
            print(f"    GPU {i}: {used:.1f} GiB used / {total_gib:.1f} GiB total")
    except Exception as e:
        print(f"  [CuPy mem_info] failed: {e}")

    # 3. nvidia-smi subprocess
    try:
        import subprocess

        print("  [nvidia-smi]")
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        for i, line in enumerate(out.strip().split("\n")[:n]):
            used_mib, total_mib = line.split(", ")
            used = float(used_mib.strip()) / 1024
            total = float(total_mib.strip().split()[0]) / 1024
            print(f"    GPU {i}: {used:.1f} GiB used / {total:.1f} GiB total")
    except Exception as e:
        print(f"  [nvidia-smi] failed: {e}")


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
    parser = argparse.ArgumentParser(description="K-Means MNMG with cuVS")
    parser.add_argument(
        "--data-path",
        "-i",
        default=None,
        help="Load dataset from directory (from generate_synthetic_dataset.py). "
        "Path must be accessible from all workers (e.g. NFS).",
    )
    parser.add_argument("--n-samples", type=int, default=60_000_000)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    data_path = args.data_path
    if data_path is not None:
        data_path = os.path.abspath(data_path)
        with open(os.path.join(data_path, "meta.json")) as f:
            meta = json.load(f)
        n_samples = meta["n_samples"]
        n_features = meta["n_features"]
        chunk_rows = meta["chunk_rows"]
        n_chunks = meta["n_chunks"]
    else:
        n_samples = args.n_samples
        n_features = args.n_features
        chunk_rows = args.chunk_rows
        n_chunks = (n_samples + chunk_rows - 1) // chunk_rows
    n_clusters = args.n_clusters

    print("=== cuVS K-Means MNMG ===\n")
    print(f"Config: n_samples={n_samples:,}, n_features={n_features}, n_clusters={n_clusters}")
    print(f"chunk_rows={chunk_rows:,} → chunk size ~{chunk_rows * n_features * 4 / 2**20:.0f} MiB")
    data_gib = n_samples * n_features * 4 / 2**30
    print(f"Total data: {data_gib:.2f} GiB (float32)\n")

    # --- Start cluster (OPG: one worker per GPU) ---
    if LocalCUDACluster is None:
        raise RuntimeError("dask-cuda required: pip install dask-cuda")
    n_gpus = cp.cuda.runtime.getDeviceCount()

    # memory_limit: default 'auto' (~38GB/worker on 503GB) can kill workers holding ~30GB of
    # chunks. Use 50GiB to avoid Dask killing workers for memory; reduce if OOM persists.
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=list(range(n_gpus)),
        n_workers=n_gpus,
        threads_per_worker=1,
        memory_limit="50GiB",
    )
    client = Client(cluster)
    n_workers = len(client.scheduler_info()["workers"])
    print(f"Dask: {n_workers} workers, {n_gpus} GPUs\n")

    # --- Memory: after cluster ---
    print("[Memory] After cluster:")
    _print_gpu_memory(client=client)
    driver_mem = _driver_memory()
    if driver_mem is not None:
        used, total = driver_mem
        print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)")

    # Worker addresses for pinning tasks (improves locality, reduces recomputation on worker loss)
    worker_addresses = sorted(client.scheduler_info()["workers"])
    n_workers = len(worker_addresses)

    if data_path is not None:
        # Load chunks from disk (workers load in parallel); pin chunk i to worker i % n_workers
        print(f"\n[Data] Loading {n_chunks} chunks from {data_path} (pinned to workers)...")
        t0 = time.perf_counter()

        def _load_chunk_on_worker(path, chunk_id, n_r, n_feat):
            """Run on worker: load chunk from disk, return numpy."""
            p = os.path.join(path, f"chunk_{chunk_id:05d}.npy")
            arr = np.load(p)
            return np.asarray(arr, dtype=np.float32).reshape(n_r, n_feat)

        def _get_chunk(f, n_r, n_c):
            arr = f.result() if hasattr(f, "result") else f
            return np.asarray(arr, dtype=np.float32).reshape(n_r, n_c)

        from dask.distributed import as_completed

        load_futures = []
        for i in range(n_chunks):
            n_r = min(chunk_rows, n_samples - i * chunk_rows)
            w = worker_addresses[i % n_workers]
            fut = client.submit(
                _load_chunk_on_worker,
                data_path,
                i,
                n_r,
                n_features,
                workers=[w],
                allow_other_workers=True,
            )
            load_futures.append(fut)

        completed = 0
        for _ in as_completed(load_futures):
            completed += 1
            print(f"  Loaded chunk {completed}/{n_chunks}...", flush=True)
        t_gen = time.perf_counter() - t0
        print(f"  Loaded all {n_chunks} chunks in {t_gen:.2f} s")
        driver_mem_after = _driver_memory()
        if driver_mem_after is not None:
            used, total = driver_mem_after
            print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)")

        blocks = []
        for i in range(n_chunks):
            n_rows_i = min(chunk_rows, n_samples - i * chunk_rows)
            block = da.from_delayed(
                delayed(_get_chunk)(load_futures[i], n_rows_i, n_features),
                shape=(n_rows_i, n_features),
                dtype=np.float32,
            )
            blocks.append(block)
        X_dask = da.concatenate(blocks, axis=0)
        print(f"  Dask array: {n_chunks} chunks (data on workers)\n")
    else:
        # Generate chunks on workers (each worker's GPU in parallel)
        print(f"\n[Data] Generating {n_chunks} chunks on worker GPUs (parallel)...")
        t0 = time.perf_counter()

        def _gen_chunk_on_worker(chunk_id, n_samp, n_feat, chunk_rows_val):
            """Run on worker: generate chunk on this worker's GPU, return numpy."""
            import cupy as cp

            cp.random.seed(42 + chunk_id)
            row_start = chunk_id * chunk_rows_val
            row_end = min((chunk_id + 1) * chunk_rows_val, n_samp)
            n_r = row_end - row_start
            gpu = cp.random.random((n_r, n_feat), dtype=cp.float32)
            cpu = cp.asnumpy(gpu)
            del gpu
            return cpu

        def _get_chunk(f, n_r, n_c):
            arr = f.result() if hasattr(f, "result") else f
            return np.asarray(arr, dtype=np.float32).reshape(n_r, n_c)

        # Pin chunk i to worker i % n_workers for locality
        from dask.distributed import as_completed

        gen_futures = []
        for i in range(n_chunks):
            w = worker_addresses[i % n_workers]
            fut = client.submit(
                _gen_chunk_on_worker,
                i,
                n_samples,
                n_features,
                chunk_rows,
                workers=[w],
                allow_other_workers=True,
            )
            gen_futures.append(fut)

        completed = 0
        for _ in as_completed(gen_futures):
            completed += 1
            print(f"  Generated chunk {completed}/{n_chunks}...", flush=True)
        t_gen = time.perf_counter() - t0
        print(f"  Generated all {n_chunks} chunks in {t_gen:.2f} s")
        scattered = gen_futures
        driver_mem_after = _driver_memory()
        if driver_mem_after is not None:
            used, total = driver_mem_after
            print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)")

        blocks = []
        for i in range(n_chunks):
            n_rows_i = min(chunk_rows, n_samples - i * chunk_rows)
            # scattered[i] is a Future; _get_chunk fetches the numpy from the worker
            block = da.from_delayed(
                delayed(_get_chunk)(scattered[i], n_rows_i, n_features),
                shape=(n_rows_i, n_features),
                dtype=np.float32,
            )
            blocks.append(block)
        X_dask = da.concatenate(blocks, axis=0)
        print(f"  Dask array: {n_chunks} chunks (data on workers)\n")

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
    _print_gpu_memory(client=client)
    driver_after = _driver_memory()
    if driver_after is not None:
        used, total = driver_after
        print(f"  Driver (CPU): {used:.1f} GiB used / {total:.1f} GiB total (system)\n")

    # --- Summary ---
    print("=== Summary ===")
    print(f"  Data: {n_samples:,} x {n_features} = {data_gib:.2f} GiB")
    print(f"  Workers: {n_workers}, Chunks: {n_chunks}")
    gen_label = "load" if data_path is not None else "gen"
    print(f"  Time: fit {t_fit:.2f} s, {gen_label} {t_gen:.2f} s")
    if driver_after is not None:
        print(f"  Driver peak: {driver_after[0]:.1f} GiB")

    # Release references before shutdown so workers can clean up faster
    del X_dask
    client.close()
    try:
        cluster.close(timeout=15)
    except TimeoutError:
        pass  # Result already captured; shutdown is best-effort


if __name__ == "__main__":
    main()
