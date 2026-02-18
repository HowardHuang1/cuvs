"""
Multi-GPU KMeans with data fully on GPU (one chunk per worker).

Data can be (1) loaded from a single-file dataset (--data-path) and partitioned
onto GPUs, or (2) generated on each worker's GPU (no --data-path). One chunk per
worker so there is no per-iteration CPU→GPU transfer and no cross-worker chunk
transfer (with locality pinning in fit_dask).

Spilling: dask-cuda spills GPU→host by default when GPU usage reaches 80% so
workloads with more data than GPU memory can run. We disable spilling with
device_memory_limit=0 so one chunk per GPU stays on GPU (no 2×, no purge OOM).

Run:
  # Load dataset from disk (from generate_synthetic_dataset_single.py):
  python python/cuvs/examples/one_chunk_per_gpu.py --data-path ./synthetic_64M_1024

  # Or generate fake data on GPUs (no --data-path):
  python python/cuvs/examples/one_chunk_per_gpu.py
"""
import argparse
import json
import os
import time

import cupy as cp
import numpy as np
import dask.array as da
from dask.delayed import delayed
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

from cuvs.cluster.kmeans import KMeansParams, fit_dask


def _return_one():
    """Run on each worker via client.run(); used to get the set of all worker addresses."""
    return 1


def _gpu_mem_info():
    """Run on each worker: return this worker's GPU memory (used, total) and device id."""
    free, total = cp.cuda.runtime.memGetInfo()
    dev = cp.cuda.runtime.getDevice()
    return {"device_id": dev, "used": total - free, "total": total}


def _free_gpu_memory():
    """Run on each worker: free CuPy GPU and pinned memory so purge has room.
    After cancel, dask-cuda purge may reload spilled keys to GPU; if GPU is full, that OOMs.
    """
    cp.cuda.Device().synchronize()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    pinned = cp.get_default_pinned_memory_pool()
    pinned.free_all_blocks()


def _create_cupy_chunk_on_gpu(n_rows, n_features, seed):
    """Run on worker: create CuPy array on this worker's GPU. Stays on GPU."""
    cp.random.seed(seed)
    return cp.random.randn(n_rows, n_features, dtype=cp.float32)


def _load_chunk_from_file(data_path, chunk_id, chunk_rows, n_features, n_samples):
    """Run on worker: load chunk [chunk_id] from data.bin (memmap) and return CuPy array on this GPU."""
    import cupy as cp
    path = os.path.join(data_path, "data.bin")
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n_samples, n_features))
    start = chunk_id * chunk_rows
    end = min(start + chunk_rows, n_samples)
    chunk = np.asarray(arr[start:end], dtype=np.float32, order="C")
    return cp.asarray(chunk)


def _get_result(f_or_arr):
    """Unwrap Future to value, or return as-is if already resolved."""
    return f_or_arr.result() if hasattr(f_or_arr, "result") else f_or_arr


def _identity(x):
    return x


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU KMeans (one chunk per GPU)")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset dir with data.bin and meta.json (from generate_synthetic_dataset_single.py). If not set, generate fake data on GPUs.",
    )
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of KMeans clusters")
    args = parser.parse_args()

    data_path = args.data_path
    n_clusters = args.n_clusters
    # --- Cluster: one worker per GPU ---
    n_gpus = cp.cuda.runtime.getDeviceCount()
    cluster = LocalCUDACluster(
        n_workers=n_gpus,
        threads_per_worker=1,
        device_memory_limit=0,
        memory_limit="80GiB",
    )
    client = Client(cluster)
    worker_addresses = sorted(client.run(_return_one).keys())
    n_workers = len(worker_addresses)
    n_chunks = n_workers
    if n_workers < n_gpus:
        print(f"Note: {n_workers} workers for {n_gpus} GPUs (using 1 chunk per worker)")
    print(f"Dask: {n_workers} workers, {n_gpus} GPUs")
    print(f"Dashboard: {client.dashboard_link}")

    if data_path is not None:
        # --- Load dataset from disk and partition onto GPUs (1 chunk per worker) ---
        data_path = os.path.abspath(data_path)
        meta_path = os.path.join(data_path, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.json not found in {data_path}. Run generate_synthetic_dataset_single.py first.")
        with open(meta_path) as f:
            meta = json.load(f)
        n_samples = meta["n_samples"]
        n_features = meta["n_features"]
        chunk_rows = (n_samples + n_chunks - 1) // n_chunks
        last_chunk_rows = n_samples - (n_chunks - 1) * chunk_rows
        print(f"\n[Data] Loading from {data_path} ({n_samples:,} x {n_features})...")
        print(f"  Partitioning into {n_chunks} chunks (~{chunk_rows:,} rows each, last chunk {last_chunk_rows:,} rows)...")

        load_futures = [
            client.submit(
                _load_chunk_from_file,
                data_path,
                i,
                chunk_rows,
                n_features,
                n_samples,
                workers=[worker_addresses[i]],
            )
            for i in range(n_chunks)
        ]
        wait(load_futures)

        block_futures = [
            client.compute(
                delayed(_get_result)(load_futures[i]),
                workers=[worker_addresses[i]],
            )
            for i in range(n_chunks)
        ]
        wait(block_futures)

        # Chunk i has rows [i*chunk_rows, min((i+1)*chunk_rows, n_samples)); last chunk may be smaller
        def _chunk_rows(i):
            start = i * chunk_rows
            end = min(start + chunk_rows, n_samples)
            return end - start

        blocks = [
            da.from_delayed(
                delayed(_identity)(block_futures[i]),
                shape=(_chunk_rows(i), n_features),
                dtype=cp.float32,
            )
            for i in range(n_chunks)
        ]
        X_dask = da.concatenate(blocks, axis=0)
        print(f"  Shape: {n_samples:,} x {n_features} ({n_chunks} chunks, data on GPU)\n")
    else:
        # --- Generate fake data on each worker's GPU (no --data-path) ---
        n_features = 1024
        chunk_rows = 7_900_000  # ~32 GB with device_memory_limit=0; use 6M–7M if OOM
        print(f"\n[Data] Creating {n_chunks} CuPy chunks on GPU (1 per worker)...")
        create_futures = [
            client.submit(
                _create_cupy_chunk_on_gpu,
                chunk_rows,
                n_features,
                42 + i,
                workers=[worker_addresses[i]],
            )
            for i in range(n_chunks)
        ]
        wait(create_futures)

        block_futures = [
            client.compute(
                delayed(_get_result)(create_futures[i]),
                workers=[worker_addresses[i]],
            )
            for i in range(n_chunks)
        ]
        wait(block_futures)

        blocks = [
            da.from_delayed(
                delayed(_identity)(block_futures[i]),
                shape=(chunk_rows, n_features),
                dtype=cp.float32,
            )
            for i in range(n_chunks)
        ]
        X_dask = da.concatenate(blocks, axis=0)
        n_samples = chunk_rows * n_chunks
        print(f"  Shape: {n_samples:,} x {n_features} ({n_chunks} chunks, data on GPU)\n")

    # GPU memory after dataset on each worker's GPU (expect high used/total per device)
    mem_by_worker = client.run(_gpu_mem_info)
    by_device = sorted(mem_by_worker.items(), key=lambda x: x[1]["device_id"])
    print("[GPU memory] After dataset on GPU (per device):")
    for addr, info in by_device:
        used_gib = info["used"] / (2**30)
        total_gib = info["total"] / (2**30)
        pct = 100.0 * info["used"] / info["total"] if info["total"] else 0
        print(f"  GPU {info['device_id']}: {used_gib:.2f} / {total_gib:.2f} GiB ({pct:.0f}% used)")
    print()

    # --- Multi-GPU KMeans (chunks already on GPU; fit_dask pins partition i to worker i) ---
    params = KMeansParams(n_clusters=n_clusters, max_iter=100, tol=1e-4)
    print("[Fit] fit_dask (data on GPU, no per-iteration transfer)...")
    centroids, inertia, n_iter = fit_dask(params, X_dask, client=client)
    print(f"  Converged in {n_iter} iterations, inertia={inertia:.4f}")
    print(f"  Centroids: {centroids.shape}\n")

    # Release persisted GPU data and wait so workers drop it before teardown.
    # Otherwise dask-cuda purge can try to reload spilled chunks to GPU and OOM.
    # (Client has no unpersist; canceling the collection releases its futures.)
    client.cancel(X_dask)
    # Let workers process the release so they drop references; then free CuPy pools
    # so purge's "key in self.data" (dask-cuda reload from host→GPU) has room.
    time.sleep(2)
    client.run(_free_gpu_memory)
    time.sleep(5)
    client.close()
    try:
        cluster.close(timeout=15)
    except TimeoutError:
        pass  # Result already captured; shutdown is best-effort


if __name__ == "__main__":
    main()
