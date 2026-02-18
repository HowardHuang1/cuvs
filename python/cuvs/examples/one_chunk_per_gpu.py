"""
Multi-GPU KMeans with data fully on GPU (one chunk per worker).

Data is created on each worker's GPU (CuPy) and stays there. One chunk per
worker so there is no per-iteration CPU→GPU transfer and no cross-worker
chunk transfer (with locality pinning in fit_dask). Default chunk size is
~4 GB so it runs on typical GPUs; increase chunk_rows for 32 GB+ GPUs.

Run from repo root: python python/cuvs/examples/test.py
"""
import time
import cupy as cp
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


def main():
    # --- Cluster: one worker per GPU (use actual worker count; may be < n_gpus on NFS/setup) ---
    n_gpus = cp.cuda.runtime.getDeviceCount()
    cluster = LocalCUDACluster(
        n_workers=n_gpus,
        threads_per_worker=1,
        memory_limit="50GiB",
    )
    client = Client(cluster)
    # client.run(f) runs f on every worker and returns {worker_address: result}; keys = all addresses
    worker_addresses = sorted(client.run(_return_one).keys())
    n_workers = len(worker_addresses)
    n_chunks = n_workers
    if n_workers < n_gpus:
        print(f"Note: {n_workers} workers for {n_gpus} GPUs (using 1 chunk per worker)")
    print(f"Dask: {n_workers} workers, {n_gpus} GPUs")
    print(f"Dashboard: {client.dashboard_link}")

    # --- One chunk per worker, created on that worker's GPU and kept there ---
    # Default 1M rows × 1024 × 4 bytes ≈ 4 GB/chunk so it runs on typical GPUs without OOM.
    # For 32 GB GPUs you can use chunk_rows = 8_000_000 (~32 GB/chunk).
    chunk_rows = 3_000_000
    n_features = 1024
    n_clusters = 8

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

    # Force "get result" to run on the same worker that created the chunk so the
    # CuPy array stays on that GPU (no transfer). Worker may receive Future or already-resolved array.
    def _get_result(f_or_arr):
        return f_or_arr.result() if hasattr(f_or_arr, "result") else f_or_arr

    block_futures = [
        client.compute(
            delayed(_get_result)(create_futures[i]),
            workers=[worker_addresses[i]],
        )
        for i in range(n_chunks)
    ]
    wait(block_futures)

    # Dask array where block i is the CuPy array on worker i. When fit_dask
    # persists, the block task runs on the worker that has the result → stays on GPU.
    def _identity(x):
        return x

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
