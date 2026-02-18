# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Dask-based multi-GPU KMeans.

Uses Dask to partition a large dataset across workers (one per GPU with
dask-cuda.LocalCUDACluster). Each iteration: (1) broadcast centroids to all
partitions, (2) per-partition assign + local reduction via cuVS predict,
(3) aggregate partial sums/counts, (4) update centroids until convergence.

The existing MPI-based fit_mg / fit_mg_sharded remain for users who prefer that
approach. This module provides a Dask-native alternative that integrates with
the broader Python distributed ecosystem.
"""

from __future__ import annotations

import sys
from collections import namedtuple
from typing import TYPE_CHECKING, Optional

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit, predict

if TYPE_CHECKING:
    from dask.distributed import Client

    import dask.array as da

FitDaskOutput = namedtuple("FitDaskOutput", "centroids inertia n_iter")

# cuVS KMeans predict uses int32 indexing; n_rows * n_features must be <= INT32_MAX.
# 2M * 1024 = 2.048e9 < 2^31-1, so we chunk predict for larger partitions.
MAX_PREDICT_ROWS = 2_000_000


def _worker_addr_normalize(addr: Optional[str]) -> Optional[tuple]:
    """Normalize worker address to (host, port) for comparison. Scheduler may use
    contact (127.0.0.1:port) while get_worker().address may be listen (0.0.0.0:port).
    """
    if not addr:
        return None
    try:
        # tcp://127.0.0.1:33167 -> host="127.0.0.1", port=33167
        part = addr.replace("tcp://", "").replace("tls://", "").strip()
        if ":" in part:
            host, port = part.rsplit(":", 1)
            return (host.strip(), int(port))
    except (ValueError, AttributeError):
        pass
    return None


def _worker_addr_match(addr1: Optional[str], addr2: Optional[str]) -> bool:
    """True if both addresses refer to the same worker (same port; localhost hosts treated equal)."""
    n1, n2 = _worker_addr_normalize(addr1), _worker_addr_normalize(addr2)
    if n1 is None or n2 is None:
        return addr1 == addr2
    h1, p1 = n1
    h2, p2 = n2
    if p1 != p2:
        return False
    localhost = {"127.0.0.1", "0.0.0.0", "localhost", ""}
    return h1 == h2 or (h1 in localhost and h2 in localhost)


def _partition_fit(
    X_partition,
    centroids: np.ndarray,
    n_clusters: int,
    metric: str = "L2Expanded",
    expected_worker: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run predict on a partition, then compute partial sums and counts.
    Runs on a Dask worker (with GPU). X_partition should be CuPy when on a
    CUDA worker; centroids is numpy (small, broadcast to all workers).
    If expected_worker is set and we're on that worker, use X_partition in place
    (no CPU transfer). Otherwise copy via host to avoid cross-process GPU pointers.
    """
    import cupy as cp
    from cuvs.cluster.kmeans import KMeansParams, predict
    from dask.distributed import get_worker

    # Use worker address as unique ID: only use array in place when this task
    # runs on the worker that owns the block (avoids cudaErrorIllegalAddress).
    worker = get_worker()
    this_worker = worker.address
    this_worker_alt = getattr(worker, "contact_address", None)
    # Normalize comparison: scheduler who_has may use contact (127.0.0.1:port),
    # get_worker().address may be listen (0.0.0.0:port) -> same worker, same port
    on_own_worker = (
        expected_worker is not None
        and (
            _worker_addr_match(this_worker, expected_worker)
            or _worker_addr_match(this_worker_alt, expected_worker)
        )
    )

    if on_own_worker and hasattr(X_partition, "get"):
        print(f"      [partition] data on this worker ({this_worker}), no transfer", flush=True)
        # Use in place when already C-contiguous to avoid 2× GPU memory (MPI path has no copy).
        # If Dask gives an array with wrong metadata, copy to a fresh buffer for cuVS predict.
        X = cp.asarray(X_partition, dtype=cp.float32)
        if not X.flags.c_contiguous:
            X = X.copy(order="C")
    elif hasattr(X_partition, "get"):
        print(f"      [partition] data from other worker, copying via host to {this_worker}", flush=True)
        X_partition = np.asarray(X_partition.get(), dtype=np.float32, order="C")
        X = cp.asarray(X_partition, dtype=cp.float32)
    else:
        print(f"      [partition] numpy: copying to this worker's GPU", flush=True)
        if isinstance(X_partition, np.ndarray) and X_partition.dtype == np.dtype("object"):
            X_partition = np.asarray(X_partition, dtype=np.float32, order="C")
        else:
            X_partition = np.asarray(X_partition, dtype=np.float32, order="C")
        X = cp.asarray(X_partition, dtype=cp.float32)
    cents = cp.asarray(centroids)
    params = KMeansParams(n_clusters=n_clusters, metric=metric)

    k, d = cents.shape
    dtype = X.dtype
    n_rows = X.shape[0]
    partial_sums = cp.zeros((k, d), dtype=dtype)
    counts = cp.zeros(k, dtype=cp.int64)
    total_inertia = 0.0

    if n_rows <= MAX_PREDICT_ROWS:
        labels, inertia = predict(params, X, cents)
        total_inertia = float(inertia)
        labels_cp = cp.asarray(labels)
        for c in range(k):
            mask = labels_cp == c
            cnt = int(mask.sum())
            counts[c] = cnt
            if cnt > 0:
                partial_sums[c] = X.T @ mask.astype(cp.float32)
    else:
        # Chunk predict to stay under cuVS int32 indexing (n_rows * n_features <= INT32_MAX).
        for start in range(0, n_rows, MAX_PREDICT_ROWS):
            end = min(start + MAX_PREDICT_ROWS, n_rows)
            X_slice = X[start:end]
            labels_slice, inertia_slice = predict(params, X_slice, cents)
            total_inertia += float(inertia_slice)
            labels_cp = cp.asarray(labels_slice)
            for c in range(k):
                mask = labels_cp == c
                cnt = int(mask.sum())
                counts[c] = counts[c] + cnt
                if cnt > 0:
                    partial_sums[c] = partial_sums[c] + (X_slice.T @ mask.astype(cp.float32))

    return (
        cp.asnumpy(partial_sums),
        cp.asnumpy(counts),
        total_inertia,
    )


def fit_dask(
    params: KMeansParams,
    X: "da.Array",
    client: Optional["Client"] = None,
    init_centroids: Optional["np.ndarray"] = None,
    random_state: Optional[int] = None,
) -> FitDaskOutput:
    """
    Multi-GPU KMeans fit using Dask. Partitions the dataset across workers
    (GPUs) and runs iterative Lloyd's algorithm with map-reduce.

    Use with dask-cuda.LocalCUDACluster for one-worker-per-GPU:

        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        cluster = LocalCUDACluster()
        client = Client(cluster)
        ...
        centroids, inertia, n_iter = fit_dask(params, X_dask, client=client)

    Parameters
    ----------
    params : KMeansParams
        KMeans parameters (n_clusters, metric, max_iter, tol, etc.).
    X : dask.array.Array
        Input data. Chunks should map to workers (typically CuPy-backed for
        GPU execution). Use e.g. dask.array.from_array(cp_array, chunks=(rows, cols)).
    client : dask.distributed.Client, optional
        Dask distributed client. If None, uses the default client.
    init_centroids : np.ndarray, optional
        Initial centroids (n_clusters, n_features). If None, uses a sample-based
        init (fit on first partition).
    random_state : int, optional
        Seed for reproducibility when sampling for init.

    Returns
    -------
    FitDaskOutput
        centroids, inertia, n_iter
    """
    from dask.distributed import default_client

    if client is None:
        try:
            client = default_client()
        except ValueError:
            raise ValueError(
                "No Dask client found. Create one with LocalCUDACluster + Client, "
                "or pass client= explicitly."
            )

    n_clusters = params.n_clusters
    max_iter = params.max_iter
    tol = params.tol
    n_features = X.shape[1]

    # Infer dtype from array meta
    dtype = getattr(X._meta, "dtype", np.float32)

    # Initialize centroids
    if init_centroids is not None:
        centroids = np.asarray(init_centroids, dtype=dtype)
        if centroids.shape != (n_clusters, n_features):
            raise ValueError(
                f"init_centroids must have shape ({n_clusters}, {n_features}), "
                f"got {centroids.shape}"
            )
    else:
        # Sample from first partition(s) and run single-GPU fit for init
        n_sample = min(1000 * n_clusters, int(X.shape[0]))
        sample = X[:n_sample].compute()
        if hasattr(sample, "get"):
            sample = np.asarray(sample.get())
        else:
            sample = np.asarray(sample)
        if sample.dtype not in (np.float32, np.float64):
            sample = sample.astype(np.float32)
        cp = __import__("cupy", fromlist=[""])
        X_sample = cp.asarray(sample)
        centroids_out, _, _ = fit(params, X_sample)
        centroids = np.asarray(cp.asnumpy(centroids_out))

    # Persist so partitions stay on workers. Then compute block futures so we get
    # keys the scheduler actually has; who_has(future.key) then returns workers.
    # (Persisted array's __dask_keys__() can differ from scheduler keys.)
    from dask.delayed import delayed
    from dask.distributed import as_completed, wait as dask_wait

    X_persisted = client.persist(X)
    dask_wait(X_persisted)
    blocks = list(X_persisted.to_delayed().flatten())
    block_futures = [client.compute(blocks[i]) for i in range(len(blocks))]
    dask_wait(block_futures)
    block_keys = [f.key for f in block_futures]
    # who_has(keys) can be empty after wait(); build key->workers from has_what() instead.
    has_what = client.has_what()
    who_has = {}
    for worker, keys in has_what.items():
        for k in keys:
            who_has.setdefault(k, set()).add(worker)

    prev_inertia = float("inf")
    metric_name = params.metric if hasattr(params, "metric") else "L2Expanded"

    n_blocks = len(blocks)
    for iteration in range(max_iter):
        hint = " (blocks ready on workers)" if iteration == 0 else ""
        print(f"  KMeans iter {iteration + 1}/{max_iter} ({n_blocks} partitions){hint}...", flush=True)

        worker_addrs = [
            next(iter(who_has.get(block_keys[i], set()))) if who_has.get(block_keys[i]) else None
            for i in range(n_blocks)
        ]
        if iteration == 0:
            n_matched = sum(1 for a in worker_addrs if a is not None)
            if n_matched < n_blocks:
                print(
                    f"    [fit_dask] who_has: {n_matched}/{n_blocks} blocks have worker. "
                    f"worker_addrs={worker_addrs!r}, block_keys_sample={block_keys[:2]!r}",
                    flush=True,
                )
            else:
                print(f"    [fit_dask] who_has: all {n_blocks} blocks have worker (pinning tasks)", flush=True)
        fit_delayed = [
            delayed(_partition_fit)(
                blocks[i], centroids, n_clusters, metric_name,
                expected_worker=worker_addrs[i],
            )
            for i in range(n_blocks)
        ]
        # Pin each task to the worker that has its block so data stays local
        futures = [
            client.compute(fit_delayed[i], workers=[worker_addrs[i]] if worker_addrs[i] else None)
            for i in range(n_blocks)
        ]
        future_to_idx = {f: i for i, f in enumerate(futures)}
        results = [None] * n_blocks
        completed = 0
        print(f"    Submitted {n_blocks} partitions, waiting for results...", flush=True)
        for future in as_completed(futures):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            print(f"    Partition {completed}/{n_blocks}...", flush=True)
            sys.stdout.flush()

        # Use float64 for aggregation to match RAFT's reduce_rows_by_key accuracy.
        # Summing float32 across partitions is non-associative and can diverge.
        total_sums = np.zeros((n_clusters, n_features), dtype=np.float64)
        total_counts = np.zeros(n_clusters, dtype=np.int64)
        inertia = 0.0

        for psum, cnt, inc in results:
            total_sums += np.asarray(psum, dtype=np.float64)
            total_counts += cnt
            inertia += inc

        # Compute new centroids (matches RAFT update_centroids: keep previous if empty)
        new_centroids = np.empty_like(centroids)
        for c in range(n_clusters):
            if total_counts[c] > 0:
                new_centroids[c] = (total_sums[c] / total_counts[c]).astype(dtype)
            else:
                new_centroids[c] = centroids[c]

        # Centroid movement (RAFT uses sqrdNormError < tol as convergence criterion)
        sqrd_norm = float(np.sum((centroids.astype(np.float64) - new_centroids.astype(np.float64)) ** 2))
        centroids[:] = new_centroids

        # Convergence: match RAFT's dual check (inertia change + centroid movement)
        if tol is not None and prev_inertia != float("inf"):
            delta = abs(prev_inertia - inertia)
            if delta < tol * prev_inertia:
                return FitDaskOutput(centroids, float(inertia), iteration + 1)
            if sqrd_norm < tol:
                return FitDaskOutput(centroids, float(inertia), iteration + 1)
        prev_inertia = inertia

    return FitDaskOutput(centroids, float(prev_inertia), max_iter)
