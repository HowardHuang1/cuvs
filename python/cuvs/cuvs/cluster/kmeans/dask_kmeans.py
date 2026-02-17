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

from collections import namedtuple
from typing import TYPE_CHECKING, Optional

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit, predict

if TYPE_CHECKING:
    from dask.distributed import Client

    import dask.array as da

FitDaskOutput = namedtuple("FitDaskOutput", "centroids inertia n_iter")


def _partition_fit(
    X_partition,
    centroids: np.ndarray,
    n_clusters: int,
    metric: str = "L2Expanded",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run predict on a partition, then compute partial sums and counts.
    Runs on a Dask worker (with GPU). X_partition should be CuPy when on a
    CUDA worker; centroids is numpy (small, broadcast to all workers).
    """
    import cupy as cp
    from cuvs.cluster.kmeans import KMeansParams, predict

    # Avoid touching Delayed objects; only convert object-dtype numpy arrays
    if isinstance(X_partition, np.ndarray) and X_partition.dtype == np.dtype("object"):
        X_partition = np.asarray(X_partition, dtype=np.float32, order="C")
    X = cp.asarray(X_partition, dtype=cp.float32)
    cents = cp.asarray(centroids)
    params = KMeansParams(n_clusters=n_clusters, metric=metric)

    labels, inertia = predict(params, X, cents)
    labels_cp = cp.asarray(labels)
    k, d = cents.shape
    dtype = X.dtype

    partial_sums = cp.zeros((k, d), dtype=dtype)
    counts = cp.zeros(k, dtype=cp.int64)

    for c in range(k):
        mask = labels_cp == c
        cnt = int(mask.sum())
        counts[c] = cnt
        if cnt > 0:
            # X.T @ mask avoids copy and instead does it in place which OOMs for large clusters
            partial_sums[c] = (X.T @ mask.astype(cp.float32))

    return (
        cp.asnumpy(partial_sums),
        cp.asnumpy(counts),
        float(inertia),
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

    # Persist so partitions stay on workers
    X_persisted = client.persist(X)
    blocks = list(X_persisted.to_delayed().flatten())

    prev_inertia = float("inf")
    metric_name = params.metric if hasattr(params, "metric") else "L2Expanded"

    n_blocks = len(blocks)
    for iteration in range(max_iter):
        # Use delayed() so _partition_fit receives computed array, not Delayed
        from dask.delayed import delayed

        hint = " (blocks computed on-demand)" if iteration == 0 else ""
        print(f"  KMeans iter {iteration + 1}/{max_iter} ({n_blocks} partitions){hint}...", flush=True)

        fit_delayed = [
            delayed(_partition_fit)(block, centroids, n_clusters, metric_name)
            for block in blocks
        ]
        results = client.compute(fit_delayed, sync=True)

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
