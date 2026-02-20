#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Verify fit_dask correctness by comparing with single-GPU fit on the same data.

Both use identical init centroids and Lloyd's algorithm. Results should match
within tolerance when deterministic GPU settings are used (see below).

Run with a Dask cluster already available (e.g. from another terminal), or this
script will create one:

    python python/cuvs/examples/verify_fit_dask.py

With existing cluster:
    from dask.distributed import default_client
    # ... after creating cluster in another process
    python -c "exec(open('verify_fit_dask.py').read()); verify(default_client())"

Note: Single-GPU (RAFT) uses atomics and block reductions, so its reduction order
can vary run-to-run. We set CUBLAS_WORKSPACE_CONFIG before importing CUDA libs to
improve cuBLAS determinism; some run-to-run variation may still occur.
"""
import os

# Set before any CUDA/cuBLAS use so cuBLAS picks deterministic algorithms where available.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit, fit_dask


def verify(client=None):
    """Compare fit_dask vs single-GPU fit on same small data with same init."""
    n_samples = 20_000
    n_features = 64
    n_clusters = 5
    seed = 42

    np.random.seed(seed)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32) * 0.5 + 0.5

    # Use very small tol so both run all max_iter (cuVS rejects tol<=0)
    max_iter = 20
    tol_tiny = 1e-10
    params = KMeansParams(n_clusters=n_clusters, max_iter=max_iter, tol=tol_tiny)

    # Get init centroids (same for both): run fit on sample, use result as init
    import cupy as cp
    n_init = min(1000 * n_clusters, n_samples)
    X_sample = cp.asarray(X_np[:n_init])
    init_out, _, _ = fit(KMeansParams(n_clusters=n_clusters, max_iter=10), X_sample)
    init_centroids = np.asarray(cp.asnumpy(init_out))

    # Single-GPU fit (full data) with same init
    params_array = KMeansParams(
        n_clusters=n_clusters, max_iter=max_iter, tol=tol_tiny, init_method="Array"
    )
    X_gpu = cp.asarray(X_np)
    centroids_single, inertia_single, n_iter_single = fit(
        params_array, X_gpu, centroids=cp.asarray(init_centroids.copy())
    )
    centroids_single = np.asarray(cp.asnumpy(centroids_single))

    # fit_dask (same data, same init)
    import dask.array as da
    X_dask = da.from_array(X_np, chunks=(5000, n_features))
    centroids_dask, inertia_dask, n_iter_dask = fit_dask(
        params, X_dask, client=client, init_centroids=init_centroids
    )

    # Compare (centroid order is deterministic with same init)
    rtol, atol = 1e-4, 1e-4
    match = np.allclose(centroids_single, centroids_dask, rtol=rtol, atol=atol)
    max_diff = np.abs(centroids_single - centroids_dask).max()

    print("=== fit_dask verification ===")
    print(f"  Single-GPU: n_iter={n_iter_single}, inertia={inertia_single:.6f}")
    print(f"  fit_dask:   n_iter={n_iter_dask}, inertia={inertia_dask:.6f}")
    print(f"  Centroids match (rtol={rtol}, atol={atol}): {match}")
    if not match:
        print(f"  max |diff|: {max_diff:.2e}")
        return False
    print("  OK")
    return True


def main():
    try:
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
    except ImportError:
        print("Need dask-cuda: pip install dask-cuda")
        return 1

    cluster = LocalCUDACluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    try:
        ok = verify(client)
    finally:
        client.close()
        cluster.close()
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
