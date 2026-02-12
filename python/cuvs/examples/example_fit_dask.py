#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Example: Multi-GPU KMeans with Dask.

Uses dask-cuda.LocalCUDACluster to create one worker per GPU, partitions
a large dataset across workers, and runs iterative Lloyd's algorithm with
map-reduce. Each partition runs cuVS predict + local reduction; results
are aggregated to update centroids.

Install optional deps first:
    pip install cuvs[dask]

Then run (adjust chunks/workers as needed):
    python python/cuvs/examples/example_fit_dask.py
"""
import os
import sys
import subprocess

# Re-launch from /tmp to avoid loading cuvs from source tree (editable install / PYTHONPATH)
if os.environ.get("CUVS_EXAMPLE_RUNNING") != "1":
    _script = os.path.abspath(__file__)
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if os.getcwd().startswith(_repo) or _repo in os.environ.get("PYTHONPATH", ""):
        env = os.environ.copy()
        env["CUVS_EXAMPLE_RUNNING"] = "1"
        env.pop("PYTHONPATH", None)  # clear PYTHONPATH if it pointed at repo
        sys.exit(subprocess.call([sys.executable, _script], cwd="/tmp", env=env))

import cupy as cp
import dask.array as da
from dask.distributed import Client

try:
    from cuvs.cluster.kmeans import KMeansParams, fit_dask
except ModuleNotFoundError as e:
    if "cydlpack" in str(e):
        print(
            "Error: cuvs loaded from source tree. Run from outside repo:\n"
            "  cd /tmp && python",
            os.path.abspath(__file__),
        )
    raise

try:
    from dask_cuda import LocalCUDACluster
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False


def main():
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

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
