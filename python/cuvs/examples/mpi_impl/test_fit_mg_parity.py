#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Verify fit_mg and fit_mg_sharded produce the same centroids (fit_mg delegates to fit_mg_sharded).
# Run: mpirun -np 4 python test_fit_mg_parity.py

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit_mg, fit_mg_sharded


def main():
    try:
        from mpi4py import MPI
    except ImportError:
        print("test_fit_mg_parity.py requires mpi4py. pip install mpi4py")
        return 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_samples = 1000
    n_features = 32
    n_clusters = 5
    seed = 42
    dtype = np.float32

    # Sharding (same as fit_mg / fit_mg_sharded)
    n_local_base = n_samples // size
    row_start = rank * n_local_base
    row_end = n_samples if rank == size - 1 else row_start + n_local_base
    n_local = row_end - row_start

    np.random.seed(seed)
    X_full = np.random.randn(n_samples, n_features).astype(dtype)
    X_local = X_full[row_start:row_end].copy()

    params = KMeansParams(n_clusters=n_clusters, n_init=2)

    # fit_mg: each rank has full X
    centroids_mg, inertia_mg, n_iter_mg = fit_mg(
        params, X_full, rank=rank, size=size
    )

    # fit_mg_sharded: each rank has only its shard
    centroids_sharded, inertia_sharded, n_iter_sharded = fit_mg_sharded(
        params, X_local, rank=rank, size=size
    )

    # Compare centroids (rank 0 prints)
    c_mg = centroids_mg.copy_to_host()
    c_sh = centroids_sharded.copy_to_host()
    match = np.allclose(c_mg, c_sh, rtol=1e-5, atol=1e-5)

    if rank == 0:
        print(
            f"fit_mg n_iter={n_iter_mg} inertia={inertia_mg:.6f}"
        )
        print(
            f"fit_mg_sharded n_iter={n_iter_sharded} inertia={inertia_sharded:.6f}"
        )
        print(f"Centroids match: {match}")
        if not match:
            print(f"  max diff: {np.abs(c_mg - c_sh).max()}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
