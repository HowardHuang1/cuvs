#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Sample script: OOM bisect for multi-GPU KMeans fit_mg_sharded.
#
# Uses fit_mg_sharded: each rank generates only its row shard (avoids host OOM).
# Matches cpp/tests/cluster/kmeans_mg.cu OOM_Bisect (n_features=1024, sharded).
#
# Run with mpirun, e.g.:
#   mpirun -np 4 python oom_bisect_fit_mg.py 100000
#   mpirun -np 4 python oom_bisect_fit_mg.py --bisect 1000000 40000000
#   mpirun -np 4 python oom_bisect_fit_mg.py --bisect 1000000 40000000 --mem  # print GPU memory
#
# Single trial: try one size, exit 0 on success and 1 on OOM (for shell bisect).
# Bisect mode: double from low until OOM, then binary search for max n_samples.
# Note: On OOM one rank may raise while others are in collectives; bisect mode
# can hang. For reliable bisect, use single-trial and a shell loop.
#
# GPU memory may not fully return between runs (CUDA/RMM caching allocator).
# The script runs gc.collect() and cudaDeviceSynchronize() between trials to
# reclaim what it can.
#
# Speed vs C++: Python generates data on host (NumPy) and copies to device each run;
# the C++ test generates directly on GPU, so it runs faster.

import argparse
import gc
import sys

import numpy as np

from cuvs.cluster.kmeans import KMeansParams, fit_mg_sharded
from cuvs.common.exceptions import CuvsException


def _reclaim_gpu_memory(rank: int, size: int) -> None:
    """Best-effort reclaim: GC + sync. CUDA caching allocator may still hold memory."""
    gc.collect()
    try:
        import ctypes
        cudart = ctypes.CDLL("libcudart.so")
        count = ctypes.c_int()
        if cudart.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1:
            dev = 0
        else:
            dev = rank
        cudart.cudaSetDevice(dev)
        cudart.cudaDeviceSynchronize()
    except Exception:
        pass


def _get_gpu_mem_str(rank: int, size: int, comm) -> str:
    """Return GPU memory free/total using cudaMemGetInfo (CUDA runtime view).
    Each rank queries its own GPU (the one it uses) and results are gathered to rank 0.
    cudaMemGetInfo is more accurate than NVML for 'available for allocation' with RMM/caching allocator.
    All ranks must call this (collective).
    """
    try:
        import ctypes
        from ctypes import byref, c_size_t, c_int, POINTER

        cudart = ctypes.CDLL("libcudart.so")
        cudaMemGetInfo = cudart.cudaMemGetInfo
        cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
        cudaMemGetInfo.restype = c_int

        count = c_int()
        if cudart.cudaGetDeviceCount(byref(count)) != 0:
            return "(cudaGetDeviceCount failed)"
        dev = 0 if count.value == 1 else rank
        if cudart.cudaSetDevice(dev) != 0:
            return "(cudaSetDevice failed)"
        if cudart.cudaDeviceSynchronize() != 0:
            pass  # best-effort sync

        free = c_size_t()
        total = c_size_t()
        if cudaMemGetInfo(byref(free), byref(total)) != 0:
            return "(cudaMemGetInfo failed)"

        free_mb = free.value // (2**20)
        total_mb = total.value // (2**20)
        # When each rank sees 1 GPU (CUDA_VISIBLE_DEVICES), label by rank; else by device index
        my_str = f"rank{rank}={free_mb}/{total_mb}MB" if count.value == 1 else f"GPU{dev}={free_mb}/{total_mb}MB"

        if size <= 1:
            return my_str
        # Gather from all ranks (each reports its GPU)
        gathered = comm.gather(my_str, root=0)
        if rank == 0:
            return " ".join(gathered)
        return ""
    except Exception as e:
        return f"(GPU mem query failed: {e})"


def main():
    parser = argparse.ArgumentParser(
        description="OOM bisect for multi-GPU KMeans fit_mg (run with mpirun)."
    )
    parser.add_argument(
        "n_samples",
        type=int,
        nargs="?",
        help="Number of samples (rows) to try. Required for single-trial mode.",
    )
    parser.add_argument(
        "--bisect",
        nargs=2,
        metavar=("LOW", "HIGH"),
        type=int,
        help="Bisect mode: find max n_samples in [LOW, HIGH] that does not OOM.",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=1024,
        help="Number of features (default: 1024, matches C++ OOM bisect test).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=100,
        help="Number of clusters (default: 100).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Data type (default: float32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data (default: 42).",
    )
    parser.add_argument(
        "--mem",
        action="store_true",
        help="Print GPU memory: Python prints 'before' (pre-alloc); C++ prints 'WITH dataset' "
        "(after alloc) and 'after fit (before freeing)' (before dealloc).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-rank row counts (n_local) for each trial.",
    )
    args = parser.parse_args()

    try:
        from mpi4py import MPI
    except ImportError:
        print("oom_bisect_fit_mg.py requires mpi4py. Install with: pip install mpi4py", file=sys.stderr)
        sys.exit(2)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    def run_one(n_samples: int) -> bool:
        """Run fit_mg_sharded with n_samples. Returns True if OK, False if OOM."""
        n_local_base = n_samples // size
        row_start = rank * n_local_base
        row_end = n_samples if rank == size - 1 else row_start + n_local_base
        n_local = row_end - row_start
        if args.debug:
            for r in range(size):
                if r == rank:
                    print(
                        f"[rank {rank}/{size}] n_samples={n_samples} -> "
                        f"n_local={n_local} (rows {row_start}:{row_end})",
                        flush=True,
                    )
                comm.Barrier()
        np.random.seed(args.seed + rank)
        X_local = np.random.randn(n_local, args.n_features).astype(dtype)
        params = KMeansParams(n_clusters=args.n_clusters)
        try:
            centroids, inertia, n_iter = fit_mg_sharded(params, X_local, rank=rank, size=size)
            return True
        except CuvsException as e:
            err = str(e).lower()
            if "memory" in err or "out of memory" in err or "oom" in err:
                return False
            raise
        except Exception as e:
            err = str(e).lower()
            if "memory" in err or "out of memory" in err or "cuda" in err:
                return False
            raise

    if args.bisect:
        low, high = args.bisect[0], args.bisect[1]
        if low < 1 or high < low:
            if rank == 0:
                print("Invalid --bisect LOW HIGH", file=sys.stderr)
            sys.exit(2)
        # Double from low until we OOM to get an upper bound
        last_ok = low
        trial = low
        while trial <= high:
            if rank == 0:
                print(f"Trying n_samples={trial} ...", flush=True)
            if args.mem:
                mem_str = _get_gpu_mem_str(rank, size, comm)
                if rank == 0:
                    print(f"  before: {mem_str}", flush=True)
            ok = run_one(trial)
            # "after" (before freeing) is printed from C++ fit_mg_sharded
            if rank == 0:
                print(f"  -> {'OK' if ok else 'OOM'}", flush=True)
            if not ok:
                high = trial - 1
                break
            _reclaim_gpu_memory(rank, size)
            last_ok = trial
            trial = min(trial * 2, high)
        else:
            if rank == 0:
                print(f"Bisect result: max n_samples = {last_ok} (all up to {high} passed)")
            sys.exit(0)
        # Binary search in [last_ok, high]
        while last_ok < high:
            mid = (last_ok + high + 1) // 2
            if rank == 0:
                print(f"Trying n_samples={mid} ...", flush=True)
            if args.mem:
                mem_str = _get_gpu_mem_str(rank, size, comm)
                if rank == 0:
                    print(f"  before: {mem_str}", flush=True)
            ok = run_one(mid)
            # "after" (before freeing) is printed from C++ fit_mg_sharded
            if rank == 0:
                print(f"  -> {'OK' if ok else 'OOM'}", flush=True)
            if ok:
                last_ok = mid
            else:
                high = mid - 1
            _reclaim_gpu_memory(rank, size)
        if rank == 0:
            print(f"Bisect result: max n_samples = {last_ok}")
        sys.exit(0)

    if args.n_samples is None:
        parser.error("n_samples required when not using --bisect")
    n_samples = args.n_samples
    if rank == 0:
        print(f"Trying n_samples={n_samples} (n_features={args.n_features}, n_clusters={args.n_clusters}) ...", flush=True)
    if args.mem:
        mem_str = _get_gpu_mem_str(rank, size, comm)
        if rank == 0:
            print(f"  before: {mem_str}", flush=True)
    ok = run_one(n_samples)
    # "after" (before freeing) is printed from C++ fit_mg_sharded
    if rank == 0:
        print("OK" if ok else "OOM", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
