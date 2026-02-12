# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


from .kmeans import KMeansParams, cluster_cost, fit, fit_mg, fit_mg_sharded, predict

# Dask multi-GPU fit (optional; requires dask, distributed, dask-cuda)
def __getattr__(name: str):
    if name == "fit_dask":
        from .dask_kmeans import fit_dask
        return fit_dask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KMeansParams",
    "cluster_cost",
    "fit",
    "fit_dask",
    "fit_mg",
    "fit_mg_sharded",
    "predict",
]
