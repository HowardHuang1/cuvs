#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/cluster/kmeans.h" nogil:
    ctypedef enum cuvsKMeansInitMethod:
        KMeansPlusPlus
        Random
        Array

    ctypedef enum cuvsKMeansType:
        CUVS_KMEANS_TYPE_KMEANS
        CUVS_KMEANS_TYPE_KMEANS_BALANCED

    ctypedef struct cuvsKMeansParams:
        cuvsDistanceType metric,
        int n_clusters,
        cuvsKMeansInitMethod init,
        int max_iter,
        double tol,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check,
        bool hierarchical,
        int hierarchical_n_iters

    ctypedef cuvsKMeansParams* cuvsKMeansParams_t

    cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* index)

    cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t index)

    cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensor* X,
                              DLManagedTensor* sample_weight,
                              DLManagedTensor * centroids,
                              double * inertia,
                              int * n_iter) except +

    cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                                  cuvsKMeansParams_t params,
                                  DLManagedTensor* X,
                                  DLManagedTensor* sample_weight,
                                  DLManagedTensor * centroids,
                                  DLManagedTensor * labels,
                                  bool normalize_weight,
                                  double * inertia)

    cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                      DLManagedTensor* X,
                                      DLManagedTensor* centroids,
                                      double* cost)

    cuvsError_t cuvsKMeansFitFromHostMG(cuvsResources_t res,
                                        cuvsKMeansParams_t params,
                                        const void* X_host,
                                        int64_t n_rows,
                                        int64_t n_cols,
                                        int is_float64,
                                        int rank,
                                        int size,
                                        DLManagedTensor* centroids,
                                        double* inertia,
                                        int* n_iter) except +

# Inject cuvsKMeansFitFromHostMGSharded declaration via verbatim C.
# Conda's kmeans.h may not have this symbol; inline declaration avoids header dependency.
cdef extern from * nogil:
    """
    #ifdef __cplusplus
    extern "C" {
    #endif
    cuvsError_t cuvsKMeansFitFromHostMGSharded(cuvsResources_t res,
                                              cuvsKMeansParams_t params,
                                              const void* X_local_host,
                                              int64_t n_local,
                                              int64_t n_cols,
                                              int is_float64,
                                              int rank,
                                              int size,
                                              DLManagedTensor* centroids,
                                              double* inertia,
                                              int* n_iter);
    #ifdef __cplusplus
    }
    #endif
    """
    cuvsError_t cuvsKMeansFitFromHostMGSharded(cuvsResources_t res,
                                              cuvsKMeansParams_t params,
                                              const void* X_local_host,
                                              int64_t n_local,
                                              int64_t n_cols,
                                              int is_float64,
                                              int rank,
                                              int size,
                                              DLManagedTensor* centroids,
                                              double* inertia,
                                              int* n_iter) except +
