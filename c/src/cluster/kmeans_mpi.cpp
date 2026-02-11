/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-GPU KMeans fit from host memory (MPI). Built only when MPI is found.
 */

#ifdef CUVS_HAVE_MPI

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <nccl.h>

#include <cuvs/cluster/kmeans.h>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/c_api.h>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include <raft/comms/std_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

namespace {

cuvs::cluster::kmeans::params convert_params(const cuvsKMeansParams& params)
{
  auto kmeans_params       = cuvs::cluster::kmeans::params();
  kmeans_params.metric     = static_cast<cuvs::distance::DistanceType>(params.metric);
  kmeans_params.init       = static_cast<cuvs::cluster::kmeans::params::InitMethod>(params.init);
  kmeans_params.n_clusters = params.n_clusters;
  kmeans_params.max_iter   = params.max_iter;
  kmeans_params.tol        = params.tol;
  kmeans_params.oversampling_factor = params.oversampling_factor;
  kmeans_params.batch_samples       = params.batch_samples;
  kmeans_params.batch_centroids     = params.batch_centroids;
  kmeans_params.inertia_check       = params.inertia_check;
  return kmeans_params;
}

#define NCCLCHECK(cmd)                                                                        \
  do {                                                                                        \
    ncclResult_t res = cmd;                                                                   \
    if (res != ncclSuccess) { RAFT_FAIL("NCCL error: %s", ncclGetErrorString(res)); }         \
  } while (0)

template <typename T>
cuvsError_t fit_from_host_mg_sharded_impl(cuvsKMeansParams_t params,
                                          const void* X_local_host,
                                          int64_t n_local,
                                          int64_t n_cols,
                                          int rank,
                                          int size,
                                          DLManagedTensor* centroids_tensor,
                                          double* inertia,
                                          int* n_iter)
{
  const int n_clusters = params->n_clusters;

  // Debug: fit_mg_sharded receives already-sharded data (each rank has only its rows).
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r == rank) {
      std::fprintf(stderr,
                   "[fit_mg_sharded] rank %d/%d: received SHARDED input. "
                   "This rank has n_local=%ld rows (n_cols=%ld). No sharding needed.\n",
                   rank,
                   size,
                   static_cast<long>(n_local),
                   static_cast<long>(n_cols));
      std::fflush(stderr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  ncclUniqueId id;
  if (rank == 0) { NCCLCHECK(ncclGetUniqueId(&id)); }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  int device_id   = (err == cudaSuccess && device_count == 1) ? 0 : rank;
  cudaSetDevice(device_id);
  ncclComm_t nccl_comm;
  NCCLCHECK(ncclCommInitRank(&nccl_comm, size, id, rank));

  raft::resources handle;
  raft::comms::build_comms_nccl_only(&handle, nccl_comm, size, rank);

  auto stream = raft::resource::get_cuda_stream(handle);

  const T* X_ptr = static_cast<const T*>(X_local_host);
  rmm::device_uvector<T> X_local(static_cast<size_t>(n_local * n_cols), stream);
  raft::copy(X_local.data(), X_ptr, n_local * n_cols, stream);

  rmm::device_uvector<T> d_centroids(static_cast<size_t>(n_clusters) * static_cast<size_t>(n_cols),
                                     stream);

  auto X_view = raft::make_device_matrix_view<const T, int64_t>(X_local.data(), n_local, n_cols);
  auto centroids_view =
    raft::make_device_matrix_view<T, int64_t>(d_centroids.data(), n_clusters, n_cols);

  T inertia_temp = 0;
  int64_t n_iter_temp = 0;

  auto kmeans_params = convert_params(*params);
  cuvs::cluster::kmeans::fit(handle,
                             kmeans_params,
                             X_view,
                             std::nullopt,
                             centroids_view,
                             raft::make_host_scalar_view<T>(&inertia_temp),
                             raft::make_host_scalar_view<int64_t>(&n_iter_temp));

  raft::resource::sync_stream(handle, stream);

  *inertia = static_cast<double>(inertia_temp);
  *n_iter  = static_cast<int>(n_iter_temp);

  auto centroids_dl = centroids_tensor->dl_tensor;
  if (centroids_dl.device.device_type == kDLCUDA && centroids_dl.dtype.code == kDLFloat &&
      static_cast<int>(centroids_dl.dtype.bits) == static_cast<int>(sizeof(T) * 8)) {
    raft::copy(static_cast<T*>(centroids_dl.data),
               d_centroids.data(),
               static_cast<size_t>(n_clusters) * static_cast<size_t>(n_cols),
               stream);
    raft::resource::sync_stream(handle, stream);
  }

  ncclCommDestroy(nccl_comm);
  return CUVS_SUCCESS;
}

// fit_mg delegates to fit_mg_sharded: compute this rank's row slice and call sharded impl.
template <typename T>
cuvsError_t fit_from_host_mg_impl(cuvsKMeansParams_t params,
                                  const void* X_host,
                                  int64_t n_rows,
                                  int64_t n_cols,
                                  int rank,
                                  int size,
                                  DLManagedTensor* centroids_tensor,
                                  double* inertia,
                                  int* n_iter)
{
  const int64_t n_local_base = n_rows / size;
  const int64_t row_start    = static_cast<int64_t>(rank) * n_local_base;
  const int64_t row_end      = (rank == size - 1) ? n_rows : (row_start + n_local_base);
  const int64_t n_local     = row_end - row_start;

  // Debug: (1) fit_mg received full dataset - one print for total size.
  if (rank == 0) {
    std::fprintf(stderr,
                 "[fit_mg] received FULL dataset: n_rows=%ld, n_cols=%ld (total across all ranks)\n",
                 static_cast<long>(n_rows),
                 static_cast<long>(n_cols));
    std::fflush(stderr);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Debug: (2) after sharding, each rank's shard size right before passing to fit_mg_sharded.
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r == rank) {
      std::fprintf(stderr,
                   "[fit_mg] rank %d/%d: after sharding -> n_local=%ld rows (rows %ld:%ld). "
                   "Passing shard to fit_mg_sharded.\n",
                   rank,
                   size,
                   static_cast<long>(n_local),
                   static_cast<long>(row_start),
                   static_cast<long>(row_end));
      std::fflush(stderr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  const T* X_local_ptr = static_cast<const T*>(X_host) + row_start * n_cols;
  return fit_from_host_mg_sharded_impl<T>(
    params, X_local_ptr, n_local, n_cols, rank, size, centroids_tensor, inertia, n_iter);
}

}  // namespace

extern "C" cuvsError_t cuvsKMeansFitFromHostMG(cuvsResources_t res,
                                               cuvsKMeansParams_t params,
                                               const void* X_host,
                                               int64_t n_rows,
                                               int64_t n_cols,
                                               int is_float64,
                                               int rank,
                                               int size,
                                               DLManagedTensor* centroids,
                                               double* inertia,
                                               int* n_iter)
{
  (void)res;
  return cuvs::core::translate_exceptions([=] {
    if (params == nullptr || X_host == nullptr || centroids == nullptr || inertia == nullptr ||
        n_iter == nullptr) {
      RAFT_FAIL("cuvsKMeansFitFromHostMG: null argument");
    }
    if (size < 1 || rank < 0 || rank >= size) {
      RAFT_FAIL("cuvsKMeansFitFromHostMG: invalid rank=%d size=%d", rank, size);
    }
    if (n_rows < 1 || n_cols < 1) {
      RAFT_FAIL("cuvsKMeansFitFromHostMG: invalid n_rows=%ld n_cols=%ld",
                static_cast<long>(n_rows),
                static_cast<long>(n_cols));
    }
    if (is_float64) {
      fit_from_host_mg_impl<double>(
        params, X_host, n_rows, n_cols, rank, size, centroids, inertia, n_iter);
    } else {
      fit_from_host_mg_impl<float>(
        params, X_host, n_rows, n_cols, rank, size, centroids, inertia, n_iter);
    }
  });
}

extern "C" cuvsError_t cuvsKMeansFitFromHostMGSharded(cuvsResources_t res,
                                                     cuvsKMeansParams_t params,
                                                     const void* X_local_host,
                                                     int64_t n_local,
                                                     int64_t n_cols,
                                                     int is_float64,
                                                     int rank,
                                                     int size,
                                                     DLManagedTensor* centroids,
                                                     double* inertia,
                                                     int* n_iter)
{
  (void)res;
  return cuvs::core::translate_exceptions([=] {
    if (params == nullptr || X_local_host == nullptr || centroids == nullptr || inertia == nullptr ||
        n_iter == nullptr) {
      RAFT_FAIL("cuvsKMeansFitFromHostMGSharded: null argument");
    }
    if (size < 1 || rank < 0 || rank >= size) {
      RAFT_FAIL("cuvsKMeansFitFromHostMGSharded: invalid rank=%d size=%d", rank, size);
    }
    if (n_local < 1 || n_cols < 1) {
      RAFT_FAIL("cuvsKMeansFitFromHostMGSharded: invalid n_local=%ld n_cols=%ld",
                static_cast<long>(n_local),
                static_cast<long>(n_cols));
    }
    if (is_float64) {
      fit_from_host_mg_sharded_impl<double>(
        params, X_local_host, n_local, n_cols, rank, size, centroids, inertia, n_iter);
    } else {
      fit_from_host_mg_sharded_impl<float>(
        params, X_local_host, n_local, n_cols, rank, size, centroids, inertia, n_iter);
    }
  });
}

#endif  // CUVS_HAVE_MPI
