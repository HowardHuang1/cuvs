/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: brute force vs CAGRA-based cluster assignment for IVF training.
 * Compares time to assign N vectors to K clusters (nearest centroid) using
 * (1) brute force 1-NN and (2) CAGRA build on centroids + k=1 search.
 */
#include <benchmark/benchmark.h>

// kmeans_balanced.cuh is under cpp/src/; CUVS_CLUSTER_ASSIGNMENT_BENCH adds that to include path
#include <cluster/kmeans_balanced.cuh>
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>

namespace {

using namespace cuvs::cluster::kmeans_balanced;

void init_random_data(raft::resources const& handle,
                      float* X,
                      int64_t n_rows,
                      int64_t dim,
                      float* centroids,
                      int64_t n_clusters)
{
  raft::random::RngState rng(12345ULL);
  raft::random::uniform(handle, rng, X, n_rows * dim, float(-1), float(1));
  raft::random::uniform(handle, rng, centroids, n_clusters * dim, float(-1), float(1));
  raft::resource::sync_stream(handle);
}

}  // namespace

static void BM_ClusterAssignment_BruteForce(benchmark::State& state)
{
  int64_t n_rows     = static_cast<int64_t>(state.range(0));
  int64_t n_clusters = static_cast<int64_t>(state.range(1));
  int64_t dim        = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> X(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                               raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<float> centroids(static_cast<size_t>(n_clusters) * static_cast<size_t>(dim),
                                       raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<uint32_t> labels(static_cast<size_t>(n_rows),
                                       raft::resource::get_cuda_stream(handle));

  init_random_data(handle, X.data(), n_rows, dim, centroids.data(), n_clusters);

  cuvs::cluster::kmeans::balanced_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;

  auto X_view = raft::make_device_matrix_view<const float, int64_t>(X.data(), n_rows, dim);
  auto centers_view =
    raft::make_device_matrix_view<const float, int64_t>(centroids.data(), n_clusters, dim);
  auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), n_rows);

  for (auto _ : state) {
    predict(handle, params, X_view, centers_view, labels_view);
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows);
}

static void BM_ClusterAssignment_CAGRA(benchmark::State& state)
{
  int64_t n_rows     = static_cast<int64_t>(state.range(0));
  int64_t n_clusters = static_cast<int64_t>(state.range(1));
  int64_t dim        = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> X(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                               raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<float> centroids(static_cast<size_t>(n_clusters) * static_cast<size_t>(dim),
                                       raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<uint32_t> labels(static_cast<size_t>(n_rows),
                                       raft::resource::get_cuda_stream(handle));

  init_random_data(handle, X.data(), n_rows, dim, centroids.data(), n_clusters);

  cuvs::cluster::kmeans::balanced_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;

  // Same timing as assign_nearest_centroid_cagra_with_index_reuse with rebuild=true each iteration.
  // float X/centroids only.
  std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_index_opt;

  for (auto _ : state) {
    cuvs::cluster::kmeans::detail::assign_nearest_centroid_cagra_with_index_reuse<int64_t,
                                                                                  uint32_t>(
      handle,
      params,
      centroids.data(),
      n_clusters,
      dim,
      X.data(),
      n_rows,
      labels.data(),
      &cagra_index_opt,
      true);
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows);
}

// Controlled sweep: hold points-per-cluster fixed at kPointsPerCluster (N = kPointsPerCluster * K)
// so K is the only independent variable. This isolates the effect of K on the brute-force-vs-CAGRA
// crossover; letting both N and K vary independently (as the old hand-picked Args list did)
// confounds the two and makes the crossover point ill-defined.
constexpr int64_t kPointsPerCluster = 5;
constexpr int64_t kDim              = 128;

// clang-format off
constexpr int64_t kClusterCounts[] = {
  1000, 2000, 4000, 8000, 16000, 32000, 65536, 131072, 262144, 500000, 1000000
};
// clang-format on

static void RegisterConstantRatioSweep()
{
  for (int64_t k : kClusterCounts) {
    int64_t n = kPointsPerCluster * k;
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({n, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({n, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Second controlled sweep: hold N fixed and vary K alone (same kClusterCounts list, so the two
// sweeps are directly comparable). This isolates the effect of K on the crossover from the effect
// of the N/K ratio tested above -- it answers "at a fixed dataset size, does ANN help more as the
// number of clusters grows?" rather than "does ANN help more as both grow together?".
// kFixedN is chosen to not collide with any kPointsPerCluster * K value from the sweep above.
constexpr int64_t kFixedN = 2000000;

static void RegisterFixedNVaryKSweep()
{
  for (int64_t k : kClusterCounts) {
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({kFixedN, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({kFixedN, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

int main(int argc, char** argv)
{
  RegisterConstantRatioSweep();
  RegisterFixedNVaryKSweep();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
