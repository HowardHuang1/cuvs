/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <optional>
#include <vector>

namespace {

void print_gpu_memory(const char* label)
{
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (e != cudaSuccess) {
    std::cerr << "[" << label << "] cudaMemGetInfo failed: " << cudaGetErrorString(e)
              << std::endl;
    return;
  }
  std::cerr << "[" << label << "] GPU memory: free=" << (free_bytes / (1024 * 1024)) << " MiB, total="
            << (total_bytes / (1024 * 1024)) << " MiB" << std::endl;
}

bool would_overflow_int32(int64_t n_samples, int64_t n_features)
{
  const int64_t prod = n_samples * n_features;
  return (prod > static_cast<int64_t>(INT_MAX)) || (prod < 0);
}

size_t size_X_bytes(int64_t n_samples, int64_t n_features, size_t sizeof_T)
{
  return static_cast<size_t>(n_samples) * static_cast<size_t>(n_features) * sizeof_T;
}

}  // namespace

namespace cuvs {

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

// template <typename DataT, typename IndexT>
// void run_cluster_cost(const raft::resources& handle,
//                       raft::device_vector_view<DataT, IndexT> minClusterDistance,
//                       rmm::device_uvector<char>& workspace,
//                       raft::device_scalar_view<DataT> clusterCost)
//{
//   cuvs::cluster::kmeans::cluster_cost(
//     handle, minClusterDistance, workspace, clusterCost, raft::add_op{});
// }

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle)),
      d_centroids(0, raft::resource::get_cuda_stream(handle)),
      d_sample_weight(0, raft::resource::get_cuda_stream(handle))
  {
  }

  //  void apiTest()
  //  {
  //    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
  //
  //    auto stream                = raft::resource::get_cuda_stream(handle);
  //    int n_samples              = testparams.n_row;
  //    int n_features             = testparams.n_col;
  //    params.n_clusters          = testparams.n_clusters;
  //    params.tol                 = testparams.tol;
  //    params.n_init              = 1;
  //    params.rng_state.seed      = 1;
  //    params.oversampling_factor = 0;
  //
  //    raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  //
  //    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
  //    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
  //
  //    raft::random::make_blobs<T, int>(X.data_handle(),
  //                                     labels.data_handle(),
  //                                     n_samples,
  //                                     n_features,
  //                                     params.n_clusters,
  //                                     stream,
  //                                     true,
  //                                     nullptr,
  //                                     nullptr,
  //                                     T(1.0),
  //                                     false,
  //                                     (T)-10.0f,
  //                                     (T)10.0f,
  //                                     (uint64_t)1234);
  //    d_labels.resize(n_samples, stream);
  //    d_labels_ref.resize(n_samples, stream);
  //    d_centroids.resize(params.n_clusters * n_features, stream);
  //    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);
  //    rmm::device_uvector<T> d_sample_weight(n_samples, stream);
  //    thrust::fill(
  //      thrust::cuda::par.on(stream), d_sample_weight.data(), d_sample_weight.data() + n_samples,
  //      1);
  //    auto weight_view =
  //      raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);
  //
  //    T inertia  = 0;
  //    int n_iter = 0;
  //    rmm::device_uvector<char> workspace(0, stream);
  //    rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
  //    rmm::device_uvector<T> inRankCp(0, stream);
  //    auto X_view = raft::make_const_mdspan(X.view());
  //    auto centroids_view =
  //      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
  //    auto miniX = raft::make_device_matrix<T, int>(handle, n_samples / 4, n_features);
  //
  //    // Initialize kmeans on a portion of X
  //    raft::cluster::kmeans::shuffle_and_gather(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<T, int>(miniX.data_handle(), miniX.extent(0),
  //      miniX.extent(1)), miniX.extent(0), params.rng_state.seed);
  //
  //    raft::cluster::kmeans::init_plus_plus(
  //      handle, params, raft::make_const_mdspan(miniX.view()), centroids_view, workspace);
  //
  //    auto minClusterDistance = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto minClusterAndDistance =
  //      raft::make_device_vector<raft::KeyValuePair<int, T>, int>(handle, n_samples);
  //    auto L2NormX           = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto clusterCostBefore = raft::make_device_scalar<T>(handle, 0);
  //    auto clusterCostAfter  = raft::make_device_scalar<T>(handle, 0);
  //
  //    raft::linalg::rowNorm(L2NormX.data_handle(),
  //                          X.data_handle(),
  //                          X.extent(1),
  //                          X.extent(0),
  //                          raft::linalg::L2Norm,
  //                          true,
  //                          stream);
  //
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostBefore.view());
  //
  //    // Run a fit of kmeans
  //    raft::cluster::kmeans::fit_main(handle,
  //                                    params,
  //                                    X_view,
  //                                    weight_view,
  //                                    centroids_view,
  //                                    raft::make_host_scalar_view(&inertia),
  //                                    raft::make_host_scalar_view(&n_iter),
  //                                    workspace);
  //
  //    // Check that the cluster cost decreased
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostAfter.view());
  //    T h_clusterCostBefore = T(0);
  //    T h_clusterCostAfter  = T(0);
  //    raft::update_host(&h_clusterCostBefore, clusterCostBefore.data_handle(), 1, stream);
  //    raft::update_host(&h_clusterCostAfter, clusterCostAfter.data_handle(), 1, stream);
  //    ASSERT_TRUE(h_clusterCostAfter < h_clusterCostBefore);
  //
  //    // Count samples in clusters using 2 methods and compare them
  //    // Fill minClusterAndDistance
  //    raft::cluster::kmeans::min_cluster_and_distance(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<const T, int>(
  //        d_centroids.data(), params.n_clusters, n_features),
  //      minClusterAndDistance.view(),
  //      L2NormX.view(),
  //      L2NormBuf_OR_DistBuf,
  //      params.metric,
  //      params.batch_samples,
  //      params.batch_centroids,
  //      workspace);
  //    raft::cluster::kmeans::KeyValueIndexOp<int, T> conversion_op;
  //    thrust::transform_iterator<raft::cluster::kmeans::KeyValueIndexOp<int, T>,
  //                               raft::KeyValuePair<int, T>*>
  //      itr(minClusterAndDistance.data_handle(), conversion_op);
  //
  //    auto sampleCountInCluster = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto weigthInCluster      = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto newCentroids = raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features);
  //    raft::cluster::kmeans::update_centroids(handle,
  //                                            X_view,
  //                                            weight_view,
  //                                            raft::make_device_matrix_view<const T, int>(
  //                                              d_centroids.data(), params.n_clusters,
  //                                              n_features),
  //                                            itr,
  //                                            weigthInCluster.view(),
  //                                            newCentroids.view());
  //    raft::cluster::kmeans::count_samples_in_cluster(handle,
  //                                                    params,
  //                                                    X_view,
  //                                                    L2NormX.view(),
  //                                                    newCentroids.view(),
  //                                                    workspace,
  //                                                    sampleCountInCluster.view());
  //
  //    ASSERT_TRUE(devArrMatch(sampleCountInCluster.data_handle(),
  //                            weigthInCluster.data_handle(),
  //                            params.n_clusters,
  //                            CompareApprox<T>(params.tol)));
  //  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
    auto stream = raft::resource::get_cuda_stream(handle);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
    }

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = raft::make_const_mdspan(X.view());

    cuvs::cluster::kmeans::fit_predict(
      handle,
      params,
      X_view,
      d_sw,
      d_centroids_view,
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, raft::resource::get_cuda_stream(handle));

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override
  {
    basicTest();
    //    apiTest();
  }

 protected:
  raft::resources handle;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  cuvs::cluster::kmeans::params params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001f, true},
                                                   {1000, 32, 5, 0.0001f, false},
                                                   {1000, 100, 20, 0.0001f, true},
                                                   {1000, 100, 20, 0.0001f, false},
                                                   {10000, 32, 10, 0.0001f, true},
                                                   {10000, 32, 10, 0.0001f, false},
                                                   {10000, 100, 50, 0.0001f, true},
                                                   {10000, 100, 50, 0.0001f, false},
                                                   {10000, 500, 100, 0.0001f, true},
                                                   {10000, 500, 100, 0.0001f, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 500, 100, 0.0001, true},
                                                    {10000, 500, 100, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;

TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));


// Single-GPU OOM bisect: find at what size OOM occurs. Run:
//   ./CLUSTER_TEST --gtest_also_run_disabled_tests --gtest_filter=*OOM_Bisect*
TEST(KmeansTests, DISABLED_OOM_Bisect_WhereItFails)
{
  const int64_t n_features = 1024;
  const int n_clusters    = 100;
  const std::vector<int64_t> sizes = {250000, 500000, 1000000, 2000000, 4000000, 8000000, 9000000, 12000000, 16000000};

  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvs::cluster::kmeans::params params;
  params.n_clusters          = n_clusters;
  params.tol                 = 0.0001f;
  params.max_iter            = 20;
  params.n_init              = 1;
  params.rng_state.seed      = 1234ULL;
  params.oversampling_factor = 1;

  for (int64_t n_samples : sizes) {
    std::cerr << "\n[OOM_Bisect] ========== n_samples=" << n_samples << " ==========" << std::endl;

    const size_t X_bytes       = size_X_bytes(n_samples, n_features, sizeof(float));
    const size_t centroids_bytes =
      static_cast<size_t>(n_clusters) * static_cast<size_t>(n_features) * sizeof(float);
    const bool overflow = would_overflow_int32(n_samples, n_features);

    std::cerr << "[OOM_Bisect] X bytes: " << (X_bytes / (1024 * 1024)) << " MiB ("
              << (X_bytes / (1024 * 1024 * 1024)) << " GiB)" << std::endl;
    std::cerr << "[OOM_Bisect] centroids bytes: " << (centroids_bytes / 1024) << " KiB"
              << std::endl;
    std::cerr << "[OOM_Bisect] n_samples*n_features overflows int32? " << (overflow ? "YES" : "no")
              << " (INT_MAX=" << INT_MAX << ")" << std::endl;
    print_gpu_memory("OOM_Bisect before alloc");

    std::cerr << "[OOM_Bisect] Step: allocate X" << std::endl;
    rmm::device_uvector<float> X_bisect(static_cast<size_t>(n_samples * n_features), stream);
    std::cerr << "[OOM_Bisect] Step: allocated X" << std::endl;
    print_gpu_memory("OOM_Bisect after alloc X");

    std::cerr << "[OOM_Bisect] Step: fill X with uniform" << std::endl;
    raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);
    raft::random::uniform(handle, rng, X_bisect.data(), X_bisect.size(), -1.0f, 1.0f);
    std::cerr << "[OOM_Bisect] Step: filled X" << std::endl;

    std::cerr << "[OOM_Bisect] Step: allocate d_centroids" << std::endl;
    rmm::device_uvector<float> centroids_bisect(
      static_cast<size_t>(n_clusters) * static_cast<size_t>(n_features), stream);
    std::cerr << "[OOM_Bisect] Step: allocated d_centroids" << std::endl;

    auto X_view_bisect =
      raft::make_device_matrix_view<const float, int64_t>(X_bisect.data(), n_samples, n_features);
    auto centroids_view_bisect =
      raft::make_device_matrix_view<float, int64_t>(centroids_bisect.data(), n_clusters, n_features);
    float inertia  = 0;
    int64_t n_iter = 0;

    std::cerr << "[OOM_Bisect] Step: call fit" << std::endl;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               X_view_bisect,
                               std::nullopt,
                               centroids_view_bisect,
                               raft::make_host_scalar_view<float>(&inertia),
                               raft::make_host_scalar_view<int64_t>(&n_iter));
    std::cerr << "[OOM_Bisect] Step: fit returned (n_iter=" << n_iter << ")" << std::endl;

    raft::resource::sync_stream(handle, stream);
    std::cerr << "[OOM_Bisect] Step: sync done" << std::endl;
  }

  std::cerr << "[OOM_Bisect] All sizes completed." << std::endl;
}

// Single-GPU stress test: 8M x 1024 (needs ~32GB GPU memory). Disabled by default; run manually:
//   ./CLUSTER_TEST --gtest_also_run_disabled_tests --gtest_filter=*8M_1024*
TEST(KmeansTests, DISABLED_8M_1024_RandomData)
{
  const int64_t n_samples   = 8000000;
  const int64_t n_features  = 1024;
  const int n_clusters     = 100;

  raft::resources handle;
  cuvs::cluster::kmeans::params params;
  params.n_clusters          = n_clusters;
  params.tol                 = 0.0001f;
  params.max_iter            = 20;
  params.n_init              = 1;
  params.rng_state.seed      = 1234ULL;
  params.oversampling_factor = 1;

  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<float> X_8m(static_cast<size_t>(n_samples * n_features), stream);
  raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);
  raft::random::uniform(handle, rng, X_8m.data(), X_8m.size(), -1.0f, 1.0f);

  rmm::device_uvector<float> centroids_8m(
    static_cast<size_t>(n_clusters) * static_cast<size_t>(n_features), stream);

  auto X_view_8m =
    raft::make_device_matrix_view<const float, int64_t>(X_8m.data(), n_samples, n_features);
  auto centroids_view_8m =
    raft::make_device_matrix_view<float, int64_t>(centroids_8m.data(), n_clusters, n_features);

  float inertia  = 0;
  int64_t n_iter = 0;

  cuvs::cluster::kmeans::fit(handle,
                             params,
                             X_view_8m,
                             std::nullopt,
                             centroids_view_8m,
                             raft::make_host_scalar_view<float>(&inertia),
                             raft::make_host_scalar_view<int64_t>(&n_iter));

  raft::resource::sync_stream(handle, stream);

  EXPECT_GE(n_iter, 1);
  EXPECT_GT(inertia, 0);
}

}  // namespace cuvs
