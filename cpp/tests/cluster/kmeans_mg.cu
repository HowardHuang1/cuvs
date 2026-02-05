/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>

#include <raft/comms/std_comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdio.h>
#include <test_utils.h>

#include <vector>

#define NCCLCHECK(cmd)                                                                        \
  do {                                                                                        \
    ncclResult_t res = cmd;                                                                   \
    if (res != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

namespace cuvs {

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : stream(raft::resource::get_cuda_stream(handle)),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream),
      d_sample_weight(0, stream)
  {
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
    const int n_ranks = 1;
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitAll(&nccl_comm, n_ranks, {0}));
    // Only attach comms when n_ranks > 1. With 1 rank, fit() uses single-GPU path so results
    // match make_blobs and ARI passes. The multi-GPU path (mg::fit) with 1 rank can produce
    // different cluster IDs and fail ARI.
    if (n_ranks > 1) {
      raft::comms::build_comms_nccl_only(&handle, nccl_comm, n_ranks, 0);
    }

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;  // match single-GPU test; 1 can change algo path and break ARI

    auto stream = raft::resource::get_cuda_stream(handle);
    rmm::device_uvector<T> X(n_samples * n_features, stream);
    rmm::device_uvector<int> labels(n_samples, stream);

    raft::random::make_blobs<T, int>(X.data(),
                                     labels.data(),
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
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
      d_sw = raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);
    }
    raft::copy(d_labels_ref.data(), labels.data(), n_samples, stream);

    raft::resource::sync_stream(handle, stream);

    T inertia  = 0;
    int n_iter = 0;

    auto X_view = raft::make_device_matrix_view<const T, int>(X.data(), n_samples, n_features);
    auto centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);

    cuvs::cluster::kmeans::fit_predict(
      handle,
      params,
      X_view,
      d_sw,
      centroids_view,
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter));
    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, stream);
    raft::resource::sync_stream(handle, stream);

    if (score < 0.99) {
      std::cout << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream)
                << std::endl;
      std::cout << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream)
                << std::endl;
      std::cout << "score = " << score << std::endl;
    }
    ncclCommDestroy(nccl_comm);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::resources handle;
  cudaStream_t stream;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  cuvs::cluster::kmeans::params params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001, true},
                                                   {1000, 32, 5, 0.0001, false},
                                                   {1000, 100, 20, 0.0001, true},
                                                   {1000, 100, 20, 0.0001, false},
                                                   {10000, 32, 10, 0.0001, true},
                                                   {10000, 32, 10, 0.0001, false},
                                                   {10000, 100, 50, 0.0001, true},
                                                   {10000, 100, 50, 0.0001, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score >= 0.99); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score >= 0.99); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

// Multi-GPU path stress test: 8M x 1024 (needs ~32GB GPU memory when run with 1 rank).
// Build requires BUILD_MG_ALGOS=ON (default). Run: ./CLUSTER_MG_TEST
//   --gtest_also_run_disabled_tests --gtest_filter=*8M_1024_MultiGPU*
// For real multi-GPU (data split across GPUs): run with MPI (e.g. mpirun -np 4) and have each
// rank allocate (8M/n_ranks) x 1024; see docs/source/developer_guide.md for comms setup.
//
// OOM debugging: 8M*1024 elements = 8,388,608,000 > INT_MAX (2^31-1), so use int64_t views to
// avoid overflow. To find where it fails, try smaller sizes (e.g. 250000, 1000000, 2000000)
// and add prints before/after allocation and before/after fit(). See README_8M_OOM.md.
TEST(KmeansTests, DISABLED_8M_1024_RandomData_MultiGPU)
{
  const int64_t n_samples   = 8000000;  // Try 250000, 1000000, 2000000 to bisect OOM
  const int64_t n_features = 1024;
  const int n_clusters     = 100;

  ncclComm_t nccl_comm;
  NCCLCHECK(ncclCommInitAll(&nccl_comm, 1, {0}));

  raft::resources handle;
  raft::comms::build_comms_nccl_only(&handle, nccl_comm, 1, 0);

  cuvs::cluster::kmeans::params params;
  params.n_clusters          = n_clusters;
  params.tol                 = 0.0001f;
  params.max_iter            = 20;
  params.n_init              = 1;
  params.rng_state.seed      = 1234ULL;
  params.oversampling_factor = 1;

  auto stream = raft::resource::get_cuda_stream(handle);

  std::cerr << "[8M_1024] About to allocate X: " << n_samples << " x " << n_features
            << " = " << (n_samples * n_features) << " floats" << std::endl;
  rmm::device_uvector<float> X(n_samples * n_features, stream);
  std::cerr << "[8M_1024] Allocated X" << std::endl;
  raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);
  raft::random::uniform(handle, rng, X.data(), X.size(), -1.0f, 1.0f);

  std::cerr << "[8M_1024] About to allocate d_centroids" << std::endl;
  rmm::device_uvector<float> d_centroids(n_clusters * n_features, stream);
  std::cerr << "[8M_1024] Allocated d_centroids" << std::endl;

  // Use int64_t index type: n_samples * n_features > INT_MAX, so int would overflow
  auto X_view =
    raft::make_device_matrix_view<const float, int64_t>(X.data(), n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<float, int64_t>(d_centroids.data(), n_clusters, n_features);

  float inertia = 0;
  int64_t n_iter = 0;

  std::cerr << "[8M_1024] About to call fit" << std::endl;
  cuvs::cluster::kmeans::fit(handle,
                             params,
                             X_view,
                             std::nullopt,
                             centroids_view,
                             raft::make_host_scalar_view<float>(&inertia),
                             raft::make_host_scalar_view<int64_t>(&n_iter));
  std::cerr << "[8M_1024] fit returned" << std::endl;

  raft::resource::sync_stream(handle, stream);
  std::cerr << "[8M_1024] sync done" << std::endl;

  EXPECT_GE(n_iter, 1);
  EXPECT_GT(inertia, 0);

  ncclCommDestroy(nccl_comm);
}

}  // end namespace cuvs
