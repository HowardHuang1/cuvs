/*
 * Declarations for MPI helpers used by kmeans_mg.cu (no mpi.h in .cu).
 */

#ifndef CUVS_TESTS_CLUSTER_KMEANS_MG_MPI_H
#define CUVS_TESTS_CLUSTER_KMEANS_MG_MPI_H

#ifdef CUVS_CLUSTER_MG_TEST_HAVE_MPI

#ifdef __cplusplus
extern "C" {
#endif

int sharded_mpi_init(int* rank, int* size);
void sharded_mpi_finalize(void);
void sharded_mpi_bcast_nccl_id(void* buf, int count);

#ifdef __cplusplus
}
#endif

#endif  // CUVS_CLUSTER_MG_TEST_HAVE_MPI
#endif  // CUVS_TESTS_CLUSTER_KMEANS_MG_MPI_H
