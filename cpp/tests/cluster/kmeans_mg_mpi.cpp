/*
 * MPI helpers for sharded k-means test. Built as C++ so mpi.h is not parsed by nvcc.
 * This file is only added to CLUSTER_MG_TEST when MPI is found, so MPI is always available here.
 */

#include <mpi.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {

// Returns 1 if size >= 2 (ready to run sharded test), 0 otherwise (caller should skip and finalize).
int sharded_mpi_init(int* rank, int* size)
{
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, size);
  return (*size >= 2) ? 1 : 0;
}

void sharded_mpi_finalize(void) { MPI_Finalize(); }

void sharded_mpi_bcast_nccl_id(void* buf, int count)
{
  MPI_Bcast(buf, count, MPI_BYTE, 0, MPI_COMM_WORLD);
}

// OOM bisect: get rank/size (init MPI if needed).
void oom_bisect_mpi_get_rank_size(int* rank, int* size)
{
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, size);
}

}  // extern "C"

// Gather each rank's log lines to rank 0 and print in GPU order (GPU 0, then GPU 1, ...).
void oom_bisect_mpi_gather_print(const std::vector<std::string>& lines, int rank, int size)
{
  if (size <= 0) return;
  std::ostringstream oss;
  for (const auto& s : lines) oss << s << '\n';
  std::string buf = oss.str();
  int len         = static_cast<int>(buf.size());

  std::vector<int> recv_lens(size, 0);
  MPI_Gather(&len, 1, MPI_INT, recv_lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(size, 0);
  int total = 0;
  for (int r = 0; r < size; ++r) {
    displs[r] = total;
    total += recv_lens[r];
  }

  std::vector<char> recv_buf(rank == 0 ? total : 0);
  MPI_Gatherv(buf.data(), len, MPI_BYTE, recv_buf.data(), recv_lens.data(), displs.data(),
              MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      if (recv_lens[r] > 0) {
        std::cerr.write(recv_buf.data() + displs[r], recv_lens[r]);
        if (recv_buf[displs[r] + recv_lens[r] - 1] != '\n') std::cerr << '\n';
      }
    }
    std::cerr << std::flush;
  }
}
