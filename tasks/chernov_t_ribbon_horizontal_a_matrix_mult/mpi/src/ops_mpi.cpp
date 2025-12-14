#include "chernov_t_ribbon_horizontal_a_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

ChernovTRibbonHorizontalAMmatrixMultMPI::ChernovTRibbonHorizontalAMmatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ChernovTRibbonHorizontalAMmatrixMultMPI::ValidationImpl() {
  const auto &input = GetInput();

  int rows_a = std::get<0>(input);
  int cols_a = std::get<1>(input);
  const auto &vec_a = std::get<2>(input);

  int rows_b = std::get<3>(input);
  int cols_b = std::get<4>(input);
  const auto &vec_b = std::get<5>(input);

  valid_ = (cols_a == rows_b) && (vec_a.size() == static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a)) &&
           (vec_b.size() == static_cast<size_t>(rows_b) * static_cast<size_t>(cols_b)) && (rows_a > 0) &&
           (cols_a > 0) && (rows_b > 0) && (cols_b > 0);

  return valid_;
}

bool ChernovTRibbonHorizontalAMmatrixMultMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!valid_) {
    return false;
  }

  if (rank == 0) {
    const auto &input = GetInput();

    rowsA_ = std::get<0>(input);
    colsA_ = std::get<1>(input);
    matrixA_ = std::get<2>(input);

    rowsB_ = std::get<3>(input);
    colsB_ = std::get<4>(input);
    matrixB_ = std::get<5>(input);
  }

  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!valid_) {
    GetOutput() = std::vector<int>();
    return false;
  }

  BroadcastMatrixSizes(rank);

  BroadcastMatrixB(rank);

  std::vector<int> local_a = ScatterMatrixA(rank, size);

  int base_rows = global_rowsA_ / size;
  int remainder = global_rowsA_ % size;
  int local_rows = base_rows + (rank < remainder ? 1 : 0);

  std::vector<int> local_c = ComputeLocalC(local_rows, local_a);

  GatherResult(rank, size, local_c);

  return true;
}

void ChernovTRibbonHorizontalAMmatrixMultMPI::BroadcastMatrixSizes(int rank) {
  std::array<int, 4> sizes{};
  if (rank == 0) {
    sizes[0] = rowsA_;
    sizes[1] = colsA_;
    sizes[2] = rowsB_;
    sizes[3] = colsB_;
  }

  MPI_Bcast(sizes.data(), sizes.size(), MPI_INT, 0, MPI_COMM_WORLD);
  global_rowsA_ = sizes[0];
  global_colsA_ = sizes[1];
  global_rowsB_ = sizes[2];
  global_colsB_ = sizes[3];
}

void ChernovTRibbonHorizontalAMmatrixMultMPI::BroadcastMatrixB(int rank) {
  if (rank != 0) {
    matrixB_.resize(static_cast<size_t>(global_rowsB_) * static_cast<size_t>(global_colsB_));
  }
  MPI_Bcast(matrixB_.data(), global_rowsB_ * global_colsB_, MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> ChernovTRibbonHorizontalAMmatrixMultMPI::ScatterMatrixA(int rank, int size) {
  int base_rows = global_rowsA_ / size;
  int remainder = global_rowsA_ % size;

  int local_rows = base_rows + (rank < remainder ? 1 : 0);
  int local_elements = static_cast<int>(static_cast<size_t>(local_rows) * static_cast<size_t>(global_colsA_));

  std::vector<int> local_a(local_elements);
  std::vector<int> sendcounts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
      int rows_for_i = base_rows + (i < remainder ? 1 : 0);
      sendcounts[i] = rows_for_i * global_colsA_;
      displacements[i] = offset;
      offset += sendcounts[i];
    }
  }

  std::vector<int> recvcounts(size);
  if (rank == 0) {
    recvcounts = sendcounts;
  }
  MPI_Bcast(recvcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? matrixA_.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displacements.data() : nullptr, MPI_INT, local_a.data(), local_elements, MPI_INT, 0,
               MPI_COMM_WORLD);

  return local_a;
}

std::vector<int> ChernovTRibbonHorizontalAMmatrixMultMPI::ComputeLocalC(int local_rows,
                                                                        const std::vector<int> &local_a) {
  std::vector<int> local_c(static_cast<size_t>(local_rows) * static_cast<size_t>(global_colsB_), 0);
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < global_colsB_; j++) {
      int sum = 0;
      for (int k = 0; k < global_colsA_; k++) {
        sum += local_a[(i * global_colsA_) + k] * matrixB_[(k * global_colsB_) + j];
      }
      local_c[(i * global_colsB_) + j] = sum;
    }
  }
  return local_c;
}

void ChernovTRibbonHorizontalAMmatrixMultMPI::GatherResult(int rank, int size, const std::vector<int> &local_c) {
  int base_rows = global_rowsA_ / size;
  int remainder = global_rowsA_ % size;

  std::vector<int> recvcounts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
      int rows_for_i = base_rows + (i < remainder ? 1 : 0);
      recvcounts[i] = rows_for_i * global_colsB_;
      displacements[i] = offset;
      offset += recvcounts[i];
    }
  }
  MPI_Bcast(recvcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> result(static_cast<size_t>(global_rowsA_) * static_cast<size_t>(global_colsB_));
  MPI_Gatherv(local_c.data(), static_cast<int>(local_c.size()), MPI_INT, result.data(), recvcounts.data(),
              displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(result.data(), global_rowsA_ * global_colsB_, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = result;
}

bool ChernovTRibbonHorizontalAMmatrixMultMPI::PostProcessingImpl() {
  matrixA_.clear();
  matrixB_.clear();

  return true;
}

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
