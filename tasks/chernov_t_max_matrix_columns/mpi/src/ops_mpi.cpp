#include "chernov_t_max_matrix_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"

namespace chernov_t_max_matrix_columns {

ChernovTMaxMatrixColumnsMPI::ChernovTMaxMatrixColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ChernovTMaxMatrixColumnsMPI::ValidationImpl() {
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  std::vector<int> &matrix = std::get<2>(GetInput());

  valid_ = (m > 0) && (n > 0) && (matrix.size() == m * n);
  return valid_;
}

bool ChernovTMaxMatrixColumnsMPI::PreProcessingImpl() {
  if (!valid_) {
    return false;
  }

  rows_ = std::get<0>(GetInput());
  cols_ = std::get<1>(GetInput());
  input_matrix_ = std::get<2>(GetInput());

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::RunImpl() {
  if (!valid_) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto cols_per_proc = static_cast<int>(cols_ / size);
  auto remainder = static_cast<int>(cols_ % size);

  int start_col = (rank * cols_per_proc) + std::min(rank, remainder);
  int num_local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<int> local_maxes(num_local_cols);

  for (int local_idx = 0; local_idx < num_local_cols; ++local_idx) {
    int global_col = start_col + local_idx;
    int max_val = input_matrix_[global_col];

    for (std::size_t row = 1; row < rows_; ++row) {
      std::size_t index = (row * cols_) + global_col;
      max_val = std::max(input_matrix_[index], max_val);
    }
    local_maxes[local_idx] = max_val;
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int process = 0; process < size; ++process) {
    int p_cols = cols_per_proc + (process < remainder ? 1 : 0);
    recvcounts[process] = p_cols;
  }

  displs[0] = 0;
  for (int process = 1; process < size; ++process) {
    displs[process] = displs[process - 1] + recvcounts[process - 1];
  }

  std::vector<int> all_local_maxes;
  if (rank == 0) {
    all_local_maxes.resize(cols_);
  }

  MPI_Gatherv(local_maxes.data(), num_local_cols, MPI_INT, all_local_maxes.data(), recvcounts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> final_result(cols_);
    for (int process = 0; process < size; ++process) {
      int p_cols = cols_per_proc + (process < remainder ? 1 : 0);
      int p_start_col = (process * cols_per_proc) + std::min(process, remainder);
      for (int j = 0; j < p_cols; ++j) {
        final_result[p_start_col + j] = all_local_maxes[displs[process] + j];
      }
    }
    GetOutput() = final_result;
  }

  int output_size = 0;
  if (rank == 0) {
    output_size = GetOutput().size();
  }
  MPI_Bcast(&output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(output_size);
  }

  MPI_Bcast(GetOutput().data(), output_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::PostProcessingImpl() {
  input_matrix_.clear();
  return true;
}

}  // namespace chernov_t_max_matrix_columns
