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

std::vector<int> ChernovTMaxMatrixColumnsMPI::CalculateLocalMaxes(int rank, int size) {
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
  return local_maxes;
}

std::vector<int> ChernovTMaxMatrixColumnsMPI::GatherResults(const std::vector<int> &local_maxes, int rank, int size) {
  auto cols_per_proc = static_cast<int>(cols_ / size);
  auto remainder = static_cast<int>(cols_ % size);

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

  MPI_Gatherv(local_maxes.data(), static_cast<int>(local_maxes.size()), MPI_INT, all_local_maxes.data(),
              recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return all_local_maxes;
}

std::vector<int> ChernovTMaxMatrixColumnsMPI::AssembleFinalResult(const std::vector<int> &all_local_maxes, int size) {
  auto cols_per_proc = static_cast<int>(cols_ / size);
  auto remainder = static_cast<int>(cols_ % size);

  std::vector<int> final_result(cols_);
  for (int process = 0; process < size; ++process) {
    int p_cols = cols_per_proc + (process < remainder ? 1 : 0);
    int p_start_col = (process * cols_per_proc) + std::min(process, remainder);
    std::vector<int>::size_type displ = 0;
    if (process > 0) {
      displ = static_cast<std::vector<int>::size_type>(p_start_col);
    }
    for (int j = 0; j < p_cols; ++j) {
      final_result[p_start_col + j] = all_local_maxes[displ + j];
    }
  }
  return final_result;
}

void ChernovTMaxMatrixColumnsMPI::BroadcastResult(const std::vector<int> &final_result, int rank, int size) {
  size_t temp_output_size = 0;
  if (rank == 0) {
    temp_output_size = final_result.size();
  }
  MPI_Bcast(&temp_output_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  output_size_ = temp_output_size;

  std::vector<int> temp_output;
  if (rank == 0) {
    temp_output = final_result;
  } else {
    temp_output.resize(output_size_);
  }
  MPI_Bcast(temp_output.data(), static_cast<int>(output_size_), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = temp_output;
}

bool ChernovTMaxMatrixColumnsMPI::RunImpl() {
  if (!valid_) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> local_maxes = CalculateLocalMaxes(rank, size);
  if (local_maxes.empty() && size > 1 && rank != 0) {
    if (size == 1) {
      return false;
    }
  }

  std::vector<int> all_local_maxes = GatherResults(local_maxes, rank, size);
  if (rank == 0 && all_local_maxes.size() != cols_) {
    return false;
  }

  std::vector<int> final_result;
  if (rank == 0) {
    final_result = AssembleFinalResult(all_local_maxes, size);
    if (final_result.size() != cols_) {
      return false;
    }
  }

  BroadcastResult(final_result, rank, size);

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::PostProcessingImpl() {
  input_matrix_.clear();
  return true;
}

}  // namespace chernov_t_max_matrix_columns
