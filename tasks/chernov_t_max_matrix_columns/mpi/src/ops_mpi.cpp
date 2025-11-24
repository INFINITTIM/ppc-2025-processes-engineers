#include "chernov_t_max_matrix_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"

namespace chernov_t_max_matrix_columns {

ChernovTMaxMatrixColumnsMPI::ChernovTMaxMatrixColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ChernovTMaxMatrixColumnsMPI::ValidationImpl() {
  auto &input = GetInput();
  std::size_t m = std::get<0>(input);
  std::size_t n = std::get<1>(input);
  std::vector<int> &matrix = std::get<2>(input);

  valid_ = (m > 0) && (n > 0) && (matrix.size() == m * n);
  return valid_;
}

bool ChernovTMaxMatrixColumnsMPI::PreProcessingImpl() {
  if (!valid_) {
    return false;
  }

  auto &input = GetInput();
  rows_ = std::get<0>(input);
  cols_ = std::get<1>(input);
  input_matrix_ = std::get<2>(input);

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!valid_) {
    GetOutput() = std::vector<int>();
    return false;
  }

  BroadcastDimensions(rank);
  std::vector<int> matrix_data = BroadcastMatrixData(rank);
  std::vector<int> local_maxima = ComputeLocalMaxima(rank, size, matrix_data);
  ComputeAndBroadcastResult(local_maxima);

  return true;
}

void ChernovTMaxMatrixColumnsMPI::BroadcastDimensions(int rank) {
  std::array<int, 2> dimensions{};
  if (rank == 0) {
    dimensions[0] = static_cast<int>(rows_);
    dimensions[1] = static_cast<int>(cols_);
  }
  MPI_Bcast(dimensions.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  total_rows_ = dimensions[0];
  total_cols_ = dimensions[1];
}

std::vector<int> ChernovTMaxMatrixColumnsMPI::BroadcastMatrixData(int rank) {
  const auto total_size = static_cast<std::size_t>(total_rows_) * static_cast<std::size_t>(total_cols_);
  std::vector<int> matrix_data(total_size);
  if (rank == 0) {
    matrix_data = input_matrix_;
  }
  MPI_Bcast(matrix_data.data(), static_cast<int>(total_size), MPI_INT, 0, MPI_COMM_WORLD);
  return matrix_data;
}

std::vector<int> ChernovTMaxMatrixColumnsMPI::ComputeLocalMaxima(int rank, int size,
                                                                 const std::vector<int> &matrix_data) const {
  std::vector<int> local_maxima(total_cols_, std::numeric_limits<int>::min());

  for (int col = rank; col < total_cols_; col += size) {
    int max_val = matrix_data[col];

    for (int row = 1; row < total_rows_; ++row) {
      const int element = matrix_data[(row * total_cols_) + col];
      max_val = std::max(element, max_val);
    }
    local_maxima[col] = max_val;
  }

  for (int col = 0; col < total_cols_; ++col) {
    if (col % size != rank) {
      local_maxima[col] = std::numeric_limits<int>::min();
    }
  }

  return local_maxima;
}

void ChernovTMaxMatrixColumnsMPI::ComputeAndBroadcastResult(const std::vector<int> &local_maxima) {
  std::vector<int> result(total_cols_);

  MPI_Reduce(local_maxima.data(), result.data(), total_cols_, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(result.data(), total_cols_, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;
}

bool ChernovTMaxMatrixColumnsMPI::PostProcessingImpl() {
  input_matrix_.clear();
  return true;
}

}  // namespace chernov_t_max_matrix_columns
