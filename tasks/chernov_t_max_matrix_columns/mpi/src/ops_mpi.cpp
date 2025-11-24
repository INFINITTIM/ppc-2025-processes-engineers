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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // Только rank 0 проверяет свои данные
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &matrix = std::get<2>(GetInput());
    valid_ = (m > 0) && (n > 0) && (matrix.size() == m * n);
  } else {
    // Процессы кроме rank 0 не имеют данных для проверки
    valid_ = true;  // Они узнают о валидности позже через Bcast
  }

  return valid_;
}

bool ChernovTMaxMatrixColumnsMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (!valid_) {
      return false;
    }
    // Только rank 0 обрабатывает свои данные
    rows_ = std::get<0>(GetInput());
    cols_ = std::get<1>(GetInput());
    input_matrix_ = std::get<2>(GetInput());
  } else {
    // Процессы кроме rank 0 не имеют данных для обработки
    // Они получат данные в RunImpl через Bcast
  }

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Синхронизируем валидность - только rank 0 знает настоящую valid_
  int global_valid = 0;
  if (rank == 0) {
    global_valid = valid_ ? 1 : 0;
  }
  MPI_Bcast(&global_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (!global_valid) {
    GetOutput() = std::vector<int>();
    return false;
  }

  // Рассылаем размеры от rank 0
  int dimensions[2] = {0, 0};
  if (rank == 0) {
    dimensions[0] = static_cast<int>(rows_);
    dimensions[1] = static_cast<int>(cols_);
  }
  MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

  int total_rows = dimensions[0];
  int total_cols = dimensions[1];

  // Рассылаем матрицу от rank 0
  std::vector<int> matrix_data(total_rows * total_cols);
  if (rank == 0) {
    matrix_data = input_matrix_;  // Только rank 0 имеет эти данные
  }
  MPI_Bcast(matrix_data.data(), total_rows * total_cols, MPI_INT, 0, MPI_COMM_WORLD);

  // Дальше все процессы работают с полученными данными
  std::vector<int> local_maxima(total_cols);

  for (int col = 0; col < total_cols; ++col) {
    int max_val = matrix_data[col];

    for (int row = 1; row < total_rows; ++row) {
      int element = matrix_data[row * total_cols + col];
      if (element > max_val) {
        max_val = element;
      }
    }
    local_maxima[col] = max_val;
  }

  if (rank == 0) {
    GetOutput().resize(total_cols);
  }

  MPI_Reduce(local_maxima.data(), GetOutput().data(), total_cols, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = std::vector<int>();
  }

  return true;
}

bool ChernovTMaxMatrixColumnsMPI::PostProcessingImpl() {
  input_matrix_.clear();
  return true;
}

}  // namespace chernov_t_max_matrix_columns
