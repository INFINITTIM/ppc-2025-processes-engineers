#include "chernov_t_ribbon_horizontal_a_matrix_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

ChernovTRibbonHorizontalAMmatrixMultSEQ::ChernovTRibbonHorizontalAMmatrixMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::ValidationImpl() {
  const auto &input = GetInput();

  int rows_a = std::get<0>(input);
  int cols_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  int rows_b = std::get<3>(input);
  int cols_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  if (cols_a != rows_b) {
    return false;
  }

  if (matrix_a.size() != static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a)) {
    return false;
  }

  if (matrix_b.size() != static_cast<size_t>(rows_b) * static_cast<size_t>(cols_b)) {
    return false;
  }

  if (rows_a <= 0 || cols_a <= 0 || rows_b <= 0 || cols_b <= 0) {
    return false;
  }

  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  int rows_a = std::get<0>(input);
  int cols_b = std::get<4>(input);

  GetOutput() = std::vector<int>(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0);
  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::RunImpl() {
  const auto &input = GetInput();

  int rows_a = std::get<0>(input);
  int cols_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  int cols_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  auto &output = GetOutput();

  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      int sum = 0;
      for (int k = 0; k < cols_a; k++) {
        sum += matrix_a[(i * cols_a) + k] * matrix_b[(k * cols_b) + j];
      }
      output[(i * cols_b) + j] = sum;
    }
  }

  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
