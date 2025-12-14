#include "chernov_t_ribbon_horizontal_a_matrix_mult/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"
#include "util/include/util.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

ChernovTRibbonHorizontalAMmatrixMultSEQ::ChernovTRibbonHorizontalAMmatrixMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::ValidationImpl() {
  const auto &input = GetInput();

  int rowsA = std::get<0>(input);
  int colsA = std::get<1>(input);
  const auto &matrixA = std::get<2>(input);

  int rowsB = std::get<3>(input);
  int colsB = std::get<4>(input);
  const auto &matrixB = std::get<5>(input);

  if (colsA != rowsB) {
    return false;
  }

  if (matrixA.size() != static_cast<size_t>(rowsA * colsA)) {
    return false;
  }

  if (matrixB.size() != static_cast<size_t>(rowsB * colsB)) {
    return false;
  }

  if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) {
    return false;
  }

  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  int rowsA = std::get<0>(input);
  int colsB = std::get<4>(input);

  GetOutput() = std::vector<int>(rowsA * colsB, 0);
  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::RunImpl() {
  const auto &input = GetInput();

  int rowsA = std::get<0>(input);
  int colsA = std::get<1>(input);
  const auto &matrixA = std::get<2>(input);

  int colsB = std::get<4>(input);
  const auto &matrixB = std::get<5>(input);

  auto &output = GetOutput();

  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      int sum = 0;
      for (int k = 0; k < colsA; k++) {
        sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
      }
      output[i * colsB + j] = sum;
    }
  }

  return true;
}

bool ChernovTRibbonHorizontalAMmatrixMultSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
