#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"
#include "chernov_t_ribbon_horizontal_a_matrix_mult/mpi/include/ops_mpi.hpp"
#include "chernov_t_ribbon_horizontal_a_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

class ChernovTFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    GetDataFromFile(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto expected =
        std::get<2>(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()));

    if (output_data.size() != expected.size()) {
      return false;
    }

    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  void GetDataFromFile(const TestType &params) {
    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chernov_t_ribbon_horizontal_a_matrix_mult, filename);

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;
    file >> rowsA >> colsA >> rowsB >> colsB;

    std::vector<int> matrixA(rowsA * colsA);
    std::vector<int> matrixB(rowsB * colsB);

    // Читаем матрицу A
    for (int i = 0; i < rowsA * colsA; i++) {
      file >> matrixA[i];
    }

    // Читаем матрицу B
    for (int i = 0; i < rowsB * colsB; i++) {
      file >> matrixB[i];
    }

    input_data_ = std::make_tuple(rowsA, colsA, matrixA, rowsB, colsB, matrixB);
  }
};

namespace {

TEST_P(ChernovTFuncTestsProcesses, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {
  std::make_tuple("Matrix_2x3_3x3", "matrix_1.txt", 
    std::vector<int>({30, 36, 42,
                      66, 81, 96})),
  std::make_tuple("Matrix_3x2_2x4", "matrix_2.txt",
    std::vector<int>({50, 60, 70, 80,
                      114, 140, 166, 192,
                      178, 220, 262, 304}))
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChernovTRibbonHorizontalAMmatrixMultMPI, InType>(
      kTestParam, 
      PPC_SETTINGS_chernov_t_ribbon_horizontal_a_matrix_mult),
    ppc::util::AddFuncTask<ChernovTRibbonHorizontalAMmatrixMultSEQ, InType>(
      kTestParam, 
      PPC_SETTINGS_chernov_t_ribbon_horizontal_a_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChernovTFuncTestsProcesses::PrintFuncTestName<ChernovTFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, ChernovTFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
