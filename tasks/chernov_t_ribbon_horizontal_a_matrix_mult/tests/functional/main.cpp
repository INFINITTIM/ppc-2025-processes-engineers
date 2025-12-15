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

class ChernovTFuncTestsMatrixMultProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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

    int rows_a = 0;
    int cols_a = 0;
    int rows_b = 0;
    int cols_b = 0;
    file >> rows_a >> cols_a >> rows_b >> cols_b;

    std::vector<int> matrix_a(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a));
    std::vector<int> matrix_b(static_cast<size_t>(rows_b) * static_cast<size_t>(cols_b));

    for (int i = 0; i < rows_a * cols_a; i++) {
      file >> matrix_a[i];
    }

    for (int i = 0; i < rows_b * cols_b; i++) {
      file >> matrix_b[i];
    }

    input_data_ = std::make_tuple(rows_a, cols_a, matrix_a, rows_b, cols_b, matrix_b);
  }
};

namespace {

TEST_P(ChernovTFuncTestsMatrixMultProcesses, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {
    std::make_tuple("Matrix_2x3_3x3", "matrix_1.txt", std::vector<int>({66, 72, 78, 156, 171, 186})),
    std::make_tuple("Matrix_3x2_2x4", "matrix_2.txt",
                    std::vector<int>({29, 32, 35, 38, 65, 72, 79, 86, 101, 112, 123, 134})),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ChernovTRibbonHorizontalAMmatrixMultMPI, InType>(
                                               kTestParam, PPC_SETTINGS_chernov_t_ribbon_horizontal_a_matrix_mult),
                                           ppc::util::AddFuncTask<ChernovTRibbonHorizontalAMmatrixMultSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_chernov_t_ribbon_horizontal_a_matrix_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ChernovTFuncTestsMatrixMultProcesses::PrintFuncTestName<ChernovTFuncTestsMatrixMultProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, ChernovTFuncTestsMatrixMultProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
