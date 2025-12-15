#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"
#include "chernov_t_ribbon_horizontal_a_matrix_mult/mpi/include/ops_mpi.hpp"
#include "chernov_t_ribbon_horizontal_a_matrix_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

class ChernovTMatrixMultPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const std::size_t kSize_ = 1100;
  InType input_data_;

  void SetUp() override {
    std::vector<int> matrix_a(kSize_ * kSize_);
    std::vector<int> matrix_b(kSize_ * kSize_);

    for (std::size_t i = 0; i < kSize_; ++i) {
      for (std::size_t j = 0; j < kSize_; ++j) {
        int value = static_cast<int>(((i * 13 + j * 29) % 100) + 1);
        matrix_a[(i * kSize_) + j] = value;
        matrix_b[(i * kSize_) + j] = static_cast<int>(((i * 17 + j * 31) % 100) + 1);
      }
    }

    input_data_ = std::make_tuple(static_cast<int>(kSize_), static_cast<int>(kSize_), matrix_a,
                                  static_cast<int>(kSize_), static_cast<int>(kSize_), matrix_b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == kSize_ * kSize_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChernovTMatrixMultPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ChernovTRibbonHorizontalAMmatrixMultMPI,
                                                       ChernovTRibbonHorizontalAMmatrixMultSEQ>(
    PPC_SETTINGS_chernov_t_ribbon_horizontal_a_matrix_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChernovTMatrixMultPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ChernovTMatrixMultPerfTests, ChernovTMatrixMultPerfTest, kGtestValues, kPerfTestName);

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
