#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"
#include "chernov_t_max_matrix_columns/mpi/include/ops_mpi.hpp"
#include "chernov_t_max_matrix_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chernov_t_max_matrix_columns {

class ChernovTPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const std::size_t kRows_ = 4000;
  const std::size_t kCols_ = 4000;
  InType input_data_{};

  void SetUp() override {
    std::vector<int> matrix_data(kRows_ * kCols_);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 1000);
    for (std::size_t i = 0; i < matrix_data.size(); ++i) {
      matrix_data[i] = dist(gen);
    }
    input_data_ = std::make_tuple(kRows_, kCols_, matrix_data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && (output_data.size() == kCols_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChernovTPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChernovTMaxMatrixColumnsMPI, ChernovTMaxMatrixColumnsSEQ>(
        PPC_SETTINGS_chernov_t_max_matrix_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChernovTPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ChernovTPerfTests, ChernovTPerfTest, kGtestValues, kPerfTestName);

}  // namespace chernov_t_max_matrix_columns
