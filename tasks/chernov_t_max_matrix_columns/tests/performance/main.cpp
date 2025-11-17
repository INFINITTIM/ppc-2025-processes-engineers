#include <gtest/gtest.h>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"
#include "chernov_t_max_matrix_columns/mpi/include/ops_mpi.hpp"
#include "chernov_t_max_matrix_columns/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chernov_t_max_matrix_columns {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChernovTMaxMatrixColumnsMPI, ChernovTMaxMatrixColumnsSEQ>(PPC_SETTINGS_chernov_t_max_matrix_columns);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chernov_t_max_matrix_columns
