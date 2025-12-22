// tasks/chernov_t_convex_hull_binary_components/tests/functional/main.cpp
#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"
#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"
#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chernov_t_convex_hull_binary_components {

class ChernovTConvexHullFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    LoadTestData(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto expected = std::get<2>(
      std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()));

    std::set<std::pair<int, int>> out_set, exp_set;
    for (const auto& hull : output_data) {
      out_set.insert(hull.begin(), hull.end());
    }
    for (const auto& hull : expected) {
      exp_set.insert(hull.begin(), hull.end());
    }
    return out_set == exp_set;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  void LoadTestData(const TestType &params) {
    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(
        PPC_ID_chernov_t_convex_hull_binary_components, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open test file: " + abs_path);
    }

    int width, height;
    file >> width >> height;
    std::vector<int> pixels(width * height);
    for (int i = 0; i < width * height; ++i) {
      file >> pixels[i];
    }
    input_data_ = std::make_tuple(width, height, pixels);
  }
};

namespace {

TEST_P(ChernovTConvexHullFuncTests, ConvexHullBinaryComponents) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {{
    std::make_tuple("Empty", "empty.txt", OutType{}),
    std::make_tuple("SinglePixel", "single.txt", OutType{{{0, 0}}}),

    std::make_tuple("Diag3x3", "diag3x3.txt", OutType{
        {{0, 0}}, {{1, 1}}, {{2, 2}}
    }),

    std::make_tuple("Square2x2", "square2x2.txt", OutType{
        {{0, 0}, {0, 1}, {1, 1}, {1, 0}}
    }),

    std::make_tuple("TwoSquares", "two_squares.txt", OutType{
        {{0, 0}, {0, 1}, {1, 1}, {1, 0}},
        {{3, 0}, {3, 1}, {4, 1}, {4, 0}}
    })
}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChernovTConvexHullBinaryComponentsMPI, InType>(
        kTestParam, PPC_SETTINGS_chernov_t_convex_hull_binary_components),
    ppc::util::AddFuncTask<ChernovTConvexHullBinaryComponentsSEQ, InType>(
        kTestParam, PPC_SETTINGS_chernov_t_convex_hull_binary_components)
);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestName = ChernovTConvexHullFuncTests::PrintFuncTestName<ChernovTConvexHullFuncTests>;

INSTANTIATE_TEST_SUITE_P(ConvexHullTests, ChernovTConvexHullFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace chernov_t_convex_hull_binary_components
