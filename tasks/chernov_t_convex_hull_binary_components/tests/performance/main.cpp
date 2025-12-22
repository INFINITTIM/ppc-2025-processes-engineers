#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"
#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"
#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chernov_t_convex_hull_binary_components {

class ChernovTConvexHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kWidth_ = 8000;
  const int kHeight_ = 8000;
  InType input_data_;

  void SetUp() override {
    std::vector<int> pixels(kWidth_ * kHeight_, 0);
    std::mt19937 gen(42);

    for (int i = 0; i < 60; ++i) {
      int w = 20 + gen() % 80;
      int h = 20 + gen() % 80;
      int x = gen() % (kWidth_ - w);
      int y = gen() % (kHeight_ - h);
      for (int dy = 0; dy < h; ++dy) {
        for (int dx = 0; dx < w; ++dx) {
          pixels[(y + dy) * kWidth_ + (x + dx)] = 1;
        }
      }
    }

    auto draw_circle = [&](int cx, int cy, int r) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          if (dx * dx + dy * dy <= r * r) {
            int x = cx + dx, y = cy + dy;
            if (x >= 0 && x < kWidth_ && y >= 0 && y < kHeight_) {
              pixels[y * kWidth_ + x] = 1;
            }
          }
        }
      }
    };
    for (int i = 0; i < 40; ++i) {
      int r = 25 + gen() % 50;
      int cx = r + gen() % (kWidth_ - 2 * r);
      int cy = r + gen() % (kHeight_ - 2 * r);
      draw_circle(cx, cy, r);
    }

    int noise_count = (kWidth_ * kHeight_) / 1000;
    for (int i = 0; i < noise_count; ++i) {
      int idx = gen() % (kWidth_ * kHeight_);
      pixels[idx] = 1;
    }

    input_data_ = InType{kWidth_, kHeight_, pixels};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChernovTConvexHullPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChernovTConvexHullBinaryComponentsSEQ, ChernovTConvexHullBinaryComponentsMPI>(
        PPC_SETTINGS_chernov_t_convex_hull_binary_components);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ChernovTConvexHullPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ConvexHullPerfTests, ChernovTConvexHullPerfTests, kGtestValues, kPerfTestName);

}  // namespace chernov_t_convex_hull_binary_components
