#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"
#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"
#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chernov_t_convex_hull_binary_components {

class ChernovTConvexHullPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kWidth_ = 6000;
  const int kHeight_ = 6000;
  InType input_data_;

  void SetUp() override {
    std::vector<int> pixels(static_cast<std::size_t>(kWidth_) * static_cast<std::size_t>(kHeight_), 0);
    std::seed_seq seed{69};
    std::mt19937 gen(seed);

    for (int i = 0; i < 60; ++i) {
      int w = 20 + static_cast<int>(gen() % 80);
      int h = 20 + static_cast<int>(gen() % 80);
      int x = static_cast<int>(gen() % (kWidth_ - w));
      int y = static_cast<int>(gen() % (kHeight_ - h));
      for (int dy = 0; dy < h; ++dy) {
        for (int dx = 0; dx < w; ++dx) {
          pixels[static_cast<std::size_t>((y + dy) * kWidth_) + static_cast<std::size_t>(x + dx)] = 1;
        }
      }
    }

    auto draw_circle = [&](int cx, int cy, int r) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          if (((static_cast<std::int64_t>(dx) * static_cast<std::int64_t>(dx)) +
               (static_cast<std::int64_t>(dy) * static_cast<std::int64_t>(dy))) <=
              static_cast<std::int64_t>(r) * static_cast<std::int64_t>(r)) {
            int x = cx + dx;
            int y = cy + dy;
            if (x >= 0 && x < kWidth_ && y >= 0 && y < kHeight_) {
              pixels[static_cast<std::size_t>(y * kWidth_) + static_cast<std::size_t>(x)] = 1;
            }
          }
        }
      }
    };
    for (int i = 0; i < 40; ++i) {
      int r = 25 + static_cast<int>(gen() % 50);
      int cx = r + static_cast<int>(gen() % (kWidth_ - 2 * r));
      int cy = r + static_cast<int>(gen() % (kHeight_ - 2 * r));
      draw_circle(cx, cy, r);
    }

    std::size_t noise_count = (static_cast<std::size_t>(kWidth_) * static_cast<std::size_t>(kHeight_)) / 1000;
    for (std::size_t i = 0; i < noise_count; ++i) {
      auto idx =
          static_cast<std::size_t>(gen() % (static_cast<std::int64_t>(kWidth_) * static_cast<std::int64_t>(kHeight_)));
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
