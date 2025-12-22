#pragma once
#include <utility>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernov_t_convex_hull_binary_components {
class ChernovTConvexHullBinaryComponentsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChernovTConvexHullBinaryComponentsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<std::vector<std::pair<int, int>>> FindConnectedComponents(int width, int height,
                                                                               const std::vector<int> &pixels);
  static std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> pts);
  static bool Clockwise(const std::pair<int, int> &a, const std::pair<int, int> &b, const std::pair<int, int> &c);
};
}  // namespace chernov_t_convex_hull_binary_components
