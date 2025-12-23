#pragma once

#include <utility>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernov_t_convex_hull_binary_components {
class ChernovTConvexHullBinaryComponentsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChernovTConvexHullBinaryComponentsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void FindConnectedComponentsMpi();
  void ExchangeBoundaryRows(bool has_top, bool has_bottom, std::vector<int> &extended_pixels, int width);
  static std::vector<std::vector<std::pair<int, int>>> ProcessExtendedRegion(const std::vector<int> &extended_pixels,
                                                                             int extended_rows, int width,
                                                                             int global_y_offset);
  static std::vector<std::pair<int, int>> ExtractComponent(int start_col, int start_ey,
                                                           const std::vector<int> &extended_pixels,
                                                           std::vector<std::vector<bool>> &visited, int width,
                                                           int extended_rows, int global_y_offset);
  void FilterLocalComponents(const std::vector<std::vector<std::pair<int, int>>> &all_components);
  void ComputeConvexHulls();
  void GatherAndBroadcastResult();

  void GatherHullsOnRank0(std::vector<int> &all_sizes, std::vector<int> &global_flat);
  void BroadcastResultToAllRanks(const std::vector<std::vector<std::pair<int, int>>> &global_hulls);

  static void SendHullsToRank0(const std::vector<int> &local_flat, const std::vector<int> &local_sizes);
  static void ReceiveHullsFromRank(int src, std::vector<int> &all_sizes, std::vector<int> &global_flat);
  static std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> pts);

  int width_ = 0;
  int height_ = 0;
  int rank_ = 0;
  int size_ = 0;
  int start_row_ = 0;
  int end_row_ = 0;
  std::vector<int> local_pixels_;
  std::vector<std::vector<std::pair<int, int>>> local_hulls_;
  bool valid_ = false;
};
}  // namespace chernov_t_convex_hull_binary_components
