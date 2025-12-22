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
  void ExchangeBoundaryRows();
  void ProcessExtendedRegion();
  void FilterLocalComponents(const std::vector<std::pair<int, int>> &comp);

  void ComputeConvexHulls();
  void GatherAndBroadcastResult();
  void SendHullsToRank0();
  void ReceiveHullsFromRank(int src);

  static std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> pts);
  static bool Clockwise(const std::pair<int, int> &a, const std::pair<int, int> &b, const std::pair<int, int> &c);

  int width_ = 0;
  int height_ = 0;
  int rank_ = 0;
  int size_ = 0;
  int start_row_ = 0;
  int end_row_ = 0;
  std::vector<int> local_pixels_;
  std::vector<std::vector<std::pair<int, int>>> local_hulls_;
  bool valid_ = false;

  // Для ExchangeBoundaryRows
  std::vector<int> top_recv_;
  std::vector<int> bottom_recv_;
  std::vector<int> extended_pixels_;
  int extended_rows_ = 0;
  int offset_top_ = 0;
};

}  // namespace chernov_t_convex_hull_binary_components
