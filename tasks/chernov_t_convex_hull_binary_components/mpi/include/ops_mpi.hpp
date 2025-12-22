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
  void ComputeConvexHulls();
  void GatherAndBroadcastResult();

  std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> pts);
  bool Clockwise(const std::pair<int, int> &a, const std::pair<int, int> &b, const std::pair<int, int> &c);

  // Новые вспомогательные функции для уменьшения сложности
  void SendReceiveNeighborRows(bool has_top, bool has_bottom, std::vector<int> &top_recv,
                               std::vector<int> &bottom_recv);
  void ProcessPixelComponent(int col, int ey, int global_y_offset, const std::vector<int> &extended_pixels,
                             std::vector<std::vector<bool>> &visited);
  void ProcessComponentQueue(std::queue<std::pair<int, int>> &q, const std::vector<int> &extended_pixels,
                             std::vector<std::vector<bool>> &visited, std::vector<std::pair<int, int>> &comp);
  void BuildLowerHull(std::vector<std::pair<int, int>> &pts, std::vector<std::pair<int, int>> &hull);
  void BuildUpperHull(std::vector<std::pair<int, int>> &pts, std::vector<std::pair<int, int>> &hull, size_t lower_len);
  void GatherWorkerData(int src, std::vector<int> &all_sizes, std::vector<int> &global_flat);
  void SendLocalData();
  void ReconstructGlobalHulls(std::vector<int> &all_sizes, std::vector<int> &flat, OutType &global_hulls);

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
