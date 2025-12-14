#pragma once

#include <vector>

#include "chernov_t_ribbon_horizontal_a_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

class ChernovTRibbonHorizontalAMmatrixMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChernovTRibbonHorizontalAMmatrixMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastMatrixSizes(int rank);
  void BroadcastMatrixB(int rank);
  std::vector<int> ScatterMatrixA(int rank, int size);
  std::vector<int> ComputeLocalC(int local_rows, const std::vector<int> &local_a);
  void GatherResult(int rank, int size, const std::vector<int> &local_c);

  int rowsA_ = 0, colsA_ = 0;
  int rowsB_ = 0, colsB_ = 0;
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;

  int global_rowsA_ = 0, global_colsA_ = 0;
  int global_rowsB_ = 0, global_colsB_ = 0;

  bool valid_ = false;
};

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
