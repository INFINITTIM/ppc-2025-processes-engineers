#pragma once

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
};

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
