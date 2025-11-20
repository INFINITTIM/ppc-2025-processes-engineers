#pragma once

#include <cstddef>
#include <vector>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernov_t_max_matrix_columns {

class ChernovTMaxMatrixColumnsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChernovTMaxMatrixColumnsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> CalculateLocalMaxes(int rank, int size);
  std::vector<int> GatherResults(const std::vector<int>& local_maxes, int rank, int size);
  void BroadcastResult(const std::vector<int>& final_result, int rank);

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<int> input_matrix_;
  bool valid_ = false;
};

}  // namespace chernov_t_max_matrix_columns
