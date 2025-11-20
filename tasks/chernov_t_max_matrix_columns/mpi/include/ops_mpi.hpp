#pragma once

#include <cstddef>
#include <vector>

#include "chernov_t_max_matrix_columns/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chernov_t_max_matrix_columns {

class ChernovTMaxMatrixColumnsMPI : public ppc::task::Task<InType, OutType> {
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
  std::vector<int> GatherResults(const std::vector<int> &local_maxes, int rank, int size);
  std::vector<int> AssembleFinalResult(const std::vector<int> &all_local_maxes, int size);
  void BroadcastResult(const std::vector<int> &final_result, int rank, int size);

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<int> input_matrix_;
  size_t output_size_ = 0;
  bool valid_ = false;
};

}  // namespace chernov_t_max_matrix_columns
