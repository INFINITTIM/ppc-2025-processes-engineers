#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

using InType = std::tuple<int, int, std::vector<int>, int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::string, std::string, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
