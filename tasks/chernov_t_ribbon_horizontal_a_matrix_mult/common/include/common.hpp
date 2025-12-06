#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace chernov_t_ribbon_horizontal_a_matrix_mult {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chernov_t_ribbon_horizontal_a_matrix_mult
