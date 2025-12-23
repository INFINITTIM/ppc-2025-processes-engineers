#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace chernov_t_convex_hull_binary_components {

using InType = std::tuple<int, int, std::vector<int>>;
using OutType = std::vector<std::vector<std::pair<int, int>>>;
using TestType = std::tuple<std::string, std::string, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chernov_t_convex_hull_binary_components
