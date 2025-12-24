#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"

namespace chernov_t_convex_hull_binary_components {

ChernovTConvexHullBinaryComponentsSEQ::ChernovTConvexHullBinaryComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ChernovTConvexHullBinaryComponentsSEQ::ValidationImpl() {
  const auto &[width, height, pixels] = GetInput();
  if (width <= 0 || height <= 0) {
    return false;
  }
  if (pixels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
    return false;
  }
  return std::ranges::all_of(pixels, [](int p) { return p == 0 || p == 1; });
}

bool ChernovTConvexHullBinaryComponentsSEQ::PreProcessingImpl() {
  return true;
}

bool ChernovTConvexHullBinaryComponentsSEQ::RunImpl() {
  const auto &[width, height, pixels] = GetInput();
  auto components = FindConnectedComponents(width, height, pixels);
  OutType hulls;
  for (auto &comp : components) {
    if (!comp.empty()) {
      hulls.push_back(ConvexHull(comp));
    }
  }
  GetOutput() = std::move(hulls);
  return true;
}

bool ChernovTConvexHullBinaryComponentsSEQ::PostProcessingImpl() {
  return true;
}

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsSEQ::ExtractComponent(
    int start_col, int start_row, const std::vector<int> &pixels, std::vector<std::vector<bool>> &visited, int width,
    int height) {
  std::vector<std::pair<int, int>> comp;
  std::queue<std::pair<int, int>> q;
  q.emplace(start_col, start_row);
  visited[static_cast<std::size_t>(start_row)][static_cast<std::size_t>(start_col)] = true;

  constexpr std::array<std::pair<int, int>, 4> kDirs = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
  while (!q.empty()) {
    auto [cx, cy] = q.front();
    q.pop();
    comp.emplace_back(cx, cy);

    for (const auto &[dx, dy] : kDirs) {
      int nx = cx + dx;
      int ny = cy + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        std::size_t idx =
            (static_cast<std::size_t>(ny) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(nx);
        if (pixels[idx] == 1 && !visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)]) {
          visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)] = true;
          q.emplace(nx, ny);
        }
      }
    }
  }
  return comp;
}

std::vector<std::vector<std::pair<int, int>>> ChernovTConvexHullBinaryComponentsSEQ::FindConnectedComponents(
    int width, int height, const std::vector<int> &pixels) {
  std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> components;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      std::size_t idx =
          (static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col);
      if (pixels[idx] == 1 && !visited[row][col]) {
        auto comp = ExtractComponent(col, row, pixels, visited, width, height);
        components.push_back(std::move(comp));
      }
    }
  }
  return components;
}

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsSEQ::ConvexHull(
    std::vector<std::pair<int, int>> pts) {
  if (pts.size() <= 1U) {
    return pts;
  }

  std::ranges::sort(pts);
  auto [first, last] = std::ranges::unique(pts);
  pts.erase(first, last);

  if (pts.size() <= 2U) {
    return pts;
  }

  std::vector<std::pair<int, int>> hull;
  hull.reserve(pts.size() + 1);

  for (const auto &p : pts) {
    while (hull.size() >= 2) {
      const auto &a = hull[hull.size() - 2];
      const auto &b = hull[hull.size() - 1];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(p.second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(p.first - a.first));
      if (cross > 0) {
        break;
      }
      hull.pop_back();
    }
    hull.push_back(p);
  }

  std::size_t lower_len = hull.size();
  for (auto it = pts.rbegin() + 1; it != pts.rend(); ++it) {
    while (hull.size() > lower_len) {
      const auto &a = hull[hull.size() - 2];
      const auto &b = hull[hull.size() - 1];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(it->second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(it->first - a.first));
      if (cross > 0) {
        break;
      }
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (hull.size() > 1) {
    hull.pop_back();
  }
  return hull;
}

}  // namespace chernov_t_convex_hull_binary_components
