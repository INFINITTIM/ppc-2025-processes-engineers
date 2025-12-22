#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <queue>
#include <utility>
#include <vector>

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
  for (int p : pixels) {
    if (p != 0 && p != 1) {
      return false;
    }
  }
  return true;
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

std::vector<std::vector<std::pair<int, int>>> ChernovTConvexHullBinaryComponentsSEQ::FindConnectedComponents(
    int width, int height, const std::vector<int> &pixels) {
  std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> components;
  const std::array<int, 4> dx = {0, 0, -1, 1};
  const std::array<int, 4> dy = {-1, 1, 0, 0};

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col)] ==
              1 &&
          !visited[row][col]) {
        std::vector<std::pair<int, int>> comp;
        std::queue<std::pair<int, int>> q;
        q.emplace(col, row);
        visited[row][col] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();
          comp.emplace_back(cx, cy);
          for (int dir = 0; dir < 4; ++dir) {
            int nx = cx + dx.at(dir);
            int ny = cy + dy.at(dir);
            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                pixels[static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx)] ==
                    1 &&
                !visited[ny][nx]) {
              visited[ny][nx] = true;
              q.emplace(nx, ny);
            }
          }
        }
        components.push_back(std::move(comp));
      }
    }
  }
  return components;
}

void ChernovTConvexHullBinaryComponentsSEQ::BuildLowerHull(std::vector<std::pair<int, int>> &hull,
                                                           const std::vector<std::pair<int, int>> &pts) {
  std::size_t k = 0;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    while (k >= 2U) {
      const auto &a = hull[k - 2];
      const auto &b = hull[k - 1];
      const auto &c = pts[i];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(c.second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(c.first - a.first));
      if (cross > 0) {
        break;
      }
      --k;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
    ++k;
  }
}

void ChernovTConvexHullBinaryComponentsSEQ::BuildUpperHull(std::vector<std::pair<int, int>> &hull,
                                                           const std::vector<std::pair<int, int>> &pts) {
  std::size_t k = hull.size();
  std::size_t t = k + 1;
  for (std::size_t i = pts.size() - 2; i != static_cast<std::size_t>(-1); --i) {
    while (k >= t) {
      const auto &a = hull[k - 2];
      const auto &b = hull[k - 1];
      const auto &c = pts[i];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(c.second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(c.first - a.first));
      if (cross > 0) {
        break;
      }
      --k;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
    ++k;
  }
  if (hull.size() > 1U) {
    hull.pop_back();
  }
}

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsSEQ::ConvexHull(
    std::vector<std::pair<int, int>> pts) {
  if (pts.size() <= 1U) {
    return pts;
  }
  std::sort(pts.begin(), pts.end());
  auto last = std::unique(pts.begin(), pts.end());
  pts.erase(last, pts.end());
  if (pts.size() == 1U) {
    return pts;
  }
  if (pts.size() == 2U) {
    if (pts[0] == pts[1]) {
      return {pts[0]};
    }
    return pts;
  }

  std::vector<std::pair<int, int>> hull;
  BuildLowerHull(hull, pts);
  BuildUpperHull(hull, pts);
  return hull;
}

bool ChernovTConvexHullBinaryComponentsSEQ::Clockwise(const std::pair<int, int> &a, const std::pair<int, int> &b,
                                                      const std::pair<int, int> &c) {
  std::int64_t cross = (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(c.second - a.second)) -
                       (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(c.first - a.first));
  return cross > 0;
}

}  // namespace chernov_t_convex_hull_binary_components
