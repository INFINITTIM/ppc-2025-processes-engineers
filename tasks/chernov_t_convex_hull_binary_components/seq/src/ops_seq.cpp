#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <queue>
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

std::vector<std::vector<std::pair<int, int>>> ChernovTConvexHullBinaryComponentsSEQ::FindConnectedComponents(
    int width, int height, const std::vector<int> &pixels) {
  std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> components;
  const std::array<int, 4> dx = {0, 0, -1, 1};
  const std::array<int, 4> dy = {-1, 1, 0, 0};

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      std::size_t idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col);
      if (pixels[idx] == 1 && !visited[row][col]) {
        std::vector<std::pair<int, int>> comp;
        std::queue<std::pair<int, int>> q;
        q.emplace(col, row);
        visited[row][col] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();
          comp.emplace_back(cx, cy);

          for (int dir = 0; dir < 4; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              std::size_t nidx =
                  static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx);
              if (pixels[nidx] == 1 && !visited[ny][nx]) {
                visited[ny][nx] = true;
                q.emplace(nx, ny);
              }
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
  for (const auto &pt : pts) {
    while (k >= 2U) {
      const auto &a = hull[k - 2];
      const auto &b = hull[k - 1];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(pt.second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(pt.first - a.first));
      if (cross > 0) {
        break;
      }
      --k;
      hull.pop_back();
    }
    hull.push_back(pt);
    ++k;
  }
}

void ChernovTConvexHullBinaryComponentsSEQ::BuildUpperHull(std::vector<std::pair<int, int>> &hull,
                                                           const std::vector<std::pair<int, int>> &pts) {
  std::size_t k = hull.size();
  std::size_t t = k + 1;
  for (std::size_t i = pts.size(); i-- > 0 && i > 0;) {
    const auto &pt = pts[i];
    while (k >= t) {
      const auto &a = hull[k - 2];
      const auto &b = hull[k - 1];
      std::int64_t cross =
          (static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(pt.second - a.second)) -
          (static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(pt.first - a.first));
      if (cross > 0) {
        break;
      }
      --k;
      hull.pop_back();
    }
    hull.push_back(pt);
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

  std::ranges::sort(pts);
  auto [first, last] = std::ranges::unique(pts);
  pts.erase(first, last);

  if (pts.size() <= 2U) {
    return pts;
  }

  std::vector<std::pair<int, int>> hull;
  BuildLowerHull(hull, pts);
  BuildUpperHull(hull, pts);
  return hull;
}

}  // namespace chernov_t_convex_hull_binary_components
