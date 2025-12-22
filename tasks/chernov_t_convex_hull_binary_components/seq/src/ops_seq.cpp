// tasks/chernov_t_convex_hull_binary_components/seq/src/ops_seq.cpp
#include "chernov_t_convex_hull_binary_components/seq/include/ops_seq.hpp"
#include <queue>
#include <vector>
#include <algorithm>

namespace chernov_t_convex_hull_binary_components {

ChernovTConvexHullBinaryComponentsSEQ::ChernovTConvexHullBinaryComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ChernovTConvexHullBinaryComponentsSEQ::ValidationImpl() {
  const auto &[width, height, pixels] = GetInput();
  if (width <= 0 || height <= 0) return false;
  if (pixels.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) return false;
  for (int p : pixels) {
    if (p != 0 && p != 1) return false;
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
  return !GetOutput().empty() || true;
}

std::vector<std::vector<std::pair<int, int>>>
ChernovTConvexHullBinaryComponentsSEQ::FindConnectedComponents(
    int width, int height, const std::vector<int>& pixels) {
  std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> components;
  const int dx[4] = {0, 0, -1, 1};
  const int dy[4] = {-1, 1, 0, 0};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (pixels[y * width + x] == 1 && !visited[y][x]) {
        std::vector<std::pair<int, int>> comp;
        std::queue<std::pair<int, int>> q;
        q.emplace(x, y);
        visited[y][x] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();
          comp.emplace_back(cx, cy);
          for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                pixels[ny * width + nx] == 1 && !visited[ny][nx]) {
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

bool ChernovTConvexHullBinaryComponentsSEQ::Clockwise(
    const std::pair<int, int>& a,
    const std::pair<int, int>& b,
    const std::pair<int, int>& c) {
  long long cross = static_cast<long long>(b.first - a.first) * (c.second - a.second) -
                    static_cast<long long>(b.second - a.second) * (c.first - a.first);
  return cross > 0;
}

std::vector<std::pair<int, int>>
ChernovTConvexHullBinaryComponentsSEQ::ConvexHull(std::vector<std::pair<int, int>> pts) {
  if (pts.size() <= 1) return pts;
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  if (pts.size() == 1) return pts;
  if (pts.size() == 2) return pts;

  std::vector<std::pair<int, int>> hull;

  for (int i = 0; i < (int)pts.size(); ++i) {
    while (hull.size() >= 2) {
      auto& a = hull[hull.size() - 2];
      auto& b = hull.back();
      auto& c = pts[i];
      long long cross = 1LL * (b.first - a.first) * (c.second - a.second) -
                        1LL * (b.second - a.second) * (c.first - a.first);
      if (cross < 0) break;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
  }

  int lower_len = hull.size();
  for (int i = (int)pts.size() - 2; i >= 0; --i) {
    while ((int)hull.size() > lower_len) {
      auto& a = hull[hull.size() - 2];
      auto& b = hull.back();
      auto& c = pts[i];
      long long cross = 1LL * (b.first - a.first) * (c.second - a.second) -
                        1LL * (b.second - a.second) * (c.first - a.first);
      if (cross < 0) break;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
  }

  if (hull.size() > 1) hull.pop_back();
  return hull;
}

}  // namespace chernov_t_convex_hull_binary_components
