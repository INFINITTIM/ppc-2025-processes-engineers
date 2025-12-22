#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace chernov_t_convex_hull_binary_components {

ChernovTConvexHullBinaryComponentsMPI::ChernovTConvexHullBinaryComponentsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool ChernovTConvexHullBinaryComponentsMPI::ValidationImpl() {
  if (rank_ == 0) {
    const auto &[w, h, p] = GetInput();
    valid_ = (w > 0) && (h > 0) && (p.size() == static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (valid_) {
      for (int px : p) {
        if (px != 0 && px != 1) {
          valid_ = false;
          break;
        }
      }
    }
  }
  MPI_Bcast(&valid_, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return valid_;
}

bool ChernovTConvexHullBinaryComponentsMPI::PreProcessingImpl() {
  if (!valid_) {
    return false;
  }

  std::array<int, 2> dims{0, 0};
  if (rank_ == 0) {
    dims[0] = std::get<0>(GetInput());
    dims[1] = std::get<1>(GetInput());
  }
  MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  width_ = dims[0];
  height_ = dims[1];

  int base_rows = height_ / size_;
  int rem = height_ % size_;
  start_row_ = (rank_ * base_rows) + std::min(rank_, rem);
  end_row_ = start_row_ + base_rows + (rank_ < rem ? 1 : 0);
  int local_rows = end_row_ - start_row_;

  if (rank_ == 0) {
    const auto &pixels = std::get<2>(GetInput());
    std::vector<int> sendcounts(size_, 0);
    std::vector<int> displs(size_, 0);
    int offset = 0;
    for (int i = 0; i < size_; ++i) {
      int rows_i = base_rows + (i < rem ? 1 : 0);
      sendcounts[i] = rows_i * width_;
      displs[i] = offset;
      offset += sendcounts[i];
    }
    local_pixels_.resize(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(width_));
    MPI_Scatterv(pixels.data(), sendcounts.data(), displs.data(), MPI_INT, local_pixels_.data(),
                 static_cast<int>(local_pixels_.size()), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    local_pixels_.resize(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(width_));
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_pixels_.data(), static_cast<int>(local_pixels_.size()),
                 MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool ChernovTConvexHullBinaryComponentsMPI::RunImpl() {
  if (!valid_) {
    GetOutput() = OutType{};
    return true;
  }

  FindConnectedComponentsMpi();
  ComputeConvexHulls();
  GatherAndBroadcastResult();

  return true;
}

bool ChernovTConvexHullBinaryComponentsMPI::PostProcessingImpl() {
  local_pixels_.clear();
  local_hulls_.clear();
  return true;
}

void ChernovTConvexHullBinaryComponentsMPI::SendReceiveNeighborRows(bool has_top, bool has_bottom,
                                                                    std::vector<int> &top_recv,
                                                                    std::vector<int> &bottom_recv) {
  std::array<MPI_Request, 4> reqs{};
  int req_count = 0;

  if (has_top) {
    top_recv.resize(static_cast<std::size_t>(width_));
    std::vector<int> top_send(local_pixels_.begin(), local_pixels_.begin() + width_);
    MPI_Irecv(top_recv.data(), width_, MPI_INT, rank_ - 1, 1, MPI_COMM_WORLD,
              &reqs[static_cast<std::size_t>(req_count++)]);
    MPI_Isend(top_send.data(), width_, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD,
              &reqs[static_cast<std::size_t>(req_count++)]);
  }
  if (has_bottom && rank_ + 1 < size_) {
    bottom_recv.resize(static_cast<std::size_t>(width_));
    std::vector<int> bottom_send(local_pixels_.end() - width_, local_pixels_.end());
    MPI_Irecv(bottom_recv.data(), width_, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD,
              &reqs[static_cast<std::size_t>(req_count++)]);
    MPI_Isend(bottom_send.data(), width_, MPI_INT, rank_ + 1, 1, MPI_COMM_WORLD,
              &reqs[static_cast<std::size_t>(req_count++)]);
  }

  if (req_count > 0) {
    std::vector<MPI_Status> statuses(static_cast<std::size_t>(req_count));
    MPI_Waitall(req_count, reqs.data(), statuses.data());
  }
}

void ChernovTConvexHullBinaryComponentsMPI::ProcessComponentQueue(std::queue<std::pair<int, int>> &q,
                                                                  const std::vector<int> &extended_pixels,
                                                                  std::vector<std::vector<bool>> &visited,
                                                                  std::vector<std::pair<int, int>> &comp) {
  const std::array<int, 4> dx = {0, 0, -1, 1};
  const std::array<int, 4> dy = {-1, 1, 0, 0};

  while (!q.empty()) {
    auto [cx, cy] = q.front();
    q.pop();
    comp.emplace_back(cx, cy);

    for (int direction = 0; direction < 4; ++direction) {
      int nx = cx + dx[static_cast<std::size_t>(direction)];
      int ny = cy + dy[static_cast<std::size_t>(direction)];
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < static_cast<int>(visited.size())) {
        std::size_t neighbor_idx =
            static_cast<std::size_t>(ny) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(nx);
        if (extended_pixels[neighbor_idx] == 1 &&
            !visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)]) {
          visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)] = true;
          q.emplace(nx, ny);
        }
      }
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::ProcessPixelComponent(int col, int ey, int global_y_offset,
                                                                  const std::vector<int> &extended_pixels,
                                                                  std::vector<std::vector<bool>> &visited) {
  std::size_t pixel_idx =
      (static_cast<std::size_t>(ey) * static_cast<std::size_t>(width_)) + static_cast<std::size_t>(col);
  if (extended_pixels[pixel_idx] == 1 && !visited[static_cast<std::size_t>(ey)][static_cast<std::size_t>(col)]) {
    std::vector<std::pair<int, int>> comp;
    std::queue<std::pair<int, int>> q;
    q.emplace(col, ey);
    visited[static_cast<std::size_t>(ey)][static_cast<std::size_t>(col)] = true;

    ProcessComponentQueue(q, extended_pixels, visited, comp);

    std::vector<std::pair<int, int>> local_comp;
    for (const auto &[px, py] : comp) {
      int global_py = global_y_offset + py;
      if (global_py >= start_row_ && global_py < end_row_) {
        local_comp.emplace_back(px, global_py);
      }
    }

    if (!local_comp.empty()) {
      std::ranges::sort(local_comp);
      auto last = std::ranges::unique(local_comp);
      local_comp.erase(last.begin(), last.end());
      local_hulls_.push_back(std::move(local_comp));
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::FindConnectedComponentsMpi() {
  int local_rows = end_row_ - start_row_;
  if (local_rows == 0) {
    return;
  }

  bool has_top = (start_row_ > 0);
  bool has_bottom = (end_row_ < height_);
  int extended_rows = local_rows + (has_top ? 1 : 0) + (has_bottom ? 1 : 0);
  std::vector<int> extended_pixels(static_cast<std::size_t>(extended_rows) * static_cast<std::size_t>(width_), 0);

  int offset = has_top ? 1 : 0;
  for (int local_row = 0; local_row < local_rows; ++local_row) {
    std::ranges::copy_n(
        local_pixels_.begin() + (static_cast<std::ptrdiff_t>(local_row) * static_cast<std::ptrdiff_t>(width_)), width_,
        extended_pixels.begin() +
            (static_cast<std::ptrdiff_t>(offset + local_row) * static_cast<std::ptrdiff_t>(width_)));
  }

  std::vector<int> top_recv, bottom_recv;
  SendReceiveNeighborRows(has_top, has_bottom, top_recv, bottom_recv);

  if (has_top) {
    std::ranges::copy(top_recv, extended_pixels.begin());
  }
  if (has_bottom && rank_ + 1 < size_) {
    std::ranges::copy(bottom_recv, extended_pixels.end() - static_cast<std::ptrdiff_t>(width_));
  }

  int global_y_offset = start_row_ - (has_top ? 1 : 0);
  std::vector<std::vector<bool>> visited(static_cast<std::size_t>(extended_rows),
                                         std::vector<bool>(static_cast<std::size_t>(width_), false));

  for (int extended_y = 0; extended_y < extended_rows; ++extended_y) {
    for (int column = 0; column < width_; ++column) {
      ProcessPixelComponent(column, extended_y, global_y_offset, extended_pixels, visited);
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::ComputeConvexHulls() {
  for (auto &comp : local_hulls_) {
    comp = ConvexHull(std::move(comp));
  }
}

bool ChernovTConvexHullBinaryComponentsMPI::Clockwise(const std::pair<int, int> &a, const std::pair<int, int> &b,
                                                      const std::pair<int, int> &c) {
  std::int64_t cross = static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(c.second - a.second) -
                       static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(c.first - a.first);
  return cross > 0;
}

void ChernovTConvexHullBinaryComponentsMPI::BuildLowerHull(std::vector<std::pair<int, int>> &pts,
                                                           std::vector<std::pair<int, int>> &hull) {
  std::size_t k = 0;
  for (auto &pt : pts) {
    while (k >= 2) {
      auto &a = hull[k - 2];
      auto &b = hull[k - 1];
      std::int64_t cross =
          static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(pt.second - a.second) -
          static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(pt.first - a.first);
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

void ChernovTConvexHullBinaryComponentsMPI::BuildUpperHull(std::vector<std::pair<int, int>> &pts,
                                                           std::vector<std::pair<int, int>> &hull,
                                                           std::size_t lower_len) {
  std::size_t k = hull.size();
  for (auto it = pts.rbegin() + 1; it != pts.rend(); ++it) {
    auto &pt = *it;
    while (k > lower_len) {
      auto &a = hull[k - 2];
      auto &b = hull[k - 1];
      std::int64_t cross =
          static_cast<std::int64_t>(b.first - a.first) * static_cast<std::int64_t>(pt.second - a.second) -
          static_cast<std::int64_t>(b.second - a.second) * static_cast<std::int64_t>(pt.first - a.first);
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

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsMPI::ConvexHull(
    std::vector<std::pair<int, int>> pts) {
  if (pts.size() <= 1) {
    return pts;
  }

  std::ranges::sort(pts);
  auto last = std::ranges::unique(pts);
  pts.erase(last.begin(), last.end());

  if (pts.size() == 1) {
    return pts;
  }
  if (pts.size() == 2) {
    if (pts[0] == pts[1]) {
      return {pts[0]};
    }
    return pts;
  }

  std::vector<std::pair<int, int>> hull;
  BuildLowerHull(pts, hull);
  BuildUpperHull(pts, hull, hull.size() + 1);

  if (hull.size() > 1) {
    hull.pop_back();
  }
  return hull;
}

void ChernovTConvexHullBinaryComponentsMPI::GatherWorkerData(int src, std::vector<int> &all_sizes,
                                                             std::vector<int> &global_flat) {
  int count = 0;
  MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (count > 0) {
    std::vector<int> sizes(static_cast<std::size_t>(count));
    MPI_Recv(sizes.data(), count, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    all_sizes.insert(all_sizes.end(), sizes.begin(), sizes.end());

    int pts = 0;
    MPI_Recv(&pts, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (pts > 0) {
      std::vector<int> pts_data(static_cast<std::size_t>(pts) * 2);
      MPI_Recv(pts_data.data(), pts * 2, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_flat.insert(global_flat.end(), pts_data.begin(), pts_data.end());
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::SendLocalData() {
  std::vector<int> local_flat;
  std::vector<int> local_sizes;
  for (const auto &hull : local_hulls_) {
    local_sizes.push_back(static_cast<int>(hull.size()));
    for (const auto &p : hull) {
      local_flat.push_back(p.first);
      local_flat.push_back(p.second);
    }
  }

  int count = static_cast<int>(local_sizes.size());
  MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  if (count > 0) {
    MPI_Send(local_sizes.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD);
    int pts = static_cast<int>(local_flat.size() / 2);
    MPI_Send(&pts, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    if (pts > 0) {
      MPI_Send(local_flat.data(), static_cast<int>(local_flat.size()), MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::ReconstructGlobalHulls(std::vector<int> &all_sizes, std::vector<int> &flat,
                                                                   OutType &global_hulls) {
  std::size_t idx = 0;
  for (int sz : all_sizes) {
    std::vector<std::pair<int, int>> hull(static_cast<std::size_t>(sz));
    for (int j = 0; j < sz; ++j, idx += 2) {
      hull[static_cast<std::size_t>(j)] = {flat[idx], flat[idx + 1]};
    }
    global_hulls.push_back(std::move(hull));
  }
}

void ChernovTConvexHullBinaryComponentsMPI::GatherAndBroadcastResult() {
  std::vector<int> local_flat;
  std::vector<int> local_sizes;
  for (const auto &hull : local_hulls_) {
    local_sizes.push_back(static_cast<int>(hull.size()));
    for (const auto &p : hull) {
      local_flat.push_back(p.first);
      local_flat.push_back(p.second);
    }
  }

  OutType global_hulls;

  if (rank_ == 0) {
    std::vector<int> all_sizes = local_sizes;
    std::vector<int> global_flat = local_flat;

    for (int src = 1; src < size_; ++src) {
      GatherWorkerData(src, all_sizes, global_flat);
    }

    ReconstructGlobalHulls(all_sizes, global_flat, global_hulls);
  } else {
    SendLocalData();
  }

  int total_hulls = static_cast<int>(global_hulls.size());
  MPI_Bcast(&total_hulls, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> all_sizes(static_cast<std::size_t>(total_hulls));
  if (rank_ == 0) {
    for (int i = 0; i < total_hulls; ++i) {
      all_sizes[static_cast<std::size_t>(i)] = static_cast<int>(global_hulls[static_cast<std::size_t>(i)].size());
    }
  }
  MPI_Bcast(all_sizes.data(), total_hulls, MPI_INT, 0, MPI_COMM_WORLD);

  int total_pts = 0;
  for (int s : all_sizes) {
    total_pts += s;
  }
  MPI_Bcast(&total_pts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat(static_cast<std::size_t>(total_pts) * 2);
  if (rank_ == 0) {
    flat.clear();
    for (const auto &h : global_hulls) {
      for (const auto &p : h) {
        flat.push_back(p.first);
        flat.push_back(p.second);
      }
    }
  }
  MPI_Bcast(flat.data(), total_pts * 2, MPI_INT, 0, MPI_COMM_WORLD);

  global_hulls.clear();
  ReconstructGlobalHulls(all_sizes, flat, global_hulls);

  GetOutput() = std::move(global_hulls);
}

}  // namespace chernov_t_convex_hull_binary_components
