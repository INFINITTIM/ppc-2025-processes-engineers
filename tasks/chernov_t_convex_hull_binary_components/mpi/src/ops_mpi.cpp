#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "chernov_t_convex_hull_binary_components/common/include/common.hpp"

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
    std::size_t src = static_cast<std::size_t>(local_row) * static_cast<std::size_t>(width_);
    std::size_t dst = static_cast<std::size_t>(offset + local_row) * static_cast<std::size_t>(width_);
    std::ranges::copy(local_pixels_.begin() + static_cast<std::ptrdiff_t>(src),
                      local_pixels_.begin() + static_cast<std::ptrdiff_t>(src + width_),
                      extended_pixels.begin() + static_cast<std::ptrdiff_t>(dst));
  }

  ExchangeBoundaryRows(has_top, has_bottom, extended_pixels, width_);

  int global_y_offset = start_row_ - (has_top ? 1 : 0);
  auto all_components = ProcessExtendedRegion(extended_pixels, extended_rows, width_, global_y_offset);
  FilterLocalComponents(all_components);
}

void ChernovTConvexHullBinaryComponentsMPI::ExchangeBoundaryRows(bool has_top, bool has_bottom,
                                                                 std::vector<int> &extended_pixels, int width) {
  std::vector<MPI_Request> reqs(4, MPI_REQUEST_NULL);
  std::vector<MPI_Status> statuses(4);
  int req_count = 0;

  std::vector<int> top_recv;
  std::vector<int> bottom_recv;

  if (has_top) {
    top_recv.resize(width);
    std::vector<int> top_send(local_pixels_.begin(), local_pixels_.begin() + width);
    MPI_Irecv(top_recv.data(), width, MPI_INT, rank_ - 1, 1, MPI_COMM_WORLD, &reqs[req_count]);
    MPI_Isend(top_send.data(), width, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD, &reqs[req_count + 1]);
    req_count += 2;
  }
  if (has_bottom && rank_ + 1 < size_) {
    bottom_recv.resize(width);
    std::vector<int> bottom_send(local_pixels_.end() - width, local_pixels_.end());
    MPI_Irecv(bottom_recv.data(), width, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD, &reqs[req_count]);
    MPI_Isend(bottom_send.data(), width, MPI_INT, rank_ + 1, 1, MPI_COMM_WORLD, &reqs[req_count + 1]);
    req_count += 2;
  }
  if (req_count > 0) {
    MPI_Waitall(req_count, reqs.data(), statuses.data());
  }
  if (has_top) {
    std::ranges::copy(top_recv, extended_pixels.begin());
  }
  if (has_bottom && rank_ + 1 < size_) {
    std::ranges::copy(bottom_recv, extended_pixels.end() - width);
  }
}

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsMPI::ExtractComponent(
    int start_col, int start_ey, const std::vector<int> &extended_pixels, std::vector<std::vector<bool>> &visited,
    int width, int extended_rows, int global_y_offset) {
  std::vector<std::pair<int, int>> comp;
  std::queue<std::pair<int, int>> q;
  q.emplace(start_col, start_ey);
  visited[static_cast<std::size_t>(start_ey)][static_cast<std::size_t>(start_col)] = true;

  constexpr std::array<std::pair<int, int>, 4> kDirs = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
  while (!q.empty()) {
    auto [cx, cy] = q.front();
    q.pop();
    comp.emplace_back(cx, global_y_offset + cy);
    for (const auto &[dx, dy] : kDirs) {
      int nx = cx + dx;
      int ny = cy + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < extended_rows) {
        std::size_t nidx =
            (static_cast<std::size_t>(ny) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(nx);
        if (extended_pixels[nidx] == 1 && !visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)]) {
          visited[static_cast<std::size_t>(ny)][static_cast<std::size_t>(nx)] = true;
          q.emplace(nx, ny);
        }
      }
    }
  }
  return comp;
}

std::vector<std::vector<std::pair<int, int>>> ChernovTConvexHullBinaryComponentsMPI::ProcessExtendedRegion(
    const std::vector<int> &extended_pixels, int extended_rows, int width, int global_y_offset) {
  std::vector<std::vector<bool>> visited(extended_rows, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> all_components;
  for (int ey = 0; ey < extended_rows; ++ey) {
    for (int col = 0; col < width; ++col) {
      std::size_t idx =
          (static_cast<std::size_t>(ey) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col);
      if (extended_pixels[idx] == 1 && !visited[static_cast<std::size_t>(ey)][static_cast<std::size_t>(col)]) {
        auto comp = ExtractComponent(col, ey, extended_pixels, visited, width, extended_rows, global_y_offset);
        all_components.push_back(std::move(comp));
      }
    }
  }
  return all_components;
}

void ChernovTConvexHullBinaryComponentsMPI::FilterLocalComponents(
    const std::vector<std::vector<std::pair<int, int>>> &all_components) {
  for (const auto &comp : all_components) {
    std::vector<std::pair<int, int>> local_comp;
    for (auto [px, py] : comp) {
      if (py >= start_row_ && py < end_row_) {
        local_comp.emplace_back(px, py);
      }
    }
    if (!local_comp.empty()) {
      std::ranges::sort(local_comp);
      auto [first, last] = std::ranges::unique(local_comp);
      local_comp.erase(first, last);
      local_hulls_.push_back(std::move(local_comp));
    }
  }
}

std::vector<std::pair<int, int>> ChernovTConvexHullBinaryComponentsMPI::ConvexHull(
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

void ChernovTConvexHullBinaryComponentsMPI::ComputeConvexHulls() {
  for (auto &comp : local_hulls_) {
    comp = ConvexHull(comp);
  }
}

void ChernovTConvexHullBinaryComponentsMPI::GatherHullsOnRank0(std::vector<int> &all_sizes,
                                                               std::vector<int> &global_flat) {
  all_sizes = std::vector<int>(local_hulls_.size());
  global_flat.clear();
  for (size_t i = 0; i < local_hulls_.size(); ++i) {
    all_sizes[i] = static_cast<int>(local_hulls_[i].size());
    for (const auto &p : local_hulls_[i]) {
      global_flat.push_back(p.first);
      global_flat.push_back(p.second);
    }
  }

  for (int src = 1; src < size_; ++src) {
    ReceiveHullsFromRank(src, all_sizes, global_flat);
  }
}

void ChernovTConvexHullBinaryComponentsMPI::BroadcastResultToAllRanks(
    const std::vector<std::vector<std::pair<int, int>>> &global_hulls) {
  int total_hulls = static_cast<int>(global_hulls.size());
  MPI_Bcast(&total_hulls, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> all_sizes(total_hulls);
  if (rank_ == 0) {
    for (int i = 0; i < total_hulls; ++i) {
      all_sizes[i] = static_cast<int>(global_hulls[static_cast<std::size_t>(i)].size());
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

  OutType result;
  std::size_t idx = 0;
  for (int i = 0; i < total_hulls; ++i) {
    int sz = all_sizes[i];
    std::vector<std::pair<int, int>> hull(static_cast<std::size_t>(sz));
    for (int j = 0; j < sz; ++j, idx += 2) {
      hull[static_cast<std::size_t>(j)] = {flat[idx], flat[idx + 1]};
    }
    result.push_back(std::move(hull));
  }
  GetOutput() = std::move(result);
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

  if (rank_ == 0) {
    std::vector<int> all_sizes;
    std::vector<int> global_flat;
    GatherHullsOnRank0(all_sizes, global_flat);

    OutType global_hulls;
    std::size_t idx = 0;
    for (int sz : all_sizes) {
      std::vector<std::pair<int, int>> hull(static_cast<std::size_t>(sz));
      for (int j = 0; j < sz; ++j, idx += 2) {
        hull[static_cast<std::size_t>(j)] = {global_flat[idx], global_flat[idx + 1]};
      }
      global_hulls.push_back(std::move(hull));
    }
    BroadcastResultToAllRanks(global_hulls);
  } else {
    SendHullsToRank0(local_flat, local_sizes);
    BroadcastResultToAllRanks({});
  }
}

void ChernovTConvexHullBinaryComponentsMPI::SendHullsToRank0(const std::vector<int> &local_flat,
                                                             const std::vector<int> &local_sizes) {
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

void ChernovTConvexHullBinaryComponentsMPI::ReceiveHullsFromRank(int src, std::vector<int> &all_sizes,
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

}  // namespace chernov_t_convex_hull_binary_components
