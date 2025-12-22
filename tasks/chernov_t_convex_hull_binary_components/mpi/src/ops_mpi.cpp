#include "chernov_t_convex_hull_binary_components/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <set>

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
    valid_ = (w > 0) && (h > 0) && (p.size() == static_cast<size_t>(w) * static_cast<size_t>(h));
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
  if (!valid_) return false;

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
  start_row_ = rank_ * base_rows + std::min(rank_, rem);
  end_row_ = start_row_ + base_rows + (rank_ < rem ? 1 : 0);
  int local_rows = end_row_ - start_row_;

  if (rank_ == 0) {
    const auto &pixels = std::get<2>(GetInput());
    std::vector<int> sendcounts(size_, 0), displs(size_, 0);
    int offset = 0;
    for (int i = 0; i < size_; ++i) {
      int rows_i = base_rows + (i < rem ? 1 : 0);
      sendcounts[i] = rows_i * width_;
      displs[i] = offset;
      offset += sendcounts[i];
    }
    local_pixels_.resize(local_rows * width_);
    MPI_Scatterv(pixels.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_pixels_.data(), local_pixels_.size(), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    local_pixels_.resize(local_rows * width_);
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                 local_pixels_.data(), local_pixels_.size(), MPI_INT, 0, MPI_COMM_WORLD);
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

void ChernovTConvexHullBinaryComponentsMPI::FindConnectedComponentsMpi() {
  int local_rows = end_row_ - start_row_;
  if (local_rows == 0) {
    return;
  }

  bool has_top = (start_row_ > 0);
  bool has_bottom = (end_row_ < height_);
  int extended_rows = local_rows + (has_top ? 1 : 0) + (has_bottom ? 1 : 0);
  std::vector<int> extended_pixels(extended_rows * width_, 0);

  int offset = has_top ? 1 : 0;
  for (int i = 0; i < local_rows; ++i) {
    std::copy_n(local_pixels_.begin() + i * width_, width_,
                extended_pixels.begin() + (offset + i) * width_);
  }

  MPI_Request reqs[4];
  int req_count = 0;
  std::vector<int> top_recv, bottom_recv;

  if (has_top) {
    top_recv.resize(width_);
    std::vector<int> top_send(local_pixels_.begin(), local_pixels_.begin() + width_);
    MPI_Irecv(top_recv.data(), width_, MPI_INT, rank_ - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(top_send.data(), width_, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
  }
  if (has_bottom) {
    bottom_recv.resize(width_);
    std::vector<int> bottom_send(local_pixels_.end() - width_, local_pixels_.end());
    MPI_Irecv(bottom_recv.data(), width_, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(bottom_send.data(), width_, MPI_INT, rank_ + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
  }

  if (req_count > 0) {
    std::vector<MPI_Status> statuses(req_count);
    MPI_Waitall(req_count, reqs, statuses.data());
  }

  if (has_top) {
    std::copy(top_recv.begin(), top_recv.end(), extended_pixels.begin());
  }
  if (has_bottom) {
    std::copy(bottom_recv.begin(), bottom_recv.end(), extended_pixels.end() - width_);
  }

  int global_y_offset = start_row_ - (has_top ? 1 : 0);
  std::vector<std::vector<bool>> visited(extended_rows, std::vector<bool>(width_, false));
  const int dx[4] = {0, 0, -1, 1};
  const int dy[4] = {-1, 1, 0, 0};

  for (int ey = 0; ey < extended_rows; ++ey) {
    for (int x = 0; x < width_; ++x) {
      if (extended_pixels[ey * width_ + x] == 1 && !visited[ey][x]) {
        std::vector<std::pair<int, int>> comp;
        std::queue<std::pair<int, int>> q;
        q.emplace(x, ey);
        visited[ey][x] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front(); q.pop();
          comp.emplace_back(cx, global_y_offset + cy);
          for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx >= 0 && nx < width_ && ny >= 0 && ny < extended_rows &&
                extended_pixels[ny * width_ + nx] == 1 && !visited[ny][nx]) {
              visited[ny][nx] = true;
              q.emplace(nx, ny);
            }
          }
        }

        std::vector<std::pair<int, int>> local_comp;
        for (auto [x, y] : comp) {
          if (y >= start_row_ && y < end_row_) {
            local_comp.emplace_back(x, y);
          }
        }
        if (!local_comp.empty()) {
          std::sort(local_comp.begin(), local_comp.end());
          local_comp.erase(std::unique(local_comp.begin(), local_comp.end()), local_comp.end());
          local_hulls_.push_back(std::move(local_comp));
        }
      }
    }
  }
}

void ChernovTConvexHullBinaryComponentsMPI::ComputeConvexHulls() {
  for (auto &comp : local_hulls_) {
    comp = ConvexHull(comp);
  }
}

bool ChernovTConvexHullBinaryComponentsMPI::Clockwise(
    const std::pair<int, int>& a,
    const std::pair<int, int>& b,
    const std::pair<int, int>& c) {
  long long cross = static_cast<long long>(b.first - a.first) * (c.second - a.second) -
                    static_cast<long long>(b.second - a.second) * (c.first - a.first);
  return cross > 0;
}

std::vector<std::pair<int, int>>
ChernovTConvexHullBinaryComponentsMPI::ConvexHull(std::vector<std::pair<int, int>> pts) {
  if (pts.size() <= 1) return pts;
  std::sort(pts.begin(), pts.end());
  auto last = std::unique(pts.begin(), pts.end());
  pts.erase(last, pts.end());
  if (pts.size() == 1) return pts;
  if (pts.size() == 2) {
    if (pts[0] == pts[1]) return {pts[0]};
    return pts;
  }

  std::vector<std::pair<int, int>> hull;
  size_t k = 0;

  // Lower hull
  for (size_t i = 0; i < pts.size(); ++i) {
    while (k >= 2) {
      auto& a = hull[k - 2];
      auto& b = hull[k - 1];
      auto& c = pts[i];
      long long cross = 1LL * (b.first - a.first) * (c.second - a.second) -
                        1LL * (b.second - a.second) * (c.first - a.first);
      if (cross > 0) break;
      --k;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
    ++k;
  }

  size_t t = k + 1;
  for (size_t i = pts.size() - 2; i != static_cast<size_t>(-1); --i) {
    while (k >= t) {
      auto& a = hull[k - 2];
      auto& b = hull[k - 1];
      auto& c = pts[i];
      long long cross = 1LL * (b.first - a.first) * (c.second - a.second) -
                        1LL * (b.second - a.second) * (c.first - a.first);
      if (cross > 0) break;
      --k;
      hull.pop_back();
    }
    hull.push_back(pts[i]);
    ++k;
  }

  if (hull.size() > 1) hull.pop_back();
  return hull;
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
      int count;
      MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (count > 0) {
        std::vector<int> sizes(count);
        MPI_Recv(sizes.data(), count, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        all_sizes.insert(all_sizes.end(), sizes.begin(), sizes.end());

        int pts;
        MPI_Recv(&pts, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (pts > 0) {
          std::vector<int> pts_data(pts * 2);
          MPI_Recv(pts_data.data(), pts * 2, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          global_flat.insert(global_flat.end(), pts_data.begin(), pts_data.end());
        }
      }
    }

    size_t idx = 0;
    for (int sz : all_sizes) {
      std::vector<std::pair<int, int>> hull(sz);
      for (int j = 0; j < sz; ++j, idx += 2) {
        hull[j] = {global_flat[idx], global_flat[idx+1]};
      }
      global_hulls.push_back(std::move(hull));
    }
  } else {
    int count = static_cast<int>(local_sizes.size());
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (count > 0) {
      MPI_Send(local_sizes.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD);
      int pts = static_cast<int>(local_flat.size() / 2);
      MPI_Send(&pts, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
      if (pts > 0) {
        MPI_Send(local_flat.data(), local_flat.size(), MPI_INT, 0, 3, MPI_COMM_WORLD);
      }
    }
  }

  int total_hulls = static_cast<int>(global_hulls.size());
  MPI_Bcast(&total_hulls, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> all_sizes(total_hulls);
  if (rank_ == 0) {
    for (int i = 0; i < total_hulls; ++i) {
      all_sizes[i] = static_cast<int>(global_hulls[i].size());
    }
  }
  MPI_Bcast(all_sizes.data(), total_hulls, MPI_INT, 0, MPI_COMM_WORLD);

  int total_pts = 0;
  for (int s : all_sizes) total_pts += s;
  MPI_Bcast(&total_pts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat(total_pts * 2);
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
  size_t idx = 0;
  for (int i = 0; i < total_hulls; ++i) {
    int sz = all_sizes[i];
    std::vector<std::pair<int, int>> hull(sz);
    for (int j = 0; j < sz; ++j, idx += 2) {
      hull[j] = {flat[idx], flat[idx+1]};
    }
    global_hulls.push_back(std::move(hull));
  }

  GetOutput() = std::move(global_hulls);
}

}  // namespace chernov_t_convex_hull_binary_components
