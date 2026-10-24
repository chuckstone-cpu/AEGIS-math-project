// ============================================================================
// apex/lidar.hpp  -  Obstacle extraction from a planar scan + lead tracking
//
// Syllabus (Unit 3): multivariate Gaussian -> 2-state Kalman filter.
//
// 1) Path corridor: the car sweeps the parabola y = -kappa x^2 / 2 (bicycle
//    model, kappa = tan(delta) / L); only returns within +-half_width of it
//    can be obstacles.
// 2) Scan clustering: from the nearest in-corridor return, region-grow along
//    the scan while neighbouring returns are < link_dist apart. A car, cone,
//    barrel or pedestrian is a SHORT segment; the road surface struck by a
//    pitching/rolling scanner, a guard-rail or a building face is a LONG one.
//    Segments longer than max_len are rejected and the next candidate tried.
// 3) Lead tracking: constant-velocity Kalman filter on s = [gap, gap_rate],
//       s_k = F s_{k-1} + w,  F = [1 dt; 0 1],  w ~ N(0, Q),
//       Q = sigma_a^2 [dt^4/4 dt^3/2; dt^3/2 dt^2]  (white-acceleration model)
//       z_k = H s_k + v,      H = [1 0],          v ~ N(0, sigma_r^2)
//    Predict: s = F s, P = F P F^T + Q.  Update: K = P H^T / (H P H^T + R),
//    s += K (z - H s), P = (I - K H) P. The closing speed is -gap_rate, with
//    variance P[1][1]; no finite differencing of a noisy range.
// ============================================================================
#pragma once

#include <cmath>

namespace apex {

// Predicted path the car will sweep, in the lidar frame (x fwd, y left).
// Preferred: the IPM lane-centre fit (follows the road curvature ahead);
// fallback: the bicycle-model arc of the current steering angle.
struct Corridor {
  float kappa = 0.0f;       // arc curvature (fallback)
  bool use_lane = false;    // lane-centre polynomial available
  double c[3] = {0, 0, 0};  // Y_c(t), t = X_cam / scale
  float scale = 20.0f;
  float cam_offset = 2.38f; // X_cam = x_lidar + cam_offset
  float lane_max_x = 40.0f; // trust the fit only where it has data
  float center(float x) const {
    if (use_lane && x + cam_offset <= lane_max_x) {
      const double t = (x + cam_offset) / scale;
      return static_cast<float>(c[0] + c[1] * t + c[2] * t * t);
    }
    return -0.5f * kappa * x * x;
  }
};

struct ScanObstacle {
  bool found = false;
  float x = 100.0f, y = 0.0f; // nearest point, sensor frame (x fwd, y left)
  float length = 0.0f;        // cluster extent [m]
  int rejected_long = 0;      // long segments discarded (road strike / walls)
};

// ranges: n returns ordered LEFT -> RIGHT over horizontal field of view fov.
inline ScanObstacle find_corridor_obstacle(const float *ranges, int n, float fov,
                                           const Corridor &cor, float half_width,
                                           float max_range = 40.0f, float min_x = 1.0f,
                                           float link_dist = 0.6f, float max_len = 6.0f) {
  ScanObstacle best;
  constexpr int kMax = 1024;
  if (n <= 1 || n > kMax)
    return best;
  float xs[kMax], ys[kMax];
  bool valid[kMax], in_corr[kMax], used[kMax];
  for (int i = 0; i < n; ++i) {
    const float r = ranges[i];
    valid[i] = std::isfinite(r) && r > 0.4f && r < 79.0f;
    const float a = 0.5f * fov - fov * i / (n - 1);
    xs[i] = valid[i] ? r * std::cos(a) : 0.0f;
    ys[i] = valid[i] ? r * std::sin(a) : 0.0f;
    // closer than min_x is the car's own bodywork, not an obstacle
    in_corr[i] = valid[i] && xs[i] > min_x && xs[i] < max_range &&
                 std::fabs(ys[i] - cor.center(xs[i])) < half_width;
    used[i] = false;
  }
  for (int attempt = 0; attempt < 8; ++attempt) {
    int k = -1;
    for (int i = 0; i < n; ++i)
      if (in_corr[i] && !used[i] && (k < 0 || xs[i] < xs[k]))
        k = i;
    if (k < 0)
      return best;
    // region growing along the scan
    int lo = k, hi = k;
    while (lo > 0 && valid[lo - 1] &&
           std::hypot(xs[lo - 1] - xs[lo], ys[lo - 1] - ys[lo]) < link_dist)
      --lo;
    while (hi < n - 1 && valid[hi + 1] &&
           std::hypot(xs[hi + 1] - xs[hi], ys[hi + 1] - ys[hi]) < link_dist)
      ++hi;
    for (int i = lo; i <= hi; ++i)
      used[i] = true;
    const float len = std::hypot(xs[hi] - xs[lo], ys[hi] - ys[lo]);
    if (len > max_len) {
      ++best.rejected_long;
      continue;
    }
    best.found = true;
    best.x = xs[k];
    best.y = ys[k];
    best.length = len;
    return best;
  }
  return best;
}

class LeadKalman {
public:
  float sigma_a = 2.0f;  // process accel noise [m/s^2]
  float sigma_r = 0.08f; // range noise [m]
  float gap = 100.0f, rate = 0.0f;
  float P[2][2] = {{1, 0}, {0, 4}};
  bool active = false;
  float lost_time = 0.0f;
  int confirm_needed = 3; // M-of-N track initiation
  float cand = -1.0f;
  int cand_hits = 0;

  void reset(float z) {
    gap = z;
    rate = 0.0f;
    P[0][0] = sigma_r * sigma_r;
    P[0][1] = P[1][0] = 0.0f;
    P[1][1] = 25.0f; // closing speed unknown at first sight: sigma 5 m/s
    active = true;
    lost_time = 0.0f;
  }

  // one filter step; has_meas = obstacle seen this frame
  // a new object must be seen in confirm_needed consecutive, consistent
  // frames before it is believed (rejects one-frame glitches)
  bool confirm(bool has_meas, float z) {
    if (!has_meas) {
      cand_hits = 0;
      return false;
    }
    if (cand_hits > 0 && std::fabs(z - cand) < 1.0f + 0.1f * z)
      ++cand_hits;
    else
      cand_hits = 1;
    cand = z;
    if (cand_hits >= confirm_needed) {
      cand_hits = 0;
      return true;
    }
    return false;
  }

  void step(float dt, bool has_meas, float z, float max_coast = 0.4f) {
    if (!active) {
      if (confirm(has_meas, z))
        reset(z);
      return;
    }
    // predict
    gap += dt * rate;
    const float q = sigma_a * sigma_a;
    const float dt2 = dt * dt;
    const float p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt2 * P[1][1] + q * dt2 * dt2 / 4;
    const float p01 = P[0][1] + dt * P[1][1] + q * dt2 * dt / 2;
    const float p11 = P[1][1] + q * dt2;
    P[0][0] = p00;
    P[0][1] = P[1][0] = p01;
    P[1][1] = p11;
    if (!has_meas) {
      lost_time += dt;
      if (lost_time > max_coast)
        active = false;
      return;
    }
    lost_time = 0.0f;
    // gate: a jump of more than 4 sigma is a different object -> re-initialise
    const float S = P[0][0] + sigma_r * sigma_r;
    const float innov = z - gap;
    if (innov * innov > 16.0f * S + 1.0f) {
      // inconsistent with the track: coast on the prediction, and switch to
      // the new object only once it has been confirmed
      lost_time += dt;
      if (confirm(true, z))
        reset(z);
      else if (lost_time > max_coast)
        active = false;
      return;
    }
    cand_hits = 0;
    const float K0 = P[0][0] / S, K1 = P[1][0] / S;
    gap += K0 * innov;
    rate += K1 * innov;
    const float n00 = (1 - K0) * P[0][0], n01 = (1 - K0) * P[0][1];
    const float n11 = P[1][1] - K1 * P[0][1];
    P[0][0] = n00;
    P[0][1] = P[1][0] = n01;
    P[1][1] = n11;
  }
  float closing_speed() const { return -rate; }
};

} // namespace apex
