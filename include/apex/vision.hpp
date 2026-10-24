// ============================================================================
// apex/vision.hpp  -  Image processing and lane tracking, from scratch
//
// Syllabus mapping:
//   * Perona-Malik anisotropic diffusion  -> Unit 2: PDE, finite differences
//   * Rank-1 SVD background (power iter.) -> Unit 1: Krylov / eigen methods
//   * Robust WLS lane fit + Gaussian fuse -> Unit 3: WLS, multivariate Gaussian,
//                                            Chebyshev inequality
// OpenCV is NOT used in this file: images are raw interleaved BGR byte arrays.
// ============================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "stats.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apex {

// ----------------------------------------------------------------------------
// Fixed-point BGR -> luminance (ITU-R BT.601 weights scaled by 256)
// ----------------------------------------------------------------------------
inline void rgb_to_gray_fixed_point(const uint8_t *__restrict bgr,
                                    uint8_t *__restrict gray, int width,
                                    int height, int channels = 3) {
  const int total = width * height;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < total; ++i) {
    const int b = bgr[i * channels + 0];
    const int g = bgr[i * channels + 1];
    const int r = bgr[i * channels + 2];
    gray[i] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
  }
}

// ----------------------------------------------------------------------------
// Perona-Malik anisotropic diffusion PDE
//      dI/dt = div( c(|grad I|) grad I ),   c(s) = 1 / (1 + (s/K)^2)
// Explicit finite-difference scheme on the 4-neighbour stencil:
//      I^{n+1} = I^n + lambda * sum_{N,S,E,W} c(d) d
// von Neumann stability of the explicit 2-D scheme requires lambda <= 1/4.
// Boundary: homogeneous Neumann (border pixels copied, zero flux).
// ----------------------------------------------------------------------------
inline void perona_malik_diffusion(float *__restrict img,
                                   float *__restrict temp_buf, int width,
                                   int height, int iters = 2,
                                   float lambda = 0.2f, float K = 20.0f) {
  lambda = std::min(lambda, 0.25f); // enforce the CFL / stability bound
  const float inv_K_sq = 1.0f / (K * K);
  float *src = img;
  float *dst = temp_buf;
  std::memcpy(dst, src, sizeof(float) * width * height);

  for (int iter = 0; iter < iters; ++iter) {
#pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; ++y) {
      const int row = y * width;
      for (int x = 1; x < width - 1; ++x) {
        const int idx = row + x;
        const float I = src[idx];
        const float dN = src[idx - width] - I;
        const float dS = src[idx + width] - I;
        const float dE = src[idx + 1] - I;
        const float dW = src[idx - 1] - I;
        const float cN = 1.0f / (1.0f + dN * dN * inv_K_sq);
        const float cS = 1.0f / (1.0f + dS * dS * inv_K_sq);
        const float cE = 1.0f / (1.0f + dE * dE * inv_K_sq);
        const float cW = 1.0f / (1.0f + dW * dW * inv_K_sq);
        dst[idx] = I + lambda * (cN * dN + cS * dS + cE * dE + cW * dW);
      }
    }
    std::swap(src, dst);
  }
  if (src != img)
    std::memcpy(img, src, sizeof(float) * width * height);
}

// ----------------------------------------------------------------------------
// Rank-1 SVD by power iteration on the Gram matrix  G = A^T A  (M x M, M small)
// A is N x M: each column is one (downscaled) video frame. The dominant right
// singular vector v satisfies G v = sigma^2 v; then u = A v / sigma.
// The rank-1 reconstruction  u (u^T p)  is the static background; |p - bg| is
// the foreground (moving obstacles). v is warm-started from the previous call,
// so 4-6 iterations are enough frame-to-frame.
// ----------------------------------------------------------------------------
inline float power_iteration_svd(const float *__restrict A, int N, int M,
                                 float *__restrict u, float *__restrict v,
                                 int iters = 6) {
  constexpr int kMaxM = 32;
  M = std::min(M, kMaxM);
  float G[kMaxM * kMaxM] = {0.0f};
  for (int k = 0; k < N; ++k) {
    const float *row = &A[k * M];
    for (int i = 0; i < M; ++i) {
      const float ri = row[i];
      for (int j = i; j < M; ++j)
        G[i * M + j] += ri * row[j];
    }
  }
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < i; ++j)
      G[i * M + j] = G[j * M + i];

  float vn = 0.0f;
  for (int i = 0; i < M; ++i)
    vn += v[i] * v[i];
  if (vn < 1e-12f)
    for (int i = 0; i < M; ++i)
      v[i] = 1.0f / std::sqrt(static_cast<float>(M));

  float lambda = 0.0f;
  for (int it = 0; it < iters; ++it) {
    float tmp[kMaxM];
    float nsq = 0.0f;
    for (int i = 0; i < M; ++i) {
      float s = 0.0f;
      for (int j = 0; j < M; ++j)
        s += G[i * M + j] * v[j];
      tmp[i] = s;
      nsq += s * s;
    }
    lambda = std::sqrt(nsq); // ||G v|| -> sigma_1^2 as v converges
    const float inv = (nsq > 1e-12f) ? 1.0f / lambda : 0.0f;
    for (int i = 0; i < M; ++i)
      v[i] = tmp[i] * inv;
  }

  float unsq = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float *row = &A[i * M];
    float s = 0.0f;
    for (int j = 0; j < M; ++j)
      s += row[j] * v[j];
    u[i] = s;
    unsq += s * s;
  }
  if (unsq > 1e-10f) {
    const float inv = 1.0f / std::sqrt(unsq);
    for (int i = 0; i < N; ++i)
      u[i] *= inv;
  }
  return std::sqrt(lambda); // sigma_1
}

// ----------------------------------------------------------------------------
// Lane tracking
//
// Stage 1 (measurement): scan image rows and produce candidate lane-centre
//   points. These colour rules are calibrated for the Webots city world:
//   yellow centre line, dark asphalt, bright kerbs, white dashes.
// Stage 2 (estimation): robust weighted least-squares quadratic
//   x(t) = c0 + c1 t + c2 t^2  through the candidates (Chebyshev rejection).
// Stage 3 (filtering): Gaussian tracker on the look-ahead point using the WLS
//   predictive variance as the measurement noise.
// Stage 4 (control): steering = atan(K * normalised lateral error), plus a
//   steering-rate limit so the wheel can never snap between frames.
// ----------------------------------------------------------------------------
struct Pt {
  int x, y;
};

struct LaneResult {
  std::vector<Pt> left_pts, right_pts, center_pts;
  std::vector<float> center_w;
  PolyFit fit;
  int y_top = 0, y_bottom = 0;
  Pt target{0, 0};
  float measurement_var = 0.0f;
  float filter_gain = 0.0f;
  float lateral_error = 0.0f; // normalised [-1, 1]
  float raw_steer = 0.0f;
  float steering_rad = 0.0f;
  bool valid = false;
};

inline bool is_yellow_line(int r, int g, int b) {
  return (r > 120 && g > 100 && b < 90 && (r - b) > 45 && (g - b) > 30);
}
inline bool is_white_paint(int r, int g, int b) {
  return (r > 145 && g > 145 && b > 145 && std::abs(r - b) < 30);
}
inline bool is_road_asphalt(int r, int g, int b) {
  const int mx = std::max({r, g, b}), mn = std::min({r, g, b});
  const int lum = (r + g + b) / 3;
  return (mx - mn) < 18 && lum >= 20 && lum <= 68;
}

class LaneTracker {
public:
  float steer_gain = 0.85f;      // tuned on city.wbt
  float max_steer = 0.38f;       // [rad]
  float max_steer_rate = 0.9f;   // [rad/s]
  float lookahead_t = 0.45f;     // fraction down the scan band
  float deadband = 0.015f;

  LaneTracker() {
    tracker.mean = 0.0f;
    tracker.var = 0.05f;
    tracker.Q = 3e-4f;
  }

  LaneResult process(const uint8_t *__restrict bgr, int width, int height,
                     float dt) {
    LaneResult res;
    res.y_top = (height * 54) / 100;
    res.y_bottom = (height * 90) / 100;
    detect_candidates(bgr, width, res);

    // ---- Stage 2: robust WLS fit in normalised coordinates ----
    const int n = static_cast<int>(res.center_pts.size());
    float t[kMaxFitPts], x[kMaxFitPts], w[kMaxFitPts];
    const int m = std::min(n, kMaxFitPts);
    const float half_w = 0.5f * width;
    for (int i = 0; i < m; ++i) {
      t[i] = static_cast<float>(res.center_pts[i].y - res.y_top) /
             static_cast<float>(res.y_bottom - res.y_top);
      x[i] = (res.center_pts[i].x - half_w) / half_w; // normalised offset
      w[i] = res.center_w[i];
    }
    res.fit = robust_wls_polyfit(t, x, w, m, 2, 2.5f, 0.01f);

    // ---- Stage 3: Gaussian tracking of the look-ahead offset ----
    tracker.predict();
    if (res.fit.ok) {
      const float meas = clampf(res.fit.eval(lookahead_t), -0.8f, 0.8f);
      // measurement variance = WLS predictive variance (+ floor for model error)
      float R = res.fit.predictive_variance(lookahead_t) + 3e-4f;
      R *= 1.0f + 0.5f * res.fit.rejected; // distrust frames with outliers
      tracker.update(meas, R);
      res.measurement_var = R;
      res.valid = true;
    } else {
      // no lane this frame: let the estimate relax slowly toward straight
      tracker.mean *= 0.97f;
    }
    res.filter_gain = tracker.last_gain;

    // ---- Stage 4: steering law + rate limit ----
    float err = tracker.mean;
    if (std::fabs(err) < deadband)
      err = 0.0f;
    res.lateral_error = err;
    res.raw_steer = std::atan(steer_gain * err);
    const float max_step = max_steer_rate * std::max(dt, 1e-3f);
    steer += clampf(res.raw_steer - steer, -max_step, max_step);
    steer = clampf(steer, -max_steer, max_steer);
    res.steering_rad = steer;

    const int ty = res.y_top +
                   static_cast<int>(lookahead_t * (res.y_bottom - res.y_top));
    res.target = Pt{static_cast<int>(half_w + tracker.mean * half_w), ty};
    return res;
  }

  float steering() const { return steer; }

private:
  GaussianTracker tracker;
  float steer = 0.0f;

  static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
  }

  // Row-scan feature extraction (calibrated half-lane offset from the yellow
  // centre line; kerb fallback; asphalt-band fallback).
  void detect_candidates(const uint8_t *__restrict bgr, int width,
                         LaneResult &res) const {
    const int y_top = res.y_top, y_bottom = res.y_bottom;
    for (int y = y_top; y < y_bottom; y += 3) {
      const uint8_t *row = bgr + static_cast<size_t>(y) * width * 3;
      const float y_norm =
          static_cast<float>(y - y_top) / static_cast<float>(y_bottom - y_top);
      const float half_lane = width * (0.23f + 0.11f * y_norm);

      // A. yellow centre line (reject crosswalks: many runs / wide span)
      int ycount = 0, first = -1, last = -1, runs = 0;
      bool in_y = false;
      for (int x = 4; x < (width * 75) / 100; x += 2) {
        const uint8_t *p = row + x * 3;
        if (is_yellow_line(p[2], p[1], p[0])) {
          ++ycount;
          if (first == -1)
            first = x;
          last = x;
          if (!in_y) {
            ++runs;
            in_y = true;
          }
        } else {
          in_y = false;
        }
      }
      bool has_left = false;
      float x_L = 0.0f;
      const int span = last - first;
      if (ycount >= 2 && ycount <= 26 && runs <= 2 && span >= 0 &&
          span < (width * 12) / 100) {
        has_left = true;
        x_L = 0.5f * (first + last);
        res.left_pts.push_back(Pt{static_cast<int>(x_L), y});
      }

      // B. outer road edge: longest run of drivable surface (paint bridged)
      const int start_x = has_left ? std::min(width - 20, static_cast<int>(x_L) + 30)
                                   : (width * 20) / 100;
      int best_s = -1, best_e = -1, cur_s = -1, cur_len = 0, max_len = 0;
      for (int x = start_x; x < width - 4; x += 2) {
        const uint8_t *p = row + x * 3;
        bool road = is_road_asphalt(p[2], p[1], p[0]);
        if (!road && (is_white_paint(p[2], p[1], p[0]) ||
                      is_yellow_line(p[2], p[1], p[0]))) {
          if (x + 18 < width) {
            const uint8_t *q = row + (x + 18) * 3;
            road = is_road_asphalt(q[2], q[1], q[0]);
          }
        }
        if (road) {
          if (cur_s == -1)
            cur_s = x;
          if (++cur_len > max_len) {
            max_len = cur_len;
            best_s = cur_s;
            best_e = x;
          }
        } else {
          cur_s = -1;
          cur_len = 0;
        }
      }
      bool has_right = false;
      float x_R = static_cast<float>(width - 1);
      if (max_len >= 18 && best_e < width - 12) {
        has_right = true;
        x_R = static_cast<float>(best_e);
        res.right_pts.push_back(Pt{static_cast<int>(x_R), y});
      }

      // C. lane-centre candidate with a confidence weight for the WLS fit
      float x_mid = 0.5f * width, weight = 0.0f;
      if (has_left) {
        x_mid = x_L + half_lane;
        if (has_right && (x_R - x_L) > width * 0.16f) {
          const float hi = x_R - 0.85f * half_lane;
          const float lo = x_L + 0.65f * half_lane;
          x_mid = std::max(lo, std::min(x_mid, hi));
        }
        weight = 1.0f;
      } else if (has_right) {
        x_mid = x_R - half_lane;
        weight = 0.6f;
      } else if (max_len >= 20) {
        x_mid = 0.5f * (best_s + best_e);
        weight = 0.25f;
      }
      if (weight > 0.0f) {
        x_mid = std::max(width * 0.10f, std::min(width * 0.90f, x_mid));
        res.center_pts.push_back(Pt{static_cast<int>(x_mid), y});
        res.center_w.push_back(weight);
      }
    }
  }
};

} // namespace apex
