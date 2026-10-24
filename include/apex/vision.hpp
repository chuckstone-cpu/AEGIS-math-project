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
// Lane tracking in METRIC ground coordinates (inverse perspective mapping)
//
// Pinhole camera, optical axis horizontal, height h above a flat road. A ground
// point at forward distance X and lateral offset Y (left positive) projects to
//      u = c_x - f Y / X ,     v = c_y + f h / X
// so each image pixel below the horizon maps back to the road plane by
//      X = f h / (v - c_y) ,   Y = -(u - c_x) X / f .
//
// Stage 1 (measurement): find yellow (centre double line) and white (lane
//   divider / road edge) paint per image row, back-project to (X, Y).
//   Every marking is a noisy measurement of the SAME lane-centre curve:
//      yellow  at  +W/2  of the lane centre (W = lane width)
//      divider at  -W/2,   road edge at  -3W/2
//   White marks are associated by gating against the previous estimate.
// Stage 2 (estimation): robust weighted least squares  Y_c(X) = c0 + c1 X + c2 X^2.
//   One pixel of error moves Y by X / f metres, so Var(Y) ~ X^2 and the
//   inverse-variance (Gauss-Markov optimal) weight is w ~ 1 / X^2.
//   Chebyshev rejection removes mis-associated marks.
// Stage 3 (filtering): Gaussian tracker on the look-ahead offset y(L_d), with
//   the WLS predictive variance as measurement noise (random-walk model).
// Stage 4 (control): pure pursuit. The circular arc from the rear axle
//   through the look-ahead point (x_t, y_t) has curvature k = 2 y_t / D^2,
//   D^2 = x_t^2 + y_t^2, so the steering angle is delta = atan(L k).
//   A steering-rate limit keeps the wheel from snapping between frames.
// ----------------------------------------------------------------------------
struct Pt {
  int x, y;
};

struct GroundMark {
  float X, Y; // metres, car camera frame (X forward, Y left)
  int type;   // 0 yellow, 1 lane divider, 2 road edge, 3 unassociated white
};

struct CameraModel {
  float f = 234.30f;       // focal length [camera px] = (w/2) / tan(fov/2)
  float cx = 127.5f, cy = 63.5f;
  int cam_w = 256, cam_h = 128;
  float height = 1.80f;    // camera height above road [m]
  float x_offset = 1.72f;  // camera ahead of the rear axle [m]

  static CameraModel from_webots(int w, int h, float fov, float height, float x_off) {
    CameraModel c;
    c.cam_w = w;
    c.cam_h = h;
    c.f = 0.5f * w / std::tan(0.5f * fov);
    c.cx = 0.5f * w - 0.5f;
    c.cy = 0.5f * h - 0.5f;
    c.height = height;
    c.x_offset = x_off;
    return c;
  }
  // image (in a view of size view_w x view_h) -> ground. false above horizon.
  bool to_ground(float u_view, float v_view, int view_w, int view_h, float &X,
                 float &Y) const {
    const float u = (u_view + 0.5f) * cam_w / view_w - 0.5f;
    const float v = (v_view + 0.5f) * cam_h / view_h - 0.5f;
    const float dv = v - cy;
    if (dv < 0.5f)
      return false;
    X = f * height / dv;
    Y = -(u - cx) * X / f;
    return true;
  }
  void to_image(float X, float Y, int view_w, int view_h, float &u_view,
                float &v_view) const {
    const float u = cx - f * Y / X, v = cy + f * height / X;
    u_view = (u + 0.5f) * view_w / cam_w - 0.5f;
    v_view = (v + 0.5f) * view_h / cam_h - 0.5f;
  }
};

struct LaneResult {
  std::vector<Pt> left_pts, right_pts, center_pts; // image overlay
  std::vector<GroundMark> marks;                   // metric measurements
  std::vector<float> cx_m, cy_m;                   // centre candidates [m]
  PolyFit fit;            // Y_c(t), t = X / kFitScale
  float fit_x_min = 0.0f, fit_x_max = 0.0f;
  Pt target{0, 0};        // look-ahead point in the view image
  float lookahead_m = 10.0f;
  float target_y_m = 0.0f;
  float curvature = 0.0f; // pure-pursuit arc curvature [1/m]
  float measurement_var = 0.0f;
  float filter_gain = 0.0f;
  float lateral_error = 0.0f; // y at look-ahead [m]
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

constexpr float kFitScale = 20.0f; // fit in t = X / 20 for conditioning

class LaneTracker {
public:
  CameraModel cam;
  float lane_width = 5.375f;    // Webots city: 21.5 m road, 4 lanes
  float wheelbase = 2.995f;     // BMW X5 [m]
  float max_steer = 0.45f;      // [rad]
  float max_steer_rate = 0.9f;  // [rad/s]
  float max_range = 40.0f;      // use marks up to 40 m
  float min_lookahead = 8.0f, lookahead_gain = 0.9f; // L_d = max(8, 4 + 0.9 v)

  LaneTracker() {
    tracker.mean = 0.0f;
    tracker.var = 0.25f;
    tracker.Q = 2.5e-3f; // (5 cm)^2 per step random walk
  }

  LaneResult process(const uint8_t *__restrict bgr, int width, int height,
                     float dt, float speed_mps) {
    LaneResult res;
    detect(bgr, width, height, res);

    // ---- Stage 2: robust WLS fit of the centre line in metres ----
    const int m = std::min(static_cast<int>(res.cx_m.size()), kMaxFitPts);
    float t[kMaxFitPts], y[kMaxFitPts], w[kMaxFitPts];
    res.fit_x_min = 1e9f;
    res.fit_x_max = 0.0f;
    for (int i = 0; i < m; ++i) {
      t[i] = res.cx_m[i] / kFitScale;
      y[i] = res.cy_m[i];
      w[i] = weight_[i] * 100.0f / (res.cx_m[i] * res.cx_m[i]); // ~ 1 / X^2
      res.fit_x_min = std::min(res.fit_x_min, res.cx_m[i]);
      res.fit_x_max = std::max(res.fit_x_max, res.cx_m[i]);
    }
    res.fit = robust_wls_polyfit(t, y, w, m, 2, 2.5f, 0.08f);
    if (res.fit.ok && res.fit.used >= 6) {
      have_prev = true;
      for (int k = 0; k < 3; ++k)
        prev_c[k] = res.fit.c[k];
    }

    // ---- Stage 3: Gaussian tracking of the look-ahead offset ----
    const float Ld = std::max(min_lookahead, 4.0f + lookahead_gain * speed_mps);
    res.lookahead_m = Ld;
    tracker.predict();
    if (res.fit.ok && res.fit.used >= 6) {
      // never extrapolate the parabola far beyond the data
      const float x_eval = std::min(Ld, std::max(res.fit_x_max, min_lookahead));
      const float tl = x_eval / kFitScale;
      const float meas = clampf(res.fit.eval(tl), -6.0f, 6.0f);
      float R = res.fit.predictive_variance(tl) + 0.01f;
      R *= 1.0f + 0.25f * res.fit.rejected;
      tracker.update(meas, R);
      res.measurement_var = R;
      res.valid = true;
    }
    res.filter_gain = tracker.last_gain;

    // ---- Stage 4: pure pursuit from the rear axle ----
    const float xt = Ld + cam.x_offset, yt = tracker.mean;
    const float D2 = xt * xt + yt * yt;
    res.curvature = 2.0f * yt / D2;
    res.target_y_m = yt;
    res.lateral_error = yt;
    // Webots convention: positive steering turns RIGHT (toward -Y)
    res.raw_steer = clampf(-std::atan(wheelbase * res.curvature), -max_steer, max_steer);
    const float max_step = max_steer_rate * std::max(dt, 1e-3f);
    steer += clampf(res.raw_steer - steer, -max_step, max_step);
    res.steering_rad = steer;

    float u, v;
    cam.to_image(Ld, yt, width, height, u, v);
    res.target = Pt{static_cast<int>(u), static_cast<int>(v)};
    return res;
  }

  float steering() const { return steer; }

private:
  GaussianTracker tracker;
  float steer = 0.0f;
  bool have_prev = false;
  double prev_c[3] = {0, 0, 0};
  std::vector<float> weight_;

  static float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
  }
  // predicted lane-centre offset at distance X from the previous fit
  float predict_center(float X) const {
    if (!have_prev)
      return tracker.mean;
    const double t = X / kFitScale;
    return static_cast<float>(prev_c[0] + prev_c[1] * t + prev_c[2] * t * t);
  }
  void add_center(LaneResult &res, float X, float Yc, float wgt, int u, int v) {
    res.cx_m.push_back(X);
    res.cy_m.push_back(Yc);
    weight_.push_back(wgt);
    res.center_pts.push_back(Pt{u, v});
  }

  void detect(const uint8_t *__restrict bgr, int width, int height, LaneResult &res) {
    weight_.clear();
    const float half = 0.5f * lane_width;
    // first row whose ground distance is within max_range
    float u0, v_far;
    cam.to_image(max_range, 0.0f, width, height, u0, v_far);
    const int y_start = std::max(static_cast<int>(v_far) + 1, height / 2 + 2);
    const int y_end = (height * 92) / 100;

    for (int yv = y_start; yv < y_end; yv += 3) {
      const uint8_t *row = bgr + static_cast<size_t>(yv) * width * 3;
      // runs of yellow / white paint along the row
      int ycount = 0, yfirst = -1, ylast = -1, yruns = 0;
      bool in_y = false;
      int w_start = -1;
      std::vector<std::pair<int, int>> whites;
      for (int x = 2; x < width - 2; x += 2) {
        const uint8_t *p = row + x * 3;
        const bool yel = is_yellow_line(p[2], p[1], p[0]);
        const bool wht = !yel && is_white_paint(p[2], p[1], p[0]);
        if (yel) {
          ++ycount;
          if (yfirst < 0)
            yfirst = x;
          ylast = x;
          if (!in_y) {
            ++yruns;
            in_y = true;
          }
        } else {
          in_y = false;
        }
        if (wht && w_start < 0)
          w_start = x;
        if (!wht && w_start >= 0) {
          whites.emplace_back(w_start, x - 2);
          w_start = -1;
        }
      }
      float X, Y;
      // A. yellow double line (reject crosswalk stripes: many runs / wide)
      const int span = ylast - yfirst;
      if (ycount >= 2 && yruns <= 2 && span >= 0 && span < (width * 12) / 100) {
        const float u = 0.5f * (yfirst + ylast);
        if (cam.to_ground(u, static_cast<float>(yv), width, height, X, Y) &&
            X < max_range) {
          res.marks.push_back(GroundMark{X, Y, 0});
          res.left_pts.push_back(Pt{static_cast<int>(u), yv});
          add_center(res, X, Y - half, 1.0f, static_cast<int>(u), yv);
        }
      }
      // B. white paint, associated by gating against the predicted centre
      for (const auto &r : whites) {
        if (r.second - r.first > width / 6) // a crosswalk bar, not a line
          continue;
        const float u = 0.5f * (r.first + r.second);
        if (!cam.to_ground(u, static_cast<float>(yv), width, height, X, Y) ||
            X >= max_range)
          continue;
        const float d = Y - predict_center(X);
        if (std::fabs(d + half) < 1.3f) {
          res.marks.push_back(GroundMark{X, Y, 1});
          res.right_pts.push_back(Pt{static_cast<int>(u), yv});
          add_center(res, X, Y + half, 0.8f, static_cast<int>(u), yv);
        } else if (std::fabs(d + 3.0f * half) < 1.6f) {
          res.marks.push_back(GroundMark{X, Y, 2});
          res.right_pts.push_back(Pt{static_cast<int>(u), yv});
          add_center(res, X, Y + 3.0f * half, 0.5f, static_cast<int>(u), yv);
        } else {
          res.marks.push_back(GroundMark{X, Y, 3});
        }
      }
    }
  }
};

} // namespace apex
