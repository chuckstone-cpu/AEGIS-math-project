// ============================================================================
// apex/stats.hpp  -  Statistical estimation used by the lane tracker
//
// Syllabus (Unit 3): inequalities of statistics (Chebyshev), multivariate
// Gaussian and weighted least squares, Markov (random-walk) state model.
// ============================================================================
#pragma once

#include "linalg.hpp"

namespace apex {

// ----------------------------------------------------------------------------
// Robust weighted least-squares polynomial fit   x(t) = c0 + c1 t + c2 t^2
//
// Model: x_i = phi(t_i)^T c + e_i,  e_i ~ N(0, sigma^2 / w_i)  (independent).
// The maximum-likelihood estimate of c under this multivariate Gaussian model
// is the WLS solution of the normal equations
//        (Phi^T W Phi) c = Phi^T W x ,
// solved with our own Cholesky factorisation.
//
// Outlier rejection uses Chebyshev's inequality:  P(|X - mu| >= k sigma) <= 1/k^2.
// It holds for ANY distribution, so with k = 2.5 at most 16% of genuine lane
// points can be discarded, while gross mis-detections (a crosswalk stripe, a
// parked car read as a curb) sit far outside k sigma and are dropped. We fit,
// reject, and refit (one round of iteratively reweighted least squares).
// ----------------------------------------------------------------------------
struct PolyFit {
  bool ok = false;
  int degree = 0;
  double c[3] = {0, 0, 0};
  double N[9] = {0};                  // Phi^T W Phi (for prediction variance)
  float sigma = 0.0f;                   // weighted residual std-dev
  int used = 0, rejected = 0;

  float eval(float t) const {
    return static_cast<float>(c[0] + c[1] * t + c[2] * t * t);
  }

  // Var[x_hat(t)] = sigma^2 * phi(t)^T (Phi^T W Phi)^{-1} phi(t)
  float predictive_variance(float t) const {
    const int m = degree + 1;
    double phi[3] = {1.0, t, static_cast<double>(t) * t};
    double z[3] = {0, 0, 0};
    if (!cholesky_solve(N, phi, z, m))
      return 1e6f;
    double q = 0.0;
    for (int i = 0; i < m; ++i)
      q += phi[i] * z[i];
    return static_cast<float>(sigma * sigma * q);
  }
};

inline bool wls_solve(const float *t, const float *x, const float *w,
                      const bool *keep, int n, int degree, PolyFit &f) {
  const int m = degree + 1;
  double N[9] = {0}, rhs[3] = {0};
  int used = 0;
  for (int i = 0; i < n; ++i) {
    if (!keep[i])
      continue;
    const double phi[3] = {1.0, t[i], static_cast<double>(t[i]) * t[i]};
    for (int r = 0; r < m; ++r) {
      rhs[r] += w[i] * phi[r] * x[i];
      for (int c = 0; c < m; ++c)
        N[r * m + c] += w[i] * phi[r] * phi[c];
    }
    ++used;
  }
  if (used < m + 1)
    return false;
  // Tiny ridge term keeps the system SPD when points are clustered.
  for (int r = 0; r < m; ++r)
    N[r * m + r] += 1e-6;
  double c[3] = {0, 0, 0};
  if (!cholesky_solve(N, rhs, c, m))
    return false;

  double ss = 0.0, ws = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!keep[i])
      continue;
    const double e = x[i] - (c[0] + c[1] * t[i] + c[2] * t[i] * t[i]);
    ss += w[i] * e * e;
    ws += w[i];
  }
  f.ok = true;
  f.degree = degree;
  for (int i = 0; i < 3; ++i)
    f.c[i] = c[i];
  std::memcpy(f.N, N, sizeof(N));
  f.sigma = static_cast<float>(std::sqrt(ss / std::max(ws, 1e-9)));
  f.used = used;
  return true;
}

constexpr int kMaxFitPts = 128;

inline PolyFit robust_wls_polyfit(const float *t, const float *x,
                                  const float *w, int n, int degree,
                                  float chebyshev_k = 2.5f,
                                  float sigma_floor = 1e-3f) {
  PolyFit f;
  n = std::min(n, kMaxFitPts);
  bool keep[kMaxFitPts];
  for (int i = 0; i < n; ++i)
    keep[i] = true;
  // Fall back to a straight line when there are too few points for a parabola.
  while (degree > 0 && n < degree + 3)
    --degree;
  if (!wls_solve(t, x, w, keep, n, degree, f))
    return f;

  const float thresh = chebyshev_k * std::max(f.sigma, sigma_floor);
  int rejected = 0;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(x[i] - f.eval(t[i])) > thresh) {
      keep[i] = false;
      ++rejected;
    }
  }
  if (rejected > 0) {
    PolyFit refit;
    if (wls_solve(t, x, w, keep, n, degree, refit))
      f = refit;
  }
  f.rejected = rejected;
  return f;
}

// ----------------------------------------------------------------------------
// Scalar Gaussian state estimator (product of Gaussians / Kalman update).
//
// State model (Markov random walk):  s_k = s_{k-1} + q_k,   q_k ~ N(0, Q)
// Measurement:                       m_k = s_k + r_k,       r_k ~ N(0, R_k)
// Predict:  P <- P + Q
// Update:   the posterior N(s | m) is the normalised product of two Gaussians:
//           K = P / (P + R),  s <- s + K (m - s),  P <- (1 - K) P
// A noisy frame (large R from the WLS predictive variance) automatically gets a
// small gain, which is what removes steering jitter without adding fixed lag.
// ----------------------------------------------------------------------------
struct GaussianTracker {
  float mean = 0.0f;
  float var = 1.0f;
  float Q = 3e-4f;
  float last_gain = 0.0f;

  void predict() { var += Q; }
  void update(float measurement, float R) {
    const float K = var / (var + R);
    mean += K * (measurement - mean);
    var *= (1.0f - K);
    last_gain = K;
  }
};

} // namespace apex
