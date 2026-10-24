// ============================================================================
// Verification suite for the APEX maths engine (no OpenCV, no Webots).
// Each solver is checked against an independent reference or a theorem:
//   CG        vs. dense Gaussian elimination
//   WLS       recovers a known polynomial; Chebyshev rejects planted outliers
//   ADMM QP   KKT residuals ~ 0, constraints satisfied, matches a long run,
//             matches the closed-form unconstrained QP when nothing is active
//   MDP       Bellman residual ~ 0, sensible policy, stationary dist sums to 1
//   PDE       maximum principle (no new extrema) and mean preservation
//   SVD       recovers the exact singular vector of a rank-1 matrix
// Build: g++ -O2 -std=c++17 -Iinclude tests/test_math.cpp -o test_math
// ============================================================================
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "apex/mdp.hpp"
#include "apex/optim.hpp"
#include "apex/vision.hpp"

using namespace apex;

static int failures = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (cond) {                                                                \
      std::printf("  [PASS] ");                                                \
    } else {                                                                   \
      std::printf("  [FAIL] ");                                                \
      ++failures;                                                              \
    }                                                                          \
    std::printf(__VA_ARGS__);                                                  \
    std::printf("\n");                                                         \
  } while (0)

// Dense Gaussian elimination with partial pivoting (reference solver)
static std::vector<double> dense_solve(std::vector<double> A,
                                       std::vector<double> b, int n) {
  for (int c = 0; c < n; ++c) {
    int p = c;
    for (int r = c + 1; r < n; ++r)
      if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c]))
        p = r;
    for (int k = 0; k < n; ++k)
      std::swap(A[c * n + k], A[p * n + k]);
    std::swap(b[c], b[p]);
    for (int r = c + 1; r < n; ++r) {
      const double f = A[r * n + c] / A[c * n + c];
      for (int k = c; k < n; ++k)
        A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  std::vector<double> x(n);
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < n; ++k)
      s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

static void test_cg() {
  std::printf("\n[1] Preconditioned Conjugate Gradient (Krylov)\n");
  const int n = 40;
  std::vector<double> A(n * n, 0.0);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> U(-1, 1);
  // SPD pentadiagonal matrix (same structure as the ADMM x-step)
  for (int i = 0; i < n; ++i) {
    A[i * n + i] = 6.0 + 2.0 * (i % 3);
    if (i + 1 < n)
      A[i * n + i + 1] = A[(i + 1) * n + i] = -1.5;
    if (i + 2 < n)
      A[i * n + i + 2] = A[(i + 2) * n + i] = 0.5;
  }
  std::vector<double> bd(n);
  float b[64], x[64] = {0}, inv_d[64];
  for (int i = 0; i < n; ++i) {
    bd[i] = U(rng);
    b[i] = static_cast<float>(bd[i]);
    inv_d[i] = static_cast<float>(1.0 / A[i * n + i]);
  }
  auto op = [&](const float *in, float *out) {
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int j = 0; j < n; ++j)
        s += A[i * n + j] * in[j];
      out[i] = static_cast<float>(s);
    }
  };
  CGStats st = pcg_solve(op, b, x, inv_d, n, 200, 1e-6f);
  auto ref = dense_solve(A, bd, n);
  double err = 0;
  for (int i = 0; i < n; ++i)
    err = std::max(err, std::fabs(ref[i] - x[i]));
  CHECK(err < 1e-4, "max |x_cg - x_dense| = %.2e  (%d iterations, n=%d)", err,
        st.iterations, n);
  CHECK(st.iterations <= n, "converged within n iterations (Krylov bound)");
}

static void test_wls() {
  std::printf("\n[2] Robust weighted least squares + Chebyshev rejection\n");
  std::mt19937 rng(7);
  std::normal_distribution<float> N(0.0f, 0.01f);
  float t[40], x[40], w[40];
  const double c0 = 0.10, c1 = -0.30, c2 = 0.25;
  for (int i = 0; i < 40; ++i) {
    t[i] = i / 39.0f;
    x[i] = static_cast<float>(c0 + c1 * t[i] + c2 * t[i] * t[i]) + N(rng);
    w[i] = 1.0f;
  }
  x[5] += 0.6f; // planted gross outliers (e.g. crosswalk read as lane)
  x[22] -= 0.5f;
  x[31] += 0.7f;
  PolyFit f = robust_wls_polyfit(t, x, w, 40, 2);
  CHECK(f.ok && std::fabs(f.c[0] - c0) < 0.02 && std::fabs(f.c[1] - c1) < 0.08 &&
            std::fabs(f.c[2] - c2) < 0.08,
        "recovered c = (%.3f, %.3f, %.3f), true (%.2f, %.2f, %.2f)", f.c[0],
        f.c[1], f.c[2], c0, c1, c2);
  CHECK(f.rejected >= 3 && f.rejected <= 6,
        "Chebyshev k=2.5 rejected %d points (3 planted)", f.rejected);
  const float var = f.predictive_variance(0.45f);
  CHECK(var > 0 && var < 1e-4f, "predictive variance at look-ahead = %.2e",
        var);

  GaussianTracker g;
  g.mean = 0.0f;
  g.var = 1.0f;
  g.Q = 1e-4f;
  for (int k = 0; k < 50; ++k) {
    g.predict();
    g.update(0.2f, 1e-3f);
  }
  CHECK(std::fabs(g.mean - 0.2f) < 1e-3f,
        "Gaussian tracker converges to the measured value (%.4f)", g.mean);
}

static void test_admm() {
  std::printf("\n[3] ADMM speed QP (L1 jerk, box + gap constraints)\n");
  SpeedPlannerADMM::Params prm;
  // ---- (a) unconstrained & lambda = 0 -> closed-form QP solution ----------
  {
    SpeedPlannerADMM::Params p = prm;
    p.lambda_jerk = 0.0f;
    p.jerk_max = 1e3f; // switch the jerk bound off for the closed form
    p.max_iters = 400;
    p.tol = 1e-6f;
    SpeedPlannerADMM pl(p);
    SpeedPlannerADMM::Problem pb;
    pb.v0 = 6.0f;
    pb.a0 = 0.0f;
    pb.v_ref = 7.0f;
    pb.v_max = 30.0f; // inactive
    auto R = pl.solve(pb);
    // Reference: (w_ref I + w_acc A2^T A2) v = w_ref r + w_acc A2^T b2
    const int n = p.n;
    const double dt = p.dt;
    std::vector<double> A(n * n, 0.0), b(n, 0.0);
    for (int i = 0; i < n; ++i) {
      A[i * n + i] += p.w_ref;
      b[i] += p.w_ref * pb.v_ref;
    }
    // A2 rows: (v_i - v_{i-1})/dt, row 0: v_0/dt, b2_0 = v0/dt
    for (int r = 0; r < n; ++r) {
      std::vector<double> row(n, 0.0);
      row[r] = 1.0 / dt;
      if (r > 0)
        row[r - 1] = -1.0 / dt;
      const double br = (r == 0) ? pb.v0 / dt : 0.0;
      for (int i = 0; i < n; ++i) {
        b[i] += p.w_acc * row[i] * br;
        for (int j = 0; j < n; ++j)
          A[i * n + j] += p.w_acc * row[i] * row[j];
      }
    }
    auto ref = dense_solve(A, b, n);
    double err = 0;
    for (int i = 0; i < n; ++i)
      err = std::max(err, std::fabs(ref[i] - R.v[i]));
    CHECK(err < 1e-2, "matches closed-form QP: max err %.2e m/s (%d iters)",
          err, R.iterations);
  }

  // ---- (b) braking for a stopped obstacle: constraints + KKT --------------
  {
    SpeedPlannerADMM pl(prm);
    SpeedPlannerADMM::Problem pb;
    pb.v0 = 8.9f; // 32 km/h
    pb.a0 = 0.0f;
    pb.v_ref = 8.9f;
    pb.v_max = 8.9f;
    pb.has_lead = true;
    pb.gap = 22.0f; // stopped car 22 m ahead
    pb.v_lead = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto R = pl.solve(pb);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float pos = 0, worst_gap = 1e9f, amin = 0, amax = 0, vmin = 1e9f;
    for (int i = 0; i < R.n; ++i) {
      pos += prm.dt * R.v[i];
      const float g = pb.gap - prm.standstill - (pos + prm.headway * R.v[i]);
      worst_gap = std::min(worst_gap, g);
      amin = std::min(amin, R.a[i]);
      amax = std::max(amax, R.a[i]);
      vmin = std::min(vmin, R.v[i]);
    }
    std::printf("      profile km/h: ");
    for (int i = 0; i < R.n; i += 4)
      std::printf("%.1f ", R.v[i] * 3.6f);
    std::printf("\n");
    CHECK(R.converged, "converged in %d iterations (%d CG steps), %.3f ms",
          R.iterations, R.cg_iterations, ms);
    CHECK(worst_gap > -0.3f, "gap constraint satisfied (min slack %.2f m)",
          worst_gap);
    CHECK(amin >= prm.a_min - 0.15f && amax <= prm.a_max + 0.15f,
          "accel within [%.1f, %.1f]: got [%.2f, %.2f] m/s^2", prm.a_min,
          prm.a_max, amin, amax);
    float jmax = 0, pa = pb.a0;
    for (int i = 0; i < R.n; ++i) {
      jmax = std::max(jmax, std::fabs(R.a[i] - pa) / prm.dt);
      pa = R.a[i];
    }
    CHECK(vmin >= -1e-3f && R.v[R.n - 1] < 2.0f,
          "slows to %.2f m/s by the end of the horizon, never reverses",
          R.v[R.n - 1]);
    CHECK(jmax <= prm.jerk_max + 0.6f, "jerk bounded: max %.2f m/s^3 (limit "
          "%.1f)", jmax, prm.jerk_max);
    CHECK(R.active_gap > 0, "KKT: gap constraint active on %d steps, max "
          "multiplier %.2f (complementary slackness)",
          R.active_gap, R.max_multiplier);
    CHECK(ms < 2.0, "fast enough for real time (%.3f ms < 2 ms)", ms);

    // long-run reference with very tight tolerance
    SpeedPlannerADMM::Params p2 = prm;
    p2.max_iters = 3000;
    p2.tol = 1e-7f;
    SpeedPlannerADMM ref(p2);
    auto RR = ref.solve(pb);
    double err = 0;
    for (int i = 0; i < R.n; ++i)
      err = std::max(err, (double)std::fabs(RR.v[i] - R.v[i]));
    CHECK(err < 0.25, "real-time solution within %.3f m/s of the 3000-iter "
          "optimum (objective %.2f vs %.2f)",
          err, R.objective, RR.objective);
  }

  // ---- (b2) infeasible: obstacle 10 m ahead at 65 km/h ---------------------
  {
    SpeedPlannerADMM pl(prm);
    SpeedPlannerADMM::Problem pb;
    pb.v0 = 18.0f;
    pb.v_ref = 0.0f;
    pb.v_max = 18.0f;
    pb.has_lead = true;
    pb.gap = 10.0f;
    auto R = pl.solve(pb);
    float amin = 0, jmax = 0, pa = 0;
    for (int i = 0; i < R.n; ++i) {
      amin = std::min(amin, R.a[i]);
      jmax = std::max(jmax, std::fabs(R.a[i] - pa) / prm.dt);
      pa = R.a[i];
    }
    CHECK(amin >= prm.a_min - 1e-3f && jmax <= prm.jerk_max + 1e-2f,
          "infeasible case: plan stays executable (a_min %.2f, max jerk %.2f) "
          "after %d iters",
          amin, jmax, R.iterations);
    // fastest physically possible deceleration: jerk ramp, then a_min
    float vb = pb.v0, ab = 0.0f;
    for (int i = 0; i < R.n; ++i) {
      ab = std::max(prm.a_min, ab - prm.jerk_max * prm.dt);
      vb = std::max(0.0f, vb + ab * prm.dt);
    }
    CHECK(amin < prm.a_min + 0.3f && R.v[R.n - 1] <= vb + 0.3f,
          "  ... and brakes as hard as physics allows (v_end %.2f, bound %.2f "
          "m/s)",
          R.v[R.n - 1], vb);
  }

  // ---- (c) warm start speeds up the next frame -----------------------------
  {
    SpeedPlannerADMM pl(prm);
    SpeedPlannerADMM::Problem pb;
    pb.v0 = 5.0f;
    pb.v_ref = 8.0f;
    pb.v_max = 8.9f;
    auto R1 = pl.solve(pb);
    pb.v0 = R1.v[0];
    pb.a0 = R1.a[0];
    auto R2 = pl.solve(pb);
    CHECK(R2.iterations <= R1.iterations,
          "warm start: %d iterations (cold %d)", R2.iterations, R1.iterations);
    bool smooth = true;
    for (int i = 0; i < R1.n; ++i)
      smooth &= (R1.a[i] <= prm.a_max + 0.15f);
    // sparsity: same problem with and without the L1 jerk term
    SpeedPlannerADMM::Params p0 = prm;
    p0.lambda_jerk = 0.0f;
    SpeedPlannerADMM pl0(p0);
    pb.v0 = 5.0f;
    pb.a0 = 0.0f;
    auto R0 = pl0.solve(pb);
    CHECK(R1.nonzero_jerk < R0.nonzero_jerk,
          "L1 term sparsifies jerk: %d / %d non-zero steps (vs %d without L1)",
          R1.nonzero_jerk, R1.n, R0.nonzero_jerk);
    CHECK(smooth && R1.v[R1.n - 1] > 7.5f,
          "accelerates smoothly 18 -> %.1f km/h within the 4 s horizon",
          R1.v[R1.n - 1] * 3.6f);
  }
}

static void test_mdp() {
  std::printf("\n[4] Markov decision process (value iteration)\n");
  DrivingMDP m;
  const char *A[] = {"ACCEL", "MAINTAIN", "BRAKE"};
  const char *D[] = {"CLEAR", "FAR", "MED", "CLOSE"};
  const char *S[] = {"SLOW", "CRUISE", "FAST"};
  for (int d = 0; d < 4; ++d) {
    std::printf("      %-5s:", D[d]);
    for (int v = 0; v < 3; ++v)
      std::printf("  %s->%-8s", S[v], A[m.policy[DrivingMDP::idx(d, v)]]);
    std::printf("\n");
  }
  CHECK(m.bellman_residual < 1e-3f,
        "Bellman residual %.2e after %d iterations (gamma=%.2f)",
        m.bellman_residual, m.iterations, m.gamma);
  // rows of every P(.|s,a) must sum to 1 (valid Markov kernel)
  double worst = 0;
  for (int a = 0; a < 3; ++a)
    for (int s = 0; s < DrivingMDP::NS; ++s) {
      double sum = 0;
      for (int s2 = 0; s2 < DrivingMDP::NS; ++s2)
        sum += m.P[a][s][s2];
      worst = std::max(worst, std::fabs(sum - 1.0));
    }
  CHECK(worst < 1e-5, "transition matrices are stochastic (|row sum-1| %.1e)",
        worst);
  CHECK(m.policy[DrivingMDP::idx(D_CLOSE, V_FAST)] == BRAKE &&
            m.policy[DrivingMDP::idx(D_CLOSE, V_CRUISE)] == BRAKE,
        "CLOSE + moving -> BRAKE");
  CHECK(m.policy[DrivingMDP::idx(D_CLEAR, V_SLOW)] == ACCEL &&
            m.policy[DrivingMDP::idx(D_CLEAR, V_CRUISE)] == ACCEL,
        "CLEAR road -> ACCEL");
  CHECK(m.policy[DrivingMDP::idx(D_FAR, V_CRUISE)] != BRAKE,
        "FAR obstacle at cruise -> no needless braking");
  double sum = 0;
  for (float p : m.stationary)
    sum += p;
  const float risk = m.stationary[DrivingMDP::idx(D_CLOSE, V_FAST)];
  CHECK(std::fabs(sum - 1.0) < 1e-4 && risk < 0.02f,
        "stationary distribution sums to %.4f, long-run P(CLOSE,FAST) = %.4f",
        sum, risk);
}

static void test_pde_svd() {
  std::printf("\n[5] Perona-Malik PDE and power-iteration SVD\n");
  const int W = 64, H = 48;
  std::vector<float> img(W * H), tmp(W * H);
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> U(0, 255);
  double mean0 = 0;
  float mn = 1e9f, mx = -1e9f;
  for (auto &p : img) {
    p = U(rng);
    mean0 += p;
    mn = std::min(mn, p);
    mx = std::max(mx, p);
  }
  mean0 /= img.size();
  perona_malik_diffusion(img.data(), tmp.data(), W, H, 5, 0.25f, 20.0f);
  float mn2 = 1e9f, mx2 = -1e9f;
  double mean1 = 0;
  for (float p : img) {
    mn2 = std::min(mn2, p);
    mx2 = std::max(mx2, p);
    mean1 += p;
  }
  mean1 /= img.size();
  CHECK(mn2 >= mn - 1e-3f && mx2 <= mx + 1e-3f,
        "maximum principle holds (range [%.1f, %.1f] within [%.1f, %.1f])",
        mn2, mx2, mn, mx);
  CHECK(std::fabs(mean1 - mean0) < 2.0, "mean intensity ~ preserved (%.2f -> "
        "%.2f)", mean0, mean1);

  // rank-1 matrix A = a b^T plus tiny noise: power iteration must recover a
  const int N = 4096, M = 10;
  std::vector<float> A(N * M), u(N), v(M, 0.0f), a(N);
  std::normal_distribution<float> G(0, 1e-3f);
  double an = 0;
  for (int i = 0; i < N; ++i) {
    a[i] = std::sin(0.01f * i) + 1.5f;
    an += a[i] * a[i];
  }
  an = std::sqrt(an);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      A[i * M + j] = a[i] * (1.0f + 0.1f * j) + G(rng);
  power_iteration_svd(A.data(), N, M, u.data(), v.data(), 8);
  double cosang = 0;
  for (int i = 0; i < N; ++i)
    cosang += u[i] * a[i] / an;
  CHECK(std::fabs(cosang) > 0.9999, "|cos(u, true singular vector)| = %.6f",
        std::fabs(cosang));
}

int main() {
  std::printf("APEX maths engine verification\n==============================");
  test_cg();
  test_wls();
  test_admm();
  test_mdp();
  test_pde_svd();
  std::printf("\n%s (%d failure%s)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
