// ============================================================================
// apex/optim.hpp  -  Constrained convex speed planning with ADMM
//
// Syllabus (Unit 1): constrained convex optimisation, formulating problems as
// QP, sparsity-inducing (L1) penalties, ADMM, Conjugate Gradient / Krylov,
// Lagrange multipliers and KKT conditions.
//
// Decision variable: v in R^n, the ego speed at t_i = (i+1) dt, i = 0..n-1.
//
//   minimise   1/2 w_ref ||v - r||^2           (track the MDP's target speed)
//            + 1/2 w_acc ||a||^2               (comfort: small accelerations)
//            + lambda    ||j||_1               (SPARSE jerk -> piecewise-
//                                               constant acceleration)
//   subject to 0 <= v_i <= v_max               (speed limit / curve limit)
//              a_min <= a_i <= a_max           (actuator limits)
//              |j_i| <= j_max dt               (bounded jerk: no sudden kicks)
//              p_i + T_h v_i <= g_i            (keep a safe gap to the lead
//                                               obstacle; linear in v!)
//   where  a_i = (v_i - v_{i-1}) / dt,  v_{-1} = v0 (measured speed)
//          j_i = a_i - a_{i-1},          a_{-1} = a0 (measured accel)
//          p_i = dt * sum_{k<=i} v_k     (distance travelled)
//          g_i = gap0 + v_lead t_i - s0  (lead position minus standstill gap)
//
// Every term is convex, every constraint is affine, so a global optimum exists
// and ADMM converges to it. We write all four terms as  A_k v - b_k = z_k  and
// run scaled-form ADMM:
//   x-step:  (w_ref I + w_acc A2^T A2 + rho sum_k A_k^T A_k) v
//              = w_ref r + w_acc A2^T b2 + rho sum_k A_k^T (z_k + b_k - u_k)
//            -> SPD, banded except for the gap block; solved MATRIX-FREE with
//               preconditioned CG (never forming the matrix).
//   z-step:  z1, z2, z4 = Euclidean projection onto boxes / half-spaces
//            z3        = clip(soft-threshold(., lambda/rho))  (prox of L1 +
//                        box: the composition is exact for separable terms)
//   u-step:  u_k += A_k v - b_k - z_k   (scaled Lagrange multipliers)
//   rho is adapted by residual balancing (Boyd et al. 2011, Sec. 3.4.1):
//   if r_primal >> r_dual, increase rho; if r_dual >> r_primal, decrease it.
//
// Feasibility safeguard (exact penalty): the speed limit and the gap
// constraint are imposed through  mu * max(0, violation)  instead of a hard
// indicator. For mu larger than the optimal Lagrange multiplier the minimiser
// is IDENTICAL to the hard-constrained one whenever the problem is feasible;
// when it is not (obstacle appears too close to stop within a_min, or a curve
// limit drops faster than the brakes allow) ADMM still converges, to the least-
// violating plan, instead of letting the multipliers diverge. The prox of
// mu max(0, x - g) is a closed-form shrink toward g.
//
// KKT conditions at the optimum: primal feasibility (A v - b = z) and dual
// feasibility (rho A^T (z - z_prev) -> 0). We report both residuals, and the
// multipliers y_k = rho u_k: a non-zero multiplier means that constraint is
// ACTIVE (complementary slackness), i.e. the car is riding the KKT boundary.
// ============================================================================
#pragma once

#include "linalg.hpp"

namespace apex {

struct SpeedPlannerParams {
  int n = 40;                // horizon steps
  float dt = 0.1f;           // horizon step [s]  -> 4 s look-ahead
  float w_ref = 1.0f;        // speed tracking weight
  float w_acc = 2.0f;        // acceleration (comfort) weight
  float lambda_jerk = 2.0f;  // L1 weight on jerk (sparsity)
  float jerk_max = 4.0f;     // |jerk| bound [m/s^3]
  float rho = 1.0f;          // initial ADMM penalty (adapted online)
  // per-block penalty scale c_k (rho_k = rho c_k), i.e. row equilibration of
  // A_k so speeds [m/s], accelerations, jerk steps and gaps [m] are comparable
  float block_scale[4] = {1.0f, 1.0f, 10.0f, 1.0f};
  float relax = 1.6f;        // over-relaxation alpha in [1.5, 1.8]
  float mu_speed = 40.0f;    // exact-penalty weight, speed limit
  float mu_gap = 80.0f;      // exact-penalty weight, gap constraint
  float a_min = -4.5f;       // max comfortable braking [m/s^2]
  float a_max = 2.0f;        // max acceleration [m/s^2]
  float headway = 1.0f;      // time headway T_h [s]
  float standstill = 4.0f;   // s0: gap kept when stopped [m]
  int max_iters = 150;
  float tol = 1e-3f;         // stop when both residuals < tol (scaled)
};

class SpeedPlannerADMM {
public:
  static constexpr int kMaxN = 64;

  using Params = SpeedPlannerParams;


  struct Problem {
    float v0 = 0.0f;     // measured speed [m/s]
    float a0 = 0.0f;     // measured acceleration [m/s^2]
    float v_ref = 0.0f;  // target speed from the MDP [m/s]
    float v_max = 0.0f;  // speed limit (curve-limited) [m/s]
    bool has_lead = false;
    float gap = 100.0f;  // distance to obstacle ahead [m]
    float v_lead = 0.0f; // estimated obstacle speed [m/s]
  };

  struct Result {
    int n = 0;
    float v[kMaxN] = {0};
    float a[kMaxN] = {0};
    float gap_limit[kMaxN] = {0}; // max speed the gap constraint alone allows
    int iterations = 0;
    int cg_iterations = 0;
    float r_primal = 0.0f, r_dual = 0.0f;
    float objective = 0.0f;
    int active_speed = 0, active_accel = 0, active_gap = 0;
    int nonzero_jerk = 0; // sparsity achieved by the L1 term
    float max_multiplier = 0.0f;
    float rho = 0.0f; // penalty after adaptation
    bool converged = false;
    float r_hist[64] = {0}; // primal residual per iteration (for the plot)
  };

  SpeedPlannerADMM() : SpeedPlannerADMM(Params()) {}
  explicit SpeedPlannerADMM(const Params &p) : P(p) {
    P.n = std::min(P.n, kMaxN);
    rho_ = P.rho;
    for (int lead = 0; lead < 2; ++lead)
      build_diagonals(lead != 0, diag_fixed[lead], diag_pen[lead]);
  }

  const Params &params() const { return P; }

  Result solve(const Problem &pb) {
    const int n = P.n;
    const float dt = P.dt;
    const bool lead = pb.has_lead;
    Result R;
    R.n = n;

    // --- constant vectors b_k and constraint data ---------------------------
    float b2[kMaxN] = {0}, b3[kMaxN] = {0}, g[kMaxN];
    b2[0] = pb.v0 / dt;
    // a measured acceleration outside the actuator box (sensor noise) would make
    // the jerk-bounded problem infeasible at step 0, so project it first
    const float a0 = clampf(pb.a0, P.a_min, P.a_max);
    b3[0] = pb.v0 / dt + a0;
    if (n > 1)
      b3[1] = -pb.v0 / dt;
    for (int i = 0; i < n; ++i) {
      const float t = (i + 1) * dt;
      g[i] = std::max(0.0f, pb.gap + pb.v_lead * t - P.standstill);
    }
    const float vmax = std::max(0.0f, pb.v_max);

    // --- warm start: shift last solution one step forward in time ----------
    if (!warm || lead != last_lead) {
      for (int i = 0; i < n; ++i) {
        v[i] = std::min(pb.v0, vmax);
        for (int k = 0; k < 4; ++k)
          u[k][i] = 0.0f;
      }
      apply_blocks(v, Av);
      sub_b(Av, b2, b3);
      for (int k = 0; k < 4; ++k)
        for (int i = 0; i < n; ++i)
          z[k][i] = Av[k][i];
    } else {
      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i + 1 < n; ++i) {
          z[k][i] = z[k][i + 1];
          u[k][i] = u[k][i + 1];
        }
      }
      for (int i = 0; i + 1 < n; ++i)
        v[i] = v[i + 1];
    }
    warm = true;
    last_lead = lead;

    // rhs constant part: w_ref r + w_acc A2^T b2
    float rhs0[kMaxN], tmp[kMaxN];
    D1T(b2, tmp);
    for (int i = 0; i < n; ++i)
      rhs0[i] = P.w_ref * pb.v_ref + P.w_acc * tmp[i] / dt;

    float z_old[4][kMaxN];
    for (int it = 0; it < P.max_iters; ++it) {
      // ---------------- x-update: solve SPD system with PCG ----------------
      float c[4][kMaxN], rhs[kMaxN];
      const float *cs_ = P.block_scale;
      for (int i = 0; i < n; ++i) {
        c[0][i] = cs_[0] * (z[0][i] - u[0][i]);
        c[1][i] = cs_[1] * (z[1][i] + b2[i] - u[1][i]);
        c[2][i] = cs_[2] * (z[2][i] + b3[i] - u[2][i]);
        c[3][i] = lead ? cs_[3] * (z[3][i] - u[3][i]) : 0.0f;
      }
      applyT_blocks(c, rhs, lead);
      const float rho = rho_;
      for (int i = 0; i < n; ++i)
        rhs[i] = rhs0[i] + rho * rhs[i];

      float inv_diag[kMaxN];
      for (int i = 0; i < n; ++i)
        inv_diag[i] = 1.0f / (diag_fixed[lead][i] + rho * diag_pen[lead][i]);
      auto op = [&](const float *in, float *out) { apply_M(in, out, lead); };
      CGStats cs = pcg_solve(op, rhs, v, inv_diag, n, 2 * n, 1e-5f);
      R.cg_iterations += cs.iterations;

      // ---------------- z-update: projections and L1 prox ------------------
      // over-relaxation: xh = alpha (A v - b) + (1 - alpha) z_old
      apply_blocks(v, Av);
      sub_b(Av, b2, b3);
      std::memcpy(z_old, z, sizeof(z_old));
      float xh[4][kMaxN];
      const float al = P.relax;
      for (int k = 0; k < 4; ++k)
        for (int i = 0; i < n; ++i)
          xh[k][i] = al * Av[k][i] + (1.0f - al) * z_old[k][i];
      const float kappa = P.lambda_jerk / (rho * cs_[2]);
      const float hmax = P.jerk_max * dt;
      const float k_speed = P.mu_speed / (rho * cs_[0]);
      const float k_gap = P.mu_gap / (rho * cs_[3]);
      for (int i = 0; i < n; ++i) {
        z[0][i] = std::max(0.0f, shrink_above(xh[0][i] + u[0][i], vmax, k_speed));
        z[1][i] = clampf(xh[1][i] + u[1][i], P.a_min, P.a_max);
        z[2][i] = clampf(soft_threshold(xh[2][i] + u[2][i], kappa), -hmax, hmax);
        z[3][i] = lead ? shrink_above(xh[3][i] + u[3][i], g[i], k_gap) : 0.0f;
      }

      // ---------------- u-update (dual ascent) + residuals -----------------
      float rp = 0.0f;
      float dz[4][kMaxN];
      for (int k = 0; k < 4; ++k) {
        if (k == 3 && !lead)
          continue;
        for (int i = 0; i < n; ++i) {
          u[k][i] += xh[k][i] - z[k][i];
          const float res = Av[k][i] - z[k][i];
          rp += cs_[k] * res * res;
          dz[k][i] = cs_[k] * (z[k][i] - z_old[k][i]);
        }
      }
      if (!lead)
        for (int i = 0; i < n; ++i)
          dz[3][i] = 0.0f;
      float sdual[kMaxN];
      applyT_blocks(dz, sdual, lead);
      R.r_primal = std::sqrt(rp);
      R.r_dual = rho * norm2(sdual, n);
      R.rho = rho_;
      R.iterations = it + 1;
      if (it < 64)
        R.r_hist[it] = R.r_primal;

      const float eps = P.tol * std::max(1.0f, pb.v0) * std::sqrt(4.0f * n);
      if (R.r_primal < eps && R.r_dual < eps && it > 2) {
        R.converged = true;
        break;
      }
      // residual balancing; scaled duals u = y / rho must be rescaled too
      if (it % 5 == 4) {
        float f = 1.0f;
        if (R.r_primal > 10.0f * R.r_dual && rho_ < 50.0f)
          f = 2.0f;
        else if (R.r_dual > 10.0f * R.r_primal && rho_ > 0.05f)
          f = 0.5f;
        if (f != 1.0f) {
          rho_ *= f;
          for (int k = 0; k < 4; ++k)
            for (int i = 0; i < n; ++i)
              u[k][i] /= f;
        }
      }
    }

    // --- feasibility restoration --------------------------------------------
    // At the iteration cap the ADMM iterate may still violate the hard
    // actuator limits slightly. A forward pass projects each step onto
    //   a_i in [max(a_min, a_{i-1} - j_max dt), min(a_max, a_{i-1} + j_max dt)],
    //   v_i >= 0
    // so the commanded profile is ALWAYS physically executable. At a converged
    // ADMM solution this pass changes nothing.
    float prev_v = pb.v0, prev_a = a0, pos = 0.0f;
    const float jstep = P.jerk_max * dt;
    for (int i = 0; i < n; ++i) {
      float a = (z[0][i] - prev_v) / dt;
      a = clampf(a, std::max(P.a_min, prev_a - jstep),
                 std::min(P.a_max, prev_a + jstep));
      R.v[i] = std::max(0.0f, prev_v + a * dt);
      R.a[i] = (R.v[i] - prev_v) / dt;
      const float jerk = R.a[i] - prev_a;
      if (std::fabs(jerk) > 0.05f)
        ++R.nonzero_jerk;
      prev_v = R.v[i];
      prev_a = R.a[i];
      pos += dt * R.v[i];
      R.gap_limit[i] = lead ? std::max(0.0f, (g[i] - (pos - dt * R.v[i])) /
                                                 (P.headway + dt))
                            : vmax;
      // complementary slackness: active constraint <=> non-zero multiplier
      const float *cs_ = P.block_scale;
      const float y0 = rho_ * cs_[0] * u[0][i], y1 = rho_ * cs_[1] * u[1][i];
      const float y3 = lead ? rho_ * cs_[3] * u[3][i] : 0.0f;
      if (std::fabs(y0) > 1e-2f)
        ++R.active_speed;
      if (std::fabs(y1) > 1e-2f)
        ++R.active_accel;
      if (y3 > 1e-2f)
        ++R.active_gap;
      R.max_multiplier =
          std::max({R.max_multiplier, std::fabs(y0), std::fabs(y1), y3});
      R.objective += 0.5f * P.w_ref * (R.v[i] - pb.v_ref) * (R.v[i] - pb.v_ref) +
                     0.5f * P.w_acc * R.a[i] * R.a[i] +
                     P.lambda_jerk * std::fabs(jerk);
    }
    return R;
  }

  void reset() {
    warm = false;
    rho_ = P.rho;
  }

private:
  Params P;
  bool warm = false, last_lead = false;
  float rho_ = 1.0f;
  float v[kMaxN] = {0};
  float z[4][kMaxN] = {{0}}, u[4][kMaxN] = {{0}}, Av[4][kMaxN] = {{0}};
  // diag(M) = diag_fixed + rho * diag_pen  (Jacobi preconditioner)
  float diag_fixed[2][kMaxN] = {{0}}, diag_pen[2][kMaxN] = {{0}};

  static float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
  }
  // prox of k * max(0, x - g):  identity below g, flat at g, shifted above
  static float shrink_above(float x, float g, float k) {
    return (x <= g) ? x : (x <= g + k ? g : x - k);
  }
  static float soft_threshold(float x, float k) {
    return (x > k) ? x - k : (x < -k ? x + k : 0.0f);
  }

  // D1: first difference with v_{-1} = 0 (the boundary value lives in b_k)
  void D1(const float *x, float *y) const {
    y[0] = x[0];
    for (int i = 1; i < P.n; ++i)
      y[i] = x[i] - x[i - 1];
  }
  void D1T(const float *x, float *y) const {
    for (int i = 0; i < P.n - 1; ++i)
      y[i] = x[i] - x[i + 1];
    y[P.n - 1] = x[P.n - 1];
  }
  // S: cumulative sum (integration), S^T: reverse cumulative sum
  void S(const float *x, float *y) const {
    float s = 0.0f;
    for (int i = 0; i < P.n; ++i)
      y[i] = (s += x[i]);
  }
  void ST(const float *x, float *y) const {
    float s = 0.0f;
    for (int i = P.n - 1; i >= 0; --i)
      y[i] = (s += x[i]);
  }

  // A_k v for the four blocks (without the -b_k shift)
  void apply_blocks(const float *x, float out[4][kMaxN]) const {
    const float inv_dt = 1.0f / P.dt;
    float d1[kMaxN], d2[kMaxN], s[kMaxN];
    D1(x, d1);
    D1(d1, d2);
    S(x, s);
    for (int i = 0; i < P.n; ++i) {
      out[0][i] = x[i];
      out[1][i] = d1[i] * inv_dt;
      out[2][i] = d2[i] * inv_dt;
      out[3][i] = P.dt * s[i] + P.headway * x[i];
    }
  }
  void sub_b(float out[4][kMaxN], const float *b2, const float *b3) const {
    for (int i = 0; i < P.n; ++i) {
      out[1][i] -= b2[i];
      out[2][i] -= b3[i];
    }
  }
  // sum_k A_k^T c_k
  void applyT_blocks(const float c[4][kMaxN], float *out, bool lead) const {
    const float inv_dt = 1.0f / P.dt;
    float t1[kMaxN], t2[kMaxN], t2b[kMaxN], t3[kMaxN];
    D1T(c[1], t1);
    D1T(c[2], t2b);
    D1T(t2b, t2);
    ST(c[3], t3);
    for (int i = 0; i < P.n; ++i) {
      out[i] = c[0][i] + inv_dt * (t1[i] + t2[i]);
      if (lead)
        out[i] += P.dt * t3[i] + P.headway * c[3][i];
    }
  }
  // M v = w_ref v + w_acc A2^T A2 v + rho sum_k A_k^T A_k v
  void apply_M(const float *x, float *out, bool lead) const {
    apply_M_parts(x, out, lead, rho_);
  }
  void apply_M_parts(const float *x, float *out, bool lead, float rho) const {
    float blocks[4][kMaxN], acc[kMaxN];
    apply_blocks(x, blocks);
    float only_a[4][kMaxN] = {{0}};
    std::memcpy(only_a[1], blocks[1], sizeof(float) * P.n);
    for (int k = 0; k < 4; ++k)
      for (int i = 0; i < P.n; ++i)
        blocks[k][i] *= (k == 3 && !lead) ? 0.0f : P.block_scale[k];
    applyT_blocks(blocks, out, lead);
    applyT_blocks(only_a, acc, false);
    for (int i = 0; i < P.n; ++i)
      out[i] = P.w_ref * x[i] + P.w_acc * acc[i] + rho * out[i];
  }
  // diag(M) split into its rho-independent and rho-proportional parts
  void build_diagonals(bool lead, float *d_fixed, float *d_pen) const {
    float e[kMaxN] = {0}, c0[kMaxN], c1[kMaxN];
    for (int j = 0; j < P.n; ++j) {
      e[j] = 1.0f;
      apply_M_parts(e, c0, lead, 0.0f);
      apply_M_parts(e, c1, lead, 1.0f);
      d_fixed[j] = c0[j];
      d_pen[j] = c1[j] - c0[j];
      e[j] = 0.0f;
    }
  }
};

} // namespace apex
