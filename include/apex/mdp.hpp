// ============================================================================
// apex/mdp.hpp  -  Longitudinal decision making as a Markov Decision Process
//
// Syllabus (Unit 3): Markov chains, Markov decision processes.
//
// State  s = (distance bin d, speed bin v):  d in {CLEAR, FAR, MED, CLOSE},
//                                             v in {SLOW, CRUISE, FAST}
// Action a in {ACCEL, MAINTAIN, BRAKE}
// Transition P(s'|s,a) = P(v'|v,a) * P(d'|d,v')  (stochastic: the actuator does
// not always achieve the intended speed change, and the obstacle ahead may
// approach, hold, or drive away).
//
// Bellman optimality equation, solved by value iteration:
//   V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V*(s') ]
// The Bellman operator is a gamma-contraction in the sup-norm, so iteration
// converges geometrically; stopping when ||V_{k+1} - V_k||_inf < eps(1-gamma)/
// (2 gamma) guarantees the greedy policy is eps-optimal.
//
// Under the optimal policy the system is a plain Markov chain with transition
// matrix P_pi. Its stationary distribution  pi = pi P_pi  (power iteration)
// tells us the long-run fraction of time spent in each state; pi(CLOSE, FAST)
// is the engine's long-run "risk" figure shown on the dashboard.
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

namespace apex {

enum DistBin { D_CLEAR = 0, D_FAR = 1, D_MED = 2, D_CLOSE = 3 };
enum SpeedBin { V_SLOW = 0, V_CRUISE = 1, V_FAST = 2 };
enum Action { ACCEL = 0, MAINTAIN = 1, BRAKE = 2 };

struct DrivingMDP {
  static constexpr int ND = 4, NV = 3, NS = ND * NV, NA = 3;

  float gamma = 0.92f;
  float P[NA][NS][NS] = {{{0}}};
  float R[NS][NA] = {{0}};
  float V[NS] = {0};
  float Q[NS][NA] = {{0}};
  int policy[NS] = {0};
  float stationary[NS] = {0};
  int iterations = 0;
  float bellman_residual = 0.0f;

  static int idx(int d, int v) { return d * NV + v; }

  DrivingMDP() {
    build_model();
    value_iteration(1e-3f);
    stationary_distribution();
  }

  void build_model() {
    // P(v' | v, a)
    float Pv[NA][NV][NV] = {{{0}}};
    for (int v = 0; v < NV; ++v) {
      const int up = std::min(v + 1, NV - 1), dn = std::max(v - 1, 0);
      Pv[ACCEL][v][up] += 0.85f;
      Pv[ACCEL][v][v] += 0.15f;
      Pv[MAINTAIN][v][v] += 0.90f;
      Pv[MAINTAIN][v][up] += 0.05f;
      Pv[MAINTAIN][v][dn] += 0.05f;
      Pv[BRAKE][v][dn] += 0.85f;
      Pv[BRAKE][v][v] += 0.15f;
    }
    // P(d' | d, v'): faster ego -> gap closes more often
    const float p_closer[NV] = {0.10f, 0.30f, 0.50f};
    const float p_farther[NV] = {0.40f, 0.20f, 0.08f};
    float Pd[NV][ND][ND] = {{{0}}};
    for (int v = 0; v < NV; ++v) {
      for (int d = 0; d < ND; ++d) {
        float pc = p_closer[v], pf = p_farther[v];
        if (d == D_CLEAR) { // an obstacle appears ahead
          pc = 0.08f + 0.04f * v;
          pf = 0.0f;
        }
        const int closer = std::min(d + 1, ND - 1), farther = std::max(d - 1, 0);
        Pd[v][d][closer] += pc;
        Pd[v][d][farther] += pf;
        Pd[v][d][d] += 1.0f - pc - pf;
      }
    }
    for (int a = 0; a < NA; ++a)
      for (int d = 0; d < ND; ++d)
        for (int v = 0; v < NV; ++v)
          for (int v2 = 0; v2 < NV; ++v2)
            for (int d2 = 0; d2 < ND; ++d2)
              P[a][idx(d, v)][idx(d2, v2)] = Pv[a][v][v2] * Pd[v2][d][d2];

    // Rewards: progress vs. safety vs. comfort
    const float progress[NV] = {0.0f, 4.0f, 6.0f};
    for (int d = 0; d < ND; ++d) {
      for (int v = 0; v < NV; ++v) {
        float base = progress[v];
        if (d == D_CLOSE)
          base += (v == V_FAST) ? -120.0f : (v == V_CRUISE ? -50.0f : 0.0f);
        if (d == D_MED)
          base += (v == V_FAST) ? -12.0f : 0.0f;
        R[idx(d, v)][ACCEL] = base - 0.5f;
        R[idx(d, v)][MAINTAIN] = base;
        R[idx(d, v)][BRAKE] =
            base - 1.5f - ((v == V_SLOW && d != D_CLOSE) ? 3.0f : 0.0f);
      }
    }
  }

  void value_iteration(float eps) {
    const float stop = eps * (1.0f - gamma) / (2.0f * gamma);
    std::memset(V, 0, sizeof(V));
    for (iterations = 1; iterations <= 2000; ++iterations) {
      float Vn[NS];
      bellman_residual = 0.0f;
      for (int s = 0; s < NS; ++s) {
        float best = -1e30f;
        for (int a = 0; a < NA; ++a) {
          float ev = 0.0f;
          for (int s2 = 0; s2 < NS; ++s2)
            ev += P[a][s][s2] * V[s2];
          Q[s][a] = R[s][a] + gamma * ev;
          if (Q[s][a] > best) {
            best = Q[s][a];
            policy[s] = a;
          }
        }
        Vn[s] = best;
        bellman_residual = std::max(bellman_residual, std::fabs(Vn[s] - V[s]));
      }
      std::memcpy(V, Vn, sizeof(V));
      if (bellman_residual < stop)
        break;
    }
  }

  // Power iteration on the closed-loop chain: pi_{k+1} = pi_k P_pi
  void stationary_distribution() {
    float pi[NS];
    for (int s = 0; s < NS; ++s)
      pi[s] = 1.0f / NS;
    for (int it = 0; it < 5000; ++it) {
      float nx[NS] = {0};
      for (int s = 0; s < NS; ++s)
        for (int s2 = 0; s2 < NS; ++s2)
          nx[s2] += pi[s] * P[policy[s]][s][s2];
      float diff = 0.0f;
      for (int s = 0; s < NS; ++s)
        diff += std::fabs(nx[s] - pi[s]);
      std::memcpy(pi, nx, sizeof(pi));
      if (diff < 1e-7f)
        break;
    }
    std::memcpy(stationary, pi, sizeof(pi));
  }

  // Discretise continuous measurements into an MDP state.
  static int distance_bin(float gap_m, float closing_mps) {
    const float ttc = closing_mps > 0.1f ? gap_m / closing_mps : 1e9f;
    if (gap_m < 8.0f || ttc < 2.0f)
      return D_CLOSE;
    if (gap_m < 18.0f || ttc < 4.0f)
      return D_MED;
    if (gap_m < 35.0f)
      return D_FAR;
    return D_CLEAR;
  }
  static int speed_bin(float v, float v_cruise) {
    if (v < 0.55f * v_cruise)
      return V_SLOW;
    if (v > 0.9f * v_cruise)
      return V_FAST;
    return V_CRUISE;
  }
};

} // namespace apex
