// ============================================================================
// apex/linalg.hpp  -  Hand-written numerical linear algebra
//
// Syllabus (Unit 1): Krylov subspace methods, Conjugate Gradient method.
// No external maths library is used: every routine below is implemented from
// first principles on raw float arrays so it can be read and verified.
// ============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

namespace apex {

inline float dot(const float *a, const float *b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

inline float norm2(const float *a, int n) { return std::sqrt(dot(a, a, n)); }

// ----------------------------------------------------------------------------
// Preconditioned Conjugate Gradient (Krylov subspace solver) for A x = b,
// A symmetric positive definite and given only as an operator  apply(in, out).
//
// After k steps, x_k minimises the A-norm error over the Krylov subspace
//   K_k(A, r0) = span{ r0, A r0, A^2 r0, ..., A^{k-1} r0 },
// so in exact arithmetic CG terminates in at most n steps. A diagonal (Jacobi)
// preconditioner M = diag(A) clusters the spectrum and cuts the iteration count.
// x holds the initial guess on entry (warm start) and the solution on exit.
// ----------------------------------------------------------------------------
struct CGStats {
  int iterations = 0;
  float residual = 0.0f; // ||b - A x||_2 at exit
};

constexpr int kMaxCG = 256;

template <class Op>
CGStats pcg_solve(Op apply, const float *b, float *x, const float *inv_diag,
                  int n, int max_iters, float tol) {
  float r[kMaxCG], z[kMaxCG], p[kMaxCG], Ap[kMaxCG];
  CGStats st;

  apply(x, Ap);
  for (int i = 0; i < n; ++i) {
    r[i] = b[i] - Ap[i];
    z[i] = inv_diag ? inv_diag[i] * r[i] : r[i];
    p[i] = z[i];
  }
  float rz = dot(r, z, n);
  const float b_norm = std::max(norm2(b, n), 1e-12f);

  for (int k = 0; k < max_iters; ++k) {
    if (norm2(r, n) <= tol * b_norm)
      break;
    apply(p, Ap);
    const float pAp = dot(p, Ap, n);
    if (pAp <= 1e-20f) // direction of (numerically) zero curvature
      break;
    const float alpha = rz / pAp;
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      z[i] = inv_diag ? inv_diag[i] * r[i] : r[i];
    }
    const float rz_new = dot(r, z, n);
    const float beta = rz_new / rz;
    for (int i = 0; i < n; ++i)
      p[i] = z[i] + beta * p[i];
    rz = rz_new;
    st.iterations = k + 1;
  }
  st.residual = norm2(r, n);
  return st;
}

// ----------------------------------------------------------------------------
// Cholesky factorisation A = L L^T and solve, for small dense SPD systems
// (used by the weighted least-squares normal equations, size <= 4).
// A is row-major n x n. Returns false if A is not positive definite.
// ----------------------------------------------------------------------------
inline bool cholesky_solve(const double *A, const double *b, double *x, int n) {
  double L[16] = {0.0};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = A[i * n + j];
      for (int k = 0; k < j; ++k)
        s -= L[i * n + k] * L[j * n + k];
      if (i == j) {
        if (s <= 1e-12)
          return false;
        L[i * n + i] = std::sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  double y[4];
  for (int i = 0; i < n; ++i) { // forward substitution  L y = b
    double s = b[i];
    for (int k = 0; k < i; ++k)
      s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) { // back substitution  L^T x = y
    double s = y[i];
    for (int k = i + 1; k < n; ++k)
      s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return true;
}

} // namespace apex
