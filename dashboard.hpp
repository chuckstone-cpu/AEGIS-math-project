// ============================================================================
// dashboard.hpp  -  Live "glass box" dashboard for the APEX engine
//
// OpenCV is used here ONLY for drawing. Every number shown is produced by the
// hand-written maths in include/apex/*.hpp, so the viewer can watch each
// algorithm (lane WLS fit, MDP policy, ADMM plan + KKT residuals) decide.
// ============================================================================
#pragma once

#include <deque>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "apex/mdp.hpp"
#include "apex/optim.hpp"
#include "apex/vision.hpp"

namespace dash {

using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;

// ---- palette (BGR) ---------------------------------------------------------
const Scalar BG(22, 18, 16);
const Scalar PANEL(38, 32, 28);
const Scalar BORDER(70, 62, 56);
const Scalar GRID(58, 52, 48);
const Scalar TEXT(235, 235, 235);
const Scalar MUTED(160, 150, 145);
const Scalar CYAN(255, 220, 60);
const Scalar YELLOW(60, 215, 255);
const Scalar ORANGE(40, 140, 255);
const Scalar GREEN(110, 220, 90);
const Scalar RED(70, 70, 235);
const Scalar MAGENTA(220, 120, 220);

struct Timing {
  const char *name;
  double ms;
};

struct Data {
  bool webots = false;
  int frame = 0;
  double sim_time = 0.0;
  double fps = 0.0;
  const Mat *camera = nullptr; // 800x600 BGR
  const apex::LaneResult *lane = nullptr;
  float steer = 0.0f, raw_steer = 0.0f;
  float speed_kmh = 0.0f, cmd_kmh = 0.0f, limit_kmh = 0.0f;
  float brake = 0.0f;
  float gap_m = 100.0f, closing_mps = 0.0f;
  bool has_lead = false;
  int dist_bin = 0, speed_bin = 0, action = 0;
  const apex::DrivingMDP *mdp = nullptr;
  const apex::SpeedPlannerADMM::Result *plan = nullptr;
  float plan_dt = 0.1f;
  float v_ref_kmh = 0.0f;
  std::vector<Timing> timings;
  // sensor inset
  const float *lidar = nullptr;
  int lidar_n = 0;
  float lidar_fov = 0.0f;
  const Mat *svd_foreground = nullptr; // 64x64 float [0,1]
  const Mat *pde_view = nullptr;       // grayscale PDE output
  float motion[3] = {0, 0, 0};
  bool emergency = false;
};

// ---- small drawing helpers ---------------------------------------------------
inline void txt(Mat &m, const std::string &s, Point p, double scale = 0.5,
                Scalar c = TEXT, int thick = 1) {
  cv::putText(m, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, c, thick, cv::LINE_AA);
}
inline std::string fmt(const char *f, double v) {
  char b[64];
  std::snprintf(b, sizeof(b), f, v);
  return b;
}
inline void panel(Mat &m, const Rect &r, const std::string &title,
                  const std::string &subtitle = "") {
  cv::rectangle(m, r, PANEL, cv::FILLED);
  cv::rectangle(m, r, BORDER, 1);
  cv::rectangle(m, Rect(r.x, r.y, 4, 28), CYAN, cv::FILLED);
  txt(m, title, Point(r.x + 14, r.y + 20), 0.56, TEXT, 1);
  if (!subtitle.empty()) {
    int base = 0;
    const cv::Size ts =
        cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.56, 1, &base);
    txt(m, subtitle, Point(r.x + 24 + ts.width, r.y + 20), 0.42, MUTED);
  }
}
inline void dashed_hline(Mat &m, int x0, int x1, int y, Scalar c) {
  for (int x = x0; x < x1; x += 10)
    cv::line(m, Point(x, y), Point(std::min(x + 5, x1), y), c, 1, cv::LINE_AA);
}
inline Scalar lerp(Scalar a, Scalar b, double t) {
  t = std::max(0.0, std::min(1.0, t));
  return Scalar(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t);
}

// ---- header ------------------------------------------------------------------
inline void draw_header(Mat &c, const Data &d) {
  cv::rectangle(c, Rect(0, 0, c.cols, 48), Scalar(30, 25, 22), cv::FILLED);
  txt(c, "AEGIS / APEX", Point(18, 32), 0.85, CYAN, 2);
  txt(c, "open mathematical driving engine", Point(214, 32), 0.55, MUTED);
  const std::string mode =
      d.webots ? "WEBOTS SIMULATION" : "DASHCAM VIDEO (offline)";
  txt(c, mode, Point(560, 32), 0.55, d.webots ? GREEN : YELLOW, 1);
  std::string right = "frame " + std::to_string(d.frame) + "   t = " +
                      fmt("%.1f s", d.sim_time) + "   " +
                      fmt("%.0f FPS", d.fps);
  int base = 0;
  cv::Size ts = cv::getTextSize(right, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
  txt(c, right, Point(c.cols - ts.width - 18, 32), 0.55, TEXT);
}

// ---- camera with lane-estimation overlay -------------------------------------
inline void draw_camera(Mat &c, const Rect &r, const Data &d) {
  Mat view = c(r);
  if (d.camera && !d.camera->empty())
    d.camera->copyTo(view);
  else
    view.setTo(Scalar(0, 0, 0));

  if (d.webots && d.lane) {
    const apex::LaneResult &L = *d.lane;
    for (const auto &p : L.left_pts)
      cv::circle(view, Point(p.x, p.y), 2, YELLOW, cv::FILLED, cv::LINE_AA);
    for (const auto &p : L.right_pts)
      cv::circle(view, Point(p.x, p.y), 2, ORANGE, cv::FILLED, cv::LINE_AA);
    for (const auto &p : L.center_pts)
      cv::circle(view, Point(p.x, p.y), 2, Scalar(200, 200, 200), cv::FILLED,
                 cv::LINE_AA);
    if (L.fit.ok) {
      // WLS quadratic lane centre and its +-2 sigma predictive band
      std::vector<Point> mid, lo, hi;
      const float hw = 0.5f * r.width;
      for (int k = 0; k <= 24; ++k) {
        const float t = k / 24.0f;
        const int y = L.y_top + static_cast<int>(t * (L.y_bottom - L.y_top));
        const float x = L.fit.eval(t);
        const float s = 2.0f * std::sqrt(L.fit.predictive_variance(t) +
                                         L.fit.sigma * L.fit.sigma);
        mid.emplace_back(static_cast<int>(hw + x * hw), y);
        lo.emplace_back(static_cast<int>(hw + (x - s) * hw), y);
        hi.emplace_back(static_cast<int>(hw + (x + s) * hw), y);
      }
      cv::polylines(view, lo, false, Scalar(140, 120, 40), 1, cv::LINE_AA);
      cv::polylines(view, hi, false, Scalar(140, 120, 40), 1, cv::LINE_AA);
      cv::polylines(view, mid, false, CYAN, 3, cv::LINE_AA);
    }
    const Point hood(r.width / 2, r.height - 18);
    const Point tgt(L.target.x, L.target.y);
    cv::arrowedLine(view, hood, tgt, GREEN, 3, cv::LINE_AA, 0, 0.06);
    cv::circle(view, tgt, 9, GREEN, 2, cv::LINE_AA);
  }

  if (d.dist_bin == apex::D_CLOSE || d.emergency) {
    cv::rectangle(view, Rect(r.width / 2 - 110, r.height / 2 - 70, 220, 140),
                  RED, 3);
    txt(view, d.emergency ? "EMERGENCY STOP" : "OBSTACLE CLOSE",
        Point(r.width / 2 - 100, r.height / 2 - 80), 0.65, RED, 2);
  }

  // translucent HUD strip
  Mat strip = view(Rect(0, 0, r.width, 34));
  Mat dark(strip.size(), strip.type(), Scalar(0, 0, 0));
  cv::addWeighted(strip, 0.45, dark, 0.55, 0, strip);
  if (d.webots && d.lane) {
    txt(view,
        "lane fit: " + std::to_string(d.lane->fit.used) + " pts, " +
            std::to_string(d.lane->fit.rejected) + " Chebyshev-rejected" +
            fmt(",  sigma %.3f", d.lane->fit.sigma) +
            fmt(",  Kalman gain %.2f", d.lane->filter_gain),
        Point(10, 22), 0.48, TEXT);
  } else {
    txt(view, "PDE-smoothed + rank-1 SVD background -> obstacle cue",
        Point(10, 22), 0.48, TEXT);
  }
  cv::rectangle(c, r, BORDER, 1);
}

// ---- steering wheel, steering trace, speed gauge -----------------------------
inline void draw_vehicle_panel(Mat &c, const Rect &r, const Data &d,
                               const std::deque<float> &raw_hist,
                               const std::deque<float> &steer_hist) {
  panel(c, r, "VEHICLE", "steering & speed");
  // steering wheel
  const Point wc(r.x + 92, r.y + 120);
  const int R = 62;
  cv::circle(c, wc, R, Scalar(90, 85, 80), 8, cv::LINE_AA);
  const double ang = d.steer * 6.0; // visual wheel ratio
  for (double base : {CV_PI / 2, CV_PI / 2 + 2.2, CV_PI / 2 - 2.2}) {
    const double a = base + ang;
    cv::line(c, wc,
             Point(wc.x + static_cast<int>(R * std::cos(a)),
                   wc.y + static_cast<int>(R * std::sin(a))),
             Scalar(120, 112, 105), 6, cv::LINE_AA);
  }
  cv::circle(c, wc, 14, Scalar(110, 100, 95), cv::FILLED, cv::LINE_AA);
  const double top = -CV_PI / 2 + ang;
  cv::circle(c,
             Point(wc.x + static_cast<int>(R * std::cos(top)),
                   wc.y + static_cast<int>(R * std::sin(top))),
             6, CYAN, cv::FILLED, cv::LINE_AA);
  txt(c, fmt("%+.1f deg", d.steer * 57.2958), Point(wc.x - 40, r.y + r.height - 14),
      0.55, TEXT);

  // steering history: raw law vs. filtered + rate limited command
  const Rect pr(r.x + 190, r.y + 40, 360, r.height - 60);
  cv::rectangle(c, pr, Scalar(30, 26, 23), cv::FILLED);
  cv::line(c, Point(pr.x, pr.y + pr.height / 2),
           Point(pr.x + pr.width, pr.y + pr.height / 2), GRID, 1);
  auto trace = [&](const std::deque<float> &h, Scalar col, int th) {
    if (h.size() < 2)
      return;
    std::vector<Point> pts;
    const float lim = 0.4f;
    for (size_t i = 0; i < h.size(); ++i) {
      const int x = pr.x + static_cast<int>(i * (pr.width - 1) / 239.0);
      const float v = std::max(-lim, std::min(lim, h[i]));
      pts.emplace_back(x, pr.y + pr.height / 2 -
                              static_cast<int>(v / lim * (pr.height / 2 - 4)));
    }
    cv::polylines(c, pts, false, col, th, cv::LINE_AA);
  };
  if (!d.webots)
    txt(c, "lane model is calibrated for Webots: steering idle on dashcam",
        Point(pr.x + 6, pr.y + pr.height - 10), 0.38, MUTED);
  trace(raw_hist, Scalar(90, 90, 160), 1);
  trace(steer_hist, CYAN, 2);
  txt(c, "steer law (raw)", Point(pr.x + 6, pr.y + 16), 0.4, Scalar(120, 120, 200));
  txt(c, "command (Gaussian-filtered, rate-limited)", Point(pr.x + 6, pr.y + 32),
      0.4, CYAN);

  // speed gauge
  const Point sc(r.x + r.width - 110, r.y + 128);
  const int SR = 72;
  const double vmax = d.webots ? 50.0 : 100.0;
  cv::ellipse(c, sc, cv::Size(SR, SR), 0, 150, 390, Scalar(70, 64, 60), 10,
              cv::LINE_AA);
  const double frac = std::max(0.0, std::min(1.0, d.speed_kmh / vmax));
  cv::ellipse(c, sc, cv::Size(SR, SR), 0, 150, 150 + 240 * frac,
              d.brake > 0.05f ? RED : GREEN, 10, cv::LINE_AA);
  const double cf = std::max(0.0, std::min(1.0, d.cmd_kmh / vmax));
  const double ca = (150 + 240 * cf) * CV_PI / 180.0;
  cv::line(c, Point(sc.x + static_cast<int>((SR - 14) * std::cos(ca)),
                    sc.y + static_cast<int>((SR - 14) * std::sin(ca))),
           Point(sc.x + static_cast<int>((SR + 10) * std::cos(ca)),
                 sc.y + static_cast<int>((SR + 10) * std::sin(ca))),
           CYAN, 3, cv::LINE_AA);
  txt(c, fmt("%.0f", d.speed_kmh), Point(sc.x - 26, sc.y + 10), 1.1, TEXT, 2);
  txt(c, "km/h", Point(sc.x - 18, sc.y + 32), 0.45, MUTED);
  txt(c, fmt("cmd %.0f", d.cmd_kmh) + (d.brake > 0.05f ? fmt("  brake %.0f%%", d.brake * 100) : ""),
      Point(sc.x - 52, r.y + r.height - 14), 0.48, d.brake > 0.05f ? RED : CYAN);
}

// ---- MDP panel ----------------------------------------------------------------
inline void draw_mdp_panel(Mat &c, const Rect &r, const Data &d) {
  panel(c, r, "MARKOV DECISION PROCESS", "value iteration -> optimal policy");
  if (!d.mdp)
    return;
  const apex::DrivingMDP &m = *d.mdp;
  const char *dn[] = {"CLEAR", "FAR", "MED", "CLOSE"};
  const char *vn[] = {"SLOW", "CRUISE", "FAST"};
  const char *an[] = {"ACCEL", "HOLD", "BRAKE"};
  const Scalar ac[] = {GREEN, CYAN, RED};

  float vmin = 1e9f, vmax = -1e9f;
  for (float v : m.V) {
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  const int gx = r.x + 74, gy = r.y + 58, cw = 88, ch = 40;
  for (int v = 0; v < 3; ++v)
    txt(c, vn[v], Point(gx + v * cw + 18, gy - 8), 0.42, MUTED);
  for (int dd = 0; dd < 4; ++dd) {
    txt(c, dn[dd], Point(r.x + 14, gy + dd * ch + 25), 0.44, MUTED);
    for (int v = 0; v < 3; ++v) {
      const int s = apex::DrivingMDP::idx(dd, v);
      const double t = (m.V[s] - vmin) / std::max(1e-6f, vmax - vmin);
      const Rect cell(gx + v * cw, gy + dd * ch, cw - 4, ch - 4);
      cv::rectangle(c, cell, lerp(Scalar(60, 40, 120), Scalar(90, 120, 40), t),
                    cv::FILLED);
      txt(c, an[m.policy[s]], Point(cell.x + 8, cell.y + 16), 0.42,
          ac[m.policy[s]]);
      txt(c, fmt("V=%.0f", m.V[s]), Point(cell.x + 8, cell.y + 32), 0.38,
          Scalar(225, 225, 225));
      if (dd == d.dist_bin && v == d.speed_bin)
        cv::rectangle(c, cell, Scalar(255, 255, 255), 3);
    }
  }
  // right column: live decision
  const int tx = gx + 3 * cw + 24;
  txt(c, "state", Point(tx, r.y + 62), 0.45, MUTED);
  txt(c, std::string(dn[d.dist_bin]) + " / " + vn[d.speed_bin],
      Point(tx, r.y + 86), 0.62, TEXT, 1);
  txt(c, "Bellman-optimal action", Point(tx, r.y + 116), 0.45, MUTED);
  txt(c, d.action == apex::ACCEL ? "ACCELERATE"
                                 : (d.action == apex::BRAKE ? "BRAKE" : "MAINTAIN"),
      Point(tx, r.y + 148), 0.95, ac[d.action], 2);
  txt(c,
      d.has_lead ? fmt("gap %.1f m", d.gap_m) + fmt("   closing %.1f m/s", d.closing_mps)
                 : std::string("no obstacle in range"),
      Point(tx, r.y + 176), 0.45, TEXT);
  txt(c,
      fmt("gamma %.2f", m.gamma) + "   " + std::to_string(m.iterations) +
          " iters" + fmt("   ||dV|| %.1e", m.bellman_residual),
      Point(tx, r.y + 202), 0.42, MUTED);
  const float risk =
      m.stationary[apex::DrivingMDP::idx(apex::D_CLOSE, apex::V_FAST)] +
      m.stationary[apex::DrivingMDP::idx(apex::D_CLOSE, apex::V_CRUISE)];
  txt(c, fmt("Markov chain stationary P(close & moving) = %.4f", risk),
      Point(tx, r.y + 224), 0.42, MUTED);
}

// ---- ADMM panel ---------------------------------------------------------------
inline void draw_admm_panel(Mat &c, const Rect &r, const Data &d) {
  panel(c, r, "ADMM SPEED PLAN", "QP + L1 jerk, solved by ADMM / CG");
  if (!d.plan)
    return;
  const auto &P = *d.plan;
  const Rect pr(r.x + 46, r.y + 44, 430, 170);
  const Rect ar(r.x + 46, r.y + 228, 430, 76);
  cv::rectangle(c, pr, Scalar(30, 26, 23), cv::FILLED);
  cv::rectangle(c, ar, Scalar(30, 26, 23), cv::FILLED);

  const double vtop = d.webots ? 45.0 : 100.0; // km/h axis
  auto vy = [&](double kmh) {
    return pr.y + pr.height - static_cast<int>(kmh / vtop * (pr.height - 6)) - 3;
  };
  auto tx = [&](int i, const Rect &rr) {
    return rr.x + static_cast<int>((i + 1) * (rr.width - 1) / double(P.n));
  };
  for (int k = 0; k <= 4; ++k) {
    const double kmh = vtop * k / 4.0;
    cv::line(c, Point(pr.x, vy(kmh)), Point(pr.x + pr.width, vy(kmh)), GRID, 1);
    txt(c, fmt("%.0f", kmh), Point(r.x + 12, vy(kmh) + 4), 0.38, MUTED);
  }
  for (int s = 1; s <= 4; ++s) {
    const int x = pr.x + s * (pr.width - 1) / 4;
    cv::line(c, Point(x, pr.y), Point(x, ar.y + ar.height), GRID, 1);
    txt(c, std::to_string(s) + "s", Point(x - 14, ar.y + ar.height + 16), 0.38,
        MUTED);
  }
  dashed_hline(c, pr.x, pr.x + pr.width, vy(d.limit_kmh), ORANGE);
  dashed_hline(c, pr.x, pr.x + pr.width, vy(d.v_ref_kmh), MAGENTA);
  if (d.has_lead) {
    std::vector<Point> gl;
    for (int i = 0; i < P.n; ++i)
      gl.emplace_back(tx(i, pr),
                      vy(std::min(vtop, (double)P.gap_limit[i] * 3.6)));
    cv::polylines(c, gl, false, RED, 1, cv::LINE_AA);
  }
  std::vector<Point> vp{Point(pr.x, vy(d.speed_kmh))};
  for (int i = 0; i < P.n; ++i)
    vp.emplace_back(tx(i, pr), vy(P.v[i] * 3.6));
  cv::polylines(c, vp, false, CYAN, 3, cv::LINE_AA);

  // acceleration subplot, +-4 m/s^2
  const int amid = ar.y + ar.height / 2;
  cv::line(c, Point(ar.x, amid), Point(ar.x + ar.width, amid), GRID, 1);
  for (int i = 0; i < P.n; ++i) {
    const int x = tx(i, ar);
    const int h = static_cast<int>(P.a[i] / 4.0 * (ar.height / 2 - 2));
    cv::line(c, Point(x, amid), Point(x, amid - h), P.a[i] < 0 ? RED : GREEN, 5);
  }
  txt(c, "accel", Point(ar.x + 6, ar.y + 14), 0.38, MUTED);

  // legend
  int ly = pr.y + 14;
  txt(c, "plan v(t)", Point(pr.x + pr.width - 120, ly), 0.38, CYAN);
  txt(c, "MDP target", Point(pr.x + pr.width - 120, ly += 14), 0.38, MAGENTA);
  txt(c, "speed limit", Point(pr.x + pr.width - 120, ly += 14), 0.38, ORANGE);
  if (d.has_lead)
    txt(c, "gap constraint", Point(pr.x + pr.width - 120, ly += 14), 0.38, RED);

  // solver diagnostics (KKT)
  const int sx = pr.x + pr.width + 24;
  int sy = r.y + 58;
  auto row = [&](const std::string &k, const std::string &v, Scalar col = TEXT) {
    txt(c, k, Point(sx, sy), 0.42, MUTED);
    txt(c, v, Point(sx + 128, sy), 0.46, col);
    sy += 24;
  };
  row("status", P.converged ? "converged" : "iter cap", P.converged ? GREEN : YELLOW);
  row("ADMM iters", std::to_string(P.iterations));
  row("CG (Krylov) steps", std::to_string(P.cg_iterations));
  row("primal resid", fmt("%.2e", P.r_primal));
  row("dual resid", fmt("%.2e", P.r_dual));
  row("rho (adaptive)", fmt("%.2f", P.rho));
  row("active: v / a", std::to_string(P.active_speed) + " / " +
                           std::to_string(P.active_accel));
  row("active: gap", std::to_string(P.active_gap),
      P.active_gap ? RED : TEXT);
  row("max multiplier", fmt("%.2f", P.max_multiplier));
  row("jerk nnz (L1)", std::to_string(P.nonzero_jerk) + " / " + std::to_string(P.n));
}

// ---- sensor + timing panel ----------------------------------------------------
inline void draw_sensor_panel(Mat &c, const Rect &r, const Data &d) {
  panel(c, r, d.webots ? "LIDAR" : "PDE + SVD", d.webots ? "top view, 1 px = 0.25 m"
                                                         : "foreground energy");
  const Rect sr(r.x + 12, r.y + 38, 220, r.height - 50);
  cv::rectangle(c, sr, Scalar(30, 26, 23), cv::FILLED);
  if (d.webots && d.lidar && d.lidar_n > 0) {
    const Point car(sr.x + sr.width / 2, sr.y + sr.height - 12);
    const float px_per_m = 4.0f;
    for (int i = 0; i < d.lidar_n; ++i) {
      float rr = d.lidar[i];
      if (!std::isfinite(rr) || rr <= 0.0f)
        continue;
      const float a = -d.lidar_fov / 2 + d.lidar_fov * i / std::max(1, d.lidar_n - 1);
      const int x = car.x + static_cast<int>(rr * std::sin(a) * px_per_m);
      const int y = car.y - static_cast<int>(rr * std::cos(a) * px_per_m);
      if (sr.contains(Point(x, y))) {
        const bool front = std::fabs(a) < d.lidar_fov * 0.04f;
        cv::circle(c, Point(x, y), 1, front ? RED : GREEN, cv::FILLED);
      }
    }
    cv::rectangle(c, Rect(car.x - 4, car.y - 8, 8, 14), CYAN, cv::FILLED);
  } else if (d.svd_foreground && !d.svd_foreground->empty()) {
    Mat fg8, big, col;
    d.svd_foreground->convertTo(fg8, CV_8U, 255.0 * 3.0);
    cv::resize(fg8, big, cv::Size(sr.width / 2 - 4, sr.height),
               0, 0, cv::INTER_NEAREST);
    cv::applyColorMap(big, col, cv::COLORMAP_INFERNO);
    col.copyTo(c(Rect(sr.x + sr.width / 2 + 2, sr.y, col.cols, col.rows)));
    if (d.pde_view && !d.pde_view->empty()) {
      Mat small, sb;
      cv::resize(*d.pde_view, small, cv::Size(sr.width / 2 - 2, sr.height));
      cv::cvtColor(small, sb, cv::COLOR_GRAY2BGR);
      sb.copyTo(c(Rect(sr.x, sr.y, sb.cols, sb.rows)));
    }
  }

  // timing bars
  const int bx = sr.x + sr.width + 24, bw = r.x + r.width - bx - 70;
  int by = r.y + 52;
  double total = 0.0, worst = 1.0;
  for (const auto &t : d.timings) {
    total += t.ms;
    worst = std::max(worst, t.ms);
  }
  txt(c, "compute time per frame (ms)", Point(bx, r.y + 40 - 4), 0.42, MUTED);
  for (const auto &t : d.timings) {
    txt(c, t.name, Point(bx, by + 12), 0.4, TEXT);
    const int w = static_cast<int>(t.ms / std::max(worst, 4.0) * (bw - 120));
    cv::rectangle(c, Rect(bx + 120, by + 2, std::max(2, w), 12), CYAN, cv::FILLED);
    txt(c, fmt("%.2f", t.ms), Point(bx + 126 + std::max(2, w), by + 13), 0.38, MUTED);
    by += 20;
  }
  txt(c, fmt("total maths %.2f ms", total), Point(bx, by + 16), 0.48,
      total < 16.0 ? GREEN : YELLOW);
}

// ---- top-level render -----------------------------------------------------------
class Dashboard {
public:
  static constexpr int W = 1600, H = 900;
  Dashboard() : canvas(H, W, CV_8UC3) {}

  const Mat &render(const Data &d) {
    raw_hist.push_back(d.raw_steer);
    steer_hist.push_back(d.steer);
    while (raw_hist.size() > 240) {
      raw_hist.pop_front();
      steer_hist.pop_front();
    }
    canvas.setTo(BG);
    draw_header(canvas, d);
    draw_camera(canvas, Rect(16, 60, 800, 600), d);
    draw_vehicle_panel(canvas, Rect(16, 672, 800, 216), d, raw_hist, steer_hist);
    draw_mdp_panel(canvas, Rect(832, 60, 752, 244), d);
    draw_admm_panel(canvas, Rect(832, 316, 752, 340), d);
    draw_sensor_panel(canvas, Rect(832, 668, 752, 220), d);
    return canvas;
  }

private:
  Mat canvas;
  std::deque<float> raw_hist, steer_hist;
};

} // namespace dash
