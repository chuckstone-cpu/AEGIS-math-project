// LEGACY: original single-file engine, kept verbatim for reference/comparison.
// Superseded by apex_engine.cpp + include/apex/*.hpp. Not built by CMake.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Conditional Webots C API inclusion
#ifdef USE_WEBOTS
#include <webots/camera.h>
#include <webots/device.h>
#include <webots/lidar.h>
#include <webots/robot.h>
#include <webots/vehicle/driver.h>
#endif

// OpenCV strictly for UI, Video I/O, and display canvas.
// ALL core mathematics (Perspective-Calibrated Lane Centering, Pure-Pursuit,
// Robust Trimmed Mean, PDE, SVD, ADMM, CG, MDP) are built from scratch.
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ============================================================================
// 1. FIXED-POINT RGB TO GRAYSCALE (Used in Dashcam Video Mode)
// ============================================================================
void rgb_to_gray_fixed_point(const uint8_t *__restrict bgr,
                             uint8_t *__restrict gray, int width, int height,
                             int channels = 3) {
  const int total_pixels = width * height;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < total_pixels; ++i) {
    int b = bgr[i * channels + 0];
    int g = bgr[i * channels + 1];
    int r = bgr[i * channels + 2];
    gray[i] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
  }
}

// ============================================================================
// 2. PERONA-MALIK ANISOTROPIC DIFFUSION PDE (Used in Dashcam Video Mode)
// ============================================================================
void perona_malik_diffusion(float *__restrict img, float *__restrict temp_buf,
                            int width, int height, int iters = 2,
                            float lambda = 0.2f, float K = 20.0f) {
  const float inv_K_sq = 1.0f / (K * K);
  float *src = img;
  float *dst = temp_buf;

  memcpy(dst, src, width * height * sizeof(float));

  for (int iter = 0; iter < iters; ++iter) {
#pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; ++y) {
      const int row_offset = y * width;
      for (int x = 1; x < width - 1; ++x) {
        const int idx = row_offset + x;
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

  if (iters % 2 != 0) {
    memcpy(img, src, width * height * sizeof(float));
  }
}

// ============================================================================
// 3. SVD BACKGROUND SUBTRACTION (Power Iteration on Gram Matrix A^T A)
// ============================================================================
void power_iteration_svd(const float *__restrict A, int N, int M,
                         float *__restrict u, float *__restrict v) {
  float ATA[64 * 64] = {0.0f};
  float temp[64] = {0.0f};

  for (int k = 0; k < N; ++k) {
    const float *row = &A[k * M];
    for (int i = 0; i < M; ++i) {
      const float r_i = row[i];
      for (int j = i; j < M; ++j) {
        ATA[i * M + j] += r_i * row[j];
      }
    }
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < i; ++j) {
      ATA[i * M + j] = ATA[j * M + i];
    }
  }

  for (int i = 0; i < M; ++i) {
    v[i] = 1.0f / std::sqrt(static_cast<float>(M));
  }

  for (int iter = 0; iter < 8; ++iter) {
    float norm_sq = 0.0f;
    for (int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < M; ++j) {
        sum += ATA[i * M + j] * v[j];
      }
      temp[i] = sum;
      norm_sq += sum * sum;
    }
    const float inv_norm =
        (norm_sq > 1e-12f) ? (1.0f / std::sqrt(norm_sq)) : 0.0f;
    for (int i = 0; i < M; ++i) {
      v[i] = temp[i] * inv_norm;
    }
  }

  float u_norm_sq = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float *row = &A[i * M];
    float sum = 0.0f;
    for (int j = 0; j < M; ++j) {
      sum += row[j] * v[j];
    }
    u[i] = sum;
    u_norm_sq += sum * sum;
  }

  if (u_norm_sq > 1e-10f) {
    const float inv_u_norm = 1.0f / std::sqrt(u_norm_sq);
    for (int i = 0; i < N; ++i) {
      u[i] *= inv_u_norm;
    }
  }
}

// ============================================================================
// 4. PERSPECTIVE-CALIBRATED LANE-CENTER DETECTOR & PURE-PURSUIT CONTROLLER
// ============================================================================
// Directly fixes the bug seen in your video at 0:06 - 0:16:
//  - Previously, the white dashed line in the middle of the right side was
//    mistaken for the outer road edge x_R, placing the Cyan midpoint at x=325
//    (to the left of the car's center x=400). That caused a -0.367 rad LEFT
//    turn across the yellow line onto the pedestrian sidewalk!
//  - Now:
//    1) We strictly distinguish dark road asphalt (lum 20..68) from the
//       brighter grey pedestrian sidewalk (lum > 78) and white dashed lines
//       (lum > 150). White dashed lines are skipped over when scanning for the
//       true outer curb/fence boundary x_R!
//    2) When the car is centered in its right-hand lane in Webots city.wbt,
//       the yellow center line sits at x_yellow_ref(y) = width * (0.28 - 0.12*y_norm).
//       Thus the exact lane center is x_L + width * (0.22 + 0.12*y_norm), which
//       places the Cyan dots Dead-Center at x = 400 (Steer = 0.000 rad) on straights!
//    3) The car's steering angle is locked directly to the Cyan target point via
//       Pure-Pursuit geometry so the car ALWAYS follows the Cyan dots!
struct LaneCenterResult {
  Point lookahead_target; // Exact Cyan target point the car steers toward
  float steering_rad;     // Pure-Pursuit steering angle [-0.38..+0.38] rad
  vector<Point> left_pts;
  vector<Point> right_pts;
  vector<Point> center_pts;
};

inline bool is_yellow_line(int r, int g, int b) {
  return (r > 120 && g > 100 && b < 90 && (r - b) > 45 && (g - b) > 30);
}

inline bool is_white_paint(int r, int g, int b) {
  return (r > 145 && g > 145 && b > 145 && std::abs(r - b) < 30);
}

inline bool is_true_road_asphalt(int r, int g, int b) {
  int max_c = std::max({r, g, b});
  int min_c = std::min({r, g, b});
  int lum = (r + g + b) / 3;
  // Crucial fix: Road asphalt in Webots city.wbt is dark (lum 20..68).
  // The pedestrian sidewalk and building walls are brighter (lum > 75).
  return ((max_c - min_c) < 18 && lum >= 20 && lum <= 68);
}

LaneCenterResult compute_calibrated_lane_center(
    const uint8_t *__restrict bgr, int width, int height,
    float &smoothed_steer) {
  LaneCenterResult res;
  res.lookahead_target = Point(width / 2, (height * 66) / 100);
  res.steering_rad = smoothed_steer;

  const int y_top = (height * 54) / 100;
  const int y_bottom = (height * 90) / 100;

  double weighted_x_sum = 0.0;
  double weight_sum = 0.0;

  for (int y = y_top; y < y_bottom; y += 3) {
    const uint8_t *row = bgr + (y * width * 3);
    float y_norm =
        static_cast<float>(y - y_top) / static_cast<float>(y_bottom - y_top);

    // Calibrated offset from the yellow dividing line to the center of the
    // right-hand driving lane. At y_norm=0 (horizon), offset is 0.23*width.
    // At y_norm=1 (hood), offset is 0.34*width.
    // When x_L is at its nominal position (0.27*w at top, 0.16*w at bottom),
    // x_L + half_lane_offset = 0.50 * width (400 px = straight ahead!).
    float half_lane_offset = width * (0.23f + 0.11f * y_norm);

    // ------------------------------------------------------------------
    // A. Detect Yellow Center Dividing Line x_L (Rejecting Crosswalks)
    // ------------------------------------------------------------------
    int yellow_count = 0;
    int first_y_x = -1, last_y_x = -1;
    int yellow_runs = 0;
    bool in_y = false;

    for (int x = 4; x < (width * 75) / 100; x += 2) {
      int b = row[x * 3 + 0];
      int g = row[x * 3 + 1];
      int r = row[x * 3 + 2];
      if (is_yellow_line(r, g, b)) {
        yellow_count++;
        if (first_y_x == -1)
          first_y_x = x;
        last_y_x = x;
        if (!in_y) {
          yellow_runs++;
          in_y = true;
        }
      } else {
        in_y = false;
      }
    }

    bool has_left_yellow = false;
    float x_L = 0.0f;
    int y_span = last_y_x - first_y_x;
    // Reject yellow crosswalks (crosswalks have >= 3 yellow stripes or wide span)
    if (yellow_count >= 2 && yellow_count <= 26 && yellow_runs <= 2 &&
        y_span >= 0 && y_span < (width * 12) / 100) {
      has_left_yellow = true;
      x_L = 0.5f * static_cast<float>(first_y_x + last_y_x);
      res.left_pts.push_back(Point(static_cast<int>(x_L), y));
    }

    // ------------------------------------------------------------------
    // B. Detect True Outer Road Edge / Fence / Curb x_R
    //    (Skips thin white dashed lane lines so x_R never triggers in the
    //     middle of the road like at 0:07 in your video!)
    // ------------------------------------------------------------------
    int start_x = has_left_yellow
                      ? std::min(width - 20, static_cast<int>(x_L) + 30)
                      : (width * 20) / 100;

    int longest_asphalt_start = -1, longest_asphalt_end = -1;
    int curr_start = -1, curr_len = 0, max_len = 0;

    for (int x = start_x; x < width - 4; x += 2) {
      int b = row[x * 3 + 0];
      int g = row[x * 3 + 1];
      int r = row[x * 3 + 2];

      // Treat dark asphalt OR thin white/yellow road paint inside the road as drivable
      bool is_road = is_true_road_asphalt(r, g, b);
      if (!is_road && (is_white_paint(r, g, b) || is_yellow_line(r, g, b))) {
        // Check if there is still dark asphalt 18 pixels to the right of this paint stripe
        if (x + 18 < width) {
          int rb = row[(x + 18) * 3 + 0];
          int rg = row[(x + 18) * 3 + 1];
          int rr = row[(x + 18) * 3 + 2];
          if (is_true_road_asphalt(rr, rg, rb)) {
            is_road = true; // It's just a dashed lane line or crosswalk stripe; keep going!
          }
        }
      }

      if (is_road) {
        if (curr_start == -1)
          curr_start = x;
        curr_len++;
        if (curr_len > max_len) {
          max_len = curr_len;
          longest_asphalt_start = curr_start;
          longest_asphalt_end = x;
        }
      } else {
        curr_start = -1;
        curr_len = 0;
      }
    }

    bool has_right_edge = false;
    float x_R = static_cast<float>(width - 1);
    if (max_len >= 18 && longest_asphalt_end < width - 12) {
      has_right_edge = true;
      x_R = static_cast<float>(longest_asphalt_end);
      res.right_pts.push_back(Point(static_cast<int>(x_R), y));
    }

    // ------------------------------------------------------------------
    // C. Compute Calibrated Lane Center x_mid(y) (Cyan Dots)
    // ------------------------------------------------------------------
    bool valid_center = false;
    float x_mid = width * 0.50f;

    if (has_left_yellow) {
      // Primary Reference: Stay exactly half_lane_offset to the right of the
      // yellow center dividing line!
      x_mid = x_L + half_lane_offset;

      // Safety Clamp against Right Fence/Curb: If the outer right fence/curb x_R
      // is visible and closer than expected (e.g. road narrows or curves left),
      // never allow x_mid to get closer than 0.85 * half_lane_offset to the fence!
      if (has_right_edge && (x_R - x_L) > width * 0.16f) {
        float max_safe_x = x_R - 0.85f * half_lane_offset;
        float min_safe_x = x_L + 0.65f * half_lane_offset;
        x_mid = std::max(min_safe_x, std::min(x_mid, max_safe_x));
      }
      valid_center = true;
    } else if (has_right_edge) {
      // Fallback when yellow line exits left of camera after a turn:
      // Stay half_lane_offset to the LEFT of the right curb / small fence!
      x_mid = x_R - half_lane_offset;
      valid_center = true;
    } else if (max_len >= 20) {
      // Crosswalk / intersection fallback: follow midpoint of detected dark asphalt band
      x_mid = 0.5f * static_cast<float>(longest_asphalt_start +
                                        longest_asphalt_end);
      valid_center = true;
    }

    if (valid_center) {
      x_mid = std::max(width * 0.10f, std::min(width * 0.90f, x_mid));
      res.center_pts.push_back(Point(static_cast<int>(x_mid), y));

      // Weight lookahead band (y_norm in [0.25 .. 0.70]) highest for smooth Pure-Pursuit
      double w = (y_norm >= 0.20f && y_norm <= 0.75f) ? 2.0 : 0.8;
      weighted_x_sum += w * static_cast<double>(x_mid);
      weight_sum += w;
    }
  }

  // ------------------------------------------------------------------
  // D. Pure-Pursuit Steering Law Locked Directly to the Cyan Dots
  // ------------------------------------------------------------------
  if (weight_sum > 1e-3) {
    float target_x = static_cast<float>(weighted_x_sum / weight_sum);
    res.lookahead_target =
        Point(static_cast<int>(target_x), (height * 66) / 100);

    // Normalized lateral error in [-1.0, +1.0]:
    //  target_x > width/2 (Cyan dots to the right) -> err > 0 -> Steer RIGHT (+)
    //  target_x < width/2 (Cyan dots to the left)  -> err < 0 -> Steer LEFT (-)
    float err_x = (target_x - (width * 0.50f)) / (width * 0.50f);

    // Deadband of 1.5% around center so the car drives rock-steady on straights
    if (std::abs(err_x) < 0.015f) {
      err_x = 0.0f;
    }

    // Pure-Pursuit curvature control law: delta = atan(Kp * err_x)
    float raw_steer = std::atan(0.85f * err_x);
    smoothed_steer = 0.55f * smoothed_steer + 0.45f * raw_steer;
  } else {
    // Decay steering gently toward straight if no features found in a single frame
    smoothed_steer *= 0.85f;
  }

  smoothed_steer = std::max(-0.38f, std::min(0.38f, smoothed_steer));
  res.steering_rad = smoothed_steer;
  return res;
}

// ============================================================================
// 5. MATRIX-FREE O(N) ADMM & CONJUGATE GRADIENT FOR TRAJECTORY
// ============================================================================
inline void multiply_Q_rho_tridiag(const float *__restrict v,
                                   float *__restrict out, int n,
                                   float diag_val) {
  out[0] = diag_val * v[0] - v[1];
  for (int i = 1; i < n - 1; ++i) {
    out[i] = -v[i - 1] + diag_val * v[i] - v[i + 1];
  }
  out[n - 1] = -v[n - 2] + diag_val * v[n - 1];
}

void conjugate_gradient_fast(const float *__restrict b, float *__restrict x,
                             int n, int max_iters, float rho) {
  float r[128];
  float p[128];
  float Qp[128];
  const float diag_val = 2.0f + rho;

  multiply_Q_rho_tridiag(x, Qp, n, diag_val);
  float rsold = 0.0f;
  for (int i = 0; i < n; ++i) {
    r[i] = b[i] - Qp[i];
    p[i] = r[i];
    rsold += r[i] * r[i];
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    if (rsold < 1e-8f)
      break;

    multiply_Q_rho_tridiag(p, Qp, n, diag_val);

    float pQp = 0.0f;
    for (int i = 0; i < n; ++i) {
      pQp += p[i] * Qp[i];
    }

    const float alpha = rsold / (pQp + 1e-8f);
    float rsnew = 0.0f;
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Qp[i];
      rsnew += r[i] * r[i];
    }

    if (rsnew < 1e-8f)
      break;

    const float beta = rsnew / rsold;
    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rsold = rsnew;
  }
}

void admm_jerk_minimization(float *__restrict traj_x, int n, int mdp_action,
                            float current_norm_speed, float target_norm_speed) {
  const float rho = 1.0f;
  float z[128] = {0.0f};
  float y[128] = {0.0f};
  float b[128] = {0.0f};

  for (int i = 0; i < n; ++i) {
    traj_x[i] = current_norm_speed;
    z[i] = current_norm_speed;
  }

  for (int iter = 0; iter < 6; ++iter) {
    for (int i = 0; i < n; ++i) {
      const float alpha_t = static_cast<float>(i) / static_cast<float>(n - 1);
      float ref_profile =
          (1.0f - alpha_t) * current_norm_speed + alpha_t * target_norm_speed;

      if (mdp_action == 2) { // BRAKE
        ref_profile *= (1.0f - 0.35f * alpha_t);
      } else if (mdp_action == 0) { // ACCEL
        ref_profile *= (0.85f + 0.25f * alpha_t);
      }

      b[i] = rho * (z[i] - y[i]) + ref_profile;
    }

    conjugate_gradient_fast(b, traj_x, n, 12, rho);

    for (int i = 0; i < n; ++i) {
      z[i] = std::max(0.15f, std::min(1.0f, traj_x[i] + y[i]));
    }

    for (int i = 0; i < n; ++i) {
      y[i] += traj_x[i] - z[i];
    }
  }

  for (int i = 0; i < n; ++i) {
    traj_x[i] = z[i];
  }
}

// ============================================================================
// 6. MARKOV DECISION PROCESS (VALUE ITERATION)
// ============================================================================
enum DistState { FAR = 0, MED = 1, CLOSE = 2 };
enum SpeedState { SPEED_SLOW = 0, SPEED_FAST = 1 };
enum Action { ACCEL = 0, MAINTAIN = 1, BRAKE = 2 };

int mdp_next_state(int dist, int speed, int action, int &next_dist,
                   int &next_speed) {
  next_speed = speed;
  if (action == ACCEL)
    next_speed = SPEED_FAST;
  if (action == BRAKE)
    next_speed = SPEED_SLOW;

  next_dist = dist;
  if (next_speed == SPEED_FAST)
    next_dist++;
  if (action == BRAKE && dist > FAR)
    next_dist--;

  next_dist = std::max(0, std::min(2, next_dist));
  return next_dist * 2 + next_speed;
}

float mdp_reward(int dist, int speed, int action) {
  float r = 0.0f;
  if (dist == CLOSE && speed == SPEED_FAST)
    r -= 100.0f;
  if (dist == CLOSE && action == BRAKE)
    r += 50.0f;
  if (dist == MED && action == MAINTAIN)
    r += 15.0f;
  if (dist == MED && speed == SPEED_FAST && action == BRAKE)
    r += 25.0f;
  if (dist == FAR && action == ACCEL)
    r += 20.0f;
  if (dist == FAR && speed == SPEED_FAST && action == MAINTAIN)
    r += 25.0f;
  return r;
}

void mdp_value_iteration(int *policy) {
  float V[6] = {0.0f};
  for (int i = 0; i < 50; ++i) {
    float V_new[6] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      for (int s = 0; s < 2; ++s) {
        int state = d * 2 + s;
        float max_q = -1e9f;
        for (int a = 0; a < 3; ++a) {
          int nd, ns;
          int nstate = mdp_next_state(d, s, a, nd, ns);
          float q = mdp_reward(d, s, a) + 0.9f * V[nstate];
          if (q > max_q) {
            max_q = q;
            policy[state] = a;
          }
        }
        V_new[state] = max_q;
      }
    }
    memcpy(V, V_new, sizeof(float) * 6);
  }
}

// ============================================================================
// MAIN APPLICATION (DUAL-MODE: WEBOTS SIMULATOR + DASHCAM VIDEO)
// ============================================================================
int main(int argc, char **argv) {
  bool use_webots = false;
  string video_path = "..\\Train_set\\dataset_video175.mp4";

  if (argc > 1) {
    string arg1 = argv[1];
    if (arg1 == "--webots" || arg1 == "-w") {
      use_webots = true;
    } else {
      video_path = arg1;
      use_webots = false;
    }
  } else {
#ifdef USE_WEBOTS
    FILE *f = fopen(video_path.c_str(), "rb");
    if (f) {
      fclose(f);
      if (getenv("WEBOTS_CONTROLLER_URL") != nullptr ||
          getenv("WEBOTS_ROBOT_NAME") != nullptr) {
        use_webots = true;
      }
    } else {
      use_webots = true;
    }
#endif
  }

  VideoCapture cap;
#ifdef USE_WEBOTS
  WbDeviceTag wb_cam = 0;
  WbDeviceTag wb_lidar = 0;
  int wb_cam_w = 0, wb_cam_h = 0;
#endif

  if (use_webots) {
#ifdef USE_WEBOTS
    cout << "[Project Apex] Initializing Webots Driver Controller..." << endl;
    wbu_driver_init();

    int basic_step = static_cast<int>(wb_robot_get_basic_time_step());
    if (basic_step <= 0)
      basic_step = 32;

    const char *cam_names[] = {"camera", "front_camera", "Camera",
                               "kinect color"};
    for (const char *name : cam_names) {
      wb_cam = wb_robot_get_device(name);
      if (wb_cam != 0)
        break;
    }
    if (wb_cam == 0) {
      int n_devs = wb_robot_get_number_of_devices();
      for (int i = 0; i < n_devs; ++i) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        if (wb_device_get_node_type(tag) == WB_NODE_CAMERA) {
          wb_cam = tag;
          break;
        }
      }
    }
    if (wb_cam != 0) {
      wb_camera_enable(wb_cam, basic_step);
      wb_cam_w = wb_camera_get_width(wb_cam);
      wb_cam_h = wb_camera_get_height(wb_cam);
      cout << "[Project Apex] Webots Camera enabled (" << wb_cam_w << "x"
           << wb_cam_h << ") - Calibrated Pure-Pursuit Active" << endl;
    }

    const char *lidar_names[] = {"Sick LMS 291", "lidar", "Lidar",
                                 "Velodyne VLP-16", "ibeo"};
    for (const char *name : lidar_names) {
      wb_lidar = wb_robot_get_device(name);
      if (wb_lidar != 0)
        break;
    }
    if (wb_lidar == 0) {
      int n_devs = wb_robot_get_number_of_devices();
      for (int i = 0; i < n_devs; ++i) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        if (wb_device_get_node_type(tag) == WB_NODE_LIDAR) {
          wb_lidar = tag;
          break;
        }
      }
    }
    if (wb_lidar != 0) {
      wb_lidar_enable(wb_lidar, basic_step);
    }

    wbu_driver_set_hazard_flashers(false);
    wbu_driver_set_dipped_beams(true);
    wbu_driver_set_antifog_lights(true);
    wbu_driver_set_wiper_mode(DOWN);
    wbu_driver_set_gear(1);
    wbu_driver_set_brake_intensity(0.0);
    wbu_driver_set_steering_angle(0.0);
    wbu_driver_set_cruising_speed(32.0);
#else
    cerr << "Error: Compiled without USE_WEBOTS support." << endl;
    return -1;
#endif
  } else {
    cap.open(video_path);
    if (!cap.isOpened()) {
      cerr << "Error: Could not open video " << video_path << endl;
      return -1;
    }
    cout << "[Project Apex] Running in Dashcam Video Mode: " << video_path
         << endl;
  }

  // UI Layout setup
  const int window_w = 1600;
  const int window_h = 900;
  Mat canvas = Mat::zeros(window_h, window_w, CV_8UC3);

  const Rect video_roi(50, 50, 800, 600);
  const Rect mdp_roi(900, 50, 650, 400);
  const Rect legend_roi(900, 480, 650, 370);
  const Rect turn_roi(50, 680, 800, 170);

  int mdp_policy[6] = {0};
  mdp_value_iteration(mdp_policy);

  const int SVD_M = 10;
  const int SVD_W = 64, SVD_H = 64;
  const int SVD_N = SVD_W * SVD_H;
  vector<float> svd_A(SVD_N * SVD_M, 0.0f);
  vector<float> svd_u(SVD_N, 0.0f);
  vector<float> svd_v(SVD_M, 0.0f);
  vector<float> bg_model(SVD_N, 0.0f);
  vector<float> pde_temp_buf(video_roi.width * video_roi.height, 0.0f);
  int svd_idx = 0;

  namedWindow("Project Apex - Autonomous Engine", WINDOW_NORMAL);
  resizeWindow("Project Apex - Autonomous Engine", window_w, window_h);

  Mat frame, resized_frame;
  Mat gray_frame(video_roi.height, video_roi.width, CV_8UC1);
  Mat float_img(video_roi.height, video_roi.width, CV_32FC1);
  Mat smoothed_gray(video_roi.height, video_roi.width, CV_8UC1);
  Mat small_gray(SVD_H, SVD_W, CV_8UC1);
  Mat small_float(SVD_H, SVD_W, CV_32FC1);

  int frame_count = 0;
  float current_speed_kmh = use_webots ? 30.0f : 65.0f;
  float smoothed_steer = 0.0f;

  while (true) {
    double t_start = static_cast<double>(getTickCount());

    if (use_webots) {
#ifdef USE_WEBOTS
      if (wbu_driver_step() == -1)
        break;

      if (wb_cam != 0) {
        const unsigned char *img_bgra = wb_camera_get_image(wb_cam);
        if (img_bgra != nullptr) {
          Mat cam_bgra(wb_cam_h, wb_cam_w, CV_8UC4,
                       const_cast<unsigned char *>(img_bgra));
          cvtColor(cam_bgra, frame, COLOR_BGRA2BGR);
        }
      }
      if (frame.empty()) {
        frame = Mat::zeros(video_roi.height, video_roi.width, CV_8UC3);
      }

      double actual_wb_speed = wbu_driver_get_current_speed();
      if (!std::isnan(actual_wb_speed) && actual_wb_speed >= 0.0 &&
          frame_count > 10) {
        current_speed_kmh = 0.65f * current_speed_kmh +
                            0.35f * static_cast<float>(actual_wb_speed);
      }
#endif
    } else {
      if (!cap.read(frame))
        break;
    }

    frame_count++;
    canvas.setTo(Scalar(20, 20, 25));

    resize(frame, resized_frame, Size(video_roi.width, video_roi.height));

    // ========================================================================
    // 1. CALIBRATED LANE-CENTER TRACKING (Follows Cyan Dots Directly!)
    // ========================================================================
    LaneCenterResult lane_res = compute_calibrated_lane_center(
        resized_frame.data, resized_frame.cols, resized_frame.rows,
        smoothed_steer);

    // ========================================================================
    // 2. OBSTACLE DETECTION & CLOSE-FENCE SAFETY MARGIN
    // ========================================================================
    float left_motion = 0.0f, center_motion = 0.0f, right_motion = 0.0f;
    int center_max_y = 0;
    float fence_nudge_rad = 0.0f;
    float min_front_dist_m = 100.0f;

    if (use_webots) {
#ifdef USE_WEBOTS
      if (wb_lidar != 0) {
        const float *ranges = wb_lidar_get_range_image(wb_lidar);
        int res = wb_lidar_get_horizontal_resolution(wb_lidar);
        if (ranges != nullptr && res > 0) {
          // Very narrow 8% dead-ahead cone (46%..54%) for genuine in-lane obstacles
          int front_start = (res * 46) / 100;
          int front_end = (res * 54) / 100;

          float min_right_bumper = 100.0f;
          float min_left_bumper = 100.0f;

          for (int i = 0; i < res; ++i) {
            float r = ranges[i];
            if (std::isinf(r) || std::isnan(r) || r <= 0.4f)
              r = 80.0f;

            if (i >= front_start && i <= front_end) {
              if (r < min_front_dist_m)
                min_front_dist_m = r;
            }
            // Front-left bumper sector (30%..45%)
            if (i >= (res * 30) / 100 && i < front_start) {
              if (r < min_left_bumper)
                min_left_bumper = r;
            }
            // Front-right bumper sector (55%..70%) - detects small outer fence after turn!
            if (i > front_end && i <= (res * 70) / 100) {
              if (r < min_right_bumper)
                min_right_bumper = r;
            }
          }

          // Subtle safety margin ONLY if the front-right or front-left bumper gets
          // closer than 2.4 meters to the small fence after turning:
          if (min_right_bumper < 2.4f) {
            // Fence is very close on the right -> gently nudge steering LEFT (-)
            fence_nudge_rad -= (2.4f - min_right_bumper) * 0.09f;
          }
          if (min_left_bumper < 2.2f) {
            // Fence is very close on the left -> gently nudge steering RIGHT (+)
            fence_nudge_rad += (2.2f - min_left_bumper) * 0.09f;
          }
          fence_nudge_rad =
              std::max(-0.15f, std::min(0.15f, fence_nudge_rad));

          if (min_front_dist_m < 7.0f) {
            center_motion = 4.0f;
            center_max_y = 56;
          } else if (min_front_dist_m < 13.0f) {
            center_motion = 1.6f;
            center_max_y = 44;
          }
        }
      }
#endif
    } else {
      // DASHCAM VIDEO MODE: Full Fixed-Point Grayscale + PDE + Rank-1 SVD Pipeline
      rgb_to_gray_fixed_point(resized_frame.data, gray_frame.data,
                              resized_frame.cols, resized_frame.rows, 3);

      gray_frame.convertTo(float_img, CV_32F);
      perona_malik_diffusion(reinterpret_cast<float *>(float_img.data),
                             pde_temp_buf.data(), float_img.cols,
                             float_img.rows, 2, 0.2f, 20.0f);
      float_img.convertTo(smoothed_gray, CV_8U);

      resize(smoothed_gray, small_gray, Size(SVD_W, SVD_H));
      small_gray.convertTo(small_float, CV_32F, 1.0 / 255.0);
      const float *curr_small_ptr =
          reinterpret_cast<const float *>(small_float.data);

      for (int i = 0; i < SVD_N; ++i) {
        svd_A[i * SVD_M + svd_idx] = curr_small_ptr[i];
      }
      svd_idx = (svd_idx + 1) % SVD_M;

      if (frame_count >= SVD_M &&
          (frame_count % 3 == 0 || frame_count == SVD_M)) {
        power_iteration_svd(svd_A.data(), SVD_N, SVD_M, svd_u.data(),
                            svd_v.data());
      }

      float proj_coeff = 0.0f;
      if (frame_count >= SVD_M) {
        for (int i = 0; i < SVD_N; ++i) {
          proj_coeff += curr_small_ptr[i] * svd_u[i];
        }
      }

      for (int i = 0; i < SVD_N; ++i) {
        const float p = curr_small_ptr[i];
        if (frame_count < SVD_M) {
          bg_model[i] =
              (frame_count == 1) ? p : (0.9f * bg_model[i] + 0.1f * p);
        } else {
          const float svd_recon = proj_coeff * svd_u[i];
          bg_model[i] = 0.85f * svd_recon + 0.15f * bg_model[i];
        }

        const float diff = std::abs(p - bg_model[i]);
        const int x = i % SVD_W;
        const int y = i / SVD_W;

        if (y > SVD_H / 2 && diff > 0.10f) {
          if (x < SVD_W / 3) {
            left_motion += diff;
          } else if (x < 2 * SVD_W / 3) {
            center_motion += diff;
            if (y > center_max_y)
              center_max_y = y;
          } else {
            right_motion += diff;
          }
        }
      }
    }

    // ========================================================================
    // 3. DIRECT STEERING COMMAND (LOCKED TO CYAN DOTS + SAFETY NUDGE)
    // ========================================================================
    float final_steering_rad = lane_res.steering_rad + fence_nudge_rad;
    final_steering_rad =
        std::max(-0.38f, std::min(0.38f, final_steering_rad));

    bool turn_left = (final_steering_rad < -0.035f);
    bool turn_right = (final_steering_rad > 0.035f);

    // ========================================================================
    // 4. MDP DECISION & ADMM JERK-MINIMIZED VELOCITY PROFILE
    // ========================================================================
    int current_dist = FAR;
    if (center_motion > 2.5f) {
      current_dist = CLOSE;
    } else if (center_motion > 1.2f) {
      current_dist = MED;
    }

    int current_speed = (current_speed_kmh > 32.0f) ? SPEED_FAST : SPEED_SLOW;
    int state = current_dist * 2 + current_speed;
    int chosen_action = mdp_policy[state];

    // Controlled speed in Webots: 32 km/h on straights, 22 km/h in curves
    // so the car completes every turn smoothly with zero understeer!
    float turn_mag = std::abs(final_steering_rad);
    float max_safe_speed =
        use_webots ? std::max(21.0f, 32.0f - turn_mag * 32.0f) : 65.0f;

    float desired_speed = max_safe_speed;
    if (current_dist == CLOSE) {
      desired_speed = use_webots ? 16.0f : 20.0f;
    } else if (current_dist == MED) {
      desired_speed = use_webots ? 24.0f : 48.0f;
    }

    const int T_STEPS = 50;
    float trajectory[T_STEPS] = {0.0f};
    admm_jerk_minimization(trajectory, T_STEPS, chosen_action,
                           current_speed_kmh / 65.0f, desired_speed / 65.0f);

    float admm_step_speed = trajectory[4] * 65.0f;
    if (admm_step_speed < current_speed_kmh) {
      current_speed_kmh = std::max(desired_speed, current_speed_kmh - 1.5f);
    } else {
      current_speed_kmh = std::min(desired_speed, current_speed_kmh + 1.2f);
    }

#ifdef USE_WEBOTS
    if (use_webots) {
      wbu_driver_set_gear(1);
      wbu_driver_set_brake_intensity(0.0);
      wbu_driver_set_cruising_speed(std::max(18.0f, current_speed_kmh));
      wbu_driver_set_steering_angle(final_steering_rad);
    }
#endif

    double fps =
        getTickFrequency() / (static_cast<double>(getTickCount()) - t_start);

    // ========================================================================
    // 5. UI RENDERING
    // ========================================================================
    Mat video_canvas = canvas(video_roi);
    if (use_webots) {
      resized_frame.copyTo(video_canvas);

      // Draw Yellow Line (Yellow dots), True Outer Road Edge (Orange dots),
      // and Calibrated Lane Centerline (Cyan dots)
      for (const Point &pt : lane_res.left_pts) {
        circle(video_canvas, pt, 3, Scalar(0, 255, 255), FILLED);
      }
      for (const Point &pt : lane_res.right_pts) {
        circle(video_canvas, pt, 3, Scalar(0, 100, 255), FILLED);
      }
      for (const Point &pt : lane_res.center_pts) {
        circle(video_canvas, pt, 4, Scalar(255, 255, 0), FILLED);
      }

      // Draw Pure-Pursuit Green Arrow directly from hood to the Cyan Target!
      Point hood_pt(video_roi.width / 2, video_roi.height - 20);
      circle(video_canvas, lane_res.lookahead_target, 8, Scalar(0, 255, 0), 2);
      arrowedLine(video_canvas, hood_pt, lane_res.lookahead_target,
                  Scalar(0, 255, 0), 3, LINE_AA);
    } else {
      cvtColor(smoothed_gray, video_canvas, COLOR_GRAY2BGR);
    }

    if (current_dist == CLOSE) {
      rectangle(video_canvas, Point(300, 200), Point(500, 400),
                Scalar(0, 0, 255), 3);
      putText(video_canvas, "OBSTACLE DETECTED", Point(300, 190),
              FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    string mode_badge =
        use_webots ? " [WEBOTS PURE-PURSUIT LOCK]" : " [VIDEO PDE + SVD]";
    putText(video_canvas,
            "Compute FPS: " + to_string(static_cast<int>(fps)) + mode_badge,
            Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 255, 0), 2);

    rectangle(canvas, video_roi, Scalar(100, 100, 100), 2);
    rectangle(canvas, mdp_roi, Scalar(100, 100, 100), 2);
    rectangle(canvas, legend_roi, Scalar(100, 100, 100), 2);
    rectangle(canvas, turn_roi, Scalar(100, 100, 100), 2);

    // MDP UI Section
    putText(canvas, "MARKOV DECISION PROCESS",
            Point(mdp_roi.x + 20, mdp_roi.y + 40), FONT_HERSHEY_TRIPLEX, 1.0,
            Scalar(255, 200, 0), 2);

    string dist_str = (current_dist == FAR)
                          ? "FAR"
                          : (current_dist == MED ? "MEDIUM" : "CLOSE");
    string speed_str = (current_speed == SPEED_FAST) ? "FAST" : "SLOW";
    string act_str =
        (chosen_action == ACCEL)
            ? "ACCELERATE"
            : (chosen_action == BRAKE ? "BRAKE (KKT BOUNDARY)" : "MAINTAIN");
    Scalar act_color =
        (chosen_action == BRAKE) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

    putText(canvas, "Current State:", Point(mdp_roi.x + 20, mdp_roi.y + 100),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
    putText(canvas, "Distance: " + dist_str,
            Point(mdp_roi.x + 50, mdp_roi.y + 140), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(200, 200, 200), 1);
    putText(canvas, "Speed: " + speed_str,
            Point(mdp_roi.x + 50, mdp_roi.y + 170), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(200, 200, 200), 1);

    putText(canvas,
            "Bellman Optimal Action:", Point(mdp_roi.x + 20, mdp_roi.y + 240),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 1);
    putText(canvas, act_str, Point(mdp_roi.x + 50, mdp_roi.y + 290),
            FONT_HERSHEY_TRIPLEX, 1.2, act_color, 2);

    int target_speed_kmh = static_cast<int>(std::round(current_speed_kmh));
    string target_speed_str =
        "Target Speed: " + to_string(target_speed_kmh) + " km/h";
    putText(canvas, target_speed_str, Point(mdp_roi.x + 50, mdp_roi.y + 350),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

    // Legend & Live ADMM Trajectory Plot Section
    putText(canvas, "SYSTEM LEGEND & ARCHITECTURE",
            Point(legend_roi.x + 20, legend_roi.y + 40), FONT_HERSHEY_TRIPLEX,
            0.9, Scalar(255, 200, 0), 2);
    int l_y = legend_roi.y + 80;
    putText(canvas,
            "- [Lane] Perspective-Calibrated Half-Lane Offset from Yellow Line",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.53,
            Scalar(255, 255, 0), 1);
    l_y += 26;
    putText(canvas,
            "- [Steer] Pure-Pursuit Control Locked Directly to Cyan Centroid",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.53,
            Scalar(255, 255, 255), 1);
    l_y += 26;
    putText(canvas,
            "- [Vision] Skips Dashed White Lines -> Locks True Outer Curb/Fence",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.53,
            Scalar(0, 200, 255), 1);
    l_y += 26;
    putText(canvas,
            "- [MDP] Bellman Value Iteration mapping states -> optimal actions",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.53,
            Scalar(255, 255, 255), 1);
    l_y += 26;
    putText(canvas,
            "- [ADMM] Matrix-Free O(N) Jerk-Minimized Velocity Profile:",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.55,
            Scalar(0, 255, 255), 1);

    Rect plot_rect(legend_roi.x + 30, l_y + 15, legend_roi.width - 60, 120);
    rectangle(canvas, plot_rect, Scalar(45, 45, 55), FILLED);
    rectangle(canvas, plot_rect, Scalar(90, 90, 100), 1);
    for (int i = 0; i < T_STEPS - 1; ++i) {
      int x1 = plot_rect.x + (i * plot_rect.width) / (T_STEPS - 1);
      int y1 = plot_rect.y + plot_rect.height -
               static_cast<int>(trajectory[i] * (plot_rect.height - 15)) - 5;
      int x2 = plot_rect.x + ((i + 1) * plot_rect.width) / (T_STEPS - 1);
      int y2 = plot_rect.y + plot_rect.height -
               static_cast<int>(trajectory[i + 1] * (plot_rect.height - 15)) -
               5;
      line(canvas, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 255), 2,
           LINE_AA);
    }

    // Turn Indicators UI Section
    putText(canvas, "AUTONOMOUS STEERING DECISION",
            Point(turn_roi.x + 20, turn_roi.y + 30), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(255, 200, 0), 1);

    Scalar left_color = turn_left ? Scalar(0, 255, 0) : Scalar(100, 100, 100);
    Scalar right_color = turn_right ? Scalar(0, 255, 0) : Scalar(100, 100, 100);

    arrowedLine(canvas, Point(turn_roi.x + 350, turn_roi.y + 100),
                Point(turn_roi.x + 150, turn_roi.y + 100), left_color, 12, 8, 0,
                0.2);
    arrowedLine(canvas, Point(turn_roi.x + 450, turn_roi.y + 100),
                Point(turn_roi.x + 650, turn_roi.y + 100), right_color, 12, 8,
                0, 0.2);

    if (turn_left || turn_right) {
      putText(canvas,
              "REASON: Pure-Pursuit Following Cyan Lane Center (Steer: " +
                  to_string(final_steering_rad).substr(0, 6) + " rad)",
              Point(turn_roi.x + 105, turn_roi.y + 150), FONT_HERSHEY_SIMPLEX,
              0.6, Scalar(0, 255, 255), 1);
    } else {
      putText(canvas,
              "REASON: Centered in Right-Hand Lane (Steer: 0.000 rad).",
              Point(turn_roi.x + 140, turn_roi.y + 150), FONT_HERSHEY_SIMPLEX,
              0.6, Scalar(150, 150, 150), 1);
    }

    imshow("Project Apex - Autonomous Engine", canvas);

    char c = static_cast<char>(waitKey(1));
    if (c == 27)
      break;
  }

  if (use_webots) {
#ifdef USE_WEBOTS
    wbu_driver_cleanup();
#endif
  } else {
    cap.release();
  }
  destroyAllWindows();
  return 0;
}
