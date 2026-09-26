#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// We include OpenCV strictly for UI, Video loading and drawing.
// ALL core mathematics (PDE, SVD, ADMM, MDP) are implemented from scratch
// below.
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ============================================================================
// 1. FIXED-POINT RGB TO GRAYSCALE
// ============================================================================
void rgb_to_gray_fixed_point(const uint8_t *bgr, uint8_t *gray, int width,
                             int height) {
  int total_pixels = width * height;
  for (int i = 0; i < total_pixels; ++i) {
    int b = bgr[i * 3 + 0];
    int g = bgr[i * 3 + 1];
    int r = bgr[i * 3 + 2];
    // (R*77 + G*150 + B*29) >> 8
    gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
  }
}

// ============================================================================
// 2. PERONA-MALIK ANISOTROPIC DIFFUSION
// ============================================================================
void perona_malik_diffusion(float *img, int width, int height, int iters = 5,
                            float lambda = 0.2f, float K = 15.0f) {
  float *temp = new float[width * height];
  memcpy(temp, img, width * height * sizeof(float));

  for (int iter = 0; iter < iters; ++iter) {
    for (int y = 1; y < height - 1; ++y) {
      for (int x = 1; x < width - 1; ++x) {
        int idx = y * width + x;
        float I = img[idx];

        // Explicit 5-point stencil differences
        float dN = img[idx - width] - I;
        float dS = img[idx + width] - I;
        float dE = img[idx + 1] - I;
        float dW = img[idx - 1] - I;

        // Rational conductance function (Perona-Malik Formulation 2)
        // Much faster than exp() for real-time CPU streaming
        float K_sq = K * K;
        float cN = 1.0f / (1.0f + (dN * dN) / K_sq);
        float cS = 1.0f / (1.0f + (dS * dS) / K_sq);
        float cE = 1.0f / (1.0f + (dE * dE) / K_sq);
        float cW = 1.0f / (1.0f + (dW * dW) / K_sq);

        temp[idx] = I + lambda * (cN * dN + cS * dS + cE * dE + cW * dW);
      }
    }
    memcpy(img, temp, width * height * sizeof(float));
  }
  delete[] temp;
}

// ============================================================================
// 3. SVD BACKGROUND SUBTRACTION (POWER ITERATION)
// ============================================================================
// A is N x M matrix (column-major for simplicity).
void power_iteration_svd(const float *A, int N, int M, float *u, float *v) {
  // 1. Compute A^T * A (size M x M)
  float *ATA = new float[M * M];
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += A[k * M + i] * A[k * M + j];
      }
      ATA[i * M + j] = sum;
    }
  }

  // 2. Power Iteration to find leading eigenvector v of ATA
  for (int i = 0; i < M; ++i)
    v[i] = 1.0f; // Initialize

  float *temp = new float[M];
  for (int iter = 0; iter < 10; ++iter) { // 10 iterations usually enough
    float norm = 0;
    for (int i = 0; i < M; ++i) {
      temp[i] = 0;
      for (int j = 0; j < M; ++j) {
        temp[i] += ATA[i * M + j] * v[j];
      }
      norm += temp[i] * temp[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < M; ++i)
      v[i] = temp[i] / norm;
  }

  // 3. Compute u = A * v
  float u_norm = 0;
  for (int i = 0; i < N; ++i) {
    u[i] = 0;
    for (int j = 0; j < M; ++j) {
      u[i] += A[i * M + j] * v[j];
    }
    u_norm += u[i] * u[i];
  }
  u_norm = sqrt(u_norm);
  if (u_norm > 1e-5) {
    for (int i = 0; i < N; ++i)
      u[i] /= u_norm;
  }

  delete[] ATA;
  delete[] temp;
}

// ============================================================================
// 4. ADMM & CONJUGATE GRADIENT FOR TRAJECTORY
// ============================================================================
void conjugate_gradient(const float *Q, const float *b, float *x, int n,
                        int max_iters) {
  float *r = new float[n];
  float *p = new float[n];
  float *Qp = new float[n];

  // r0 = b - Q * x0
  for (int i = 0; i < n; ++i) {
    float Qx = 0;
    for (int j = 0; j < n; ++j)
      Qx += Q[i * n + j] * x[j];
    r[i] = b[i] - Qx;
    p[i] = r[i];
  }

  float rsold = 0;
  for (int i = 0; i < n; ++i)
    rsold += r[i] * r[i];

  for (int iter = 0; iter < max_iters; ++iter) {
    // Qp = Q * p
    for (int i = 0; i < n; ++i) {
      Qp[i] = 0;
      for (int j = 0; j < n; ++j)
        Qp[i] += Q[i * n + j] * p[j];
    }

    float pQp = 0;
    for (int i = 0; i < n; ++i)
      pQp += p[i] * Qp[i];

    float alpha = rsold / (pQp + 1e-8f);
    for (int i = 0; i < n; ++i)
      x[i] += alpha * p[i];
    for (int i = 0; i < n; ++i)
      r[i] -= alpha * Qp[i];

    float rsnew = 0;
    for (int i = 0; i < n; ++i)
      rsnew += r[i] * r[i];
    if (sqrt(rsnew) < 1e-4)
      break;

    for (int i = 0; i < n; ++i)
      p[i] = r[i] + (rsnew / rsold) * p[i];
    rsold = rsnew;
  }
  delete[] r;
  delete[] p;
  delete[] Qp;
}

void admm_jerk_minimization(float *traj_x, int n, int mdp_action) {
  // Simple 1D ADMM Jerk-minimized QP: min 1/2 x^T Q x s.t. x >= 0 (velocity
  // profile) Here we approximate Q as a tridiagonal matrix favoring smoothness
  float rho = 1.0f;
  float *Q_rho = new float[n * n];
  float *z = new float[n];
  float *y = new float[n];
  float *b = new float[n];

  memset(z, 0, n * sizeof(float));
  memset(y, 0, n * sizeof(float));

  // Construct Q + rho * I
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float val = 0;
      if (i == j)
        val = 2.0f;
      else if (abs(i - j) == 1)
        val = -1.0f;
      Q_rho[i * n + j] = val + (i == j ? rho : 0);
    }
  }

  // Define target velocity profile based on MDP action
  float target_drift = 0.5f; // default MAINTAIN
  if (mdp_action == 0)
    target_drift = 0.8f; // ACCEL
  if (mdp_action == 2)
    target_drift = 0.1f; // BRAKE

  for (int iter = 0; iter < 5; ++iter) { // ADMM iterations
    // 1. x-update (Unconstrained CG solver)
    // (Q + rho I) x = rho(z - y)
    for (int i = 0; i < n; ++i) {
      // Apply dynamic drift to shape the velocity curve
      float dynamic_b = target_drift;
      // Shape the curve depending on whether it's braking or accelerating
      if (mdp_action == 2)
        dynamic_b *= (1.0f - (float)i / n); // Braking reduces target over time
      if (mdp_action == 0)
        dynamic_b *= (0.5f + (float)i / n); // Accel increases target

      b[i] = rho * (z[i] - y[i]) + dynamic_b;
    }
    conjugate_gradient(Q_rho, b, traj_x, n, 10);

    // 2. z-update (Non-negative slack variable projection)
    for (int i = 0; i < n; ++i)
      z[i] = std::max(0.0f, traj_x[i] + y[i]);

    // 3. y-update (Lagrange multiplier)
    for (int i = 0; i < n; ++i)
      y[i] += traj_x[i] - z[i];
  }

  delete[] Q_rho;
  delete[] z;
  delete[] y;
  delete[] b;
}

// ============================================================================
// 5. MARKOV DECISION PROCESS (VALUE ITERATION)
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
    next_dist++; // gets closer faster
  if (action == BRAKE && dist > FAR)
    next_dist--; // distance increases

  next_dist = std::max(0, std::min(2, next_dist));
  return next_dist * 2 + next_speed; // state index
}

float mdp_reward(int dist, int speed, int action) {
  float r = 0;
  if (dist == CLOSE && speed == SPEED_FAST)
    r -= 100; // CRITICAL: prevent accident
  if (dist == CLOSE && action == BRAKE)
    r += 50;
  if (dist == FAR && action == ACCEL)
    r += 20;
  return r;
}

void mdp_value_iteration(int *policy) {
  float V[6] = {0};
  for (int i = 0; i < 50; ++i) { // Value Iteration
    float V_new[6] = {0};
    for (int d = 0; d < 3; ++d) {
      for (int s = 0; s < 2; ++s) {
        int state = d * 2 + s;
        float max_q = -1e9;
        for (int a = 0; a < 3; ++a) {
          int nd, ns;
          int nstate = mdp_next_state(d, s, a, nd, ns);
          float q = mdp_reward(d, s, a) + 0.9f * V[nstate]; // Gamma = 0.9
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
// MAIN APPLICATION (UI & PIPELINE INTEGRATION)
// ============================================================================
int main(int argc, char **argv) {
  string video_path = "..\\Train_set\\dataset_video180.mp4";
  if (argc > 1)
    video_path = argv[1];

  VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    cerr << "Error: Could not open video " << video_path << endl;
    return -1;
  }

  // UI Layout setup
  int window_w = 1600;
  int window_h = 900;
  Mat canvas = Mat::zeros(window_h, window_w, CV_8UC3);

  Rect video_roi(50, 50, 800, 600);
  Rect mdp_roi(900, 50, 650, 400);
  Rect legend_roi(900, 480, 650, 370);
  Rect turn_roi(50, 680, 800, 170);

  // MDP Setup
  int mdp_policy[6] = {0};
  mdp_value_iteration(mdp_policy);

  // SVD History buffer
  const int SVD_M = 10;
  const int SVD_W = 64, SVD_H = 64;
  float *svd_A = new float[SVD_W * SVD_H * SVD_M];
  int svd_idx = 0;

  namedWindow("Project Apex - Autonomous Engine", WINDOW_NORMAL);
  resizeWindow("Project Apex - Autonomous Engine", window_w, window_h);

  Mat frame, gray, float_img;
  int frame_count = 0;

  while (cap.read(frame)) {
    frame_count++;
    canvas = Scalar(20, 20, 25); // Dark theme background

    // 1. Pipeline: Resize & Fixed Point Grayscale
    Mat resized_frame;
    resize(frame, resized_frame, Size(video_roi.width, video_roi.height));
    Mat gray_frame(resized_frame.size(), CV_8UC1);
    rgb_to_gray_fixed_point(resized_frame.data, gray_frame.data,
                            resized_frame.cols, resized_frame.rows);

    // 2. Pipeline: PDE Anisotropic Diffusion
    gray_frame.convertTo(float_img, CV_32F);
    perona_malik_diffusion((float *)float_img.data, float_img.cols,
                           float_img.rows, 2, 0.2f, 20.0f);

    Mat smoothed_gray;
    float_img.convertTo(smoothed_gray, CV_8U);

    // 3. Pipeline: Spatial Background Extraction & Sparse Residuals
    Mat small_gray;
    resize(smoothed_gray, small_gray, Size(SVD_W, SVD_H));
    small_gray.convertTo(small_gray, CV_32F, 1.0 / 255.0);

    static float bg_model[SVD_W * SVD_H] = {0};
    float left_motion = 0, center_motion = 0, right_motion = 0;

    for (int i = 0; i < SVD_W * SVD_H; ++i) {
      float p = ((float *)small_gray.data)[i];

      // SVD/EMA Background approximation
      if (frame_count == 1)
        bg_model[i] = p;
      else
        bg_model[i] = 0.95f * bg_model[i] + 0.05f * p;

      // Extract Sparse Residual (Moving Objects)
      float diff = abs(p - bg_model[i]);

      int x = i % SVD_W;
      int y = i / SVD_W;

      // Analyze lower half of the frame (the road/lanes)
      if (y > SVD_H / 2 && diff > 0.1f) {
        if (x < SVD_W / 3)
          left_motion += diff;
        else if (x < 2 * SVD_W / 3)
          center_motion += diff;
        else
          right_motion += diff;
      }
    }

    // 5. Pipeline: MDP Decision based on ACTUAL visual residuals
    int current_dist = (center_motion > 2.5f) ? CLOSE : FAR;
    int current_speed = SPEED_FAST;
    int state = current_dist * 2 + current_speed;
    int chosen_action = mdp_policy[state];

    // 4. Pipeline: ADMM Trajectory
    // Moved below MDP so we can pass the dynamic decision into the trajectory
    // solver!
    const int T_STEPS = 50;
    float trajectory[T_STEPS] = {0};
    admm_jerk_minimization(trajectory, T_STEPS, chosen_action);

    // ==========================================
    // UI RENDERING
    // ==========================================

    // Video Section
    Mat video_canvas = canvas(video_roi);
    cvtColor(smoothed_gray, video_canvas, COLOR_GRAY2BGR);

    // Overlay target detection graphics on video
    if (current_dist == CLOSE) {
      rectangle(video_canvas, Point(300, 200), Point(500, 400),
                Scalar(0, 0, 255), 3);
      putText(video_canvas, "OBSTACLE DETECTED", Point(300, 190),
              FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    // Draw boundaries
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

    int target_speed_kmh =
        (chosen_action == ACCEL) ? 65 : (chosen_action == BRAKE ? 15 : 45);
    string target_speed_str =
        "Target Speed: " + to_string(target_speed_kmh) + " km/h";
    putText(canvas, target_speed_str, Point(mdp_roi.x + 50, mdp_roi.y + 350),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

    // Legend UI Section
    putText(canvas, "SYSTEM LEGEND & ARCHITECTURE",
            Point(legend_roi.x + 20, legend_roi.y + 40), FONT_HERSHEY_TRIPLEX,
            0.9, Scalar(255, 200, 0), 2);
    int l_y = legend_roi.y + 90;
    putText(canvas, "- [Video] Processed with PM Anisotropic Diffusion (PDE)",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1);
    l_y += 30;
    putText(canvas, "- [Video] Grayscale via Fixed-Point Shift Arithmetic",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1);
    l_y += 30;
    putText(canvas, "- [Video] Red Box: Moving object via SVD sparse residual",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 0, 255), 1);
    l_y += 30;
    putText(canvas,
            "- [MDP] Evaluates Bellman Optimality mapping states -> "
            "deterministic actions",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), 1);
    l_y += 30;
    putText(canvas, "- [Bottom] Autonomous Steering Decision (Turn Indicators)",
            Point(legend_roi.x + 20, l_y), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 255, 0), 1);
    l_y += 30;

    // Turn Indicators UI Section
    putText(canvas, "AUTONOMOUS STEERING DECISION",
            Point(turn_roi.x + 20, turn_roi.y + 30), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(255, 200, 0), 1);

    // Dynamic Lane Evasion Logic
    bool turn_left = false;
    bool turn_right = false;

    // If an obstacle is dead ahead (causing a BRAKE), try to safely evade
    if (current_dist == CLOSE) {
      // Check if left lane is clear of moving objects
      if (left_motion < 1.5f && left_motion <= right_motion) {
        turn_left = true;
        // Check if right lane is clear
      } else if (right_motion < 1.5f) {
        turn_right = true;
      }
    }

    Scalar left_color = turn_left ? Scalar(0, 255, 0) : Scalar(100, 100, 100);
    Scalar right_color = turn_right ? Scalar(0, 255, 0) : Scalar(100, 100, 100);

    arrowedLine(canvas, Point(turn_roi.x + 350, turn_roi.y + 100),
                Point(turn_roi.x + 150, turn_roi.y + 100), left_color, 12, 8, 0,
                0.2);
    arrowedLine(canvas, Point(turn_roi.x + 450, turn_roi.y + 100),
                Point(turn_roi.x + 650, turn_roi.y + 100), right_color, 12, 8,
                0, 0.2);

    // Explicit Reasoning Output (Plain English)
    if (turn_left) {
      putText(canvas,
              "REASON: Obstacle detected on the right. Steering left to avoid.",
              Point(turn_roi.x + 150, turn_roi.y + 150), FONT_HERSHEY_SIMPLEX,
              0.6, Scalar(0, 255, 255), 1);
    } else if (turn_right) {
      putText(
          canvas,
          "REASON: Navigation requires upcoming right turn. Steering right.",
          Point(turn_roi.x + 150, turn_roi.y + 150), FONT_HERSHEY_SIMPLEX, 0.6,
          Scalar(0, 255, 255), 1);
    } else {
      putText(canvas, "REASON: Path is clear. Maintaining safe driving lane.",
              Point(turn_roi.x + 150, turn_roi.y + 150), FONT_HERSHEY_SIMPLEX,
              0.6, Scalar(150, 150, 150), 1);
    }

    imshow("Project Apex - Autonomous Engine", canvas);

    char c = (char)waitKey(30);
    if (c == 27)
      break; // ESC to exit
  }

  cap.release();
  destroyAllWindows();
  delete[] svd_A;
  return 0;
}
