// ============================================================================
// AEGIS / APEX  -  Open mathematical driving engine
//
// Pipeline (every block is hand-written in include/apex/):
//   camera  -> lane candidates -> robust WLS fit -> Gaussian filter -> steering
//   lidar / PDE+SVD video cue  -> distance state -> MDP policy -> target speed
//   target speed + constraints -> ADMM (CG inner solver) -> smooth speed plan
//
// Modes:  apex_engine                 (Webots controller, auto-detected)
//         apex_engine --webots
//         apex_engine <video.mp4> [--headless] [--frames N]
//         options: --record dash.avi   save the dashboard as a video
//                  --snap N file.png   save the dashboard at frame N
// Build with -DAPEX_NO_GUI to drop the OpenCV window (record/snap still work).
// ============================================================================
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_WEBOTS
#include <webots/camera.h>
#include <webots/device.h>
#include <webots/gps.h>
#include <webots/lidar.h>
#include <webots/robot.h>
#include <webots/vehicle/driver.h>
#endif

#include <opencv2/opencv.hpp>

#include "apex/lidar.hpp"
#include "apex/mdp.hpp"
#include "apex/optim.hpp"
#include "apex/vision.hpp"
#include "dashboard.hpp"

using namespace std;
using namespace cv;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

constexpr float kWheelbase = 2.995f;         // BMW X5 [m]
constexpr float kCorridorHalfWidth = 1.35f;  // half car width + margin [m]
constexpr float kLatAccelMax = 2.0f;         // comfortable lateral accel [m/s^2]
constexpr float kLidarAheadOfCamera = 2.38f; // SICK x 4.10 m - camera x 1.72 m

} // namespace

int main(int argc, char **argv) {
  bool use_webots = false;
  bool headless = false;
  long max_frames = -1;
  string video_path = "..\\Train_set\\dataset_video175.mp4";
  bool explicit_video = false;
  string record_path, snap_path, log_path, jpg_dir;
  int ui_every = 1;
  string scan_dump_dir;
  double last_dump = -10.0;
  float cruise_arg = -1.0f;
  long snap_frame = -1;

  for (int i = 1; i < argc; ++i) {
    const string a = argv[i];
    if (a == "--webots" || a == "-w")
      use_webots = true;
    else if (a == "--headless")
      headless = true;
    else if (a == "--frames" && i + 1 < argc)
      max_frames = std::atol(argv[++i]);
    else if (a == "--record-jpg" && i + 1 < argc)
      jpg_dir = argv[++i];
    else if (a == "--record" && i + 1 < argc)
      record_path = argv[++i];
    else if (a == "--dump-scans" && i + 1 < argc)
      scan_dump_dir = argv[++i];
    else if (a == "--cruise" && i + 1 < argc)
      cruise_arg = static_cast<float>(std::atof(argv[++i]));
    else if (a == "--log" && i + 1 < argc)
      log_path = argv[++i];
    else if (a == "--ui-every" && i + 1 < argc)
      ui_every = std::max(1, std::atoi(argv[++i]));
    else if (a == "--snap" && i + 2 < argc) {
      snap_frame = std::atol(argv[++i]);
      snap_path = argv[++i];
    }
    else {
      video_path = a;
      explicit_video = true;
    }
  }
#ifdef APEX_NO_GUI
  headless = true;
#endif
#ifndef USE_WEBOTS
  (void)explicit_video;
#endif
#ifdef USE_WEBOTS
  // Launched by Webots as a controller: Webots sets WEBOTS_ROBOT_NAME /
  // WEBOTS_CONTROLLER_URL. Otherwise fall back to video mode if a file exists.
  if (!use_webots && !explicit_video) {
    FILE *f = fopen(video_path.c_str(), "rb");
    if (f)
      fclose(f);
    if (!f || getenv("WEBOTS_CONTROLLER_URL") || getenv("WEBOTS_ROBOT_NAME"))
      use_webots = true;
  }
#endif

  VideoCapture cap;
  apex::CameraModel cam_model;
  double frame_dt = 1.0 / 25.0;
#ifdef USE_WEBOTS
  WbDeviceTag wb_cam = 0, wb_lidar = 0, wb_gps = 0;
  int wb_cam_w = 0, wb_cam_h = 0;
#endif

  if (use_webots) {
#ifdef USE_WEBOTS
    cout << "[APEX] Initializing Webots driver controller..." << endl;
    wbu_driver_init();
    int basic_step = static_cast<int>(wb_robot_get_basic_time_step());
    if (basic_step <= 0)
      basic_step = 32;
    frame_dt = basic_step / 1000.0;

    const char *cam_names[] = {"camera", "front_camera", "Camera", "kinect color"};
    for (const char *name : cam_names)
      if ((wb_cam = wb_robot_get_device(name)) != 0)
        break;
    const char *lidar_names[] = {"Sick LMS 291", "lidar", "Lidar",
                                 "Velodyne VLP-16", "ibeo"};
    for (const char *name : lidar_names)
      if ((wb_lidar = wb_robot_get_device(name)) != 0)
        break;
    const int n_devs = wb_robot_get_number_of_devices();
    for (int i = 0; i < n_devs; ++i) {
      WbDeviceTag tag = wb_robot_get_device_by_index(i);
      if (!wb_cam && wb_device_get_node_type(tag) == WB_NODE_CAMERA)
        wb_cam = tag;
      if (!wb_lidar && wb_device_get_node_type(tag) == WB_NODE_LIDAR)
        wb_lidar = tag;
      if (!wb_gps && wb_device_get_node_type(tag) == WB_NODE_GPS)
        wb_gps = tag;
    }
    if (wb_cam) {
      wb_camera_enable(wb_cam, basic_step);
      wb_cam_w = wb_camera_get_width(wb_cam);
      wb_cam_h = wb_camera_get_height(wb_cam);
      // BMW X5 roof slot: camera 1.80 m above the road, 1.72 m ahead of the
      // rear axle (from BmwX5.proto: slot z 1.45 + body 0.40 - mount 0.05)
      cam_model = apex::CameraModel::from_webots(
          wb_cam_w, wb_cam_h, static_cast<float>(wb_camera_get_fov(wb_cam)), 1.80f, 1.72f);
      cout << "[APEX] Camera " << wb_cam_w << "x" << wb_cam_h << ", f = " << cam_model.f
           << " px" << endl;
    } else {
      cerr << "[APEX] WARNING: no camera found - lane keeping disabled" << endl;
    }
    if (wb_gps)
      wb_gps_enable(wb_gps, basic_step);
    if (wb_lidar)
      wb_lidar_enable(wb_lidar, basic_step);
    else
      cerr << "[APEX] WARNING: no lidar found - obstacle braking disabled" << endl;

    wbu_driver_set_hazard_flashers(false);
    wbu_driver_set_dipped_beams(true);
    wbu_driver_set_antifog_lights(true);
    wbu_driver_set_wiper_mode(DOWN);
    wbu_driver_set_gear(1);
    wbu_driver_set_brake_intensity(0.0);
    wbu_driver_set_steering_angle(0.0);
    wbu_driver_set_cruising_speed(0.0);
#else
    cerr << "Error: compiled without USE_WEBOTS support." << endl;
    return -1;
#endif
  } else {
    cap.open(video_path);
    if (!cap.isOpened()) {
      cerr << "Error: could not open video " << video_path << endl;
      return -1;
    }
    const double fps = cap.get(CAP_PROP_FPS);
    if (fps > 1.0)
      frame_dt = 1.0 / fps;
    cout << "[APEX] Dashcam video mode: " << video_path << endl;
  }

  // ---------------------------------------------------------------- engine state
  const float v_cruise_kmh =
      cruise_arg > 0.0f ? cruise_arg : (use_webots ? 32.0f : 65.0f);
  apex::DrivingMDP mdp; // value iteration + stationary distribution at start-up
  apex::SpeedPlannerADMM planner;
  apex::LaneTracker lane_tracker;
  lane_tracker.cam = cam_model;
  dash::Dashboard dashboard;

  const int SVD_M = 10, SVD_W = 64, SVD_H = 64, SVD_N = SVD_W * SVD_H;
  vector<float> svd_A(SVD_N * SVD_M, 0.0f), svd_u(SVD_N, 0.0f),
      svd_v(SVD_M, 0.0f), bg_model(SVD_N, 0.0f);
  const int CAM_W = 800, CAM_H = 600;
  vector<float> pde_buf(CAM_W * CAM_H);
  Mat frame, cam_view, gray(CAM_H, CAM_W, CV_8UC1), fimg(CAM_H, CAM_W, CV_32FC1),
      smoothed(CAM_H, CAM_W, CV_8UC1), small_gray, small_f,
      foreground(SVD_H, SVD_W, CV_32FC1, Scalar(0));
  int svd_idx = 0;

  float speed_mps = use_webots ? 0.0f : v_cruise_kmh / 3.6f;
  float accel = 0.0f, steer_cmd = 0.0f, plan_a_prev = 0.0f;
  int speed_bin_prev = apex::V_SLOW, dist_bin_prev = apex::D_CLEAR;
  float gap_f = 100.0f, closing_f = 0.0f;
  int lidar_rejected = 0;
  apex::LeadKalman lead_kf;
  long frame_count = 0;
  double sim_time = 0.0, fps_ema = 0.0;
  apex::SpeedPlannerADMM::Result plan;
  apex::LaneResult lane;

  const string win = "AEGIS / APEX - Open Mathematical Driving Engine";
#ifndef APEX_NO_GUI
  if (!headless) {
    namedWindow(win, WINDOW_NORMAL | WINDOW_KEEPRATIO);
    resizeWindow(win, dash::Dashboard::W, dash::Dashboard::H);
  }
#endif
  VideoWriter recorder;
  if (!record_path.empty()) {
    recorder.open(record_path, VideoWriter::fourcc('M', 'J', 'P', 'G'),
                  1.0 / (frame_dt * ui_every),
                  Size(dash::Dashboard::W, dash::Dashboard::H));
    if (!recorder.isOpened())
      cerr << "[APEX] WARNING: cannot record to " << record_path << endl;
  }
  // JPEG sequence survives the controller being killed when Webots quits
  // (an AVI would be left without its index)
  const bool draw = !headless || recorder.isOpened() || snap_frame > 0 || !jpg_dir.empty();
  long jpg_count = 0;
  FILE *log = log_path.empty() ? nullptr : fopen(log_path.c_str(), "w");
  if (log)
    fprintf(log, "t,x,y,speed_kmh,cmd_kmh,accel,steer,raw_steer,lane_err,lane_ok,"
                 "has_lead,gap,closing,dist_bin,action,admm_iters,admm_conv,"
                 "active_gap,brake,math_ms\n");
  double sum_draw_ms = 0.0;
  double sum_math_ms = 0.0, max_math_ms = 0.0;

  while (max_frames < 0 || frame_count < max_frames) {
    const auto t_loop = Clock::now();
    vector<dash::Timing> timings;

    // ------------------------------------------------------------ acquisition
    const float *lidar = nullptr;
    int lidar_n = 0;
    float lidar_fov = 0.0f;
    if (use_webots) {
#ifdef USE_WEBOTS
      if (wbu_driver_step() == -1)
        break;
      if (wb_cam) {
        const unsigned char *img = wb_camera_get_image(wb_cam);
        if (img) {
          Mat bgra(wb_cam_h, wb_cam_w, CV_8UC4, const_cast<unsigned char *>(img));
          cvtColor(bgra, frame, COLOR_BGRA2BGR);
        }
      }
      if (frame.empty())
        frame = Mat::zeros(CAM_H, CAM_W, CV_8UC3);
      const double v_kmh = wbu_driver_get_current_speed();
      if (std::isfinite(v_kmh) && v_kmh >= 0.0) {
        const float v_new = static_cast<float>(v_kmh / 3.6);
        accel = 0.8f * accel + 0.2f * (v_new - speed_mps) / frame_dt;
        speed_mps = v_new;
      }
      if (wb_lidar) {
        lidar = wb_lidar_get_range_image(wb_lidar);
        lidar_n = wb_lidar_get_horizontal_resolution(wb_lidar);
        lidar_fov = static_cast<float>(wb_lidar_get_fov(wb_lidar));
      }
#endif
    } else {
      if (!cap.read(frame))
        break;
    }
    ++frame_count;
    sim_time += frame_dt;
    auto t = Clock::now();
    resize(frame, cam_view, Size(CAM_W, CAM_H));
    timings.push_back({"frame resize", ms_since(t)});

    // --------------------------------------------- longitudinal: obstacle range
    float fence_nudge = 0.0f;
    bool has_lead = false;
    float gap_meas = 100.0f;
    float motion[3] = {0, 0, 0};
    if (use_webots && lidar && lidar_n > 0) {
      t = Clock::now();
      // corridor + scan clustering + ground-strike/wall rejection (apex/lidar.hpp)
      apex::Corridor cor;
      cor.kappa = std::tan(steer_cmd) / kWheelbase;
      cor.cam_offset = kLidarAheadOfCamera;
      if (lane.valid && lane.fit.ok && lane.fit.used >= 10) { // previous frame's fit
        cor.use_lane = true;
        for (int k = 0; k < 3; ++k)
          cor.c[k] = lane.fit.c[k];
        cor.scale = apex::kFitScale;
        cor.lane_max_x = lane.fit_x_max;
      }
      const apex::ScanObstacle ob =
          apex::find_corridor_obstacle(lidar, lidar_n, lidar_fov, cor, kCorridorHalfWidth);
      lidar_rejected = ob.rejected_long;
      // debug: raw scan whenever something is reported closer than 4 m
      if (!scan_dump_dir.empty() && ob.found && ob.x < 4.0f && sim_time - last_dump > 1.0) {
        last_dump = sim_time;
        char name[96];
        std::snprintf(name, sizeof(name), "/scan_%07.2f.csv", sim_time);
        if (FILE *fs_ = fopen((scan_dump_dir + name).c_str(), "w")) {
          fprintf(fs_, "# ob_x %.3f ob_y %.3f len %.3f steer %.4f lane_ok %d\n", ob.x, ob.y,
                  ob.length, steer_cmd, cor.use_lane ? 1 : 0);
          for (int i = 0; i < lidar_n; ++i)
            fprintf(fs_, "%d,%.4f\n", i, lidar[i]);
          fclose(fs_);
        }
      }
      if (ob.found) {
        has_lead = true;
        gap_meas = ob.x;
      }
      // kerb / fence keep-out: gentle nudge away from a close side obstacle
      const int fs = (lidar_n * 46) / 100, fe = (lidar_n * 54) / 100;
      float min_left = 100.0f, min_right = 100.0f;
      for (int i = (lidar_n * 30) / 100; i <= (lidar_n * 70) / 100; ++i) {
        const float r = lidar[i];
        if (!std::isfinite(r) || r <= 0.4f)
          continue;
        if (i < fs)
          min_left = std::min(min_left, r);
        else if (i > fe)
          min_right = std::min(min_right, r);
      }
      if (min_right < 2.4f)
        fence_nudge -= (2.4f - min_right) * 0.09f;
      if (min_left < 2.2f)
        fence_nudge += (2.2f - min_left) * 0.09f;
      fence_nudge = clampf(fence_nudge, -0.15f, 0.15f);
      timings.push_back({"lidar cluster", ms_since(t)});
    }

    // ------------------------------------------------- lateral: lane + steering
    if (use_webots) {
      t = Clock::now();
      // occlusion masking: paint beyond the car ahead is hidden by it, and its
      // lights / plate must not be read as lane markings
      lane_tracker.max_range =
          has_lead ? clampf(gap_meas + kLidarAheadOfCamera - 1.5f, 8.0f, 40.0f) : 40.0f;
      lane = lane_tracker.process(cam_view.data, CAM_W, CAM_H,
                                  static_cast<float>(frame_dt), speed_mps);
      timings.push_back({"IPM lane WLS", ms_since(t)});
    }

    if (!use_webots) {
      // PDE smoothing -> 64x64 -> rank-1 SVD background -> foreground energy
      t = Clock::now();
      apex::rgb_to_gray_fixed_point(cam_view.data, gray.data, CAM_W, CAM_H, 3);
      gray.convertTo(fimg, CV_32F);
      apex::perona_malik_diffusion(reinterpret_cast<float *>(fimg.data),
                                   pde_buf.data(), CAM_W, CAM_H, 2, 0.2f, 20.0f);
      fimg.convertTo(smoothed, CV_8U);
      timings.push_back({"gray + PDE", ms_since(t)});

      t = Clock::now();
      resize(smoothed, small_gray, Size(SVD_W, SVD_H), 0, 0, INTER_AREA);
      small_gray.convertTo(small_f, CV_32F, 1.0 / 255.0);
      const float *p = reinterpret_cast<const float *>(small_f.data);
      for (int i = 0; i < SVD_N; ++i)
        svd_A[i * SVD_M + svd_idx] = p[i];
      svd_idx = (svd_idx + 1) % SVD_M;
      if (frame_count >= SVD_M)
        apex::power_iteration_svd(svd_A.data(), SVD_N, SVD_M, svd_u.data(),
                                  svd_v.data(), 4);
      float proj = 0.0f;
      if (frame_count >= SVD_M)
        for (int i = 0; i < SVD_N; ++i)
          proj += p[i] * svd_u[i];
      float *fg = reinterpret_cast<float *>(foreground.data);
      for (int i = 0; i < SVD_N; ++i) {
        if (frame_count < SVD_M)
          bg_model[i] = (frame_count == 1) ? p[i] : 0.9f * bg_model[i] + 0.1f * p[i];
        else
          bg_model[i] = 0.85f * proj * svd_u[i] + 0.15f * bg_model[i];
        const float diff = std::fabs(p[i] - bg_model[i]);
        fg[i] = diff;
        const int x = i % SVD_W, y = i / SVD_W;
        if (y > SVD_H / 2 && diff > 0.10f)
          motion[x < SVD_W / 3 ? 0 : (x < 2 * SVD_W / 3 ? 1 : 2)] += diff;
      }
      // Monocular cue -> pseudo range (heuristic, NOT a measurement): more
      // centre-column foreground energy = shorter time headway. The recorded
      // car's speed is fixed by the footage, so scale by the cruise speed,
      // never by our simulated speed (that would create a feedback loop).
      if (motion[1] > 0.8f) {
        has_lead = true;
        const float headway_s = clampf(0.7f + (2.5f - motion[1]) * 1.0f, 0.7f, 2.4f);
        gap_meas = headway_s * v_cruise_kmh / 3.6f;
      }
      timings.push_back({"SVD background", ms_since(t)});
    }

    // 2-state Kalman filter on [gap, gap rate] (apex/lidar.hpp)
    lead_kf.step(static_cast<float>(frame_dt), has_lead, gap_meas);
    has_lead = lead_kf.active;
    if (has_lead) {
      gap_f = std::max(0.0f, lead_kf.gap);
      closing_f = lead_kf.closing_speed();
    } else {
      gap_f = 100.0f;
      closing_f = 0.0f;
    }

    // ----------------------------------------- final steering (rate limited)
    float steer_target = use_webots ? lane.steering_rad + fence_nudge : 0.0f;
    steer_target = clampf(steer_target, -lane_tracker.max_steer, lane_tracker.max_steer);
    const float max_step = lane_tracker.max_steer_rate * static_cast<float>(frame_dt);
    steer_cmd += clampf(steer_target - steer_cmd, -max_step, max_step);

    // ----------------------------------------------------- MDP decision
    t = Clock::now();
    // curve speed limit from lateral acceleration: v^2 |kappa| <= a_lat_max
    const float kappa_abs = std::max(std::fabs(lane.curvature),
                                     std::fabs(std::tan(steer_cmd)) / kWheelbase);
    const float limit_kmh =
        use_webots ? std::max(15.0f, std::min(v_cruise_kmh,
                                              3.6f * std::sqrt(kLatAccelMax / std::max(kappa_abs, 1e-4f))))
                   : v_cruise_kmh;
    dist_bin_prev = has_lead ? apex::DrivingMDP::distance_bin(gap_f, closing_f, dist_bin_prev)
                             : static_cast<int>(apex::D_CLEAR);
    const int dist_bin = dist_bin_prev;
    const float v_lead_est = has_lead ? std::max(0.0f, speed_mps - closing_f) : speed_mps;
    speed_bin_prev = apex::DrivingMDP::speed_bin(speed_mps * 3.6f, v_cruise_kmh, speed_bin_prev);
    const int speed_bin = speed_bin_prev;
    const int action = mdp.policy[apex::DrivingMDP::idx(dist_bin, speed_bin)];
    const float v_lim = limit_kmh / 3.6f;
    float v_ref = v_lim;
    if (action == apex::MAINTAIN)  // FAST bin = "at cruise": hold the cruise target
      v_ref = (speed_bin == apex::V_FAST) ? v_lim
                                          : std::min(std::max(speed_mps, 0.3f * v_lim), v_lim);
    else if (action == apex::BRAKE)
      // brake TOWARD the obstacle's speed (car-following), not to zero, unless
      // it is (nearly) stationary; the ADMM gap constraint guarantees safety
      v_ref = (dist_bin == apex::D_CLOSE)
                  ? (v_lead_est < 1.0f ? 0.0f : 0.8f * v_lead_est)
                  : std::max(0.0f, std::min({speed_mps - 1.5f, v_lead_est, v_lim}));
    timings.push_back({"MDP policy", ms_since(t)});

    // ------------------------------------------------- ADMM speed plan
    t = Clock::now();
    apex::SpeedPlannerADMM::Problem pb;
    pb.v0 = speed_mps;
    // seed jerk continuity with the previous PLAN's acceleration (receding
    // horizon); the differentiated speed sensor is too noisy and feeds back
    pb.a0 = use_webots ? plan_a_prev : clampf(accel, -6.0f, 3.0f);
    pb.v_ref = v_ref;
    pb.v_max = v_lim;
    pb.has_lead = has_lead;
    pb.gap = gap_f;
    pb.v_lead = std::max(0.0f, speed_mps - closing_f);
    plan = planner.solve(pb);
    const double admm_ms = ms_since(t);
    timings.push_back({"ADMM + CG", admm_ms});

    // receding horizon: apply the start of the plan, re-plan next frame
    const float plan_dt = planner.params().dt;
    const float a_cmd = plan.a[0];
    plan_a_prev = plan.a[0];
    const float v_cmd =
        clampf(plan.v[std::min(2, plan.n - 1)], 0.0f, v_lim);
    float brake = 0.0f;
    if (a_cmd < -1.0f)
      brake = clampf((-a_cmd - 1.0f) / 5.0f, 0.0f, 1.0f);
    const bool emergency = has_lead && gap_f < 3.0f && speed_mps > 0.3f;
    if (emergency)
      brake = 1.0f;

    if (use_webots) {
#ifdef USE_WEBOTS
      wbu_driver_set_steering_angle(steer_cmd);
      wbu_driver_set_cruising_speed(emergency ? 0.0 : v_cmd * 3.6);
      wbu_driver_set_brake_intensity(brake);
#endif
    } else {
      speed_mps = std::max(0.0f, speed_mps + a_cmd * static_cast<float>(frame_dt));
      accel = a_cmd;
    }

    double math_ms = 0.0;
    for (const auto &tm : timings)
      math_ms += tm.ms;
    sum_math_ms += math_ms;
    max_math_ms = std::max(max_math_ms, math_ms);

    // ------------------------------------------------- dashboard
    if (log) {
      double gx = 0.0, gy = 0.0;
#ifdef USE_WEBOTS
      if (use_webots && wb_gps) {
        const double *g = wb_gps_get_values(wb_gps);
        if (g) {
          gx = g[0];
          gy = g[1];
        }
      }
#endif
      fprintf(log, "%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.4f,%.4f,%.4f,%d,%d,%.2f,%.2f,"
                   "%d,%d,%d,%d,%d,%.2f,%.3f\n",
              sim_time, gx, gy, speed_mps * 3.6f, v_cmd * 3.6f, plan.a[0],
              steer_cmd, use_webots ? lane.raw_steer : 0.0f,
              use_webots ? lane.lateral_error : 0.0f, use_webots && lane.valid,
              has_lead, gap_f, closing_f, dist_bin, action, plan.iterations,
              plan.converged, plan.active_gap, brake, math_ms);
    }

    if (draw && (frame_count % ui_every == 0 || frame_count == snap_frame)) {
      const auto t_draw = Clock::now();
      dash::Data d;
      d.webots = use_webots;
      d.frame = static_cast<int>(frame_count);
      d.sim_time = sim_time;
      d.fps = fps_ema;
      d.camera = &cam_view;
      d.lane = use_webots ? &lane : nullptr;
      d.cam = &lane_tracker.cam;
      d.steer = steer_cmd;
      d.raw_steer = use_webots ? lane.raw_steer + fence_nudge : 0.0f;
      d.speed_kmh = speed_mps * 3.6f;
      d.cmd_kmh = v_cmd * 3.6f;
      d.limit_kmh = limit_kmh;
      d.brake = brake;
      d.gap_m = gap_f;
      d.closing_mps = closing_f;
      d.has_lead = has_lead;
      d.dist_bin = dist_bin;
      d.speed_bin = speed_bin;
      d.action = action;
      d.mdp = &mdp;
      d.plan = &plan;
      d.plan_dt = plan_dt;
      d.v_ref_kmh = v_ref * 3.6f;
      d.timings = timings;
      d.lidar = lidar;
      d.lidar_n = lidar_n;
      d.lidar_fov = lidar_fov;
      d.svd_foreground = use_webots ? nullptr : &foreground;
      d.pde_view = use_webots ? nullptr : &smoothed;
      std::copy(motion, motion + 3, d.motion);
      d.emergency = emergency;
      const Mat &ui = dashboard.render(d);
      if (recorder.isOpened())
        recorder.write(ui);
      if (!jpg_dir.empty()) {
        char name[64];
        std::snprintf(name, sizeof(name), "/%06ld.jpg", jpg_count++);
        imwrite(jpg_dir + name, ui, {IMWRITE_JPEG_QUALITY, 88});
      }
      if (frame_count == snap_frame)
        imwrite(snap_path, ui);
      sum_draw_ms += ms_since(t_draw);
#ifndef APEX_NO_GUI
      if (!headless) {
        imshow(win, ui);
        if (static_cast<char>(waitKey(1)) == 27)
          break;
      }
#endif
    }
    const double loop_ms = ms_since(t_loop);
    fps_ema = (fps_ema == 0.0) ? 1000.0 / loop_ms : 0.9 * fps_ema + 0.1 * 1000.0 / loop_ms;
  }

  if (frame_count > 0)
    printf("[APEX] %ld frames, maths %.2f ms/frame avg, %.2f ms worst, "
           "dashboard %.2f ms/frame, loop %.1f FPS\n",
           frame_count, sum_math_ms / frame_count, max_math_ms,
           sum_draw_ms / frame_count, fps_ema);
  if (log)
    fclose(log);
  if (use_webots) {
#ifdef USE_WEBOTS
    wbu_driver_cleanup();
#endif
  } else {
    cap.release();
  }
#ifndef APEX_NO_GUI
  if (!headless)
    destroyAllWindows();
#endif
  return 0;
}
