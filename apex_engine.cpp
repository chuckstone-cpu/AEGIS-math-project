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
#include <webots/lidar.h>
#include <webots/robot.h>
#include <webots/vehicle/driver.h>
#endif

#include <opencv2/opencv.hpp>

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

} // namespace

int main(int argc, char **argv) {
  bool use_webots = false;
  bool headless = false;
  long max_frames = -1;
  string video_path = "..\\Train_set\\dataset_video175.mp4";
  bool explicit_video = false;
  string record_path, snap_path;
  long snap_frame = -1;

  for (int i = 1; i < argc; ++i) {
    const string a = argv[i];
    if (a == "--webots" || a == "-w")
      use_webots = true;
    else if (a == "--headless")
      headless = true;
    else if (a == "--frames" && i + 1 < argc)
      max_frames = std::atol(argv[++i]);
    else if (a == "--record" && i + 1 < argc)
      record_path = argv[++i];
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
  double frame_dt = 1.0 / 25.0;
#ifdef USE_WEBOTS
  WbDeviceTag wb_cam = 0, wb_lidar = 0;
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
    }
    if (wb_cam) {
      wb_camera_enable(wb_cam, basic_step);
      wb_cam_w = wb_camera_get_width(wb_cam);
      wb_cam_h = wb_camera_get_height(wb_cam);
      cout << "[APEX] Camera " << wb_cam_w << "x" << wb_cam_h << endl;
    } else {
      cerr << "[APEX] WARNING: no camera found - lane keeping disabled" << endl;
    }
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
  const float v_cruise_kmh = use_webots ? 32.0f : 65.0f;
  apex::DrivingMDP mdp; // value iteration + stationary distribution at start-up
  apex::SpeedPlannerADMM planner;
  apex::LaneTracker lane_tracker;
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
  float accel = 0.0f, steer_cmd = 0.0f;
  float gap_f = 100.0f, closing_f = 0.0f, prev_gap = -1.0f;
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
                  1.0 / frame_dt, Size(dash::Dashboard::W, dash::Dashboard::H));
    if (!recorder.isOpened())
      cerr << "[APEX] WARNING: cannot record to " << record_path << endl;
  }
  const bool draw = !headless || recorder.isOpened() || snap_frame > 0;
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

    // ------------------------------------------------- lateral: lane + steering
    float fence_nudge = 0.0f;
    if (use_webots) {
      t = Clock::now();
      lane = lane_tracker.process(cam_view.data, CAM_W, CAM_H,
                                  static_cast<float>(frame_dt));
      timings.push_back({"lane WLS+Gauss", ms_since(t)});
    }

    // --------------------------------------------- longitudinal: obstacle range
    bool has_lead = false;
    float gap_meas = 100.0f;
    float motion[3] = {0, 0, 0};
    if (use_webots && lidar && lidar_n > 0) {
      t = Clock::now();
      const int fs = (lidar_n * 46) / 100, fe = (lidar_n * 54) / 100;
      float min_front = 100.0f, min_left = 100.0f, min_right = 100.0f;
      for (int i = 0; i < lidar_n; ++i) {
        float r = lidar[i];
        if (!std::isfinite(r) || r <= 0.4f)
          r = 80.0f;
        if (i >= fs && i <= fe)
          min_front = std::min(min_front, r);
        else if (i >= (lidar_n * 30) / 100 && i < fs)
          min_left = std::min(min_left, r);
        else if (i > fe && i <= (lidar_n * 70) / 100)
          min_right = std::min(min_right, r);
      }
      // kerb / fence keep-out: gentle nudge away from a close side obstacle
      if (min_right < 2.4f)
        fence_nudge -= (2.4f - min_right) * 0.09f;
      if (min_left < 2.2f)
        fence_nudge += (2.2f - min_left) * 0.09f;
      fence_nudge = clampf(fence_nudge, -0.15f, 0.15f);
      if (min_front < 45.0f) {
        has_lead = true;
        gap_meas = min_front;
      }
      timings.push_back({"lidar scan", ms_since(t)});
    } else if (!use_webots) {
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

    // smooth the range signal and estimate the closing speed d(gap)/dt
    if (has_lead) {
      gap_f = (prev_gap < 0.0f) ? gap_meas : 0.6f * gap_f + 0.4f * gap_meas;
      if (prev_gap >= 0.0f)
        closing_f = 0.85f * closing_f +
                    0.15f * static_cast<float>((prev_gap - gap_f) / frame_dt);
      prev_gap = gap_f;
    } else {
      gap_f = 100.0f;
      closing_f = 0.0f;
      prev_gap = -1.0f;
    }

    // ----------------------------------------- final steering (rate limited)
    float steer_target = use_webots ? lane.steering_rad + fence_nudge : 0.0f;
    steer_target = clampf(steer_target, -0.38f, 0.38f);
    const float max_step = lane_tracker.max_steer_rate * static_cast<float>(frame_dt);
    steer_cmd += clampf(steer_target - steer_cmd, -max_step, max_step);

    // ----------------------------------------------------- MDP decision
    t = Clock::now();
    const float turn = std::fabs(steer_cmd);
    const float limit_kmh =
        use_webots ? std::max(21.0f, v_cruise_kmh - turn * 32.0f) : v_cruise_kmh;
    const int dist_bin = has_lead ? apex::DrivingMDP::distance_bin(gap_f, closing_f)
                                  : static_cast<int>(apex::D_CLEAR);
    const int speed_bin = apex::DrivingMDP::speed_bin(speed_mps * 3.6f, v_cruise_kmh);
    const int action = mdp.policy[apex::DrivingMDP::idx(dist_bin, speed_bin)];
    const float v_lim = limit_kmh / 3.6f;
    float v_ref = v_lim;
    if (action == apex::MAINTAIN)
      v_ref = std::min(std::max(speed_mps, 0.3f * v_lim), v_lim);
    else if (action == apex::BRAKE)
      v_ref = (dist_bin == apex::D_CLOSE) ? 0.0f
                                          : std::max(0.0f, std::min(speed_mps, v_lim) - 2.5f);
    timings.push_back({"MDP policy", ms_since(t)});

    // ------------------------------------------------- ADMM speed plan
    t = Clock::now();
    apex::SpeedPlannerADMM::Problem pb;
    pb.v0 = speed_mps;
    pb.a0 = clampf(accel, -6.0f, 3.0f);
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
    const float v_cmd = std::max(0.0f, plan.v[std::min(2, plan.n - 1)]);
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
    if (draw) {
      const auto t_draw = Clock::now();
      dash::Data d;
      d.webots = use_webots;
      d.frame = static_cast<int>(frame_count);
      d.sim_time = sim_time;
      d.fps = fps_ema;
      d.camera = &cam_view;
      d.lane = use_webots ? &lane : nullptr;
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
