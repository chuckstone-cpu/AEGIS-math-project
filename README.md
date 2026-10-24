# AEGIS / APEX — an open mathematical driving engine

A self-driving controller in C++ where **every decision is made by maths we wrote ourselves**: no
black-box neural network and no maths library. OpenCV is used only to read video and draw the dashboard.

```
camera ─► lane candidates ─► robust WLS fit ─► Gaussian filter ─► steering (rate-limited)
lidar / PDE+SVD cue ─► distance state ─► MDP policy ─► target speed
target speed + constraints ─► ADMM (inner CG/Krylov solver) ─► smooth, safe speed plan
```

## Syllabus map (22MAT220)

| Unit | Topic | Where | What it does |
|---|---|---|---|
| 1 | Krylov subspace, Conjugate Gradient | `include/apex/linalg.hpp` `pcg_solve` | Matrix-free preconditioned CG solves the ADMM x-step |
| 1 | Constrained convex optimisation, QP | `include/apex/optim.hpp` | 4 s speed plan: track target, minimise acceleration, obey limits |
| 1 | Sparsity-inducing penalty | `optim.hpp` (L1 on jerk, soft-threshold prox) | Sparse jerk gives piecewise-constant, comfortable acceleration |
| 1 | ADMM | `optim.hpp` `SpeedPlannerADMM::solve` | Over-relaxed, residual-balanced ADMM, warm-started every frame |
| 1 | Lagrange multipliers, KKT | `optim.hpp` (dual variables, residuals) | Primal/dual residuals and active-constraint multipliers shown live |
| 1 | Exact penalty | `optim.hpp` `shrink_above` | Speed and gap limits stay solvable even when an obstacle appears too close |
| 2 | PDE, finite differences | `include/apex/vision.hpp` `perona_malik_diffusion` | Edge-preserving smoothing (explicit scheme, CFL bound λ ≤ 1/4) |
| 1/2 | Eigen / SVD (power iteration) | `vision.hpp` `power_iteration_svd` | Rank-1 background model gives a moving-obstacle cue in dashcam mode |
| 3 | Weighted least squares, multivariate Gaussian | `include/apex/stats.hpp` | Lane-centre quadratic fit plus predictive variance σ²φᵀ(ΦᵀWΦ)⁻¹φ |
| 3 | Inequalities of statistics (Chebyshev) | `stats.hpp` `robust_wls_polyfit` | Rejects lane outliers beyond kσ; at most 1/k² of good points are lost, for any distribution |
| 3 | Markov chain / Gaussian state model | `stats.hpp` `GaussianTracker` | Random-walk state model with product-of-Gaussians update (removes steering jitter) |
| 3 | Markov decision process | `include/apex/mdp.hpp` | 12-state stochastic MDP, value iteration to an ε-optimal policy, plus the stationary distribution of the closed loop |

## Build

Requirements: a C++17 compiler, CMake ≥ 3.16, and OpenCV. Webots is optional; without it the
engine builds in video-only mode.

```bash
cmake -S . -B build -G "MinGW Makefiles"
```
```bash
cmake --build build -j 8
```
```bash
build/test_math.exe
```

The CMake script auto-detects `C:/opencv_mingw`, `C:/opencv/build` and MSYS2 (`C:/msys64/ucrt64`).
Webots is found through `WEBOTS_HOME` or at `C:/Program Files/Webots`.

> **MSYS2 note:** MSYS2's OpenCV opens its window through Qt6. If the program exits with
> `0xC0000135` (DLL not found), either install Qt (`pacman -S mingw-w64-ucrt-x86_64-qt6-base`) or
> build with `-DAPEX_NO_GUI=ON` and use `--record` to watch the dashboard as a video.

## Run

| Command | Mode |
|---|---|
| `apex_engine ..\Train_set\dataset_video175.mp4` | dashcam video, live dashboard |
| `apex_engine video.mp4 --headless --frames 600` | benchmark only |
| `apex_engine video.mp4 --record dash.avi` | save the dashboard as a video (for presentations) |
| `apex_engine video.mp4 --snap 400 shot.png` | save one dashboard frame |
| set as the car's controller in Webots | simulation (auto-detected) |

## Webots simulation

```bash
powershell -ExecutionPolicy Bypass -File webots/run_demo.ps1 -Duration 185 -Run demo
```

The script generates the world, pre-fetches assets, runs Webots (fast mode, recording), analyses the
logs, and composes `webots/output/<run>/<run>_final.mp4`: the Webots 3D chase view on the left and
the live engine dashboard on the right. `-Watch` opens Webots in real time instead.

| Piece | File |
|---|---|
| World, generated from Webots' `city.wbt` (reproducible) | `webots/tools/build_world.py` → `webots/worlds/apex_city.wbt` |
| Scenarios and map dressing, placed on the driven lane path | `webots/tools/scenarios.py`, `scenario_plan.json`, `worlds/ref_path.csv` |
| Ego controller (this engine, plus bundled DLLs) | `webots/controllers/apex_engine/` (CMake post-build deploy) |
| Supervisor: movie, stop & go, moving actors, ground-truth log | `webots/controllers/scenario_supervisor/scenario_supervisor.c` |
| Asset pre-fetch (works around Webots' download stall) | `webots/tools/prefetch_assets.py` |
| Metrics and report figure | `webots/tools/analyze_run.py` |

Scenarios along the ~760 m lap: a pedestrian on the zebra crossing, a stock oil barrel in the lane,
parked cars in the outer lane (must be ignored), a slow lead car at 12 km/h, a cut-in from the outer
lane, a stalled car, and a cone/barrel lane closure. Map dressing adds street lights, trees, bus stops,
benches, bins and roadwork signs.

### Simulation-side maths (added while bringing the car up in Webots)
- **Inverse perspective mapping:** the pinhole model x = f·h/(v − c_y), y = −(u − c_x)·x/f
  back-projects lane paint to metres on the road. The yellow line, lane divider and road edge are
  then fused in **one** robust WLS fit with inverse-variance weights w ∝ 1/x².
- **Pure pursuit:** curvature κ = 2y/D² to a speed-dependent look-ahead point, steering δ = atan(Lκ).
  The curve speed limit comes from lateral acceleration, v ≤ √(a_lat/|κ|).
- **Path-corridor obstacle test:** lidar returns are kept only inside the parabola y = −κx²/2 that
  the car will sweep. A fixed cone sees guard-rails on every bend.
- **Ground-strike rejection:** a pitching planar scanner cuts the road along a line *across* the
  scan; obstacles are short segments, so returns wider than 7 m are rejected.

## Verified results

`test_math` checks each solver against an independent reference or a theorem. All 30 checks pass:

- **CG** matches dense Gaussian elimination to 5e-8, in 9 iterations for n = 40.
- **WLS** recovers the true polynomial and rejects exactly the 3 planted outliers.
- **ADMM** matches the closed-form QP to 3e-4 m/s, and lands within 0.02 m/s of a 3000-iteration
  optimum in about 58 iterations / 0.4 ms. It respects the speed, acceleration, jerk and gap
  limits; with warm start it needs 5 iterations instead of 17 from cold.
- **Infeasible scenario** (obstacle 10 m ahead at 65 km/h): the plan stays physically executable
  and brakes exactly at the physical limit.
- **MDP**: Bellman residual 4e-5, transition matrices are stochastic, and the policy makes sense.
- **PDE** obeys the maximum principle; **SVD** recovers the true singular vector (cos = 1.000000).

Speed (Intel i5-13420H laptop, 800×600): all maths together takes **2.2–2.4 ms per frame** on full 1200-frame
dashcam videos.

Steering jitter (synthetic Webots-style frames, a third of them corrupted; same detections fed to both
versions): total steering variation on a straight road goes from **0.426 rad (old) to 0.000 rad
(new)**, with the same response time to a lane shift.

## Bugs fixed relative to `legacy/apex_engine_original.cpp`

1. **ADMM was mathematically broken.** Its system matrix had zero-Dirichlet ends and no tracking
   term, so the plan always saturated at the 65 km/h clamp, whatever the action (see
   `legacy/`). The controller therefore ignored ADMM and changed speed in jumps.
2. The car **could never stop**: the commanded speed had a floor of 18 km/h and the brake was
   always 0.
3. The ADMM "trajectory" was in normalised units and never fed back. It now drives the car
   (receding horizon).
4. The MDP was deterministic with 6 states, and value iteration ran a fixed 50 sweeps with no
   convergence test. It is now stochastic with 12 states, uses the contraction-based stopping rule,
   and reports the stationary distribution.
5. `-ffast-math` broke the `isnan`/`isinf` lidar checks (GCC assumes NaN never occurs under
   fast-math). It has been removed.
6. Steering was smoothed with a fixed exponential average on a plain weighted mean, so one bad row
   moved the wheel. It is now a robust WLS fit, a variance-weighted Gaussian update and a rate limit.
7. The SVD Gram buffer was a fixed `float[64*64]` with no bounds check. It is now guarded, and
   power iteration is warm-started.

See `SIMULATION_PLAN.md` for the Webots roadmap.
