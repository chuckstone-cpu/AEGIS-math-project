# Simulation plan — Webots

Status: the maths engine is fixed, tested and fast (see README). The simulation itself has
**not been changed yet**. This document says what to do next and in what order.

## 0. First blocker: the Webots world is missing from this folder

There is no `.wbt` world and no `controllers/` folder in any git branch, and Webots is not installed
on this PC. (`Simulation/` on the `apex-engine-v2` and `autopilot` branches is the CARLA Python
client, not a Webots world.) The code comments show the car was driven in Webots' **built-in
`city.wbt`** sample, with the controller pointed at `apex_engine.exe`. So step 1 creates our own
copy of `city.wbt`, which the lane detector is calibrated on (yellow centre line, dark asphalt,
white dashes), and commits it. That way the world becomes part of the project and can be extended.

## 1. Webots or CARLA? → stay with Webots

| | Webots | CARLA |
|---|---|---|
| Install / hardware | ~0.5 GB, runs on a laptop iGPU/dGPU | ~20 GB+, Unreal Engine, needs a strong GPU (8 GB VRAM recommended) |
| C++ controller | Native C API, already used by `apex_engine.cpp` (`wbu_driver_*`) | Python-first; C++ client (LibCarla) is painful to build on Windows |
| Determinism | Lock-step simulation (`synchronization TRUE`) | Needs synchronous mode + fixed delta setup |
| Assets | City, highway, village worlds; road, crossroad and roundabout PROTOs; pedestrians, cones, traffic lights; SUMO traffic | Photorealistic towns, Traffic Manager, ScenarioRunner |
| Visual quality | Good, not photoreal | Excellent |
| Cost to switch | 0 | Rewrite the whole I/O layer, recalibrate the lane detector |

Verdict: **Webots.** The project is graded on maths, and the engine already speaks Webots' C API.
CARLA would mean a rewrite and a GPU that a laptop may not have. `carla_settings.txt`
(Mini Cooper S) can stay as a "future work" note.

## 2. Project layout to create

```
AEGIS-math-project/
  webots/
    worlds/apex_city.wbt            <- copy of Webots city.wbt, then extended (step 4-5)
    controllers/apex_engine/
        apex_engine.exe             <- CMake post-build copy (add a step to CMakeLists)
        runtime.ini                 <- adds the OpenCV bin folder to PATH for Webots
    controllers/scenario_supervisor/  <- spawns obstacles on a timeline (step 4)
```

`runtime.ini` (avoids DLL-not-found errors when Webots launches the controller):
```ini
[environment variables with paths]
PATH = C:/msys64/ucrt64/bin
```

The car (`BmwX5` or `TeslaModel3`) gets `controller "apex_engine"`. Its `sensorsSlotFront` holds a
`Camera { name "camera" }` and a `Lidar` named `"Sick LMS 291"`; the engine auto-discovers both.

## 3. Remove jitter from the *visual* driving

The engine side is done (WLS + Gaussian filter + steering-rate limit + ADMM speed plan). The
simulation-side causes of jitter are:

| Cause | Fix |
|---|---|
| Camera follows the car rigidly, so every physics wobble shows on screen | `Viewpoint { follow "vehicle" followType "Pan and Tilt Shot" followSmoothness 0.5 }` |
| Physics step too coarse for tyres | `WorldInfo { basicTimeStep 16 }`; keep sensors at 32 ms (`wb_camera_enable(cam, 32)`) |
| Dashboard slows the simulation (`imshow` every step) and the real-time factor drops | Draw the dashboard every 2nd step, or run it in a thread; watch the RT factor in the Webots toolbar |
| Camera aspect ratio ≠ 800×600 (resize distorts the lane geometry) | Set the camera to 512×384 (4:3), or recalibrate the `0.23 + 0.11·y` half-lane offset |
| Speed jumps at turns | Already fixed: the curve limit enters ADMM as a constraint, so the plan decelerates *before* the turn |

**Acceptance check:** log `steer`, `speed`, `accel` to CSV for 2 minutes of driving.
Targets: steering total variation < 1 rad/min on straights, |jerk| ≤ 4 m/s³, and no
oscillation visible from the follow camera.

## 4. More obstacles (each one a scenario that shows off a piece of maths)

| # | Scenario | Webots nodes | What it demonstrates |
|---|---|---|---|
| S1 | Stopped car in the lane | `BmwX5Simple` / `ToyotaPriusSimple` (no controller) | ADMM gap constraint active, KKT multipliers > 0, smooth stop at 4 m |
| S2 | Slow lead car (ACC) | Second car with a tiny controller holding 15 km/h | MDP MED/CLOSE ↔ HOLD/BRAKE; plan converges to the lead's speed |
| S3 | Pedestrian crossing at a crosswalk | `Pedestrian` PROTO + Webots' `pedestrian` controller (trajectory argument) | Sudden obstacle → exact-penalty (infeasible-safe) braking |
| S4 | Lane closure with cones | 6–8 `TrafficCone` in a taper | Lidar side-sector nudge + MDP slowdown |
| S5 | Cut-in | Car on the adjacent lane steering in at t = 20 s | Closing-speed estimate, TTC-based distance bin |
| S6 | Random traffic | SUMO via `SumoInterface` (Webots ships it) | Long unscripted runs for statistics |

A `scenario_supervisor` controller (Supervisor API, `wb_supervisor_field_import_mf_node_from_string`)
spawns these at fixed times. Every run is then repeatable, which makes the demo and the report
reproducible.

## 5. More map elements

Starting from `city.wbt`, add:
- **Roads:** `Roundabout`, `CurvedRoadSegment` (gentle and sharp), `RoadIntersection`, and a
  two-lane straight for S5. Keep the same road textures (yellow centre line), because the lane
  detector depends on them.
- **Traffic control:** `GenericTrafficLight` with Webots' `crossroads_traffic_lights` controller,
  `StopSign`, `SpeedLimitSign`.
- **Scenery:** buildings (`Building`, `CommercialBuilding`), `Forest`, street lights, `Fog` (weather
  stress test for the PDE smoothing).
- **Parking:** parked cars along the kerb. These test that the narrow lidar cone does not brake for
  cars in the parking lane.

## 6. Engine follow-ups the simulation enables (still syllabus-relevant)

1. **Traffic-light state in the MDP:** extend the state to (distance, speed, light), i.e. 36 states.
   Value iteration is unchanged.
2. **Lateral MPC:** replace `atan(K·e)` with a steering QP over the horizon (lane error + steering-rate
   penalty), solved by the same ADMM/CG code.
3. **Pedestrian intention as a Markov chain:** states {waiting, crossing, cleared}, with transition
   probabilities estimated from the scenario logs.
4. **Fourier method (Unit 2):** an FFT heat-equation smoother as a comparison against the
   finite-difference Perona–Malik scheme (accuracy vs. speed table for the report).

## 7. Order of work (each step has a visible result)

1. Install the latest Webots release, then copy `projects/vehicles/worlds/city.wbt` to `webots/worlds/apex_city.wbt`.
2. Add the controller folder, `runtime.ini` and the CMake copy step; confirm the car drives the default loop.
3. Apply the viewpoint and timestep settings (§3); record a 2-minute CSV and check the jitter targets.
4. Add S1 and S2 (the strongest ADMM/KKT demo), then S3–S5.
5. Add map elements (§5) and re-check lane tracking on the new roads.
6. Optional: SUMO traffic and the engine follow-ups (§6).
