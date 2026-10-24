# Webots work: status (paused 2026-10-25)

## Working
- Webots R2025a installed per-user (`%LOCALAPPDATA%\Programs\Webots`); Qt6 installed in MSYS2.
- Build: `cmake -S . -B build_wb -G "MinGW Makefiles"` with `WEBOTS_HOME` set, then `cmake --build build_wb`.
  It deploys the controller and its DLLs to `webots/controllers/apex_engine/`.
- `python webots/tools/prefetch_assets.py webots/worlds/apex_city.wbt` must run after any world
  change. Webots' own asset downloader stalls on this network.
- World generator, scenarios, supervisor (movie, stop & go, moving actors), analyser and video
  composer are all working.
- Tests: 39 checks, all passing (`build_wb/test_math.exe`).
- Best clean-lap result (calib4, no scenarios): constant 29 km/h, steering variation 2.3 rad/min,
  ADMM converged on 98% of frames.
- Demo 5 (all scenarios): the barrel, lead car, cut-in and stalled car are all handled. The car
  then stops before the roadworks (the open bug below).

## Open bug: phantom obstacle before the roadworks (next step)
- Reproduce in about 1 minute:
  `python webots/tools/build_world.py --stage full --duration 40 --movie dbg.mp4 --ego-record --run dbg1 --start-s 600`,
  then run Webots on the world.
- The car stops at x = 27.7 on the top straight. The raw scan (`output/dbg1/scans/`) shows ONE
  isolated ray, index 93 (straight ahead), returning 1.52 m, with cluster length 0.
- The only nearby world object is the DivergentIndicator at (18.57, 110.92). That's 3.2 m right of
  the lane, so it should not be the cause. Next: check whether the SICK at bumper height
  (translation 0.25 0 -0.42 in the front slot) hits its own mount, the plate or the ground.
- Candidate fixes:
  - require a cluster of at least 2 rays (single-ray returns = noise / self-hit), or
  - check this ray's range against the predicted ground-hit distance.

## After the bug
1. Full demo: `powershell -ExecutionPolicy Bypass -File webots/run_demo.ps1 -Duration 185 -Run demo`.
   Output: `webots/output/demo/demo_final.mp4` (3D view | dashboard) plus `run_report.png`.
2. Smooth the car-following (some brake chatter remains behind the lead car).
3. Update README results and SIMULATION_PLAN, then commit (do not push).
