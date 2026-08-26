# Apex Engine (AEGIS Math Project)

This repository contains the core implementations for the Apex Engine, a computer vision and convex optimization pipeline for autonomous navigation. 

## Included Modules
- **Phase 1 (Vision Ingestion):** Grayscale conversion and preprocessing (`demo_step1_ingestion.cpp`).
- **Phase 2 (PDE Smoothing):** Anisotropic diffusion for noise reduction (`demo_step2_pdesmoothing.cpp`).
- **Phase 3 (Optimization & Perception):** Robust Hough Transform, Edge Density collision detection, and Quadratic Programming (Conjugate Gradient) trajectory generation (`demo_step3_optimization.cpp`, `vision_layer.hpp`, `control_layer.hpp`).

## Build Instructions (Windows / MinGW)

Because compiled binaries and build artifacts are completely ignored in version control, you must generate the `build` directory locally on your machine before running the engine.

### Prerequisites
1. **CMake** installed and added to your system PATH.
2. **MinGW-w64** (with `g++` and `mingw32-make`) installed and added to your system PATH.
3. **OpenCV 4.x** installed and configured for MinGW.

### Compilation Steps

1. Open a terminal (PowerShell or Command Prompt) in the root of this project directory.
2. Create a new `build` directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
3. Generate the Makefiles using CMake (specifying the MinGW generator):
   ```bash
   cmake -G "MinGW Makefiles" ..
   ```
4. Compile the executables:
   ```bash
   mingw32-make
   ```
   *(Alternatively, you can compile a specific module, e.g., `mingw32-make demo_step3_optimization`)*

### Execution
Once compiled, you can run the generated `.exe` files directly from the `build` directory:
```bash
.\demo_step3_optimization.exe
```

*(Note: The `Train_set` video data must be located in the expected directory structure relative to the executable for the demonstrations to run correctly).*
