cmake_minimum_required(VERSION 3.10)
project(ProjectApex)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the path to our custom compiled OpenCV for MinGW
set(OpenCV_DIR "C:/opencv_mingw")
set(OpenCV_STATIC ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Executable for Step 1
add_executable(demo_step1_ingestion demo_step1_ingestion.cpp)
target_link_libraries(demo_step1_ingestion ${OpenCV_LIBS})

# Executable for Step 2
add_executable(demo_step2_pdesmoothing demo_step2_pdesmoothing.cpp)
target_link_libraries(demo_step2_pdesmoothing ${OpenCV_LIBS})

# Executable for Step 3
add_executable(demo_step3_optimization demo_step3_optimization.cpp)
target_link_libraries(demo_step3_optimization ${OpenCV_LIBS})
