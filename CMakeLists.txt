cmake_minimum_required(VERSION 3.10)
project(ProjectApex)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funroll-loops")

# Find OpenCV, you may need to set OpenCV_DIR if CMake cannot find it.
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(apex_engine apex_engine.cpp)
target_link_libraries(apex_engine ${OpenCV_LIBS})
