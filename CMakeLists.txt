cmake_minimum_required(VERSION 3.16)
project(ApexEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(APEX_NO_GUI "Build without an OpenCV window (record/snap only)" OFF)

# ============================================================================
# 1. MATHS ENGINE (header-only, no external maths library) + its test suite
# ============================================================================
add_library(apex_math INTERFACE)
target_include_directories(apex_math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math PRIVATE apex_math)
enable_testing()
add_test(NAME apex_math_tests COMMAND test_math)

# ============================================================================
# 2. OPENCV (UI, video I/O and drawing only) - auto-detect common Windows paths
# ============================================================================
if(NOT DEFINED OpenCV_DIR AND NOT DEFINED ENV{OpenCV_DIR})
    find_path(AUTO_OPENCV_DIR
        NAMES OpenCVConfig.cmake opencv-config.cmake
        PATHS
            "C:/opencv_mingw" "C:/opencv/build" "C:/tools/opencv/build"
            "C:/Program Files/opencv/build" "C:/msys64/ucrt64" "C:/msys64/mingw64"
            "$ENV{USERPROFILE}/opencv/build"
        PATH_SUFFIXES
            x64/mingw/staticlib x64/mingw/lib x64/vc17/lib x64/vc16/lib
            lib/cmake/opencv4 share/opencv4 ""
    )
    if(AUTO_OPENCV_DIR)
        set(OpenCV_DIR "${AUTO_OPENCV_DIR}" CACHE PATH "Auto-detected OpenCV" FORCE)
    endif()
endif()
find_package(OpenCV REQUIRED)
message(STATUS "[ApexEngine] OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")
find_package(OpenMP)

add_executable(apex_engine apex_engine.cpp)
target_include_directories(apex_engine PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(apex_engine PRIVATE apex_math ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(apex_engine PRIVATE OpenMP::OpenMP_CXX)
endif()
if(APEX_NO_GUI)
    target_compile_definitions(apex_engine PRIVATE APEX_NO_GUI)
endif()
if(MSVC)
    target_compile_options(apex_engine PRIVATE /O2 /arch:AVX2)
else()
    # no -ffast-math: the solvers rely on IEEE isfinite/NaN checks
    target_compile_options(apex_engine PRIVATE -O3 -march=native)
endif()

# ============================================================================
# 3. WEBOTS (optional): controller + car library
# ============================================================================
if(DEFINED ENV{WEBOTS_HOME})
    file(TO_CMAKE_PATH "$ENV{WEBOTS_HOME}" WEBOTS_HOME)
elseif(EXISTS "C:/Program Files/Webots")
    set(WEBOTS_HOME "C:/Program Files/Webots")
endif()

if(DEFINED WEBOTS_HOME AND EXISTS "${WEBOTS_HOME}/include/controller/c")
    message(STATUS "[ApexEngine] Webots at ${WEBOTS_HOME}")
    target_compile_definitions(apex_engine PRIVATE USE_WEBOTS)
    target_include_directories(apex_engine PRIVATE "${WEBOTS_HOME}/include/controller/c")
    find_library(WEBOTS_CONTROLLER_LIB NAMES Controller libController
                 PATHS "${WEBOTS_HOME}/lib/controller" NO_DEFAULT_PATH)
    find_library(WEBOTS_DRIVER_LIB NAMES driver libdriver car libcar
                 PATHS "${WEBOTS_HOME}/lib/controller" NO_DEFAULT_PATH)
    if(WEBOTS_CONTROLLER_LIB AND WEBOTS_DRIVER_LIB)
        target_link_libraries(apex_engine PRIVATE "${WEBOTS_CONTROLLER_LIB}" "${WEBOTS_DRIVER_LIB}")
    else()
        target_link_directories(apex_engine PRIVATE "${WEBOTS_HOME}/lib/controller")
        target_link_libraries(apex_engine PRIVATE Controller driver)
    endif()
    # Deploy into the Webots project: webots/controllers/apex_engine/
    get_filename_component(_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_qt_plugins "${_cxx_bin}/../share/qt6/plugins")
    add_custom_command(TARGET apex_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DEXE=$<TARGET_FILE:apex_engine>
            "-DDEST=${CMAKE_CURRENT_SOURCE_DIR}/webots/controllers/apex_engine"
            "-DSEARCH_DIRS=${_cxx_bin};${OpenCV_DIR}/../bin"
            "-DQT_PLUGINS=${_qt_plugins}"
            -Wno-dev -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deploy_controller.cmake"
        VERBATIM)

    # Scenario supervisor (plain C, Webots Supervisor API)
    add_executable(scenario_supervisor webots/controllers/scenario_supervisor/scenario_supervisor.c)
    target_include_directories(scenario_supervisor PRIVATE "${WEBOTS_HOME}/include/controller/c")
    target_link_libraries(scenario_supervisor PRIVATE "${WEBOTS_CONTROLLER_LIB}")
    set_target_properties(scenario_supervisor PROPERTIES RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/webots/controllers/scenario_supervisor")
else()
    message(WARNING "[ApexEngine] Webots not found: building video-only mode.")
endif()

# Copy OpenCV DLLs next to apex_engine.exe so it runs without PATH changes
if(WIN32 AND DEFINED OpenCV_DIR)
    file(GLOB OPENCV_DLLS
        "${OpenCV_DIR}/../bin/*.dll"
        "C:/opencv_mingw/x64/mingw/bin/*.dll")
    foreach(CV_DLL ${OPENCV_DLLS})
        add_custom_command(TARGET apex_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CV_DLL}"
                    "$<TARGET_FILE_DIR:apex_engine>")
    endforeach()
endif()
