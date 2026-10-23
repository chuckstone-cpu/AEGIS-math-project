cmake_minimum_required(VERSION 3.16)
project(ApexEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 1. AUTO-DETECT OPENCV ON WINDOWS IF OpenCV_DIR IS NOT SET
# ============================================================================
if(NOT DEFINED OpenCV_DIR AND NOT DEFINED ENV{OpenCV_DIR})
    # Search common Windows OpenCV installation directories
    file(GLOB OPENCV_SEARCH_PATHS
        "C:/opencv_mingw/x64/mingw/staticlib"
        "C:/opencv_mingw"
        "C:/opencv/build"
        "C:/opencv/build/x64/vc*/lib"
        "C:/opencv/build/x64/mingw/lib"
        "C:/tools/opencv/build"
        "C:/Program Files/opencv/build"
        "C:/vcpkg/installed/x64-windows/share/opencv*"
        "C:/msys64/mingw64/lib/cmake/opencv4"
        "$ENV{USERPROFILE}/opencv/build"
        "$ENV{USERPROFILE}/Downloads/opencv/build"
    )
    find_path(AUTO_OPENCV_DIR
        NAMES OpenCVConfig.cmake opencv-config.cmake
        PATHS ${OPENCV_SEARCH_PATHS}
        PATH_SUFFIXES
            x64/mingw/staticlib
            x64/mingw/lib
            x64/vc17/lib
            x64/vc16/lib
            x64/vc15/lib
            lib/cmake/opencv4
            share/opencv4
            ""
    )
    if(AUTO_OPENCV_DIR)
        set(OpenCV_DIR "${AUTO_OPENCV_DIR}" CACHE PATH "Auto-detected OpenCV directory" FORCE)
        message(STATUS "[ApexEngine] Auto-detected OpenCV_DIR: ${OpenCV_DIR}")
    endif()
endif()

find_package(OpenCV REQUIRED)
message(STATUS "[ApexEngine] Found OpenCV ${OpenCV_VERSION} at: ${OpenCV_DIR}")

# ============================================================================
# 2. FIND OPENMP
# ============================================================================
find_package(OpenMP)

# ============================================================================
# 3. LOCATE WEBOTS INSTALLATION (Default: C:/Program Files/Webots)
# ============================================================================
if(DEFINED ENV{WEBOTS_HOME})
    file(TO_CMAKE_PATH "$ENV{WEBOTS_HOME}" WEBOTS_HOME)
elseif(EXISTS "C:/Program Files/Webots")
    set(WEBOTS_HOME "C:/Program Files/Webots")
endif()

add_executable(apex_engine apex_engine.cpp)

target_include_directories(apex_engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(apex_engine PRIVATE ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(apex_engine PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(apex_engine PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(apex_engine PRIVATE -O3 -march=native -ffast-math)
endif()

# ============================================================================
# 4. LINK WEBOTS CONTROLLER & DRIVER LIBRARIES
# ============================================================================
if(DEFINED WEBOTS_HOME AND EXISTS "${WEBOTS_HOME}/include/controller/c")
    message(STATUS "[ApexEngine] Found Webots at: ${WEBOTS_HOME}")
    target_compile_definitions(apex_engine PRIVATE USE_WEBOTS)
    target_include_directories(apex_engine PRIVATE "${WEBOTS_HOME}/include/controller/c")

    find_library(WEBOTS_CONTROLLER_LIB
        NAMES Controller libController
        PATHS "${WEBOTS_HOME}/lib/controller"
        NO_DEFAULT_PATH
    )
    find_library(WEBOTS_DRIVER_LIB
        NAMES driver libdriver vehicle libvehicle
        PATHS "${WEBOTS_HOME}/lib/controller"
        NO_DEFAULT_PATH
    )

    if(WEBOTS_CONTROLLER_LIB AND WEBOTS_DRIVER_LIB)
        target_link_libraries(apex_engine PRIVATE "${WEBOTS_CONTROLLER_LIB}" "${WEBOTS_DRIVER_LIB}")
    else()
        target_link_directories(apex_engine PRIVATE "${WEBOTS_HOME}/lib/controller")
        target_link_libraries(apex_engine PRIVATE Controller driver)
    endif()

    # Copy Webots DLLs next to apex_engine.exe
    if(WIN32)
        file(GLOB WEBOTS_DLLS "${WEBOTS_HOME}/lib/controller/*.dll")
        foreach(DLL_FILE ${WEBOTS_DLLS})
            add_custom_command(TARGET apex_engine POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL_FILE}"
                "$<TARGET_FILE_DIR:apex_engine>"
            )
        endforeach()
    endif()
else()
    message(WARNING "[ApexEngine] Webots not found! Building in Video-Only mode.")
endif()

# Copy OpenCV DLLs next to apex_engine.exe so it runs without PATH issues
if(WIN32 AND DEFINED OpenCV_DIR)
    get_filename_component(OPENCV_LIB_PARENT "${OpenCV_DIR}" DIRECTORY)
    file(GLOB OPENCV_DLLS
        "${OpenCV_DIR}/../bin/*.dll"
        "${OPENCV_LIB_PARENT}/bin/*.dll"
        "C:/opencv/build/x64/vc*/bin/*.dll"
        "C:/opencv_mingw/x64/mingw/bin/*.dll"
    )
    foreach(CV_DLL ${OPENCV_DLLS})
        add_custom_command(TARGET apex_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CV_DLL}"
            "$<TARGET_FILE_DIR:apex_engine>"
        )
    endforeach()
endif()