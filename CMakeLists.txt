cmake_minimum_required(VERSION 3.20)

project(cuda-fluids CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    message(STATUS "default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

include(FetchContent)

# GLFW
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# OpenGL loader
find_package(GLEW)
if(NOT GLEW_FOUND)
    # try pkg-config
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GLEW REQUIRED glew)
    else()
        message(FATAL_ERROR "GLEW not found. Install with: apt install libglew-dev (Linux) or brew install glew (macOS)")
    endif()
endif()

# ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.8
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a static lib
add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${GLEW_INCLUDE_DIRS}
)
target_link_libraries(imgui_lib PUBLIC glfw ${GLEW_LIBRARIES})

# OpenGL
find_package(OpenGL REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# main executable
add_executable(fluid
  src/main.cpp
  src/fluid.cu
)

target_include_directories(fluid PRIVATE 
  src
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(fluid PRIVATE
  imgui_lib
  ${GLEW_LIBRARIES}
  OpenGL::GL
  CUDA::cudart
)

set_target_properties(fluid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA optimization flags
target_compile_options(fluid PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extra-device-vectorization
        -lineinfo
    >
)

# Release build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(fluid PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --ptxas-options=-v
        >
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(fluid PRIVATE dl pthread)
    set_target_properties(fluid PROPERTIES
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

message(STATUS "architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA ver: ${CUDAToolkit_VERSION}")
