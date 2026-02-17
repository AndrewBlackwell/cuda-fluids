cmake_minimum_required(VERSION 3.20)

# macOS note: CMake initializes compiler flags from environment variables like CFLAGS/CXXFLAGS. 
# if your shell has `-fopenmp` in CXXFLAGS, AppleClang will fail.
# clear init flags BEFORE enabling languages (before `project()`).
if(APPLE)
  set(CMAKE_C_FLAGS_INIT "")
  set(CMAKE_CXX_FLAGS_INIT "")
endif()

project(fluid_mac CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS OpenGL deprecation silencer
add_compile_definitions(GL_SILENCE_DEPRECATION)

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

# ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.8
)
FetchContent_MakeAvailable(imgui)

# build ImGui as a static lib
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
)
target_link_libraries(imgui_lib PUBLIC glfw)

# OpenGL
find_package(OpenGL REQUIRED)

# main app target: executable stored at ./build/fluid
add_executable(fluid
  src/main.cpp
  src/fluid.cpp
)

target_include_directories(fluid PRIVATE src)

target_link_libraries(fluid PRIVATE
  imgui_lib
  OpenGL::GL
)

# OpenMP (optional, reccommended for performance on large simulations)
option(FLUID_ENABLE_OPENMP "Enable OpenMP (requires libomp on macOS)" ON)

if(FLUID_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled")
    target_link_libraries(fluid PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(fluid PRIVATE FLUID_USE_OPENMP=1)
  else()
    message(WARNING "OpenMP requested but not found; building without OpenMP")
  endif()
else()
  message(STATUS "OpenMP disabled (default)")
endif()
