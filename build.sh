#!/usr/bin/env bash

# Build script for CUDA + OpenGL fluid simulator
# Supports: Linux (primary), macOS (very limited CUDA support)
set -euo pipefail

BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
CLR=$'\033[2K' # clear the line
OK="✓"; BAD="✗"

say()  { printf "${BOLD}%s${RESET}\n" "$*"; }
pass() { printf "%s %s\n" "$OK" "$*"; }
fail() { printf "%s %s\n" "$BAD" "$*" >&2; exit 1; }
info() { printf "%s\n" "$*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGET="${TARGET:-fluid}"
LOG_DIR="${BUILD_DIR}/logs"

# detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

# clean up
rm -rf "$BUILD_DIR"
rm -f imgui.ini
mkdir -p "$LOG_DIR"

# status updates
run_config_one_line() {
  local msg="$1" logfile="$2"; shift 2
  say "$msg"
  : > "$logfile"

  set +e
  ("$@") 2>&1 | tee "$logfile" | awk -v dim="$DIM" -v reset="$RESET" -v clr="$CLR" -v ok="$OK" '
    function draw(s){ printf("\r%s%s%s%s", clr, dim, s, reset); fflush(); }
    BEGIN{ last=""; }
    /^-- / {
      line=$0; sub(/^-- /,"",line);
      last=line;
      draw(last);
      next
    }
    /CMake Error|fatal error|error:|undefined reference|ld: / {
      print ""; print $0; fflush();
      if(last!="") draw(last);
      next
    }
    END{
      if(last!=""){ draw(last " " ok); print ""; fflush(); }
    }
  ' &
  local awk_pid=$!

  # a spinner to appease the impatient user
  local spin="|/-\\" i=0
  while kill -0 "$awk_pid" 2>/dev/null; do
    local ch=${spin:i%${#spin}:1}
    printf "\r%s%s %s%s" "$CLR" "$DIM$ch Working..." "$RESET"
    sleep 0.1
    ((i++))
  done
  
  wait "$awk_pid"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    printf "\n"
    say "Configure failed (tail):"
    tail -n 140 "$logfile" >&2
    info ""
    info "Full log: $logfile"
    exit $rc
  fi
}

# one-line status updates; this one is for building
run_build_one_line() {
  local msg="$1" logfile="$2"; shift 2
  say "$msg"
  : > "$logfile"

  set +e
  ("$@") 2>&1 | tee "$logfile" | awk -v dim="$DIM" -v reset="$RESET" -v clr="$CLR" -v ok="$OK" '
    function draw(s){ printf("\r%s%s%s%s", clr, dim, s, reset); fflush(); }
    BEGIN{ last=""; }
    /^\[[0-9]+\/[0-9]+\]/ {
      last=$0;
      draw(last);
      next
    }
    /CMake Error|fatal error|error:|undefined reference|ld: / {
      print ""; print $0; fflush();
      if(last!="") draw(last);
      next
    }
    END{
      if(last!=""){ draw(last " " ok); print ""; fflush(); }
    }
  '
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    printf "\n"
    say "Build failed (tail):"
    tail -n 180 "$logfile" >&2
    info ""
    info "Full log: $logfile"
    exit $rc
  fi
}

say "Checking dependencies for ${OS}…"
command -v cmake >/dev/null || fail "cmake not found (install via apt/brew)"
command -v git  >/dev/null || fail "git not found"

# check for CUDA
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    pass "CUDA Toolkit: $CUDA_VERSION"
else
    fail "CUDA Toolkit not found (nvcc missing). Install CUDA 11.8 or later."
fi

# platform-specific checks
if [[ "$OS" == "linux" ]]; then
    if command -v gcc >/dev/null && command -v g++ >/dev/null; then
        GCC_VERSION=$(gcc --version | head -n1)
        pass "Compiler: $GCC_VERSION"
    else
        fail "GCC/G++ not found (install build-essential)"
    fi
    
    if ! ldconfig -p | grep -q libGL.so; then
        fail "OpenGL libraries not found (install libgl1-mesa-dev)"
    fi
    if ! ldconfig -p | grep -q libGLEW.so; then
        fail "GLEW not found (install libglew-dev)"
    fi
    pass "OpenGL + GLEW: OK"
    
elif [[ "$OS" == "macos" ]]; then
    info "macOS detected: CUDA support is deprecated on macOS"
    info "For full CUDA support, use Linux"
    
    [[ -x /usr/bin/clang ]]   || fail "Apple clang missing (install Xcode CLI Tools)"
    [[ -x /usr/bin/clang++ ]] || fail "Apple clang++ missing"
    pass "Compiler: Apple Clang"
fi

# generator: use ninja if present
GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-G Ninja)
    pass "Generator: Ninja"
else
    pass "Generator: Unix Makefiles"
fi

# pretty status format
export NINJA_STATUS="[%f/%t] "

CMAKE_ARGS=(
    -S .
    -B "$BUILD_DIR"
    "${GEN_ARGS[@]}"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    --log-level=WARNING
    -Wno-dev
)

if [[ "$OS" == "linux" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER=gcc
        -DCMAKE_CXX_COMPILER=g++
    )
fi

# configure
run_config_one_line "Configuring CUDA + OpenGL build…" "${LOG_DIR}/configure.log" \
    cmake "${CMAKE_ARGS[@]}"

# dependencies
run_build_one_line "Building dependencies (GLFW, ImGui)…" "${LOG_DIR}/deps.log" \
    cmake --build "$BUILD_DIR" --target glfw imgui_lib --parallel

# main targer
run_build_one_line "Building CUDA fluid simulator…" "${LOG_DIR}/app.log" \
    cmake --build "$BUILD_DIR" --target "$TARGET" --parallel

pass "Build successful!"
echo ""
say "To run:"
info "  $ ./${BUILD_DIR}/${TARGET}"
echo ""
info "System info:"
info "  OS: ${OS}"
if [[ "$OS" == "linux" ]]; then
    info "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
fi
