#!/usr/bin/env bash

# build script, edit at your own risk!!! :)
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

# clean up first
rm -rf "$BUILD_DIR"
rm -f imgui.ini
mkdir -p "$LOG_DIR"

# helper to run commands with one-line status updates; this one is for Configuration
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

# helper to run commands with one-line status updates; this one is for Building
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

# depenency checks ...
say "Checking dependencies…"
command -v cmake >/dev/null || fail "cmake not found"
command -v git  >/dev/null || fail "git not found (FetchContent needs it)"
[[ -x /usr/bin/clang ]]   || fail "Apple clang missing (install Xcode Command Line Tools)"
[[ -x /usr/bin/clang++ ]] || fail "Apple clang++ missing (install Xcode Command Line Tools)"
pass "toolchain OK"

# OpenMP optional (libomp)
OPENMP_FLAG=OFF
OMP_ARGS=()
if command -v brew >/dev/null 2>&1 && brew list --formula libomp >/dev/null 2>&1; then
  PREFIX="$(brew --prefix libomp)"
  OPENMP_FLAG=ON
  OMP_ARGS+=(
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${PREFIX}/include"
    -DOpenMP_CXX_LIB_NAMES=omp
    -DOpenMP_omp_LIBRARY="${PREFIX}/lib/libomp.dylib"
  )
  pass "OpenMP: ON"
else
  pass "OpenMP: OFF"
fi

# generator: uses ninja if present, otherwise system default
GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GEN_ARGS=(-G Ninja)
fi

# pretty status format
export NINJA_STATUS="[%f/%t] "

# configuration step (this is where flags are set)
run_config_one_line "Configuring graphics… (please be patient!)" "${LOG_DIR}/configure.log" \
  env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
  cmake -S . -B "$BUILD_DIR" \
    "${GEN_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DFLUID_ENABLE_OPENMP="$OPENMP_FLAG" \
    "${OMP_ARGS[@]}" \
    --log-level=WARNING -Wno-dev

# building both the dependencies and the main target
run_build_one_line "Building dependencies…" "${LOG_DIR}/deps.log" \
  cmake --build "$BUILD_DIR" --target glfw imgui_lib --parallel

run_build_one_line "Building executable…" "${LOG_DIR}/app.log" \
  cmake --build "$BUILD_DIR" --target "$TARGET" --parallel

pass "Success! to run executable: $ ./${BUILD_DIR}/${TARGET}"
